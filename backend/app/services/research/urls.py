"""URL normalisation and safety checks for retrieved web material.

Every URL the Researcher sees originates from a search engine, which
means every URL is *untrusted input*. This module is the boundary that
decides what may be fetched.

Why this exists
---------------
The Researcher accepts arbitrary search results and requests them
server-side. Without checks that turns a public API into a server-side
request forgery gadget: a caller could ask for ``http://127.0.0.1:11434``
and reach the local Ollama server, or ``http://192.168.1.1/admin`` and
reach the router. A redirect is equally dangerous, because an allowed URL
can redirect to a disallowed one.

Defences applied, in order
--------------------------
1. **Scheme allowlist.** Only ``http`` and ``https``. This alone rejects
   ``file://``, ``data://``, ``javascript://``, ``ftp://`` and friends.
2. **Host checks.** Rejects ``localhost``, ``*.localhost``, ``*.local``
   (mDNS), bare hostnames without a dot, and the IPv6 unspecified
   address.
3. **DNS resolution and IP inspection.** The hostname is resolved and
   *every* returned address is checked against the private, loopback,
   link-local, reserved and multicast ranges. All addresses are checked,
   not just the first, because a name resolving to both a public and a
   private address is exactly the rebinding case worth refusing.
4. **Per-hop revalidation.** :class:`~app.services.research.fetching.http.HTTPSourceFetcher`
   re-runs these checks on every redirect target.

Nothing here fetches anything. The checks are pure and synchronous so
they can be unit-tested without a network.

Documented limitations
----------------------
* Resolution is a best-effort snapshot. A hostname could resolve publicly
  here and privately later, so this reduces risk rather than eliminating
  it. Eliminating it entirely requires pinning the resolved address into
  the connection, which is out of scope here.
* There is no allowlist of trusted publishers yet; every public host is
  permitted, which is what makes general research possible.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from app.core.logging import get_logger

logger = get_logger("research.urls")

#: Only these schemes may ever be fetched. Everything else, including
#: ``file``, ``data``, ``javascript`` and ``ftp``, is refused outright.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Query parameters that only exist to track a click or a campaign. They
#: are dropped when normalising, so two search results pointing at the
#: same article deduplicate against each other.
TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "spm",
        "yclid",
    }
)

#: Parameter *prefixes* that mark analytics or campaign tracking. Matched
#: as prefixes rather than enumerated, because the ``utm_`` family grows
#: every year and listing today's members would silently miss tomorrow's.
TRACKING_PARAM_PREFIXES = ("utm_", "pk_", "mtm_", "hsa_", "_hs")

#: Host names that never refer to a public site. Checked before DNS so a
#: blocked lookup does not cost a round trip.
BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)

#: Suffixes reserved for local networks: mDNS (``.local``) and the
#: Kubernetes internal service domain.
BLOCKED_SUFFIXES = (".local", ".localhost", ".internal", ".home.arpa")


class UnsafeURLError(ValueError):
    """A URL must not be fetched.

    Raised before any connection is attempted. The message names the
    reason, which is useful in logs; callers that surface an error to a
    client should use their own generic wording instead.
    """

    code = "unsafe_url"


#: Resolves a hostname to a list of textual IP addresses. Injected in
#: tests so the safety checks can be exercised without DNS.
HostResolver = Callable[[str], Iterable[str]]


# ---------- Normalisation ----------


def normalize_url(url: str) -> str:
    """Return a canonical form of ``url`` for comparison and storage.

    Lowercases the scheme and host, drops a leading ``www.``, removes the
    fragment, removes tracking parameters and drops a trailing slash from
    the path. Query parameters that carry meaning are preserved and
    sorted, since ``?b=2&a=1`` and ``?a=1&b=2`` are the same resource.

    The result is a valid absolute URL: the normalised form is what gets
    fetched and stored, so a caller sees the URL that was actually used.
    Raises :class:`UnsafeURLError` for anything that is not an absolute
    http(s) URL.
    """
    parts = urlsplit(url.strip())

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Unsupported URL scheme {parts.scheme!r}.")

    if not parts.netloc:
        raise UnsafeURLError("URL has no host.")

    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURLError("URL has no host.")

    if host.startswith("www."):
        host = host[4:]

    path = parts.path or "/"
    # A trailing slash is not a different resource, except for the root.
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if not _is_tracking_param(key)
    ]
    query = urlencode(sorted(query_pairs), doseq=True)

    netloc = host
    # An explicit port is only meaningful when it is not the scheme's
    # default, so ``https://a:443`` and ``https://a`` compare equal.
    default_port = {"http": 80, "https": 443}.get(parts.scheme.lower())
    if parts.port is not None and parts.port != default_port:
        netloc = f"{host}:{parts.port}"

    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def domain_of(url: str) -> str:
    """Host part of ``url``, lowercased and without a leading ``www.``.

    Used as the unit of source independence when grading confidence, so
    it deliberately ignores the scheme, port and path: ten articles on one
    site are one voice, not ten.
    """
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def registrable_domain(url: str) -> str:
    """A domain grouped well enough to spot mirrors.

    Crude on purpose: it keeps the last two labels (``news.bbc.co.uk``
    becomes ``co.uk``, which over-groups) because a full public-suffix list
    is a large dependency for a deduplication hint. Only used for
    near-duplicate detection, never for security.
    """
    host = domain_of(url)
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    return ".".join(labels[-2:])


def build_search_url(url: str) -> str:
    """Re-encode a path for a URL that has been rewritten by a redirect.

    Redirect targets frequently contain unescaped spaces. Encoding the
    path keeps the URL parseable rather than letting the client guess.
    """
    parts = urlsplit(url)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            quote(parts.path, safe="/%"),
            parts.query,
            "",
        )
    )


# ---------- Safety ----------


def assert_structurally_safe_url(url: str) -> str:
    """Validate ``url`` without touching the network, and normalise it.

    This is the half of :func:`assert_safe_url` that needs no name
    resolution: scheme, host shape, and IP literals (which are inspected
    directly, since there is nothing to look up).

    It exists because a caller on the event loop must be able to filter a
    list of candidates cheaply. Deduplication filters up to a dozen search
    results per run, and resolving every hostname there would make a
    blocking ``getaddrinfo`` call the price of a tidiness step. Any host
    that passes this check and is not an IP literal is still unverified
    until the fetcher resolves it, which is why the fetcher re-checks
    rather than trusting this result.

    Raises:
        UnsafeURLError: for a non-http(s) scheme, a local or reserved
            host name, a host that is not fully qualified, or an IP
            literal in a non-public range.
    """
    normalised = normalize_url(url)
    parts = urlsplit(normalised)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()

    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Unsupported URL scheme {scheme!r}.")

    if host in BLOCKED_HOSTS:
        raise UnsafeURLError(f"Host {host!r} is a local address.")

    if any(host.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
        raise UnsafeURLError(f"Host {host!r} is in a reserved local domain.")

    # A bare hostname such as "intranet" has no public meaning; requiring a
    # dot keeps search results on the public internet.
    if "." not in host and not _is_ip_literal(host):
        raise UnsafeURLError(f"Host {host!r} is not a fully qualified domain name.")

    if _is_ip_literal(host):
        reason = _ip_rejection_reason(host)
        if reason is not None:
            raise UnsafeURLError(f"Host {host!r} is {reason}.")

    return normalised


def assert_safe_url(url: str, *, resolver: HostResolver | None = None) -> str:
    """Validate ``url`` for fetching and return its normalised form.

    Applies, in order: the structural checks of
    :func:`assert_structurally_safe_url`, then IP inspection of every
    address the hostname resolves to.

    Args:
        url: the candidate URL.
        resolver: override for hostname resolution, used by tests.

    Raises:
        UnsafeURLError: with a message naming the specific reason. The
            reason is for logs; do not forward it verbatim to a client,
            since it describes the server's own network layout.
    """
    normalised = assert_structurally_safe_url(url)
    host = (urlsplit(normalised).hostname or "").lower()

    addresses = _resolve(host, resolver=resolver)

    # Fail closed. A name that resolves to nothing was never shown to be
    # public, and accepting it because the inspection loop had nothing to
    # look at would silently allow whatever the resolver would return
    # later. Refusing is the conservative reading.
    if not addresses:
        raise UnsafeURLError(
            f"Host {host!r} did not resolve to any address, so it cannot be "
            "confirmed as a public source."
        )

    for address in addresses:
        reason = _ip_rejection_reason(address)
        if reason is not None:
            raise UnsafeURLError(
                f"Host {host!r} resolves to {address}, which is {reason}."
            )

    return normalised


def is_safe_url(url: str, *, resolver: HostResolver | None = None) -> bool:
    """Whether ``url`` may be fetched. Convenient for filtering lists."""
    try:
        assert_safe_url(url, resolver=resolver)
    except UnsafeURLError:
        return False
    return True


# ---------- Internals ----------


def _resolve(host: str, *, resolver: HostResolver | None) -> list[str]:
    """Resolve ``host`` to IP addresses, returning ``[]`` when it is literal.

    An IP literal needs no lookup and is inspected directly. DNS failures
    return an empty list, which the caller treats as a refusal: see the
    fail-closed check in :func:`assert_safe_url`.
    """
    if _is_ip_literal(host):
        return [host]

    lookup = resolver or default_resolver

    try:
        return [str(address) for address in lookup(host)]
    except (OSError, UnicodeError) as exc:
        logger.info("dns_resolution_failed host=%s error=%s", host, type(exc).__name__)
        return []


def default_resolver(host: str) -> list[str]:
    """Resolve ``host`` with the system resolver.

    Wrapped in a helper so it can be swapped out wholesale, and so the
    blocking ``getaddrinfo`` call is visible to callers that must move it
    off the event loop.
    """
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [info[4][0] for info in infos]


def _is_ip_literal(host: str) -> bool:
    """Whether ``host`` is already an IP address rather than a name."""
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def _ip_rejection_reason(address: str) -> str | None:
    """Describe why ``address`` must not be contacted, or ``None`` if it may.

    Order matters, because Python's classifications overlap: ``0.0.0.0``
    is both *unspecified* and *private*, and reporting it as private would
    describe it inaccurately. The more specific classification is checked
    first so the message names what is actually true.
    """
    try:
        ip = ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return "not a valid IP address"

    if ip.is_unspecified:
        # Covers both 0.0.0.0 and ::.
        return "an unspecified address"
    if ip.is_loopback:
        return "a loopback address"
    if ip.is_link_local:
        # 169.254.169.254 is the cloud metadata service, so this check is
        # the one that stops credential theft against a cloud deployment.
        return "a link-local address"
    if ip.is_private:
        return "a private network address"
    if ip.is_multicast:
        return "a multicast address"
    if ip.is_reserved:
        return "a reserved address"

    # Catch-all. Special-purpose ranges that are neither public nor covered
    # by a specific flag above are refused too: carrier-grade NAT
    # (100.64.0.0/10) is the live example, where Python reports
    # ``is_private`` False *and* ``is_global`` False. Requiring an address
    # to be globally routable closes that gap without having to enumerate
    # every reserved block by hand.
    if not ip.is_global:
        return "not a publicly routable address"

    return None


def _is_tracking_param(name: str) -> bool:
    """Whether a query parameter is analytics or campaign tracking."""
    lowered = name.lower()
    return lowered in TRACKING_PARAMS or lowered.startswith(TRACKING_PARAM_PREFIXES)

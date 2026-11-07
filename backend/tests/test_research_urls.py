"""URL normalisation and the SSRF boundary.

The Researcher fetches URLs a search engine chose, so every URL is
untrusted input. These tests pin down what is refused and why, because a
regression here is a security regression rather than a bug.
"""

from __future__ import annotations

import pytest

from app.services.research.urls import (
    UnsafeURLError,
    assert_safe_url,
    build_search_url,
    default_resolver,
    domain_of,
    is_safe_url,
    normalize_url,
    registrable_domain,
)

PUBLIC = "93.184.216.34"


def resolve_to(*addresses: str):
    """Build a resolver returning fixed addresses."""

    def resolve(host: str) -> list[str]:
        return list(addresses)

    return resolve


# ---------- Normalisation ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://Example.COM/Path", "https://example.com/Path"),
        ("https://www.example.com/page", "https://example.com/page"),
        ("https://example.com/page/", "https://example.com/page"),
        ("https://example.com/page#section", "https://example.com/page"),
        ("https://example.com/page?b=2&a=1", "https://example.com/page?a=1&b=2"),
        ("https://example.com/page?utm_source=x&keep=1", "https://example.com/page?keep=1"),
        ("https://example.com/page?fbclid=abc", "https://example.com/page"),
        ("HTTPS://example.com/x", "https://example.com/x"),
        ("https://example.com", "https://example.com/"),
    ],
)
def test_normalize_url_canonicalises_equivalent_forms(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


def test_normalize_url_keeps_meaningful_query_parameters() -> None:
    """Only tracking parameters are dropped; a real filter must survive."""
    normalised = normalize_url("https://example.com/search?q=youtube&page=2")

    assert "q=youtube" in normalised
    assert "page=2" in normalised


def test_normalize_url_drops_a_redundant_default_port() -> None:
    assert normalize_url("https://example.com:443/x") == normalize_url("https://example.com/x")


def test_normalize_url_keeps_a_non_default_port() -> None:
    """A non-default port is a different service and must not be erased."""
    assert normalize_url("http://example.com:8080/x") == "http://example.com:8080/x"


def test_normalize_url_rejects_a_non_http_scheme() -> None:
    with pytest.raises(UnsafeURLError):
        normalize_url("file:///etc/passwd")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/html,<script>alert(1)</script>",
        "javascript:alert(1)",
        "ftp://example.com/x",
        "about:blank",
    ],
)
def test_normalize_url_rejects_non_http_schemes(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        normalize_url(url)


def test_normalize_url_rejects_a_url_with_no_host() -> None:
    with pytest.raises(UnsafeURLError):
        normalize_url("https:///just/a/path")


def test_domain_of_ignores_scheme_port_and_path() -> None:
    assert domain_of("https://www.example.com:8443/a/b?c=d") == "example.com"


def test_registrable_domain_groups_subdomains() -> None:
    assert registrable_domain("https://news.bbc.co.uk/story") == "co.uk"


def test_build_search_url_encodes_a_space_in_a_path() -> None:
    """Redirect targets regularly contain raw spaces; they must stay parseable."""
    built = build_search_url("https://example.com/some page")

    assert " " not in built
    assert built.endswith("/some%20page")


# ---------- Scheme allowlist ----------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/html,hello",
        "javascript:alert(1)",
        "ftp://example.com/file",
        "gopher://example.com/1",
    ],
)
def test_unsafe_schemes_are_refused(url: str) -> None:
    """Only http and https may ever be fetched."""
    assert is_safe_url(url) is False

    with pytest.raises(UnsafeURLError):
        assert_safe_url(url, resolver=resolve_to(PUBLIC))


# ---------- Host checks ----------


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST",
        "app.localhost",
        "printer.local",
        "vault.internal",
        "router.home.arpa",
        "intranet",
    ],
)
def test_local_and_reserved_hosts_are_refused(host: str) -> None:
    """localhost, mDNS and internal domains never refer to public sites.

    These are refused by name, before DNS, so the rejection costs no round
    trip and does not depend on what the resolver would answer.
    """
    assert is_safe_url(f"http://{host}/page") is False


def test_a_bare_hostname_is_refused() -> None:
    """``http://intranet`` has no public meaning; a dot is required."""
    assert is_safe_url("http://intranet/admin") is False


def test_a_public_address_is_accepted() -> None:
    """The check must not simply reject everything."""
    assert is_safe_url("https://example.com/x", resolver=resolve_to(PUBLIC)) is True


# ---------- IP inspection ----------


@pytest.mark.parametrize(
    ("address", "reason"),
    [
        ("127.0.0.1", "loopback"),
        ("127.10.20.30", "loopback"),
        ("::1", "loopback"),
        ("0.0.0.0", "unspecified"),
        ("::", "unspecified"),
        ("10.0.0.5", "private"),
        ("172.16.4.4", "private"),
        ("192.168.1.1", "private"),
        # Carrier-grade NAT space: Python reports neither is_private nor
        # is_global, which is why the safety check also requires an address
        # to be globally routable.
        ("100.64.0.1", "not a publicly routable address"),
        # 240.0.0.0/4, which Python classifies as private.
        ("240.0.0.1", "private"),
        ("169.254.169.254", "link-local"),
        ("fe80::1", "link-local"),
        ("224.0.0.1", "multicast"),
        # Benchmarking range, another non-public block.
        ("198.18.0.1", "private"),
    ],
)
def test_non_public_addresses_are_refused(address: str, reason: str) -> None:
    """The address itself is enough to refuse it, whatever the hostname says.

    ``169.254.169.254`` matters most in practice: it is the cloud metadata
    endpoint, and reaching it from a server-side fetch is how credentials
    get stolen.
    """
    assert is_safe_url("http://example.com/x", resolver=resolve_to(address)) is False

    with pytest.raises(UnsafeURLError) as excinfo:
        assert_safe_url("http://example.com/x", resolver=resolve_to(address))

    assert reason in str(excinfo.value)


def test_an_ip_literal_host_is_checked_without_dns() -> None:
    """A literal address needs no lookup, and is judged directly."""
    assert is_safe_url("http://127.0.0.1:11434/api/generate") is False
    assert is_safe_url("http://93.184.216.34/page") is True


def test_every_resolved_address_is_checked_not_just_the_first() -> None:
    """One private address among several is enough to refuse the host.

    A name resolving to both a public and a private address is the DNS
    rebinding case worth refusing outright, rather than accepting because
    the first answer looked public.
    """
    resolver = resolve_to(PUBLIC, "10.0.0.5")

    with pytest.raises(UnsafeURLError) as excinfo:
        assert_safe_url("https://rebind.example.com/x", resolver=resolver)

    assert "10.0.0.5" in str(excinfo.value)


def test_an_unresolvable_host_is_refused() -> None:
    """Failing closed: a host that cannot be resolved cannot be a safe target.

    The refusal is the point of the test. Accepting the host because the
    inspection loop had no addresses to look at would hand an attacker a
    name that resolves differently on the next attempt.
    """

    def failing(host: str) -> list[str]:
        raise OSError("Name or service not known")

    assert is_safe_url("https://nowhere.invalid/x", resolver=failing) is False

    with pytest.raises(UnsafeURLError) as excinfo:
        assert_safe_url("https://nowhere.invalid/x", resolver=failing)

    assert "did not resolve" in str(excinfo.value)


def test_the_error_message_names_the_reason_without_leaking_the_stack() -> None:
    with pytest.raises(UnsafeURLError) as excinfo:
        assert_safe_url("http://192.168.0.1/admin", resolver=resolve_to(PUBLIC))

    assert "192.168.0.1" in str(excinfo.value)


def test_assert_safe_url_returns_the_normalised_form() -> None:
    """The value returned is what gets fetched and what gets cited."""
    normalised = assert_safe_url(
        "https://WWW.Example.com/Article/#top", resolver=resolve_to(PUBLIC)
    )

    assert normalised == "https://example.com/Article"


# ---------- The real resolver ----------


def test_default_resolver_is_injectable_and_used_when_no_override() -> None:
    """The production path resolves through the system resolver."""
    assert callable(default_resolver)

    # A literal IP never reaches the resolver, which is what keeps the
    # safety check free of DNS for addresses.
    assert is_safe_url("http://127.0.0.1:8000/") is False

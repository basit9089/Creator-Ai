"""HTTP retrieval of source documents.

This is the only place in the project that opens a connection to a URL
chosen by a search engine. Everything hostile about that input is handled
here, which is why the safety logic lives in
:mod:`app.services.research.urls` and is re-applied at every step.

Bounded by design
-----------------
A page on the open web is untrusted in both directions: it may be
hostile, and it may be enormous. Four limits apply to every request.

* **Timeout.** Connect plus read, so a server that accepts a connection
  and then goes quiet cannot hold the run open.
* **Size cap.** Enforced *while streaming*, not after. The response is
  abandoned as soon as it exceeds the cap, so a two-gigabyte file is
  never fully transferred.
* **Redirect budget.** Followed manually rather than by the HTTP client,
  because the default of "follow everything" is precisely the hole an
  attacker uses to smuggle a request past a URL check. Each hop is
  re-validated, and each hop requests the target exactly as the server
  wrote it rather than in normalised form.
* **Content-type allowlist.** Only documents are downloaded. Executable
  and media types are refused before a byte of body is read.

Nothing downloaded is ever executed or rendered. The body is decoded to
text and handed to the extractor; scripts are stripped there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from urllib.parse import urlsplit

import httpx

from app.core.logging import get_logger
from app.services.research.fetching.base import (
    FetchError,
    FetchedDocument,
    FetchTimeoutError,
    HttpStatusError,
    SourceFetcher,
    SourceTooLargeError,
    TooManyRedirectsError,
    UnsafeUrlFetchError,
    UnsupportedContentTypeError,
)
from app.services.research.urls import (
    HostResolver,
    UnsafeURLError,
    assert_safe_url,
    build_search_url,
    domain_of,
)

logger = get_logger("research.fetching.http")

#: Identifies the tool to the sites it contacts. Honest rather than
#: impersonating a browser, so an operator can filter or block it.
DEFAULT_USER_AGENT = "CreatorAI-Researcher/1.0 (+local research tool)"

#: Media types worth extracting text from. Anything else is refused
#: without reading the body: an executable, an archive or a video has no
#: place in a text-evidence pipeline, and downloading one would be
#: pointless at best.
ALLOWED_MEDIA_TYPES: frozenset[str] = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "text/markdown",
        "application/xml",
        "text/xml",
    }
)

#: Statuses that redirect. 303 becomes a GET, which is what browsers do;
#: 301 and 307 preserve the method. All our requests are GETs anyway, so
#: the distinction only matters for documentation.
REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})

#: Media types decoded with a fixed charset when the server omits one.
#: HTML5 defaults to UTF-8 and these formats are UTF-8 in practice.
_DEFAULT_CHARSET = "utf-8"


class HTTPSourceFetcher(SourceFetcher):
    """Fetches web documents over HTTP with explicit bounds."""

    name = "http"

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_size: int = 2_000_000,
        max_redirects: int = 5,
        allowed_media_types: Sequence[str] = tuple(sorted(ALLOWED_MEDIA_TYPES)),
        resolver: HostResolver | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        Args:
            timeout: seconds allowed for one request, covering connect,
                read and write. Applied per hop, so a redirect chain can
                legitimately take several times this in total.
            max_size: bytes accepted for one document. The connection is
                dropped as soon as this is exceeded.
            max_redirects: redirect hops allowed. Zero disables
                following.
            allowed_media_types: media types to download.
            resolver: override for hostname resolution, so safety checks
                can be tested without DNS.
            client: an existing HTTP client to reuse; tests supply one
                built on a mock transport.
        """
        self._timeout = timeout
        self._max_size = max_size
        self._max_redirects = max_redirects
        self._allowed = frozenset(m.lower() for m in allowed_media_types)
        self._resolver = resolver
        self._user_agent = user_agent
        self._client = client
        self._owns_client = client is None

    # ---------- SourceFetcher ----------

    async def fetch(self, url: str) -> FetchedDocument:
        """Retrieve ``url`` and return its decoded text.

        Raises:
            UnsafeUrlFetchError: a URL or redirect target failed a safety
                check. Nothing was requested.
            FetchTimeoutError: the server did not respond in time.
            SourceTooLargeError: the body exceeded the size cap.
            UnsupportedContentTypeError: not a readable document type.
            HttpStatusError: the server answered with an error status.
            TooManyRedirectsError: the redirect budget was exhausted.
        """
        return await self._fetch_with_budget(url, self._max_redirects)

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ---------- Internals ----------

    async def _fetch_with_budget(
        self,
        url: str,
        redirects_left: int,
        *,
        requested_url: str | None = None,
        redirected: bool = False,
    ) -> FetchedDocument:
        """Perform one request, following up to ``redirects_left`` hops.

        The remaining budget is passed down as an argument rather than
        held on ``self``. The Researcher fetches several sources at once,
        so mutable per-request state on a shared fetcher would let one
        page's redirect chain consume another's budget, and concurrent
        requests would race on it.

        ``requested_url`` and ``redirected`` are threaded through for the
        same reason. Whether a redirect occurred is a property of the
        chain, not of the last hop, so comparing the final URL with the
        current hop's URL would report ``False`` for every redirected
        fetch. Threading the original URL likewise keeps
        :attr:`FetchedDocument.url` meaning *the URL that was requested*
        while :attr:`FetchedDocument.final_url` is the one that was read.
        """
        try:
            # Validated before the first request, and again on every hop:
            # an allowed URL is free to redirect somewhere disallowed.
            #
            # The URL is then requested exactly as given, and the
            # normalised form returned by the check is deliberately
            # discarded. Normalisation treats a trailing slash as
            # insignificant, but many servers do not: blog.youtube
            # answers 308 to ``/inside-youtube`` and points at
            # ``/inside-youtube/``. Rewriting the redirect target on the
            # next hop therefore sends the request straight back to where
            # it started, and the chain only ends at the redirect budget.
            #
            # Where a resource lives is the server's call, not this
            # tool's. Normalisation exists to compare search results
            # against each other; it has no business overruling a server
            # that has just answered the question.
            assert_safe_url(url, resolver=self._resolver)
            current = url
            original = requested_url or current

            async with (await self._get_client()).stream(
                "GET",
                current,
                headers=self._headers(),
                timeout=self._timeout,
                # Redirects are followed by hand so each target can be
                # re-checked; the client's own following would bypass that.
                follow_redirects=False,
            ) as response:
                if response.status_code in REDIRECT_STATUSES:
                    return await self._follow(
                        current, response, redirects_left, original
                    )

                return await self._read(
                    current,
                    response,
                    requested_url=original,
                    redirected=redirected,
                )
        except UnsafeURLError as exc:
            raise UnsafeUrlFetchError(
                f"{domain_of(url)} is not a permitted source: {exc}"
            ) from exc
        except FetchError:
            raise
        except httpx.TimeoutException as exc:
            raise FetchTimeoutError(
                f"{domain_of(url)} did not respond in time."
            ) from exc
        except httpx.HTTPError as exc:
            # The transport message can contain the resolved address and
            # other local detail, so only the host is surfaced.
            raise FetchError(
                f"Could not retrieve {domain_of(url)}: {type(exc).__name__}."
            ) from exc

    async def _follow(
        self,
        url: str,
        response: httpx.Response,
        redirects_left: int,
        requested_url: str,
    ) -> FetchedDocument:
        """Resolve one redirect hop, bounded by the remaining budget.

        The new target is validated on the next call, before any request
        to it is made. A redirect into the private network is the classic
        way to slip past a URL check, which is why the client's built-in
        redirect following is disabled entirely.
        """
        location = response.headers.get("location")
        if not location:
            raise FetchError("The source redirected without a target.")

        if redirects_left <= 0:
            raise TooManyRedirectsError("The source redirected too many times.")

        target = build_search_url(_absolute(location, url))

        logger.info(
            "source_redirect from_domain=%s to_domain=%s",
            domain_of(url),
            domain_of(target),
        )

        return await self._fetch_with_budget(
            target, redirects_left - 1, requested_url=requested_url, redirected=True
        )

    async def _read(
        self,
        url: str,
        response: httpx.Response,
        *,
        requested_url: str,
        redirected: bool,
    ) -> FetchedDocument:
        """Validate the response and read it under the size cap."""
        if response.status_code >= 400:
            # The body of an error page is not evidence and can be very
            # large, so it is never read.
            raise HttpStatusError(
                f"{domain_of(url)} returned HTTP {response.status_code}.",
                status_code=response.status_code,
            )

        content_type = response.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()

        if media_type not in self._allowed:
            raise UnsupportedContentTypeError(
                f"{domain_of(url)} served {media_type or 'an unknown content type'}, "
                "which is not a readable document."
            )

        charset = _charset_of(content_type)

        body = bytearray()
        truncated = False
        async for chunk in response.aiter_bytes():
            # Checked per chunk rather than after the fact, so an oversized
            # response is abandoned rather than fully downloaded.
            if len(body) + len(chunk) > self._max_size:
                remaining = self._max_size - len(body)
                if remaining > 0:
                    body.extend(chunk[:remaining])
                truncated = True
                break
            body.extend(chunk)

        text = _decode(bytes(body), charset)

        if not text.strip():
            raise FetchError(f"{domain_of(url)} returned an empty document.")

        logger.info(
            "source_fetched domain=%s status=%s bytes=%d truncated=%s",
            domain_of(url),
            response.status_code,
            len(body),
            truncated,
        )

        return FetchedDocument(
            url=requested_url,
            final_url=str(response.url) or url,
            status_code=response.status_code,
            content_type=content_type or media_type,
            charset=charset,
            text=text,
            size_bytes=len(body),
            retrieved_at=datetime.now().astimezone(),
            truncated=truncated,
            redirected=redirected,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self._user_agent}
        # Text-only content. Declaring this is not a security control in
        # itself, but it avoids downloading binary bodies at all when a
        # server honours it.
        headers["Accept"] = "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client


def _absolute(location: str, base: str) -> str:
    """Resolve a possibly relative ``Location`` header against ``base``."""
    if location.startswith(("http://", "https://")):
        return location
    if location.startswith("//"):
        return f"https:{location}"
    if location.startswith("/"):
        scheme = urlsplit(base).scheme or "https"
        return f"{scheme}://{domain_of(base)}{location}"
    return f"{base.rstrip('/')}/{location}"


def _charset_of(content_type: str) -> str | None:
    """Read the charset parameter out of a Content-Type header."""
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "charset":
            return value.strip().strip('"').strip("'").lower() or None
    return None


def _decode(body: bytes, charset: str | None) -> str:
    """Decode a response body without ever raising.

    A page that lies about its encoding, or omits one, must not abort the
    retrieval: the bytes are decoded with the declared charset when it
    works, then with the default, then lossily. Lossy is acceptable here
    because the text is evidence for a summary, not data to round-trip.
    """
    for candidate in (charset, _DEFAULT_CHARSET):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode(_DEFAULT_CHARSET, errors="replace")

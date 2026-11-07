"""Free web search via DuckDuckGo's HTML endpoint.

Why this backend
----------------
The Researcher must run on the user's own machine for free, with no API
key and no account. That rules out every commercial search API, and
self-hosting an engine such as SearXNG needs Docker, which this project
deliberately avoids. DuckDuckGo's ``html.duckduckgo.com/html/`` endpoint
is a plain HTML results page: no key, no registration, no quota.

The trade-off is stated plainly rather than hidden. There is no official
API here, so this is an HTML interface that could change or rate-limit at
any time. It is isolated in this one module precisely so that replacing
it later touches nothing else: the agent depends on
:class:`~app.services.research.search.base.SearchProvider`, and a
supported API can be added beside this class without touching it.

No extra dependency is used. Parsing goes through the standard library's
:class:`html.parser.HTMLParser`, so the parsing logic is explicit and
testable against a saved fixture instead of hiding inside a scraping
library.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx

from app.core.logging import get_logger
from app.services.research.search.base import (
    SearchProvider,
    SearchQuery,
    SearchRateLimitedError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
)

logger = get_logger("research.search.duckduckgo")

#: The no-JavaScript results page. The ``lite`` host resets connections
#: from some networks; this one serves the full result markup.
ENDPOINT = "https://html.duckduckgo.com/html/"

#: DuckDuckGo serves results to clients that look like a browser. Sending
#: an honest, static user agent is deliberate: identifying the client is
#: not the same as impersonating a browser to evade blocks.
DEFAULT_USER_AGENT = (
    "CreatorAI-Researcher/1.0 (+local research tool; "
    "https://github.com/local/creator-ai)"
)

#: DuckDuckGo's ``kl`` parameter selects region and language. Only a few
#: mappings are listed: anything unrecognised simply omits the parameter,
#: which returns general results rather than failing.
_KL_BY_LANGUAGE: dict[str, str] = {
    "en": "us-en",
    "de": "de-de",
    "es": "es-es",
    "fr": "fr-fr",
    "it": "it-it",
    "nl": "nl-nl",
    "pt": "pt-pt",
    "pl": "pl-pl",
    "ru": "ru-ru",
}

#: DuckDuckGo returns at most this many results per page regardless of
#: the requested limit.
MAX_RESULTS_PER_REQUEST = 30

#: Markers of the interstitial served instead of results when DuckDuckGo
#: is suspicious of the client.
_CHALLENGE_MARKERS = ("anomaly.js", "captcha", "unusual traffic", "blocked")

#: HTML elements that never have a closing tag. Encountering their end tag
#: is a parse error to be ignored, and they must not open a scope.
_VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    }
)

#: Elements that imply a line break in the rendered text. Used so a title
#: split across nested markup does not run its words together.
_BLOCK_ELEMENTS = frozenset(
    {"address", "article", "aside", "blockquote", "br", "div", "dd", "dl",
     "dt", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5",
     "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
     "table", "td", "th", "tr", "ul"}
)


class DuckDuckGoSearchProvider(SearchProvider):
    """Web search backed by DuckDuckGo's keyless HTML endpoint."""

    name = "duckduckgo"
    returns_web_results = True

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
        min_interval: float = 1.0,
        max_length: int = 4000,
    ) -> None:
        """
        Args:
            timeout: seconds allowed for one search request.
            client: an existing HTTP client to reuse. Tests inject a
                client built on a mock transport; in production one is
                created lazily and closed by :meth:`close`.
            min_interval: minimum seconds between two searches from this
                instance. DuckDuckGo rate-limits aggressively, and the
                Researcher is the only caller, so a simple per-instance
                pause is enough protection.
            max_length: truncation limit for the query string.
        """
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._min_interval = min_interval
        self._max_length = max_length
        self._last_request_at = 0.0

    # ---------- SearchProvider ----------

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        """Return up to ``limit`` results for ``query``.

        Raises:
            SearchUnavailableError: the endpoint could not be reached.
            SearchTimeoutError: the request took too long.
            SearchRateLimitedError: the endpoint served a challenge or a
                rate-limit response.
        """
        if limit <= 0:
            return []

        await self._throttle()

        params: dict[str, str] = {
            "q": query.terms[: self._max_length],
            "kl": _KL_BY_LANGUAGE.get(query.language.lower(), ""),
        }
        # An empty value would be sent literally, so drop it.
        if not params["kl"]:
            params.pop("kl")

        client = await self._get_client()

        try:
            response = await client.get(
                ENDPOINT,
                params=params,
                headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html"},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise SearchTimeoutError(
                "The search backend did not respond in time."
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchUnavailableError(
                "The search backend could not be reached."
            ) from exc

        if response.status_code in {202, 429}:
            raise SearchRateLimitedError(
                "The search backend is rate limiting requests. Try again shortly."
            )
        if response.status_code >= 400:
            raise SearchUnavailableError(
                f"The search backend returned HTTP {response.status_code}."
            )

        body = response.text
        if _looks_like_challenge(body):
            raise SearchRateLimitedError(
                "The search backend returned an anti-bot challenge instead of results."
            )

        results = parse_results(body)
        logger.info(
            "duckduckgo_search ok parsed=%d returned=%d depth=%s",
            len(results),
            min(limit, len(results)),
            query.depth.value,
        )

        return results[: min(limit, MAX_RESULTS_PER_REQUEST)]

    async def close(self) -> None:
        """Close the HTTP client, but only if this instance created it.

        An injected client belongs to the caller, which may still be using
        it for other requests.
        """
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ---------- Internals ----------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            )
        return self._client

    async def _throttle(self) -> None:
        """Pause so this instance does not hammer the endpoint.

        Uses a monotonic clock: a wall-clock jump backwards would
        otherwise produce a negative interval and a long, pointless sleep.
        """
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self._min_interval:
            await anyio.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()


def parse_results(html: str) -> list[SearchResult]:
    """Extract search results from a DuckDuckGo results page.

    A module-level function rather than a method so tests can feed it a
    saved page directly, with no client, no network and no fixture
    directory.
    """
    parser = _DuckDuckGoResultParser()
    parser.feed(html)
    parser.close()
    return parser.results()


def unwrap_redirect(href: str) -> str | None:
    """Recover the real destination from a DuckDuckGo redirect link.

    Result links are of the form
    ``//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=...``.
    The real URL lives in the ``uddg`` parameter; the rest is DuckDuckGo's
    own click tracking. Returns ``None`` when the link carries nothing
    usable, so a malformed entry is dropped rather than fetched.
    """
    candidate = href.strip()
    if not candidate:
        return None

    # Protocol-relative and path-only forms are resolved against the
    # search host, which is what the browser would do.
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    elif candidate.startswith("/"):
        candidate = f"https://html.duckduckgo.com{candidate}"
    elif "://" not in candidate:
        return None

    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"}:
        return None

    target = parse_qs(parts.query).get("uddg")
    if target:
        real = target[0].strip()
        return real or None

    # No wrapper: the link already points at the destination.
    if "duckduckgo.com" in parts.netloc.lower():
        return None
    return candidate


def _looks_like_challenge(body: str) -> bool:
    """Whether the response is an anti-bot interstitial, not results."""
    lowered = body.lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


@dataclass(slots=True)
class _Entry:
    """Mutable parse buffer for one result.

    Mutable because the snippet arrives after the title and has to be
    filled in; :class:`SearchResult` is frozen and is only constructed
    once the whole page has been read.
    """

    title: str
    url: str
    snippet: str = ""


class _DuckDuckGoResultParser(HTMLParser):
    """Pull title, link and snippet out of the results page.

    Only anchors whose class contains ``result__a`` start a result. The
    page repeats each result's URL in several decorative anchors (a favicon
    and a bare domain link); keying on the class is what keeps those from
    being mistaken for additional results.

    Two flags rather than a tag stack, because the page contains void
    elements such as ``<img>`` and ``<br>`` that are never closed. A
    naive push/pop stack desynchronises on the first one of those and then
    mis-assigns every following title and snippet.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._entries: list[_Entry] = []
        self._buffer: list[str] = []

        self._in_title = False
        self._title_href = ""
        self._in_snippet = False
        self._snippet_entry: _Entry | None = None

    def results(self) -> list[SearchResult]:
        """Build immutable results, dropping duplicates.

        Dedup here as well as in the agent: the page occasionally repeats a
        destination, and an identical duplicate is never worth fetching
        twice.
        """
        seen: set[str] = set()
        results: list[SearchResult] = []

        for entry in self._entries:
            if entry.url in seen:
                continue
            seen.add(entry.url)
            results.append(
                SearchResult(
                    title=entry.title,
                    url=entry.url,
                    snippet=entry.snippet,
                    rank=len(results) + 1,
                )
            )

        return results

    # -- HTMLParser hooks --

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_ELEMENTS:
            return

        attributes = {key: (value or "") for key, value in attrs}
        classes = attributes.get("class", "").split()

        if tag == "a" and "result__a" in classes and not self._in_title:
            self._in_title = True
            self._title_href = attributes.get("href", "")
            self._buffer = []
            return

        if tag == "a" and "result__snippet" in classes and not self._in_snippet:
            # The snippet anchor follows its title anchor, so it completes
            # the most recently opened result.
            self._in_snippet = True
            self._snippet_entry = self._entries[-1] if self._entries else None
            self._buffer = []
            return

        if tag in _BLOCK_ELEMENTS:
            self._buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_ELEMENTS:
            return

        if tag == "a" and self._in_title:
            self._in_title = False
            url = unwrap_redirect(self._title_href)
            title = normalise_text("".join(self._buffer))
            self._buffer = []
            if url and title:
                self._entries.append(_Entry(title=title, url=url))
            return

        if tag == "a" and self._in_snippet:
            self._in_snippet = False
            text = normalise_text("".join(self._buffer))
            self._buffer = []
            if self._snippet_entry is not None and text:
                self._snippet_entry.snippet = text
            self._snippet_entry = None
            return

        if tag in _BLOCK_ELEMENTS:
            self._buffer.append(" ")

    def handle_data(self, data: str) -> None:
        self._buffer.append(data)


def normalise_text(value: str) -> str:
    """Collapse whitespace and trim.

    Page indentation and line breaks in the source HTML would otherwise
    end up inside titles and snippets, which then get compared and sent to
    the model.
    """
    return " ".join(value.split())


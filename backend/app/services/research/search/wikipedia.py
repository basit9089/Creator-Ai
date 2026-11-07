"""Wikipedia full-text search as a second, keyless source of candidates.

Why a second backend
--------------------
DuckDuckGo's HTML endpoint is free but unversioned: it can be rate
limited, throttled, or changed without notice. Wikipedia's search API is a
documented, supported, keyless JSON API. Neither is reliable alone, so the
Researcher can fall back from one to the other.

What this backend is and is not
-------------------------------
It searches one encyclopaedia, so it is not a substitute for web search.
It earns its place as a *fallback* and as a source of well-maintained
background on historical and scientific topics. The
:attr:`SearchProvider.returns_web_results` flag records the distinction,
so a report can say which kind of evidence it is looking at.

Honest limitation
-----------------
Wikipedia is unreachable from some networks. On the development machine
for this project an intercepting proxy with an expired certificate blocks
it, so this backend has never completed a live request here; it is
covered by tests against recorded API responses instead. That is a
deliberate trade: a documented interface that is exercised by tests is
more useful than an undocumented scraper that happens to work on one
network.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urlencode

import httpx

from app.core.logging import get_logger
from app.services.research.search.base import (
    SearchProvider,
    SearchQuery,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
)
from app.services.research.search.duckduckgo import normalise_text

logger = get_logger("research.search.wikipedia")

#: Wikipedia's API is stable and documented, unlike an HTML scrape.
API_PATH = "/w/api.php"

#: Mapping from an ISO 639-1 language code to a Wikipedia subdomain.
#: Anything not listed falls back to English rather than failing, since a
#: result in the wrong language still beats no result at all.
_SUBDOMAIN_BY_LANGUAGE: dict[str, str] = {
    "en": "en",
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "nl": "nl",
    "pt": "pt",
    "pl": "pl",
    "ru": "ru",
    "ja": "ja",
    "zh": "zh",
}

#: The API caps ``srlimit`` at 500 for anonymous clients; the Researcher
#: never asks for anything close to that.
MAX_LIMIT = 50

#: Wikipedia marks matched terms with ``<span class="searchmatch">``; the
#: tags are stripped but the words are kept, since they are the match.
_TAG_RE = re.compile(r"<[^>]+>")


class WikipediaSearchProvider(SearchProvider):
    """Search Wikipedia through its public JSON API."""

    name = "wikipedia"

    #: One encyclopaedia, not the open web. Recorded so a report can be
    #: explicit about what kind of evidence a candidate came from.
    returns_web_results = False

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        """Return up to ``limit`` matching Wikipedia articles.

        Raises:
            SearchUnavailableError: the API could not be reached, or
                answered with an error.
            SearchTimeoutError: the request took too long.
        """
        if limit <= 0:
            return []

        subdomain = _SUBDOMAIN_BY_LANGUAGE.get(query.language.lower(), "en")
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query.terms,
            "srlimit": str(min(limit, MAX_LIMIT)),
            # formatversion 2 returns a stable, plain-object shape instead
            # of the legacy mixed key/value dictionary.
            "format": "json",
            "formatversion": "2",
        }

        client = await self._get_client()

        try:
            response = await client.get(
                f"https://{subdomain}.wikipedia.org{API_PATH}",
                params=urlencode(params),
                headers={"Accept": "application/json"},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise SearchTimeoutError("Wikipedia did not respond in time.") from exc
        except httpx.HTTPError as exc:
            raise SearchUnavailableError("Wikipedia could not be reached.") from exc

        if response.status_code >= 400:
            raise SearchUnavailableError(
                f"Wikipedia returned HTTP {response.status_code}."
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise SearchUnavailableError(
                "Wikipedia returned a response that was not JSON."
            ) from exc

        results = parse_search_payload(payload, subdomain=subdomain)
        logger.info(
            "wikipedia_search ok parsed=%d language=%s", len(results), subdomain
        )
        return results[:limit]

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=True)
        return self._client


def parse_search_payload(payload: Any, *, subdomain: str = "en") -> list[SearchResult]:
    """Turn a Wikipedia API response into search results.

    Separated from the request so it can be tested against a recorded
    response. Returns an empty list for any shape it does not recognise,
    which is preferable to raising: one unfamiliar response should cost a
    few candidates, not the whole research run.
    """
    if not isinstance(payload, dict):
        return []

    query = payload.get("query")
    if not isinstance(query, dict):
        return []

    hits = query.get("search")
    if not isinstance(hits, list):
        return []

    results: list[SearchResult] = []
    for rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            continue

        title = normalise_text(str(hit.get("title") or ""))
        if not title:
            continue

        results.append(
            SearchResult(
                title=title,
                url=article_url(title, subdomain=subdomain),
                snippet=strip_markup(str(hit.get("snippet") or "")),
                rank=rank,
            )
        )

    return results


def article_url(title: str, *, subdomain: str = "en") -> str:
    """Canonical article URL for a title.

    Spaces become underscores and everything else is percent-encoded, which
    is what the site itself does.

    ``safe=""`` is essential rather than cosmetic. With the default
    ``safe="/"`` a title such as ``AC/DC`` keeps its slash and the URL
    addresses ``/wiki/AC`` with a subpage ``DC``, quietly returning the
    wrong article. Wikipedia titles genuinely contain slashes, so the
    default would be wrong exactly where it matters most.
    """
    return f"https://{subdomain}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}"


def strip_markup(value: str) -> str:
    """Remove HTML tags from a snippet and tidy the whitespace.

    Wikipedia highlights matched terms with ``<span class="searchmatch">``.
    The tags go; the words stay, because they are the reason the article
    matched.
    """
    return normalise_text(_TAG_RE.sub("", value))

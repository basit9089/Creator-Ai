"""A search provider that tries several backends in order.

Resilience, not cleverness. Every free, keyless search option has a way
to fail: an HTML endpoint can be rate limited, and a single encyclopaedia
can be blocked by a network. Rather than have research fail outright, the
composite asks each backend in turn and keeps what it gets until it has
enough.

The first backend that produces results is normally enough. The others are
only consulted when a backend fails or comes back empty, so this does not
multiply the requests sent to any one service.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.logging import get_logger
from app.services.research.search.base import (
    SearchError,
    SearchProvider,
    SearchQuery,
    SearchResult,
)

logger = get_logger("research.search.composite")


class CompositeSearchProvider(SearchProvider):
    """Try each provider in order and return the first usable answer.

    Results from several providers are merged only when a backend returned
    fewer results than were asked for, which is the case where a second
    opinion genuinely adds coverage. Merging unconditionally would send
    duplicate requests to every backend on every search.
    """

    name = "composite"

    def __init__(self, providers: Sequence[SearchProvider]) -> None:
        if not providers:
            raise ValueError("CompositeSearchProvider requires at least one provider.")
        self._providers = tuple(providers)

    @property
    def providers(self) -> tuple[SearchProvider, ...]:
        return self._providers

    @property
    def returns_web_results(self) -> bool:
        """True when any constituent provider performs a real web search."""
        return any(provider.returns_web_results for provider in self._providers)

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        """Return results from the first provider that supplies any.

        A backend that raises :class:`SearchError` is recorded and skipped:
        one unavailable provider must not end the research run. An
        unexpected exception is *not* swallowed, because a bug in one
        provider should be visible rather than silently degrading the
        result.
        """
        merged: list[SearchResult] = []
        seen_urls: set[str] = set()

        for provider in self._providers:
            if len(merged) >= limit:
                break

            remaining = limit - len(merged)
            try:
                results = await provider.search(query, limit=remaining)
            except SearchError as exc:
                logger.warning(
                    "search_provider_unavailable provider=%s code=%s",
                    provider.name,
                    exc.code,
                )
                continue

            added = 0
            for result in results:
                if result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                merged.append(result)
                added += 1

            logger.info(
                "search_provider_consulted provider=%s returned=%d accepted=%d",
                provider.name,
                len(results),
                added,
            )

            if not results:
                continue

            # Stop once the request is satisfied. A single provider
            # answering well should not cost extra calls elsewhere.
            if len(merged) >= limit:
                break

        return merged[:limit]

    async def close(self) -> None:
        for provider in self._providers:
            try:
                await provider.close()
            except Exception:  # noqa: BLE001 - closing must never raise
                logger.debug(
                    "search_provider_close_failed provider=%s", provider.name, exc_info=True
                )

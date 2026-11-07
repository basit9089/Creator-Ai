"""Shared FastAPI dependencies.

Every collaborator an agent needs is resolved here and cached for the
process. Tests override the relevant accessor to substitute a fake, which
is what keeps the suite offline and deterministic.

Caching matters for the HTTP-backed services in particular: creating a
fresh ``httpx.AsyncClient`` per request would drop the connection pool
and re-run the TLS handshake for every source fetch.

The factories below read configuration rather than hardcoding limits, so
the deployment's bounds apply without any agent being aware of them.
"""

from __future__ import annotations

from functools import lru_cache

from app.agents.runner import AgentRunner
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.ai.ollama_provider import OllamaProvider
from app.services.ai.provider import AIProvider
from app.services.research.extraction.base import ContentExtractor
from app.services.research.extraction.html import HtmlContentExtractor
from app.services.research.fetching.base import SourceFetcher
from app.services.research.fetching.http import HTTPSourceFetcher
from app.services.research.search.base import SearchProvider
from app.services.research.search.composite import CompositeSearchProvider
from app.services.research.search.duckduckgo import DuckDuckGoSearchProvider
from app.services.research.search.wikipedia import WikipediaSearchProvider

logger = get_logger("dependencies")


@lru_cache(maxsize=1)
def get_ai_provider() -> AIProvider:
    """Return the shared AI provider instance.

    Constructing the provider performs no network I/O; the connection is
    established lazily on first use, so this stays fast and safe to call
    at import time.
    """
    settings: Settings = get_settings()
    return OllamaProvider(settings)


@lru_cache(maxsize=1)
def get_search_provider() -> SearchProvider:
    """Return the shared search provider.

    ``SEARCH_PROVIDER`` selects the backend. The default is a composite
    rather than a single engine: DuckDuckGo's HTML endpoint is free and
    keyless but can be rate limited, and a composite keeps research
    working when it is. Wikipedia is the fallback because it is a
    documented, keyless API.

    Returns an object satisfying
    :class:`~app.services.research.search.base.SearchProvider`; nothing
    here is visible to the Researcher.
    """
    settings = get_settings()
    timeout = settings.search_timeout

    match settings.search_provider:
        case "duckduckgo":
            provider: SearchProvider = DuckDuckGoSearchProvider(timeout=timeout)
        case "wikipedia":
            provider = WikipediaSearchProvider(timeout=timeout)
        case "fallback":
            provider = CompositeSearchProvider(
                [
                    DuckDuckGoSearchProvider(timeout=timeout),
                    WikipediaSearchProvider(timeout=timeout),
                ]
            )
        case _:  # pragma: no cover - Settings validates the value
            raise ValueError(f"Unknown search provider {settings.search_provider!r}.")

    logger.info(
        "search_provider_configured provider=%s returns_web_results=%s timeout=%ss",
        provider.name,
        provider.returns_web_results,
        timeout,
    )
    return provider


@lru_cache(maxsize=1)
def get_source_fetcher() -> SourceFetcher:
    """Return the shared source fetcher."""
    settings = get_settings()
    return HTTPSourceFetcher(
        timeout=settings.source_fetch_timeout,
        max_size=settings.max_source_size,
        max_redirects=settings.max_source_redirects,
    )


@lru_cache(maxsize=1)
def get_content_extractor() -> ContentExtractor:
    """Return the shared content extractor."""
    settings = get_settings()
    return HtmlContentExtractor(max_text=settings.max_extracted_text)


@lru_cache(maxsize=1)
def get_agent_runner() -> AgentRunner:
    """Return the shared agent runner.

    One runner for the process so execution history is continuous and a
    single agent execution can be retried later through
    :meth:`~app.agents.runner.AgentRunner.retry`.
    """
    settings = get_settings()
    return AgentRunner(execution_timeout=settings.ai_request_timeout)


def reset_provider_cache() -> None:
    """Clear cached providers and the runner. Used by tests."""
    get_ai_provider.cache_clear()
    get_search_provider.cache_clear()
    get_source_fetcher.cache_clear()
    get_content_extractor.cache_clear()
    get_agent_runner.cache_clear()


async def close_services() -> None:
    """Release every cached service that holds a connection.

    Called from the application lifespan on shutdown. Failures are logged
    and swallowed: shutdown must not fail because a socket would not close
    cleanly.
    """
    for accessor in (get_search_provider, get_source_fetcher, get_ai_provider):
        try:
            service = accessor()
            await service.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.debug("service_close_failed service=%s", accessor.__name__, exc_info=True)

    reset_provider_cache()
    logger.info("services_closed")

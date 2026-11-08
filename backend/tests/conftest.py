"""Shared pytest fixtures.

Unit tests never contact a real Ollama server and never touch the
network. The AI provider, search backend and source fetcher are all
replaced with fakes, so the suite runs offline and deterministically.
Two opt-in integration suites exercise the real local Ollama server and
the real web search; neither is required for the unit suite to pass.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.registry import get_registry
from app.core.config import reset_settings_cache
from app.dependencies import get_ai_provider, reset_provider_cache
from app.main import create_app
from app.services.ai.provider import (
    AIProvider,
    GenerationOptions,
    GenerationResult,
    HealthStatus,
    Message,
    ProviderHealth,
)
from app.services.research.extraction.base import ContentExtractor, ExtractedContent
from app.services.research.fetching.base import FetchedDocument, SourceFetcher
from app.services.research.search.base import SearchProvider, SearchQuery, SearchResult
from app.services.research.urls import domain_of


class FakeAIProvider(AIProvider):
    """In-memory provider used by unit tests.

    Records the messages it received so tests can assert that request
    construction is correct, and can be told to fail so that error
    handling can be exercised.
    """

    name = "fake"

    def __init__(
        self,
        *,
        text: str = "Fake response",
        health: ProviderHealth | None = None,
        error: Exception | None = None,
    ) -> None:
        self._text = text
        self._error = error
        self._health = health or ProviderHealth(
            status=HealthStatus.HEALTHY,
            provider=self.name,
            model="fake-model",
            reachable=True,
            model_available=True,
            detail="Fake provider ready.",
            latency_ms=0.1,
        )
        self.calls: list[tuple[Sequence[Message], GenerationOptions | None]] = []
        self.streamed: list[str] = []

    @property
    def default_model(self) -> str:
        return "fake-model"

    def generate(
        self,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        self.calls.append((messages, options))
        if self._error is not None:
            raise self._error
        return GenerationResult(
            text=self._text,
            model=self.default_model,
            prompt_tokens=11,
            completion_tokens=7,
            total_duration_ms=1.5,
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
    ) -> Iterator[str]:
        self.calls.append((messages, options))
        if self._error is not None:
            raise self._error
        for piece in self._text.split(" "):
            self.streamed.append(piece)
            yield piece + " "

    def health_check(self) -> ProviderHealth:
        return self._health


#: A complete, schema-valid Researcher analysis. Reused so that tests can
#: focus on the behaviour under test rather than on assembling a valid
#: payload every time.
RESEARCH_ANALYSIS_PAYLOAD: dict = {
    "topic": "The history of YouTube",
    "summary": "YouTube began as a video-sharing site and was acquired by Google.",
    "key_points": [
        {"text": "YouTube was founded in February 2005.", "source_ids": ["source_01"]},
    ],
    "claims": [
        {
            "id": "claim_01",
            "statement": "YouTube launched in February 2005.",
            "source_ids": ["source_01", "source_02"],
        },
        {
            "id": "claim_02",
            "statement": "YouTube was acquired by Google in 2006.",
            "source_ids": ["source_01", "source_03"],
        },
    ],
    "conflicts": [
        {
            "id": "conflict_01",
            "statement": "The year of the Google acquisition is disputed.",
            "source_ids": ["source_01"],
            "contradicting_source_ids": ["source_02"],
            "note": "One source says 2006, another 2007.",
        }
    ],
    "uncertainties": [
        {"description": "Early upload numbers are not corroborated.", "source_ids": []}
    ],
    "statistics": [
        {"id": "stat_01", "label": "Founding year", "value": "2005", "source_ids": ["source_01"]}
    ],
}


class FakeSearchProvider(SearchProvider):
    """Search backend returning a fixed list of candidates.

    Records the queries it received so a test can assert that the topic,
    the depth and the language actually reached the search layer.
    """

    name = "fake-search"

    def __init__(
        self,
        results: Sequence[SearchResult] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._results = list(results or [])
        self._error = error
        self.queries: list[SearchQuery] = []
        self.limits: list[int] = []

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        self.queries.append(query)
        self.limits.append(limit)
        if self._error is not None:
            raise self._error
        return self._results[:limit]

    async def close(self) -> None:
        return None


class FakeSourceFetcher(SourceFetcher):
    """Fetcher returning canned documents, with per-URL failures.

    The default behaviour is to fail every URL, so a test that forgets to
    declare its documents sees an obvious empty result rather than an
    accidental pass.
    """

    name = "fake-fetcher"

    def __init__(
        self,
        documents: dict[str, str] | None = None,
        *,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self._documents = dict(documents or {})
        self._failures = dict(failures or {})
        self.requested: list[str] = []
        self.closed = False

    async def fetch(self, url: str) -> FetchedDocument:
        self.requested.append(url)

        failure = self._failures.get(url)
        if failure is not None:
            raise failure

        if url not in self._documents:
            from app.services.research.fetching.base import FetchError

            raise FetchError(f"No document registered for {url}.")

        return FetchedDocument(
            url=url,
            final_url=url,
            status_code=200,
            content_type="text/html; charset=utf-8",
            charset="utf-8",
            text=self._documents[url],
            size_bytes=len(self._documents[url]),
            retrieved_at=datetime.now().astimezone(),
        )

    async def close(self) -> None:
        self.closed = True


class FakeContentExtractor(ContentExtractor):
    """Extractor returning a fixed amount of text per document.

    Lets the retrieval tests exercise the pipeline without building HTML
    documents, so a failure points at retrieval rather than at parsing.
    """

    name = "fake-extractor"

    def __init__(self, *, length: int = 1200) -> None:
        self._length = length

    def supports(self, content_type: str) -> bool:
        return True

    def extract(self, document: FetchedDocument) -> ExtractedContent:
        body = "x" * self._length
        return ExtractedContent(
            title="Fixture title",
            text=body,
            original_length=self._length,
        )


def make_search_result(url: str, title: str = "", *, snippet: str = "") -> SearchResult:
    """Build a search result, filling in a sensible default title."""
    return SearchResult(
        title=title or f"Result for {domain_of(url)}",
        url=url,
        snippet=snippet,
        domain=domain_of(url),
    )


def build_research_agent(
    *,
    provider: FakeAIProvider | None = None,
    search: FakeSearchProvider | None = None,
    fetcher: FakeSourceFetcher | None = None,
    extractor: ContentExtractor | None = None,
    **settings_kwargs: object,
):
    """Assemble a Researcher with every collaborator replaced.

    The point of this factory is that no test ever needs a network, an
    Ollama server or a live search backend. Everything the agent touches
    is an explicit argument.
    """
    from app.agents.researcher.agent import ResearcherAgent
    from app.core.config import get_settings

    overrides = {
        "search_provider": "duckduckgo",
        "search_timeout": 5.0,
        "max_research_sources": 12,
        "min_usable_sources": 2,
        "max_concurrent_source_fetches": 4,
        "source_fetch_timeout": 5.0,
        "max_source_size": 100_000,
        "max_extracted_text": 4000,
    }
    overrides.update(settings_kwargs)

    return ResearcherAgent(
        provider=provider or FakeAIProvider(text=json.dumps(RESEARCH_ANALYSIS_PAYLOAD)),
        search_provider=search or FakeSearchProvider(),
        fetcher=fetcher or FakeSourceFetcher(),
        extractor=extractor or FakeContentExtractor(),
        settings=get_settings().model_copy(update=overrides),
    )


def analysis_payload(**overrides: object) -> str:
    """A schema-valid analysis payload as JSON, with optional overrides."""
    payload = {**RESEARCH_ANALYSIS_PAYLOAD, **overrides}
    return json.dumps(payload)


@pytest.fixture
def fake_provider() -> FakeAIProvider:
    """A provider that succeeds without touching the network."""
    return FakeAIProvider()


@pytest.fixture
def client(fake_provider: FakeAIProvider) -> Iterator[TestClient]:
    """Test client wired to the fake provider."""
    app = create_app()
    app.dependency_overrides[get_ai_provider] = lambda: fake_provider

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def client_factory():
    """Build a client with a specific fake provider behaviour.

    Returns a context manager factory so a test can inject failures or a
    custom health state without depending on the default fixtures::

        with client_factory(provider) as client:
            ...
    """

    @contextmanager
    def _factory(provider: FakeAIProvider) -> Iterator[TestClient]:
        app = create_app()
        app.dependency_overrides[get_ai_provider] = lambda: provider

        try:
            with TestClient(app, raise_server_exceptions=False) as test_client:
                yield test_client
        finally:
            app.dependency_overrides.clear()

    return _factory


@pytest.fixture
def anyio_backend() -> str:
    """Run ``@pytest.mark.anyio`` tests on asyncio.

    Declared here so the agent tests need no extra plugin: anyio ships its
    own pytest plugin and is already a runtime dependency.
    """
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_caches() -> Iterator[None]:
    """Clear cached settings and providers between tests.

    Without this, a test that changes an environment variable would leak
    that value into every test that follows, through the settings cache.
    The shared agent registry is cleared too: it holds agent instances
    that cache their resolved settings (storage roots) on first use.
    """
    reset_settings_cache()
    reset_provider_cache()
    get_registry.cache_clear()
    yield
    reset_settings_cache()
    reset_provider_cache()
    get_registry.cache_clear()


@pytest.fixture
def integration_enabled() -> bool:
    """Whether opt-in tests against a real Ollama server should run.

    Enable with::

        RUN_OLLAMA_INTEGRATION=1 pytest -m integration
    """
    return os.getenv("RUN_OLLAMA_INTEGRATION") == "1"


@pytest.fixture
def live_research_enabled() -> bool:
    """Whether opt-in tests that touch the real web should run.

    Enable with::

        RUN_RESEARCH_INTEGRATION=1 pytest -m live_research

    These tests search the internet and fetch real pages, so they are kept
    out of the default run entirely.
    """
    return os.getenv("RUN_RESEARCH_INTEGRATION") == "1"


@pytest.fixture
def public_resolver():
    """A hostname resolver that maps every name to a public address.

    Tests must never depend on DNS. With this installed, a URL is judged
    on its *host* rather than on what this machine's resolver happens to
    return today.
    """

    def resolve(host: str) -> list[str]:
        return ["93.184.216.34"]

    return resolve


@pytest.fixture(scope="session")
def backend_dir() -> Path:
    """Absolute path to the backend project directory."""
    return Path(__file__).resolve().parent.parent
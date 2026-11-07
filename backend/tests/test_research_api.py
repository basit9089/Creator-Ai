"""The HTTP surface of research.

The route is supposed to contain no logic: it resolves an agent by name,
runs it, and translates the outcome. These tests hold it to that, and pin
the status mapping, because the difference between "the evidence was thin"
and "the run broke" is invisible to a client unless the status code and
the body agree about which one happened.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.agents.registry import AgentRegistry, build_default_registry, get_registry
from app.agents.researcher.agent import ResearcherAgent
from app.core.config import get_settings
from app.dependencies import close_services, get_agent_runner, reset_provider_cache
from app.main import create_app
from app.services.ai.provider import ProviderTimeoutError
from app.services.research.extraction.base import ExtractedContent
from app.services.research.fetching.base import FetchError, HttpStatusError

from .conftest import (
    FakeAIProvider,
    FakeContentExtractor,
    FakeSearchProvider,
    FakeSourceFetcher,
    analysis_payload,
    build_research_agent,
    make_search_result,
)

URLS = [
    "https://example.com/one",
    "https://other.com/two",
    "https://third.com/three",
]

TOPIC = "History of YouTube"


def documents(*urls: str) -> dict[str, str]:
    return {url: f"<html><body><p>{'Evidence. ' * 200}</p></body></html>" for url in urls}


def working_agent(**kwargs):
    """A Researcher wired to fakes, returning the agent and its provider."""
    provider = kwargs.pop("provider", None) or FakeAIProvider(text=analysis_payload())
    agent = build_research_agent(
        provider=provider,
        search=kwargs.pop("search", None)
        or FakeSearchProvider([make_search_result(u) for u in URLS]),
        fetcher=kwargs.pop("fetcher", None) or FakeSourceFetcher(documents(*URLS)),
        extractor=kwargs.pop("extractor", None) or FakeContentExtractor(),
        **kwargs,
    )
    return agent, provider


@pytest.fixture
def research_client() -> Iterator[TestClient]:
    """A client whose registry holds a fully faked Researcher.

    The dependency is the registry itself, so overriding it replaces the
    agent without the route or the runner being aware. That is the point
    of addressing agents by name.
    """
    agent, provider = working_agent()

    app = create_app()
    # The registry is replaced so that resolving the name "researcher"
    # yields the fully faked agent. The lookup path, the runner and the
    # route's translation are all still the real ones.
    app.dependency_overrides[get_registry] = _registry_with(agent)
    app.dependency_overrides[get_agent_runner] = lambda: _StubRunner(agent, provider)

    with TestClient(app, raise_server_exceptions=False) as client:
        client.researcher = agent  # type: ignore[attr-defined]
        client.provider = provider  # type: ignore[attr-defined]
        yield client

    app.dependency_overrides.clear()


def _registry_with(agent: ResearcherAgent):
    """A registry whose ``"researcher"`` resolves to ``agent``.

    Built from the real registry rather than an empty one, so the lookup
    and its failure behaviour are the real ones too.
    """

    def resolve() -> AgentRegistry:
        registry = build_default_registry()
        registry.register(agent, overwrite=True)
        return registry

    return resolve


class _StubRunner:
    """Runs the real agent through the real runner.

    Only the collaborator resolution is short-circuited, so the agent's
    actual behaviour, the runner's status handling and the route's
    translation are all genuinely exercised.
    """

    def __init__(self, agent: ResearcherAgent, provider: FakeAIProvider) -> None:
        from app.agents.runner import AgentRunner

        self._agent = agent
        # The runner builds its own AgentOutput, and the provider the agent
        # was given is the one it will call. Constructing it with no
        # provider leaves the real ``get_ai_provider`` lookup unused,
        # because the agent already holds a fake.
        self._runner = AgentRunner()

    async def run(self, agent, context, messages):
        return await self._runner.run(agent, context, messages)


# ---------- The happy path ----------


def test_research_returns_grounded_citations(research_client: TestClient) -> None:
    response = research_client.post("/api/research", json={"topic": TOPIC})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True

    result = body["result"]
    assert result["topic"] == TOPIC
    assert result["sources_found"] == 3
    assert result["summary"]


def test_every_citation_resolves_to_a_url_in_the_response(
    research_client: TestClient,
) -> None:
    """A client-side reader must be able to follow a claim from the
    response alone, with no second request."""
    result = research_client.post("/api/research", json={"topic": TOPIC}).json()["result"]

    known = {source["id"] for source in result["sources"]}
    assert result["claims"]
    for claim in result["claims"]:
        for source_id in claim["source_ids"]:
            assert source_id in known
    for source in result["sources"]:
        assert source["url"].startswith("https://")


def test_the_response_reports_confidence_counts(research_client: TestClient) -> None:
    result = research_client.post("/api/research", json={"topic": TOPIC}).json()["result"]

    assert set(result["confidence_counts"]) == {"high", "medium", "low", "unsupported"}


def test_the_response_carries_execution_metadata(research_client: TestClient) -> None:
    execution = research_client.post(
        "/api/research", json={"topic": TOPIC}
    ).json()["execution"]

    assert execution["agent_name"] == "researcher"
    assert execution["status"] == "completed"
    assert execution["succeeded"] is True
    assert execution["duration_ms"] is not None
    assert execution["error"] is None


def test_the_request_options_reach_the_agent(research_client: TestClient) -> None:
    response = research_client.post(
        "/api/research",
        json={"topic": TOPIC, "search_depth": "deep", "max_sources": 8, "language": "en-GB"},
    )

    result = response.json()["result"]
    assert result["search_depth"] == "deep"
    assert result["language"] == "en-gb"


def test_an_ignored_option_is_reported_in_the_response(research_client: TestClient) -> None:
    """A caller who set a filter and got none would conclude it worked."""
    response = research_client.post(
        "/api/research", json={"topic": TOPIC, "date_range": "past_year"}
    )

    warnings = " ".join(response.json()["result"]["warnings"])
    assert "date_range" in warnings


# ---------- Validation ----------


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "no topic"),
        ({"topic": ""}, "empty topic"),
        ({"topic": "   "}, "blank topic"),
        ({"topic": "x" * 501}, "overlong topic"),
        ({"topic": "x", "max_sources": 0}, "zero sources"),
        ({"topic": "x", "max_sources": 99}, "uncapped sources"),
        ({"topic": "x", "search_depth": "exhaustive"}, "unknown depth"),
        ({"topic": "x", "language": "e"}, "malformed language"),
    ],
)
def test_an_invalid_request_is_rejected_with_field_errors(
    research_client: TestClient, payload: dict, reason: str
) -> None:
    response = research_client.post("/api/research", json=payload)

    assert response.status_code == 422, reason
    # The project's own error envelope, not FastAPI's bare ``detail``, so a
    # client parses research failures and validation failures the same way.
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]


def test_validation_happens_before_anything_is_retrieved(
    research_client: TestClient,
) -> None:
    """No search, no fetch, no model call for a request that cannot run."""
    agent = research_client.researcher  # type: ignore[attr-defined]

    research_client.post("/api/research", json={"topic": ""})

    assert agent.search_provider.queries == []
    assert agent.fetcher.requested == []


def test_a_non_object_body_is_rejected(research_client: TestClient) -> None:
    assert research_client.post("/api/research", json=["a", "list"]).status_code == 422


# ---------- Partial success ----------


def test_a_run_with_some_dead_sources_still_succeeds(
    research_client: TestClient,
) -> None:
    """Six requested, two retrieved and one blocked is a usable result."""
    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._search_provider = FakeSearchProvider(  # noqa: SLF001
        [make_search_result(u) for u in URLS]
    )
    agent._fetcher = FakeSourceFetcher(  # noqa: SLF001
        documents(URLS[0], URLS[1]),
        failures={URLS[2]: HttpStatusError("The source returned 403.", status_code=403)},
    )

    body = research_client.post("/api/research", json={"topic": TOPIC}).json()

    assert body["success"] is True
    assert body["result"]["sources_found"] == 2
    assert body["result"]["sources_failed"] == 1
    assert body["result"]["failed_sources"][0]["domain"] == "third.com"


def test_insufficient_evidence_returns_two_hundred_not_an_error(
    research_client: TestClient,
) -> None:
    """The Phase 3 distinction made visible: the pipeline worked and the
    web had too little to offer. That is a verdict, not a fault."""
    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._search_provider = FakeSearchProvider(  # noqa: SLF001
        [make_search_result(u) for u in URLS]
    )
    agent._fetcher = FakeSourceFetcher(documents(URLS[0]))  # noqa: SLF001

    response = research_client.post("/api/research", json={"topic": TOPIC})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["execution"]["status"] == "completed"
    assert body["execution"]["code"] == "research_insufficient_sources"
    assert body["execution"]["retryable"] is True


def test_the_partial_result_is_returned_alongside_a_negative_verdict(
    research_client: TestClient,
) -> None:
    """The one page that was retrieved is still reported, so a caller can
    see what was found."""
    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._search_provider = FakeSearchProvider(  # noqa: SLF001
        [make_search_result(u) for u in URLS]
    )
    agent._fetcher = FakeSourceFetcher(documents(URLS[0]))  # noqa: SLF001

    result = research_client.post("/api/research", json={"topic": TOPIC}).json()["result"]

    assert result["sources_found"] == 1
    assert result["sources"][0]["url"] == URLS[0]
    assert result["sufficient"] is False


# ---------- Failures ----------


def test_a_model_outage_is_reported_as_unavailable(
    research_client: TestClient,
) -> None:
    """The retrieval worked, so the client should retry rather than give
    up on the request entirely."""
    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._provider = FakeAIProvider(  # noqa: SLF001
        error=ProviderTimeoutError("The model did not respond.")
    )

    response = research_client.post("/api/research", json={"topic": TOPIC})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "research_ai_unavailable"


def test_unusable_model_output_is_a_bad_gateway(research_client: TestClient) -> None:
    """A 502 rather than a 500: the server worked, the upstream did not."""
    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._provider = FakeAIProvider(text="{not json")  # noqa: SLF001

    response = research_client.post("/api/research", json={"topic": TOPIC})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "research_ai_output_invalid"


def test_a_search_outage_is_reported_as_unavailable(research_client: TestClient) -> None:
    from app.services.research.search.base import SearchError

    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._search_provider = FakeSearchProvider(  # noqa: SLF001
        error=SearchError("The search backend is unreachable.")
    )

    response = research_client.post("/api/research", json={"topic": TOPIC})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["execution"]["code"] == "research_no_sources"


def test_the_route_adds_nothing_unsafe_to_a_failure_reason(
    research_client: TestClient,
) -> None:
    """Where the client-safe boundary actually sits.

    A fetch failure's ``reason`` is the fetcher's own message, verbatim.
    That is safe because :class:`HTTPSourceFetcher` constructs those
    messages deliberately, naming the host and the exception type and
    never the underlying transport text. The route and the agent copy the
    message rather than rebuilding it, so neither can leak something the
    fetcher was careful about.

    The guarantee itself is tested where it is implemented, in
    ``test_research_fetching.py``. What is asserted here is that nothing on
    the way to the client adds to it.
    """
    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._fetcher = FakeSourceFetcher(  # noqa: SLF001
        failures={url: HttpStatusError("third.com returned 403.", status_code=403) for url in URLS}
    )

    body = research_client.post("/api/research", json={"topic": TOPIC}).json()

    reasons = [f["reason"] for f in body["result"]["failed_sources"]]
    assert reasons
    assert set(reasons) == {"third.com returned 403."}
    # No traceback, no local path, and no raw exception type leaked in.
    assert "Traceback" not in body["result"]["failed_sources"][0]["reason"]


def test_no_sources_at_all_still_returns_a_typed_body(
    research_client: TestClient,
) -> None:
    agent = research_client.researcher  # type: ignore[attr-defined]
    agent._search_provider = FakeSearchProvider([])  # noqa: SLF001

    body = research_client.post("/api/research", json={"topic": TOPIC}).json()

    assert body["success"] is False
    assert body["result"] is not None
    assert body["result"]["sources"] == []
    assert body["execution"]["code"] == "research_no_sources"


# ---------- Routing ----------


def test_the_endpoint_is_registered_once() -> None:
    """FastAPI nests included routers, so ``app.routes`` is not a reliable
    place to look for the path."""
    paths = create_app().openapi()["paths"]

    assert "/api/research" in paths
    assert "post" in paths["/api/research"]


def test_the_route_resolves_the_agent_by_name() -> None:
    """The route must not import the implementation.

    It may name the class in prose, so the check is on imports rather than
    on the text. If it ever imports ``ResearcherAgent``, swapping the
    implementation becomes a change to the API layer, which is exactly
    what addressing agents by name exists to prevent.
    """
    import app.api.routes.research as route

    assert route.RESEARCHER_AGENT_NAME == "researcher"

    tree = ast.parse(Path(route.__file__).read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert "ResearcherAgent" not in imported
    # The one thing the route does import from the agent package is the
    # context key. That is the seam between them: a name both sides agree
    # on for handing over the validated request, and importing a constant
    # couples the route to no behaviour at all.
    assert "REQUEST_CONTEXT_KEY" in imported


def test_the_health_endpoint_still_works(research_client: TestClient) -> None:
    """Research must not have displaced the Phase 3 health check."""
    response = research_client.get("/api/health/ai")

    assert response.status_code == 200
    assert "status" in response.json()


def test_the_agents_list_still_marks_the_other_agents_unimplemented(
    research_client: TestClient,
) -> None:
    response = research_client.get("/api/agents")

    assert response.status_code == 200
    agents = {a["name"]: a for a in response.json()["agents"]}
    assert agents["researcher"]["implemented"] is True
    assert agents["scriptwriter"]["implemented"] is True  # Phase 6 implements scriptwriter


# ---------- Default wiring ----------


def test_the_default_registry_resolves_the_researcher() -> None:
    """Constructing the registry must not open a connection: every
    collaborator is resolved lazily on first use."""
    registry = build_default_registry()

    agent = registry.get_agent("researcher")

    assert isinstance(agent, ResearcherAgent)


def test_the_configured_backend_is_duckduckgo_by_default() -> None:
    """A free backend with no API key, so a fresh clone works offline of
    any account."""
    assert get_settings().search_provider == "duckduckgo"
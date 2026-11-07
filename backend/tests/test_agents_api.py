"""Tests for the agent API and the AI routes' new options."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.base import AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry, build_default_registry
from app.core.config import get_settings
from app.services.ai.provider import AIProvider, GenerationOptions


def test_list_agents_returns_ok(client: TestClient) -> None:
    response = client.get("/api/agents")

    assert response.status_code == 200


def test_list_agents_reports_the_total(client: TestClient) -> None:
    response = client.get("/api/agents")

    body = response.json()
    assert body["success"] is True
    assert body["total"] == len(body["agents"])


def test_list_agents_includes_the_six_planned_agents(client: TestClient) -> None:
    """Every planned name is discoverable before the logic exists."""
    body = client.get("/api/agents").json()

    names = [agent["name"] for agent in body["agents"]]

    assert names == [
        "researcher",
        "fact_checker",
        "scriptwriter",
        "video_producer",
        "captions_thumbnail",
        "publisher",
    ]


def test_list_agents_in_pipeline_order(client: TestClient) -> None:
    body = client.get("/api/agents").json()

    assert [agent["name"] for agent in body["agents"]][0] == "researcher"
    assert [agent["name"] for agent in body["agents"]][-1] == "publisher"


def test_list_agents_reports_versions(client: TestClient) -> None:
    body = client.get("/api/agents").json()

    assert all(agent["version"] == "1.0.0" for agent in body["agents"])


def test_list_agents_marks_placeholders_as_not_implemented(client: TestClient) -> None:
    """Agents without an implementation yet must say so.

    Since Phase 4 the Researcher is real, so the assertion is about the
    agents still waiting rather than about all six.
    """
    from app.agents.registry import IMPLEMENTED_AGENT_NAMES

    body = client.get("/api/agents").json()

    agents = {agent["name"]: agent for agent in body["agents"]}
    assert agents, "expected the registry to report agents"

    for name, agent in agents.items():
        if name in IMPLEMENTED_AGENT_NAMES:
            assert agent["status"] == "available"
            assert agent["implemented"] is True
        else:
            assert agent["status"] == "not_implemented"
            assert agent["implemented"] is False


def test_list_agents_includes_descriptions(client: TestClient) -> None:
    body = client.get("/api/agents").json()

    for agent in body["agents"]:
        assert agent["description"]


def test_list_agents_includes_capabilities(client: TestClient) -> None:
    body = client.get("/api/agents").json()

    for agent in body["agents"]:
        assert isinstance(agent["capabilities"], list)


def test_list_agents_is_read_only(client: TestClient) -> None:
    """The endpoint must not execute anything or contact the model."""
    assert client.post("/api/agents").status_code == 405
    assert client.delete("/api/agents").status_code == 405


def test_only_the_researcher_has_an_execution_endpoint(client: TestClient) -> None:
    """Phase 4 added exactly one execution endpoint, and it is the Researcher's.

    The generic "run any agent by name" route stays absent: executing a
    placeholder would fail confusingly, and a generic route would let any
    future agent be triggered without a contract of its own.
    """
    assert client.post("/api/research", json={"topic": "  "}).status_code == 422
    assert client.post("/api/agents/run", json={"agent": "researcher"}).status_code == 404


def test_agents_endpoint_uses_an_isolated_registry() -> None:
    """Registering an agent in one registry must not affect the shared one."""
    from app.agents.registry import get_registry

    build_default_registry().register_placeholder("transcriber")

    assert "transcriber" not in get_registry()


def test_list_agents_reflects_a_custom_registry() -> None:
    """A caller may substitute its own registry without touching the app."""
    from app.agents.registry import get_registry
    from app.main import create_app

    registry = AgentRegistry()
    registry.register_placeholder("custom_agent", version="2.0.0")

    app = create_app()
    app.dependency_overrides[get_registry] = lambda: registry

    try:
        with TestClient(app) as custom_client:
            body = custom_client.get("/api/agents").json()
    finally:
        app.dependency_overrides.clear()

    assert body["total"] == 1
    assert body["agents"][0]["name"] == "custom_agent"
    assert body["agents"][0]["version"] == "2.0.0"


def test_agents_endpoint_needs_no_provider(client: TestClient) -> None:
    """Listing agents must work even when Ollama is down.

    The client fixture supplies a fake provider; this confirms the route
    performs no provider call of its own.
    """
    assert client.get("/api/agents").status_code == 200


def test_agents_endpoint_is_documented_in_the_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/api/agents" in schema["paths"]
    assert "get" in schema["paths"]["/api/agents"]


# ---------- Thinking mode over the API ----------


def test_generate_defaults_to_thinking_off(client: TestClient, fake_provider) -> None:
    """The deployment default reaches the provider.

    The fake applies the same merge logic as a real provider, so an unset
    value here means Ollama would receive think=False too.
    """
    client.post("/api/ai/generate", json={"prompt": "Hello"})

    _, options = fake_provider.calls[-1]
    assert options.think is None

    resolved = fake_provider.resolve_options(options)
    assert resolved.think is False


def test_generate_reports_the_thinking_mode(client: TestClient) -> None:
    response = client.post("/api/ai/generate", json={"prompt": "Hello"})

    assert response.json()["thinking"] is False


def test_generate_accepts_thinking_true(client: TestClient) -> None:
    """A caller can opt into reasoning for a single request."""
    response = client.post(
        "/api/ai/generate", json={"prompt": "Hello", "think": True}
    )

    assert response.status_code == 200
    assert response.json()["thinking"] is True


def test_generate_accepts_thinking_false_explicitly(client: TestClient) -> None:
    response = client.post(
        "/api/ai/generate", json={"prompt": "Hello", "think": False}
    )

    assert response.json()["thinking"] is False


def test_thinking_must_be_a_boolean(client: TestClient) -> None:
    """A string is rejected rather than silently coerced."""
    response = client.post(
        "/api/ai/generate", json={"prompt": "Hello", "think": "yes"}
    )

    assert response.status_code == 422


def test_thinking_is_applied_to_streaming_too(client: TestClient, fake_provider) -> None:
    response = client.post(
        "/api/ai/generate/stream", json={"prompt": "Hello", "think": True}
    )

    assert response.status_code == 200
    _, options = fake_provider.calls[-1]
    assert options.think is True


# ---------- Independence from the backend ----------


def test_agents_endpoint_is_reachable_without_ollama_installed() -> None:
    """The agent layer must import without the Ollama SDK present.

    This is what makes a cloud provider a drop-in replacement: nothing in
    the agent or discovery path requires a local backend to exist.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, app.agents, app.prompts; "
            "print('ollama' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_agent_modules_have_no_ollama_dependency_at_runtime(
    fake_provider,
) -> None:
    """Agents resolve their provider through the abstraction only."""

    class ExampleAgent(BaseAgent):
        name = "example"
        version = "1.0.0"

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return AgentOutput.ok()

    agent = ExampleAgent(provider=fake_provider)

    assert agent.provider is fake_provider


def test_base_agent_resolves_options_through_the_provider(fake_provider) -> None:
    """An agent that only wants reasoning inherits the configured defaults."""
    from app.agents.base import BaseAgent

    class ExampleAgent(BaseAgent):
        name = "example"
        version = "1.0.0"

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return AgentOutput.ok()

    agent = ExampleAgent(provider=fake_provider)

    resolved = agent.resolve_options(GenerationOptions(think=True))

    assert resolved.think is True
    assert resolved.temperature == get_settings().default_ai_temperature


def test_base_agent_lazily_resolves_the_configured_provider() -> None:
    """Constructing an agent performs no I/O."""
    from app.agents.base import BaseAgent

    class ExampleAgent(BaseAgent):
        name = "example"
        version = "1.0.0"

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return AgentOutput.ok()

    agent = ExampleAgent()

    assert isinstance(agent.provider, AIProvider)

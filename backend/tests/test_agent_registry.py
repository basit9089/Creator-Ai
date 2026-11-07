"""Tests for AgentRegistry: discovery, lookup, versioning and unknown names."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.agents.base import (
    Agent,
    AgentNotFoundError,
    AgentOutput,
    Capability,
    PlaceholderAgent,
)
from app.agents.registry import (
    IMPLEMENTED_AGENT_NAMES,
    PLANNED_AGENT_NAMES,
    AgentAvailability,
    AgentRegistry,
    build_default_registry,
    get_registry,
)


class DummyAgent(Agent):
    name = "dummy"
    version = "1.0.0"
    description = "Does nothing useful."

    async def run(self, context, messages):  # noqa: ANN001, ANN201
        return AgentOutput.ok("done")


def named(name: str, version: str = "1.0.0", description: str = "") -> Agent:
    """Build a trivially executable agent with a given identity."""

    class _Agent(Agent):
        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return AgentOutput.ok(name)

    _Agent.name = name
    _Agent.version = version
    _Agent.description = description
    return _Agent()


# ---------- Registration ----------


def test_registry_starts_empty() -> None:
    assert len(AgentRegistry()) == 0


def test_register_makes_an_agent_available() -> None:
    registry = AgentRegistry()

    registry.register(DummyAgent())

    assert registry.has("dummy")
    assert len(registry) == 1


def test_get_agent_returns_the_registered_instance() -> None:
    """Identity matters: a caller may hold state on the agent."""
    registry = AgentRegistry()
    agent = DummyAgent()

    registry.register(agent)

    assert registry.get_agent("dummy") is agent


def test_duplicate_registration_is_rejected() -> None:
    """Silently replacing an agent would make behaviour unpredictable."""
    registry = AgentRegistry()
    registry.register(named("researcher", "1.0.0"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(named("researcher", "2.0.0"))


def test_duplicate_registration_can_be_overwritten_explicitly() -> None:
    registry = AgentRegistry()
    registry.register(named("researcher", "1.0.0"))
    replacement = named("researcher", "2.0.0")

    registry.register(replacement, overwrite=True)

    assert registry.get_agent("researcher") is replacement


def test_registering_an_empty_name_is_rejected() -> None:
    registry = AgentRegistry()

    with pytest.raises(ValueError, match="must not be empty"):
        registry.register(named("   "))


def test_unregister_removes_an_agent() -> None:
    registry = AgentRegistry()
    registry.register(DummyAgent())

    registry.unregister("dummy")

    assert not registry.has("dummy")


def test_unregistering_an_unknown_name_is_harmless() -> None:
    AgentRegistry().unregister("never-registered")


def test_registry_iterates_over_names() -> None:
    registry = AgentRegistry()
    registry.register(named("scriptwriter"))
    registry.register(named("researcher"))

    assert sorted(registry) == ["researcher", "scriptwriter"]


def test_membership_check_supports_in() -> None:
    registry = AgentRegistry()
    registry.register(DummyAgent())

    assert "dummy" in registry
    assert "missing" not in registry


# ---------- Lookup and unknown names ----------


def test_unknown_agent_raises_not_found() -> None:
    registry = AgentRegistry()

    with pytest.raises(AgentNotFoundError):
        registry.get_agent("nonexistent")


def test_unknown_agent_error_lists_what_is_available() -> None:
    """A typo should be actionable without a second lookup."""
    registry = AgentRegistry()
    registry.register(named("researcher"))

    with pytest.raises(AgentNotFoundError) as excinfo:
        registry.get_agent("reseracher")

    assert "researcher" in str(excinfo.value)


def test_unknown_agent_error_says_none_when_registry_is_empty() -> None:
    with pytest.raises(AgentNotFoundError) as excinfo:
        AgentRegistry().get_agent("researcher")

    assert "none" in str(excinfo.value)


def test_get_descriptor_raises_for_unknown_agent() -> None:
    with pytest.raises(AgentNotFoundError):
        AgentRegistry().get_descriptor("nonexistent")


# ---------- Descriptors and versioning ----------


def test_descriptor_reports_name_and_version() -> None:
    registry = AgentRegistry()
    registry.register(named("researcher", "1.2.0"))

    descriptor = registry.get_descriptor("researcher")

    assert descriptor.name == "researcher"
    assert descriptor.version == "1.2.0"
    assert descriptor.qualified_name == "researcher@1.2.0"


def test_registered_agent_is_available() -> None:
    registry = AgentRegistry()
    registry.register(DummyAgent())

    descriptor = registry.get_descriptor("dummy")

    assert descriptor.status is AgentAvailability.AVAILABLE
    assert descriptor.implemented is True


def test_version_can_change_when_an_agent_is_overwritten() -> None:
    """A version bump signals that behaviour changed."""
    registry = AgentRegistry()
    registry.register(named("scriptwriter", "1.0.0"))
    registry.register(named("scriptwriter", "1.1.0"), overwrite=True)

    assert registry.get_descriptor("scriptwriter").version == "1.1.0"


def test_descriptor_lists_capabilities_sorted() -> None:
    agent = named("fact_checker")
    agent.capabilities = frozenset(
        {Capability.REQUIRES_REASONING, Capability.STRUCTURED_OUTPUT}
    )
    registry = AgentRegistry()
    registry.register(agent)

    assert registry.get_descriptor("fact_checker").capabilities == [
        "requires_reasoning",
        "structured_output",
    ]


def test_descriptors_cover_every_registration() -> None:
    registry = AgentRegistry()
    registry.register(named("researcher"))
    registry.register(named("scriptwriter"))

    assert len(registry.descriptors()) == 2


def test_disabled_agent_is_not_implemented() -> None:
    registry = AgentRegistry()
    registry.register(DummyAgent())
    registry.disable("dummy")

    descriptor = registry.get_descriptor("dummy")

    assert descriptor.status is AgentAvailability.DISABLED
    assert descriptor.implemented is False


def test_disabling_an_unknown_agent_raises() -> None:
    with pytest.raises(AgentNotFoundError):
        AgentRegistry().disable("nonexistent")


def test_agent_describe_matches_registry_descriptor() -> None:
    agent = DummyAgent()
    registry = AgentRegistry()
    registry.register(agent)

    assert agent.describe()["name"] == registry.get_descriptor("dummy").name


# ---------- Placeholders ----------


def test_placeholder_is_registered_as_not_implemented() -> None:
    registry = AgentRegistry()

    registry.register_placeholder("researcher")

    descriptor = registry.get_descriptor("researcher")
    assert descriptor.status is AgentAvailability.NOT_IMPLEMENTED
    assert descriptor.implemented is False


def test_placeholder_can_be_looked_up() -> None:
    """Names resolve before the logic exists, so wiring can be written now."""
    registry = AgentRegistry()
    registry.register_placeholder("researcher")

    assert isinstance(registry.get_agent("researcher"), PlaceholderAgent)


def test_placeholder_has_a_version() -> None:
    registry = AgentRegistry()

    registry.register_placeholder("researcher", version="1.0.0")

    assert registry.get_descriptor("researcher").version == "1.0.0"


def test_placeholder_gets_a_default_description() -> None:
    registry = AgentRegistry()

    registry.register_placeholder("researcher")

    assert registry.get_descriptor("researcher").description


def test_real_implementation_supersedes_a_placeholder() -> None:
    """Shipping the researcher must not require removing a placeholder."""
    registry = AgentRegistry()
    registry.register_placeholder("researcher")

    registry.register(named("researcher", "1.0.0"))

    descriptor = registry.get_descriptor("researcher")
    assert descriptor.status is AgentAvailability.AVAILABLE


def test_placeholder_cannot_replace_a_real_implementation() -> None:
    """Registering a placeholder over working code must fail loudly.

    A silent no-op would hide a wiring mistake; an exception surfaces it at
    startup rather than as a mysteriously unimplemented agent later.
    """
    registry = AgentRegistry()
    registry.register(named("researcher", "1.0.0"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register_placeholder("researcher", version="9.9.9")

    assert registry.get_descriptor("researcher").version == "1.0.0"
    assert registry.get_descriptor("researcher").implemented is True


def test_placeholder_cannot_replace_another_placeholder() -> None:
    registry = AgentRegistry()
    registry.register_placeholder("researcher", version="1.0.0")

    with pytest.raises(ValueError):
        registry.register_placeholder("researcher", version="2.0.0")


# ---------- Default registry ----------


def test_default_registry_registers_the_six_planned_agents() -> None:
    registry = build_default_registry()

    for name in PLANNED_AGENT_NAMES:
        assert registry.has(name), f"{name} should be registered"


def test_planned_agents_are_the_expected_six() -> None:
    """The pipeline shape is fixed for now; changing it is a deliberate act."""
    assert PLANNED_AGENT_NAMES == (
        "researcher",
        "fact_checker",
        "scriptwriter",
        "video_producer",
        "captions_thumbnail",
        "publisher",
    )


def test_default_registry_registers_implemented_agents_as_available() -> None:
    """Phase 4 replaces the Researcher placeholder with the real agent.

    The other five stay placeholders until their phases. This test is the
    one place that asserts which side of that line each agent is on, so a
    future phase that implements an agent has to update it deliberately.
    """
    registry = build_default_registry()

    for name in IMPLEMENTED_AGENT_NAMES:
        assert registry.get_descriptor(name).implemented is True, (
            f"{name} should be registered as available"
        )


def test_default_registry_keeps_unimplemented_agents_as_placeholders() -> None:
    """The placeholders must survive: they keep discovery and lookup working."""
    registry = build_default_registry()

    unimplemented = [n for n in PLANNED_AGENT_NAMES if n not in IMPLEMENTED_AGENT_NAMES]

    assert unimplemented, "expected some agents to still be placeholders"

    for name in unimplemented:
        descriptor = registry.get_descriptor(name)
        assert descriptor.implemented is False, f"{name} should still be a placeholder"
        assert descriptor.status is AgentAvailability.NOT_IMPLEMENTED


def test_default_registry_researcher_is_the_real_agent() -> None:
    """The Researcher resolves to a real implementation, not a placeholder."""
    from app.agents.researcher.agent import ResearcherAgent

    registry = build_default_registry()

    agent = registry.get_agent("researcher")

    assert isinstance(agent, ResearcherAgent)
    assert not isinstance(agent, PlaceholderAgent)


def test_implemented_agent_names_are_a_subset_of_planned_names() -> None:
    """An implemented agent outside the pipeline shape would be a mistake."""
    assert set(IMPLEMENTED_AGENT_NAMES) <= set(PLANNED_AGENT_NAMES)


def test_default_registry_lists_in_pipeline_order() -> None:
    """Discovery output should read as the pipeline, not alphabetically."""
    registry = build_default_registry()

    assert registry.names() == PLANNED_AGENT_NAMES


def test_default_registry_descriptions_are_populated() -> None:
    registry = build_default_registry()

    for name in PLANNED_AGENT_NAMES:
        assert registry.get_descriptor(name).description


def test_default_registry_versions_are_v1() -> None:
    registry = build_default_registry()

    for name in PLANNED_AGENT_NAMES:
        assert registry.get_descriptor(name).version == "1.0.0"


def test_unknown_agents_sort_after_planned_ones() -> None:
    registry = build_default_registry()
    registry.register(named("transcriber"))

    names = registry.names()

    assert names[-1] == "transcriber"
    assert names[:-1] == PLANNED_AGENT_NAMES


def test_shared_registry_is_cached() -> None:
    assert get_registry() is get_registry()


def test_shared_registry_is_independent_of_a_fresh_build() -> None:
    """Building a new registry must not mutate the shared one."""
    build_default_registry().register(named("transcriber"))

    assert not get_registry().has("transcriber")


def test_two_registries_do_not_share_state() -> None:
    """Tests rely on isolation between registry instances."""
    first = AgentRegistry()
    second = AgentRegistry()

    first.register(DummyAgent())

    assert not second.has("dummy")


# ---------- Independence from the AI backend ----------


def test_agent_modules_do_not_import_a_backend_sdk() -> None:
    """No agent module may import a concrete AI backend.

    Importing ``ollama`` from an agent would tie the agent to one backend,
    and a cloud provider could no longer be substituted later. Docstrings
    are allowed to mention Ollama; actual imports are not.
    """
    import ast
    import app.agents.base
    import app.agents.registry
    import app.agents.runner

    modules = (app.agents.base, app.agents.registry, app.agents.runner)

    for module in modules:
        source = module.__file__
        assert source is not None
        with open(source, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())

        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        for name in imported:
            assert not name.startswith("ollama"), (
                f"{module.__name__} imports {name}"
            )
            assert "ollama_provider" not in name, f"{module.__name__} imports {name}"


def test_registry_import_does_not_pull_in_the_ollama_sdk() -> None:
    """Importing the agent layer must not construct a backend client."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.agents; "
            "print('ollama' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_agent_can_be_defined_without_a_provider() -> None:
    """Construction performs no I/O, so agents are cheap to build."""

    class Standalone(Agent):
        name = "standalone"
        version = "1.0.0"

        async def run(self, context, messages: Sequence[object]) -> AgentOutput:
            return AgentOutput.ok()

    agent = Standalone()

    assert agent.qualified_name == "standalone@1.0.0"

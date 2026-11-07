"""Discovery of available agents.

The registry exists so agent selection is data, not code. A route or a
pipeline asks for ``"researcher"`` and receives an agent; it never
imports a concrete class. That indirection is what lets the six planned
agents be registered as placeholders today and swapped for real
implementations later without touching their callers.

Versioning
----------
Every registration carries a version. It is part of the agent's identity
in results and logs, so a finished execution always records which
behaviour produced it, even after the implementation moves on.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import (
    Agent,
    AgentNotFoundError,
    Capability,
    PlaceholderAgent,
)
from app.core.logging import get_logger

logger = get_logger("agents.registry")


class AgentAvailability(StrEnum):
    """Whether a registered agent can actually be executed.

    Separate from :class:`~app.agents.base.AgentStatus`, which describes a
    single *execution*. Availability is a property of the registration
    and does not change between runs.
    """

    #: Registered and executable.
    AVAILABLE = "available"

    #: Name and contract exist, logic does not. Executing it fails with a
    #: retryable error, since the implementation is still to come.
    NOT_IMPLEMENTED = "not_implemented"

    #: Registered but deliberately disabled.
    DISABLED = "disabled"


class AgentDescriptor(BaseModel):
    """Serialisable description of one registered agent."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str = ""
    status: AgentAvailability = AgentAvailability.AVAILABLE
    capabilities: list[str] = Field(default_factory=list)

    @property
    def implemented(self) -> bool:
        """Whether the agent can be executed today."""
        return self.status is AgentAvailability.AVAILABLE

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.version}"


class AgentRegistry:
    """A name-to-agent mapping.

    Instances are independent: tests build their own rather than
    mutating the shared registry returned by :func:`get_registry`.
    """

    __slots__ = ("_agents", "_availability")

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._availability: dict[str, AgentAvailability] = {}

    # ---------- Registration ----------

    def register(self, agent: Agent, *, overwrite: bool = False) -> Agent:
        """Register an executable agent.

        Args:
            overwrite: replace an existing registration with the same name.

        Raises:
            ValueError: the name is taken and ``overwrite`` is False.
        """
        self._store(agent.name, agent, AgentAvailability.AVAILABLE, overwrite=overwrite)
        logger.info("agent_registered agent=%s version=%s", agent.name, agent.version)
        return agent

    def register_placeholder(
        self,
        name: str,
        *,
        version: str = "1.0.0",
        description: str = "",
        capabilities: Sequence[Capability] = (),
        overwrite: bool = False,
    ) -> PlaceholderAgent:
        """Register a contract whose implementation does not exist yet.

        Never replaces an existing registration. A real implementation
        registered under the same name takes precedence, so a shipped
        agent cannot be silently downgraded to a placeholder by a later
        call to this method.
        """
        agent = PlaceholderAgent()
        agent.name = name
        agent.version = version
        agent.description = description or "Planned agent. Not implemented yet."
        agent.capabilities = frozenset(capabilities)

        self._store(
            name,
            agent,
            AgentAvailability.NOT_IMPLEMENTED,
            overwrite=overwrite,
            # A placeholder never replaces anything. It exists to fill a
            # gap, so registering one must not clobber real behaviour.
            allow_replacing_placeholder=False,
        )
        logger.info("agent_placeholder_registered agent=%s version=%s", name, version)
        return agent

    def disable(self, name: str) -> None:
        """Mark a registered agent as present but not runnable."""
        self._require(name)
        self._availability[name] = AgentAvailability.DISABLED

    def unregister(self, name: str) -> None:
        """Remove a registration. Unknown names are ignored."""
        self._agents.pop(name, None)
        self._availability.pop(name, None)

    # ---------- Lookup ----------

    def get_agent(self, name: str) -> Agent:
        """Return the agent registered under ``name``.

        Raises:
            AgentNotFoundError: nothing is registered under that name.
                The message lists what *is* registered, which turns a typo
                into an immediately actionable error.
        """
        try:
            return self._agents[name]
        except KeyError:
            available = ", ".join(self.names()) or "none"
            raise AgentNotFoundError(
                f"No agent registered under {name!r}. Available agents: {available}."
            ) from None

    def get_descriptor(self, name: str) -> AgentDescriptor:
        """Return the serialisable description of one agent."""
        agent = self.get_agent(name)
        return self.describe(agent)

    def describe(self, agent: Agent) -> AgentDescriptor:
        """Build a descriptor for ``agent`` using its registration status."""
        return AgentDescriptor(
            name=agent.name,
            version=agent.version,
            description=agent.description,
            status=self._availability.get(agent.name, AgentAvailability.AVAILABLE),
            capabilities=sorted(capability.value for capability in agent.capabilities),
        )

    def has(self, name: str) -> bool:
        return name in self._agents

    def names(self) -> tuple[str, ...]:
        """Registered names in pipeline order, then alphabetically."""
        known = [name for name in PLANNED_AGENT_NAMES if name in self._agents]
        extra = sorted(name for name in self._agents if name not in PLANNED_AGENT_NAMES)
        return (*known, *extra)

    def descriptors(self) -> tuple[AgentDescriptor, ...]:
        """Every registration, described."""
        return tuple(self.describe(self._agents[name]) for name in self.names())

    # ---------- Container protocol ----------

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._agents)

    # ---------- Internals ----------

    def _store(
        self,
        name: str,
        agent: Agent,
        status: AgentAvailability,
        *,
        overwrite: bool,
        allow_replacing_placeholder: bool = True,
    ) -> None:
        if not name.strip():
            raise ValueError("Agent name must not be empty")

        existing = self._agents.get(name)
        if existing is not None and not overwrite:
            replacing_placeholder = (
                allow_replacing_placeholder
                and self._availability.get(name) is AgentAvailability.NOT_IMPLEMENTED
            )
            if not replacing_placeholder:
                raise ValueError(
                    f"Agent {name!r} is already registered as "
                    f"{existing.qualified_name}. Pass overwrite=True to replace it."
                )

        self._agents[name] = agent
        self._availability[name] = status

    def _require(self, name: str) -> Agent:
        if name not in self._agents:
            raise AgentNotFoundError(f"No agent registered under {name!r}.")
        return self._agents[name]


#: The pipeline order Creator AI is planned around. Used for a stable,
#: readable listing and to define the placeholders registered by default.
PLANNED_AGENT_NAMES: tuple[str, ...] = (
    "researcher",
    "fact_checker",
    "scriptwriter",
    "video_producer",
    "captions_thumbnail",
    "publisher",
)

#: One-line purpose per planned agent, surfaced by the agents endpoint.
PLANNED_AGENT_DESCRIPTIONS: dict[str, str] = {
    "researcher": "Gathers background material on a topic and extracts structured findings.",
    "fact_checker": "Verifies claims against the supplied material and reports verdicts.",
    "scriptwriter": "Turns researched material into a narration script for a video.",
    "video_producer": "Plans or renders the video assets for a script.",
    "captions_thumbnail": "Produces captions and thumbnail concepts for a finished video.",
    "publisher": "Prepares a finished video for upload and publishes it.",
}

#: Agents that have a real implementation. Everything else in
#: :data:`PLANNED_AGENT_NAMES` is still a placeholder until its phase.
IMPLEMENTED_AGENT_NAMES: tuple[str, ...] = ("researcher", "fact_checker", "scriptwriter")


def build_default_registry() -> AgentRegistry:
    """Create a registry populated with the implemented agents.

    Implemented agents are registered as executable; the remaining
    planned names stay placeholders, so every name still resolves and
    discovery keeps working while the later agents are built.

    Registration order is deliberate: implementations go in first, and the
    placeholder loop skips them. Registering a placeholder over a real
    implementation is refused by
    :meth:`AgentRegistry.register_placeholder`, so a shipped agent cannot
    be silently downgraded by a later edit to this function.
    """
    from app.agents.scriptwriter.agent import ScriptwriterAgent
    from app.agents.fact_checker.agent import FactCheckerAgent
    from app.agents.researcher.agent import ResearcherAgent

    registry = AgentRegistry()
    registry.register(ScriptwriterAgent())
    registry.register(FactCheckerAgent())
    registry.register(ResearcherAgent())

    for name in PLANNED_AGENT_NAMES:
        if name in IMPLEMENTED_AGENT_NAMES:
            continue
        registry.register_placeholder(
            name,
            version="1.0.0",
            description=PLANNED_AGENT_DESCRIPTIONS[name],
        )

    return registry


@lru_cache(maxsize=1)
def get_registry() -> AgentRegistry:
    """Return the shared registry.

    Cached because registrations are process-wide configuration. Tests
    that need isolation construct their own :class:`AgentRegistry`.
    """
    return build_default_registry()

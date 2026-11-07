"""Core agent contracts.

Everything here is backend-independent and side-effect free. In
particular this module must never import a provider implementation: an
agent depends on :class:`~app.services.ai.provider.AIProvider`, which is
what allows a cloud provider to be substituted later without touching a
single agent.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from app.services.ai.provider import AIProvider, GenerationOptions, Message


class AgentStatus(StrEnum):
    """Lifecycle state of a single agent execution.

    ``PENDING`` and ``CANCELLED`` complete the set: an execution that has
    an identifier but has not started is ``PENDING``, and a run
    abandoned before completion is ``CANCELLED`` rather than ``FAILED``,
    since nothing went wrong.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition is expected."""
        return self in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }


class Capability(StrEnum):
    """Optional features an agent may support.

    Informational in this phase: the registry reports capabilities so a
    caller can see what an agent offers, without any agent actually
    implementing streaming or structured output yet.
    """

    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    REQUIRES_REASONING = "requires_reasoning"
    RETRYABLE = "retryable"


class AgentError(Exception):
    """Base class for agent failures.

    Carries a machine-readable ``code`` and a client-safe ``message``.
    The original exception, when there is one, stays on ``__cause__`` for
    server-side logs.
    """

    code = "agent_error"
    #: Whether re-running the agent unchanged could plausibly succeed.
    #: Infrastructure failures are retryable; bad input usually is not.
    retryable = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class AgentNotFoundError(AgentError):
    """No agent is registered under the requested name."""

    code = "agent_not_found"
    retryable = False


class AgentNotImplementedError(AgentError):
    """The agent is registered as a placeholder and has no logic yet."""

    code = "agent_not_implemented"
    #: Implementations land in later phases, so a retry may well succeed.
    retryable = True


def new_execution_id() -> str:
    """Generate a unique identifier for one agent execution.

    A UUID4 is sufficient and dependency-free. If execution records later
    move to a database, a prefixed ULID would order better, but nothing
    depends on ordering today.
    """
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Current time as an aware UTC datetime.

    Aware datetimes throughout, so durations are never ambiguous when
    comparing timestamps from different systems.
    """
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything an agent may need beyond its own input.

    Values are passed explicitly; there is no authentication or database
    lookup yet. Identifiers are optional because an agent can be run
    standalone, outside any project.
    """

    execution_id: str = field(default_factory=new_execution_id)

    project_id: str | None = None
    channel_id: str | None = None
    user_id: str | None = None

    #: Free-form agent configuration, for example temperature overrides.
    config: dict[str, Any] = field(default_factory=dict)

    #: Files available to this run. Contents are not loaded here.
    available_files: tuple[str, ...] = ()

    #: Outputs from earlier pipeline stages, keyed by stage name.
    previous_stage_outputs: dict[str, Any] = field(default_factory=dict)

    #: Arbitrary caller metadata, carried through to the result.
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_execution_id(self, execution_id: str) -> AgentContext:
        """Return a copy bound to a specific execution."""
        return replace(self, execution_id=execution_id)

    def option(self, key: str, default: Any = None) -> Any:
        """Read a configuration value."""
        return self.config.get(key, default)

    def previous_output(self, stage: str, default: Any = None) -> Any:
        """Read a previous stage's output."""
        return self.previous_stage_outputs.get(stage, default)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Outcome of one agent execution.

    Success and status are related but not redundant: ``success`` answers
    "did the agent produce usable output?", which an agent may report as
    ``False`` while still completing normally, for example a Fact Checker
    that found the material insufficient.
    """

    agent_name: str
    agent_version: str
    execution_id: str
    status: AgentStatus

    success: bool = False
    data: Any = None
    error: str | None = None

    project_id: str | None = None
    channel_id: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_retryable(self) -> bool:
        """Whether re-running this agent unchanged is worth attempting.

        Derived from metadata rather than from the exception, because the
        runner records the failure reason in metadata while the original
        exception may not survive.
        """
        return bool(self.metadata.get("retryable", False))

    def failure_reason(self) -> str | None:
        """Human-readable failure summary, safe to log."""
        if self.status is AgentStatus.FAILED:
            error_type = self.metadata.get("error_type")
            return f"{error_type}: {self.error}" if error_type else self.error
        return None

    @classmethod
    def from_success(
        cls,
        *,
        agent_name: str,
        agent_version: str,
        execution_id: str,
        data: Any = None,
        project_id: str | None = None,
        channel_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Build a successful, completed result."""
        return cls(
            agent_name=agent_name,
            agent_version=agent_version,
            execution_id=execution_id,
            status=AgentStatus.COMPLETED,
            success=True,
            data=data,
            project_id=project_id,
            channel_id=channel_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    @classmethod
    def from_failure(
        cls,
        *,
        agent_name: str,
        agent_version: str,
        execution_id: str,
        error: str,
        status: AgentStatus = AgentStatus.FAILED,
        data: Any = None,
        project_id: str | None = None,
        channel_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Build a failed (or cancelled) result.

        ``data`` is available for an agent that completed but could not
        deliver a usable verdict: a Researcher that retrieved four of the
        six sources it wanted has produced real work worth keeping. It is
        only ever set alongside a terminal status, never on a success.
        """
        return cls(
            agent_name=agent_name,
            agent_version=agent_version,
            execution_id=execution_id,
            status=status,
            success=False,
            data=data,
            error=error,
            project_id=project_id,
            channel_id=channel_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class AgentOutput:
    """What an agent hands back to the runner.

    Distinct from :class:`AgentResult`: this is the agent's *answer*,
    while the result wraps it with execution metadata the runner owns.
    Keeping them separate means an agent never has to fabricate a
    duration or status.
    """

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any = None, **metadata: Any) -> AgentOutput:
        """Successful output carrying ``data``."""
        return cls(success=True, data=data, metadata=dict(metadata))

    @classmethod
    def failed(cls, error: str, **metadata: Any) -> AgentOutput:
        """Unsuccessful output carrying a client-safe reason."""
        return cls(success=False, error=error, metadata=dict(metadata))


class Agent(ABC):
    """Base class for a versioned, independently retryable agent.

    A single agent performs one job. Agents do not call each other; the
    runner invokes them one at a time, so a failure is isolated and the
    failing stage alone can be retried.
    """

    #: Stable identifier used for registry lookup, e.g. ``"researcher"``.
    name: str = "unnamed"

    #: Semantic version of this agent's behaviour and prompts. Bump when
    #: behaviour changes so an old execution stays interpretable.
    version: str = "1.0.0"

    #: One-line description surfaced by the agents endpoint.
    description: str = ""

    capabilities: frozenset[Capability] = frozenset()

    @abstractmethod
    async def run(self, context: AgentContext, messages: Sequence[Message]) -> AgentOutput:
        """Perform the agent's work.

        Args:
            context: execution identifiers, configuration and any previous
                stage outputs.
            messages: the assembled prompt. Supplied by the runner so the
                agent does not rebuild its own prompt.

        Returns:
            An :class:`AgentOutput`. Raising is also acceptable: the runner
            converts an exception into a failed result either way.
        """

    @property
    def qualified_name(self) -> str:
        """Name and version, for logs and results."""
        return f"{self.name}@{self.version}"

    def describe(self) -> dict[str, Any]:
        """Serialisable description, used by the agents endpoint."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": sorted(c.value for c in self.capabilities),
        }


class PlaceholderAgent(Agent):
    """A registered agent whose implementation is not written yet.

    Registered so the six planned agent names resolve today, keeping
    discovery and lookup working. Executing one raises
    :class:`AgentNotImplementedError`, which the runner records as a
    retryable failure: the logic is coming in a later phase.
    """

    async def run(self, context: AgentContext, messages: Sequence[Message]) -> AgentOutput:
        raise AgentNotImplementedError(
            f"Agent {self.qualified_name!r} is registered but not implemented yet."
        )


class BaseAgent(Agent):
    """Convenience base that injects a provider into the agent.

    Agents depend on :class:`~app.services.ai.provider.AIProvider` rather
    than on Ollama. This base simply resolves the configured provider, so
    an agent can call ``self.provider`` without repeating the lookup. It
    performs no I/O at construction time.
    """

    def __init__(self, provider: AIProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> AIProvider:
        """The configured AI provider, resolved on first access.

        Typed as the abstraction, never a concrete backend. An agent
        written against this cannot tell Ollama from a cloud API, which is
        exactly what makes the later migration a configuration change.
        """
        if self._provider is None:
            from app.dependencies import get_ai_provider

            self._provider = get_ai_provider()
        return self._provider

    def resolve_options(self, overrides: GenerationOptions | None = None) -> GenerationOptions:
        """Merge caller overrides with the provider's deployment defaults.

        Agents should use this rather than constructing full options, so an
        agent that only wants reasoning enabled inherits the configured
        temperature and output cap.
        """
        return self.provider.resolve_options(overrides)

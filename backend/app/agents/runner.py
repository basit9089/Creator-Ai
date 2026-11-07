"""In-process execution of a single agent.

The runner owns everything an agent should not have to think about:
execution identifiers, lifecycle status, timing, failure capture and
structured logging. An agent only implements
:meth:`~app.agents.base.Agent.run` and returns data.

Why a separate layer
--------------------
Keeping execution concerns out of the agents is what makes a later swap
to a real task queue cheap. :class:`AgentRunner` is called one agent at a
time and knows nothing about ordering, so replacing it with a Redis or
Celery worker means reimplementing this class, not the agents.

Execution is sequential by design. The target hardware is an i5-10210U
with 16 GB of RAM and integrated graphics, running a single 8B model
locally: two agents generating at once would contend for the same CPU and
memory and make both slower, not faster.

Failure isolation
-----------------
A failed agent does not invalidate earlier stages. Each execution is
recorded with the context it ran under, so a later retry can re-run that
one agent with its inputs intact instead of restarting the pipeline. No
automatic retry is performed here: silently repeating work on a local
machine with a shared model is a worse default than surfacing the
failure.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import anyio

from app.agents.base import (
    Agent,
    AgentContext,
    AgentError,
    AgentOutput,
    AgentResult,
    AgentStatus,
    new_execution_id,
    utc_now,
)
from app.core.logging import get_logger

logger = get_logger("agents.runner")

#: How many finished executions to remember. Enough for a developer to
#: inspect and retry recent work; old records are dropped rather than
#: growing without bound in a long-lived process.
DEFAULT_HISTORY_LIMIT = 200


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Everything known about one execution, kept for retry and inspection.

    The context is retained so a retry reuses the same inputs. It is an
    immutable snapshot, so mutating the caller's original context later
    cannot change what a retry replays.
    """

    agent_name: str
    agent_version: str
    execution_id: str
    status: AgentStatus
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = None
    context: AgentContext | None = None
    messages: tuple[Any, ...] = ()
    result: AgentResult | None = None
    attempt: int = 1

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


@dataclass(slots=True)
class AgentRunner:
    """Executes agents and records their outcomes.

    Attributes:
        execution_timeout: upper bound in seconds for a single execution.
            ``None`` uses the configured ``AI_REQUEST_TIMEOUT``.
        history_limit: number of finished executions retained for retry.
        sequential: kept as an explicit attribute to document the current
            single-agent-at-a-time policy.
    """

    execution_timeout: float | None = None
    history_limit: int = DEFAULT_HISTORY_LIMIT
    sequential: bool = True

    _history: "OrderedDict[str, ExecutionRecord]" = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    # ---------- Public API ----------

    async def run(
        self,
        agent: Agent,
        context: AgentContext | None = None,
        messages: Sequence[Any] | None = None,
        *,
        attempt: int = 1,
    ) -> AgentResult:
        """Execute ``agent`` once and return its result.

        Never raises for agent failures: an exception becomes a
        :class:`~app.agents.base.AgentResult` with ``FAILED`` status, so
        a caller recording progress through a pipeline does not have to
        guard every call.
        """
        execution_id = new_execution_id()
        run_context = (context or AgentContext()).with_execution_id(execution_id)
        # Copied into a list so the agent receives a plain, mutable
        # sequence matching the ``Sequence[Message]`` contract, and so
        # mutating it cannot reach back into the caller's data.
        prompt: list[Any] = list(messages or ())

        record = ExecutionRecord(
            agent_name=agent.name,
            agent_version=agent.version,
            execution_id=execution_id,
            status=AgentStatus.RUNNING,
            started_at=utc_now(),
            context=run_context,
            messages=tuple(prompt),
            attempt=attempt,
        )
        self._remember(record)

        logger.info(
            "agent_started agent=%s version=%s execution_id=%s project_id=%s "
            "channel_id=%s attempt=%d prompt_messages=%d",
            agent.name,
            agent.version,
            execution_id,
            run_context.project_id,
            run_context.channel_id,
            attempt,
            len(prompt),
        )

        started = time.perf_counter()
        try:
            output = await self._invoke(agent, run_context, prompt)
        except anyio.get_cancelled_exc_class():
            # The task was cancelled by the caller. Nothing went wrong, so
            # the execution is CANCELLED rather than FAILED. Cancellation
            # is re-raised so it still propagates to the caller.
            cancelled = self._finish(
                record,
                status=AgentStatus.CANCELLED,
                error="Execution cancelled before completion.",
                started=started,
                extra_metadata={"retryable": True, "error_type": "Cancelled"},
            )
            self._remember(_finalise_record(record, cancelled))
            logger.info(
                "agent_cancelled agent=%s version=%s execution_id=%s project_id=%s "
                "duration_ms=%.2f",
                agent.name,
                agent.version,
                execution_id,
                run_context.project_id,
                self._elapsed_ms(started),
            )
            raise
        except Exception as exc:  # noqa: BLE001 - captured, converted to a result
            result = self._record_failure(agent, record, exc, started)
        else:
            result = self._record_output(agent, record, output, started)

        self._remember(_finalise_record(record, result))
        return result

    async def run_named(
        self,
        name: str,
        context: AgentContext | None = None,
        messages: Sequence[Any] | None = None,
        *,
        registry: Any = None,
    ) -> AgentResult:
        """Resolve ``name`` through the registry and execute it.

        An unknown name produces a ``FAILED`` result rather than an
        exception, matching :meth:`run`. Agents are addressed by name so
        API code never hardcodes a class.
        """
        from app.agents.base import AgentNotFoundError
        from app.agents.registry import get_registry

        active_registry = registry if registry is not None else get_registry()
        try:
            agent = active_registry.get_agent(name)
        except AgentNotFoundError as exc:
            stub = _StubAgent(name)
            record = ExecutionRecord(
                agent_name=name,
                agent_version=stub.version,
                execution_id=new_execution_id(),
                status=AgentStatus.RUNNING,
                started_at=utc_now(),
                context=context,
                messages=tuple(messages or ()),
            )
            started = time.perf_counter()
            result = self._record_failure(stub, record, exc, started)
            self._remember(_finalise_record(record, result))
            return result

        return await self.run(agent, context, messages)

    async def retry(
        self,
        execution_id: str,
        *,
        agent: Agent | None = None,
        registry: Any = None,
    ) -> AgentResult:
        """Re-run a single failed execution, leaving other stages untouched.

        The original agent, context and prompt are reused so the retry is
        genuinely the same work. A new execution ID is assigned and
        ``attempt`` is incremented, keeping the first attempt's history
        intact.
        """
        record = self._history.get(execution_id)
        if record is None:
            raise KeyError(f"No execution recorded for id {execution_id!r}")

        target = agent
        if target is None:
            from app.agents.registry import get_registry

            active_registry = registry if registry is not None else get_registry()
            target = active_registry.get_agent(record.agent_name)

        return await self.run(
            target,
            record.context,
            list(record.messages),
            attempt=record.attempt + 1,
        )

    def record(self, execution_id: str) -> ExecutionRecord | None:
        """Return a recorded execution, if still retained."""
        return self._history.get(execution_id)

    def history(self) -> tuple[ExecutionRecord, ...]:
        """Recorded executions, oldest first."""
        return tuple(self._history.values())

    # ---------- Internals ----------

    async def _invoke(
        self,
        agent: Agent,
        context: AgentContext,
        messages: list[Any],
    ) -> AgentOutput:
        """Call the agent, applying the execution timeout."""
        timeout = self.execution_timeout
        if timeout is None:
            from app.core.config import get_settings

            timeout = get_settings().ai_request_timeout

        with anyio.fail_after(timeout):
            outcome = await agent.run(context, messages)

        # An agent may legitimately return None or a bare payload; wrap
        # anything that is not already an AgentOutput so downstream code
        # has one shape to handle.
        if isinstance(outcome, AgentOutput):
            return outcome

        return AgentOutput(success=outcome is not None, data=outcome)

    def _record_output(
        self,
        agent: Agent,
        record: ExecutionRecord,
        output: AgentOutput,
        started: float,
    ) -> AgentResult:
        """Turn a returned output into a completed or failed result.

        An agent that returns ``success=False`` is a legitimate outcome
        (for example, material that cannot support a claim), so status
        depends on the agent's own verdict rather than on the absence of
        an exception.
        """
        duration_ms = self._elapsed_ms(started)

        if output.success:
            logger.info(
                "agent_completed agent=%s version=%s execution_id=%s "
                "project_id=%s duration_ms=%.2f",
                agent.name,
                agent.version,
                record.execution_id,
                record.context.project_id if record.context else None,
                duration_ms,
            )
            return AgentResult.from_success(
                agent_name=agent.name,
                agent_version=agent.version,
                execution_id=record.execution_id,
                data=output.data,
                project_id=record.context.project_id if record.context else None,
                channel_id=record.context.channel_id if record.context else None,
                started_at=record.started_at,
                completed_at=utc_now(),
                duration_ms=duration_ms,
                metadata={
                    **_context_metadata(record),
                    **output.metadata,
                    "attempt": record.attempt,
                },
            )

        logger.warning(
            "agent_unsuccessful agent=%s version=%s execution_id=%s "
            "project_id=%s duration_ms=%.2f reason=%s",
            agent.name,
            agent.version,
            record.execution_id,
            record.context.project_id if record.context else None,
            duration_ms,
            output.error or "no reason given",
        )
        # The agent's data is preserved: an execution that completed but
        # could not deliver a usable verdict still produced real work, and
        # a pipeline stage or a retry needs to see it.
        return AgentResult.from_failure(
            agent_name=agent.name,
            agent_version=agent.version,
            execution_id=record.execution_id,
            error=output.error or "Agent reported failure without a reason.",
            status=AgentStatus.COMPLETED,
            data=output.data,
            project_id=record.context.project_id if record.context else None,
            channel_id=record.context.channel_id if record.context else None,
            started_at=record.started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms,
            metadata={
                **_context_metadata(record),
                **output.metadata,
                "attempt": record.attempt,
                "retryable": bool(output.metadata.get("retryable", False)),
            },
        )

    def _record_failure(
        self,
        agent: Agent,
        record: ExecutionRecord,
        exc: BaseException,
        started: float,
    ) -> AgentResult:
        """Convert an exception into a failed result.

        ``result.error`` stays client-safe. The full exception text is
        kept in ``metadata['error_detail']`` for server-side
        diagnostics and must never be sent to a client.
        """
        duration_ms = self._elapsed_ms(started)

        if isinstance(exc, AgentError):
            safe_message = exc.message
            retryable = exc.retryable
        elif isinstance(exc, TimeoutError):
            safe_message = "Agent execution exceeded its time limit."
            retryable = True
        else:
            # An unexpected error: the type is safe to name, the message
            # is not assumed to be.
            safe_message = f"{type(exc).__name__} raised during agent execution."
            retryable = False

        logger.exception(
            "agent_failed agent=%s version=%s execution_id=%s project_id=%s "
            "duration_ms=%.2f error_type=%s retryable=%s",
            agent.name,
            agent.version,
            record.execution_id,
            record.context.project_id if record.context else None,
            duration_ms,
            type(exc).__name__,
            retryable,
        )

        return AgentResult.from_failure(
            agent_name=agent.name,
            agent_version=agent.version,
            execution_id=record.execution_id,
            error=safe_message,
            status=AgentStatus.FAILED,
            project_id=record.context.project_id if record.context else None,
            channel_id=record.context.channel_id if record.context else None,
            started_at=record.started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms,
            metadata={
                **_context_metadata(record),
                "error_type": type(exc).__name__,
                # The agent's own machine-readable code, when it has one.
                # An agent that classified its failure (the Researcher does)
                # has already chosen what a client should branch on, so the
                # classification is carried through rather than discarded
                # and replaced with a generic 500 at the HTTP layer.
                "code": exc.code if isinstance(exc, AgentError) else None,
                # Server-side only. Never expose this over HTTP.
                "error_detail": str(exc),
                "retryable": retryable,
                "attempt": record.attempt,
            },
        )

    def _finish(
        self,
        record: ExecutionRecord,
        *,
        status: AgentStatus,
        error: str,
        started: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Build a terminal result for a non-agent-error outcome."""
        duration_ms = self._elapsed_ms(started)
        return AgentResult.from_failure(
            agent_name=record.agent_name,
            agent_version=record.agent_version,
            execution_id=record.execution_id,
            error=error,
            status=status,
            project_id=record.context.project_id if record.context else None,
            channel_id=record.context.channel_id if record.context else None,
            started_at=record.started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms,
            metadata={
                **_context_metadata(record),
                **(extra_metadata or {}),
                "attempt": record.attempt,
            },
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        """Milliseconds since ``started``, using a monotonic clock."""
        return round((time.perf_counter() - started) * 1000, 2)

    @staticmethod
    def _project_id(record: ExecutionRecord) -> str | None:
        return record.context.project_id if record.context else None

    def _remember(self, record: ExecutionRecord) -> None:
        """Store a record, evicting the oldest once the limit is reached."""
        self._history[record.execution_id] = record
        while len(self._history) > self.history_limit:
            self._history.popitem(last=False)


class _StubAgent(Agent):
    """Stand-in used to report a failure for an agent that does not exist.

    Lets an unknown name produce a normal ``FAILED`` result with the same
    shape as any other failure, without inventing a registered agent.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.version = "unknown"
        self.description = "Placeholder for an unregistered agent name."

    async def run(self, context: AgentContext, messages: Sequence[Any]) -> AgentOutput:
        from app.agents.base import AgentNotFoundError

        raise AgentNotFoundError(f"No agent registered under {self.name!r}.")


def _context_metadata(record: ExecutionRecord) -> dict[str, Any]:
    """Caller metadata from the context, or an empty mapping."""
    return dict(record.context.metadata) if record.context else {}


def _finalise_record(record: ExecutionRecord, result: AgentResult) -> ExecutionRecord:
    """Return ``record`` updated from ``result``.

    ``ExecutionRecord`` is frozen, so a new instance is produced rather
    than mutating the in-flight one.
    """
    return replace(
        record,
        status=result.status,
        completed_at=result.completed_at,
        duration_ms=result.duration_ms,
        result=result,
    )

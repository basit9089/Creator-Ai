"""Tests for AgentRunner: execution, timing, failure capture and retry."""

from __future__ import annotations

import time
from collections.abc import Sequence

import anyio
import pytest

from app.agents.base import (
    Agent,
    AgentContext,
    AgentError,
    AgentOutput,
    AgentStatus,
)
from app.agents.registry import AgentRegistry
from app.agents.runner import AgentRunner
from app.services.ai.provider import Message

PROMPT = [Message(role="user", content="Research recursion")]


class RecordingAgent(Agent):
    """Succeeds and remembers what it was given."""

    name = "recording"
    version = "1.0.0"

    def __init__(self, data: object = None) -> None:
        self.data = data if data is not None else {"topic": "recursion"}
        self.seen_context: AgentContext | None = None
        self.seen_messages: Sequence[Message] | None = None

    async def run(self, context, messages):  # noqa: ANN001, ANN201
        self.seen_context = context
        self.seen_messages = messages
        return AgentOutput.ok(self.data)


class RaisingAgent(Agent):
    """Always fails, with a configurable exception."""

    name = "raising"
    version = "2.0.0"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("model exploded")

    async def run(self, context, messages):  # noqa: ANN001, ANN201
        raise self.error


class UnsuccessfulAgent(Agent):
    """Completes normally but reports an unusable outcome."""

    name = "unsuccessful"
    version = "1.0.0"

    async def run(self, context, messages):  # noqa: ANN001, ANN201
        return AgentOutput.failed("no material supplied")


class SlowAgent(Agent):
    name = "slow"
    version = "1.0.0"

    def __init__(self, seconds: float = 0.5) -> None:
        self.seconds = seconds

    async def run(self, context, messages):  # noqa: ANN001, ANN201
        await anyio.sleep(self.seconds)
        return AgentOutput.ok("late")


@pytest.fixture
def runner() -> AgentRunner:
    return AgentRunner()


# ---------- Successful execution ----------


@pytest.mark.anyio
async def test_successful_agent_completes(runner: AgentRunner) -> None:
    result = await runner.run(RecordingAgent())

    assert result.success is True
    assert result.status is AgentStatus.COMPLETED
    assert result.data == {"topic": "recursion"}
    assert result.error is None


@pytest.mark.anyio
async def test_result_carries_agent_name_and_version(runner: AgentRunner) -> None:
    result = await runner.run(RecordingAgent())

    assert result.agent_name == "recording"
    assert result.agent_version == "1.0.0"


@pytest.mark.anyio
async def test_each_run_gets_a_fresh_execution_id(runner: AgentRunner) -> None:
    first = await runner.run(RecordingAgent())
    second = await runner.run(RecordingAgent())

    assert first.execution_id != second.execution_id


@pytest.mark.anyio
async def test_context_receives_the_execution_id(runner: AgentRunner) -> None:
    """The agent's context must carry the id the result reports.

    Otherwise an agent cannot log or annotate its own execution.
    """
    agent = RecordingAgent()

    result = await runner.run(agent, AgentContext(project_id="p1"))

    assert agent.seen_context is not None
    assert agent.seen_context.execution_id == result.execution_id


@pytest.mark.anyio
async def test_prompt_is_passed_through_untouched(runner: AgentRunner) -> None:
    agent = RecordingAgent()

    await runner.run(agent, None, PROMPT)

    assert agent.seen_messages == PROMPT


@pytest.mark.anyio
async def test_result_carries_project_and_channel_ids(runner: AgentRunner) -> None:
    result = await runner.run(
        RecordingAgent(), AgentContext(project_id="p1", channel_id="c1")
    )

    assert result.project_id == "p1"
    assert result.channel_id == "c1"


@pytest.mark.anyio
async def test_context_metadata_flows_into_the_result(runner: AgentRunner) -> None:
    result = await runner.run(RecordingAgent(), AgentContext(metadata={"trigger": "manual"}))

    assert result.metadata["trigger"] == "manual"


@pytest.mark.anyio
async def test_output_metadata_flows_into_the_result(runner: AgentRunner) -> None:
    class WithMetadata(Agent):
        name = "with_metadata"
        version = "1.0.0"

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return AgentOutput.ok("data", tokens=42)

    result = await runner.run(WithMetadata())

    assert result.metadata["tokens"] == 42


# ---------- Execution timing ----------


@pytest.mark.anyio
async def test_duration_is_recorded(runner: AgentRunner) -> None:
    result = await runner.run(SlowAgent(0.05))

    assert result.duration_ms is not None
    assert result.duration_ms >= 45


@pytest.mark.anyio
async def test_duration_is_monotonic_accurate(runner: AgentRunner) -> None:
    result = await runner.run(SlowAgent(0.15))

    assert result.duration_ms is not None
    assert 140 <= result.duration_ms < 1500


@pytest.mark.anyio
async def test_start_and_completion_times_are_recorded(runner: AgentRunner) -> None:
    result = await runner.run(RecordingAgent())

    assert result.started_at is not None
    assert result.completed_at is not None
    assert result.completed_at >= result.started_at


@pytest.mark.anyio
async def test_duration_is_recorded_on_failure(runner: AgentRunner) -> None:
    """A failure still has to report how long it ran for."""
    result = await runner.run(RaisingAgent())

    assert result.duration_ms is not None
    assert result.duration_ms >= 0


@pytest.mark.anyio
async def test_uses_a_monotonic_clock_for_duration() -> None:
    """Wall-clock jumps must not produce negative durations."""
    result = await AgentRunner().run(RecordingAgent())

    assert result.duration_ms is not None
    assert result.duration_ms >= 0
    assert result.completed_at >= result.started_at


# ---------- Failure capture ----------


@pytest.mark.anyio
async def test_exception_produces_failed_status(runner: AgentRunner) -> None:
    result = await runner.run(RaisingAgent())

    assert result.success is False
    assert result.status is AgentStatus.FAILED
    assert result.is_terminal


@pytest.mark.anyio
async def test_failure_never_raises_to_the_caller(runner: AgentRunner) -> None:
    """A pipeline records progress by reading results, not by catching.

    An exception escaping here would force every caller into a try/except.
    """
    result = await runner.run(RaisingAgent(RuntimeError("boom")))

    assert result.status is AgentStatus.FAILED


@pytest.mark.anyio
async def test_failure_records_the_error_type(runner: AgentRunner) -> None:
    result = await runner.run(RaisingAgent())

    assert result.metadata["error_type"] == "RuntimeError"


@pytest.mark.anyio
async def test_failure_message_does_not_leak_the_exception_text(runner: AgentRunner) -> None:
    """``result.error`` is treated as client-safe.

    The original text is kept in metadata for server logs only.
    """
    result = await runner.run(RaisingAgent(RuntimeError("connection to 10.0.0.5 refused")))

    assert "10.0.0.5" not in result.error
    assert result.metadata["error_detail"] == "connection to 10.0.0.5 refused"


@pytest.mark.anyio
async def test_agent_error_message_is_used_verbatim(runner: AgentRunner) -> None:
    """Agent authors write their own client-safe wording, so it is trusted."""
    error = AgentError("The supplied topic is empty.")
    error.retryable = True

    result = await runner.run(RaisingAgent(error))

    assert result.error == "The supplied topic is empty."
    assert result.is_retryable is True


@pytest.mark.anyio
async def test_unexpected_errors_are_not_marked_retryable(runner: AgentRunner) -> None:
    """A bug in an agent will not fix itself on a retry."""
    result = await runner.run(RaisingAgent(RuntimeError("NoneType has no attribute")))

    assert result.is_retryable is False


@pytest.mark.anyio
async def test_unexpected_failure_names_the_exception_class(runner: AgentRunner) -> None:
    result = await runner.run(RaisingAgent(RuntimeError("secret detail")))

    assert result.error == "RuntimeError raised during agent execution."


@pytest.mark.anyio
async def test_failure_reason_combines_type_and_message(runner: AgentRunner) -> None:
    result = await runner.run(RaisingAgent())

    assert result.failure_reason() == (
        "RuntimeError: RuntimeError raised during agent execution."
    )


# ---------- Unsuccessful but completed ----------


@pytest.mark.anyio
async def test_agent_can_complete_with_success_false(runner: AgentRunner) -> None:
    """A Fact Checker finding insufficient material is not a crash.

    The agent finished its work and reached a negative conclusion, so the
    status is COMPLETED while ``success`` is False.
    """
    result = await runner.run(UnsuccessfulAgent())

    assert result.success is False
    assert result.status is AgentStatus.COMPLETED
    assert result.error == "no material supplied"


@pytest.mark.anyio
async def test_unsuccessful_output_can_declare_itself_retryable(
    runner: AgentRunner,
) -> None:
    class Retryable(Agent):
        name = "retryable"
        version = "1.0.0"

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return AgentOutput.failed("provider timed out", retryable=True)

    result = await runner.run(Retryable())

    assert result.is_retryable is True


@pytest.mark.anyio
async def test_unsuccessful_output_without_a_reason_still_reports(runner: AgentRunner) -> None:
    class NoReason(Agent):
        name = "no_reason"
        version = "1.0.0"

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return AgentOutput(success=False)

    result = await runner.run(NoReason())

    assert result.success is False
    assert result.error


@pytest.mark.anyio
async def test_agent_returning_bare_data_is_wrapped(runner: AgentRunner) -> None:
    """Tolerating a plain return value keeps early agent code simple."""

    class ReturnsData(Agent):
        name = "returns_data"
        version = "1.0.0"

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            return {"summary": "s"}

    result = await runner.run(ReturnsData())

    assert result.success is True
    assert result.data == {"summary": "s"}


# ---------- Timeout ----------


@pytest.mark.anyio
async def test_execution_timeout_fails_the_agent() -> None:
    runner = AgentRunner(execution_timeout=0.05)

    result = await runner.run(SlowAgent(1.0))

    assert result.status is AgentStatus.FAILED
    assert "time limit" in result.error


@pytest.mark.anyio
async def test_timeout_is_retryable(runner: AgentRunner) -> None:
    """A timeout is the canonical retryable infrastructure failure."""
    timed_out = AgentRunner(execution_timeout=0.05)
    result = await timed_out.run(SlowAgent(1.0))

    assert result.is_retryable is True


@pytest.mark.anyio
async def test_timeout_does_not_leak_internal_details() -> None:
    runner = AgentRunner(execution_timeout=0.05)

    result = await runner.run(SlowAgent(1.0))

    assert result.error == "Agent execution exceeded its time limit."


# ---------- Cancellation ----------


@pytest.mark.anyio
async def test_cancellation_propagates(runner: AgentRunner) -> None:
    """Cancelling must actually cancel, not be swallowed as a failure."""

    with anyio.move_on_after(0.05):
        await runner.run(SlowAgent(5.0))

    assert True  # reaching here without a hang is the assertion


@pytest.mark.anyio
async def test_cancelled_execution_is_recorded(runner: AgentRunner) -> None:
    with anyio.CancelScope() as scope:
        scope.cancel()

        await runner.run(SlowAgent(5.0))

    records = runner.history()
    assert records[-1].status is AgentStatus.CANCELLED


# ---------- History and retry ----------


@pytest.mark.anyio
async def test_history_records_completed_execution(runner: AgentRunner) -> None:
    result = await runner.run(RecordingAgent())

    record = runner.record(result.execution_id)

    assert record is not None
    assert record.status is AgentStatus.COMPLETED
    assert record.agent_name == "recording"
    assert record.result is result


@pytest.mark.anyio
async def test_history_records_failed_execution(runner: AgentRunner) -> None:
    result = await runner.run(RaisingAgent())

    record = runner.record(result.execution_id)

    assert record is not None
    assert record.status is AgentStatus.FAILED


@pytest.mark.anyio
async def test_history_records_cancelled_execution(runner: AgentRunner) -> None:
    with anyio.CancelScope() as scope:
        scope.cancel()
        await runner.run(SlowAgent(1.0))

    assert runner.history()[-1].status is AgentStatus.CANCELLED


@pytest.mark.anyio
async def test_history_is_bounded(runner: AgentRunner) -> None:
    runner.history_limit = 3

    for _ in range(6):
        await runner.run(RecordingAgent())

    assert len(runner.history()) == 3


@pytest.mark.anyio
async def test_history_drops_the_oldest_first(runner: AgentRunner) -> None:
    runner.history_limit = 2

    first = await runner.run(RecordingAgent())
    await runner.run(RecordingAgent())
    last = await runner.run(RecordingAgent())

    assert runner.record(first.execution_id) is None
    assert runner.record(last.execution_id) is not None


@pytest.mark.anyio
async def test_retry_reuses_the_same_context_and_prompt(runner: AgentRunner) -> None:
    """A retry must replay the original work, not invent new inputs."""
    registry = AgentRegistry()
    registry.register(RecordingAgent())
    agent = RecordingAgent()
    registry.register(agent, overwrite=True)

    original = await runner.run_named(
        "recording", AgentContext(project_id="p1"), PROMPT, registry=registry
    )

    retried = await runner.retry(original.execution_id, registry=registry)

    assert agent.seen_context is not None
    assert agent.seen_context.project_id == "p1"
    assert agent.seen_messages == PROMPT
    assert retried.success is True


@pytest.mark.anyio
async def test_retry_gets_a_new_execution_id(runner: AgentRunner) -> None:
    registry = AgentRegistry()
    registry.register(RecordingAgent())

    original = await runner.run_named("recording", registry=registry)
    retried = await runner.retry(original.execution_id, registry=registry)

    assert retried.execution_id != original.execution_id


@pytest.mark.anyio
async def test_retry_increments_the_attempt_counter(runner: AgentRunner) -> None:
    registry = AgentRegistry()
    registry.register(RaisingAgent())

    original = await runner.run_named("raising", registry=registry)
    retried = await runner.retry(original.execution_id, registry=registry)

    assert original.metadata["attempt"] == 1
    assert retried.metadata["attempt"] == 2


@pytest.mark.anyio
async def test_retry_only_reruns_the_requested_agent(runner: AgentRunner) -> None:
    """A later stage failing must not force earlier stages to run again."""
    registry = AgentRegistry()
    researcher = RecordingAgent()
    researcher.name = "researcher"
    scriptwriter = RaisingAgent()
    scriptwriter.name = "scriptwriter"
    registry.register(researcher)
    registry.register(scriptwriter)

    context = AgentContext(project_id="p1", previous_stage_outputs={"topic": "T"})

    research = await runner.run_named("researcher", context, registry=registry)
    researcher_calls_before = 1

    script = await runner.run_named("scriptwriter", context, registry=registry)
    retried = await runner.retry(script.execution_id, registry=registry)

    assert research.status is AgentStatus.COMPLETED
    assert script.status is AgentStatus.FAILED
    assert retried.status is AgentStatus.FAILED
    assert researcher_calls_before == 1


@pytest.mark.anyio
async def test_retry_of_an_unknown_execution_raises(runner: AgentRunner) -> None:
    with pytest.raises(KeyError):
        await runner.retry("does-not-exist")


@pytest.mark.anyio
async def test_retry_can_succeed_after_a_failure(runner: AgentRunner) -> None:
    """The realistic case: a transient failure that clears on retry."""
    registry = AgentRegistry()
    flaky = FlakyAgent()
    registry.register(flaky)

    first = await runner.run_named("flaky", registry=registry)
    second = await runner.retry(first.execution_id, registry=registry)

    assert first.status is AgentStatus.FAILED
    assert second.status is AgentStatus.COMPLETED


class FlakyAgent(Agent):
    """Fails once, then succeeds."""

    name = "flaky"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, context, messages):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("provider unreachable")
        return AgentOutput.ok("recovered")


# ---------- Named execution ----------


@pytest.mark.anyio
async def test_run_named_resolves_through_the_registry(runner: AgentRunner) -> None:
    registry = AgentRegistry()
    agent = RecordingAgent()
    registry.register(agent)

    result = await runner.run_named("recording", registry=registry)

    assert result.success is True
    assert agent.seen_context is not None


@pytest.mark.anyio
async def test_unknown_agent_name_produces_a_failed_result(
    runner: AgentRunner,
) -> None:
    """An unknown name must not raise.

    Callers select agents by name from configuration; a typo should
    surface as a recorded failure, not an unhandled exception.
    """
    registry = AgentRegistry()

    result = await runner.run_named("nonexistent", registry=registry)

    assert result.status is AgentStatus.FAILED
    assert result.agent_name == "nonexistent"
    assert result.metadata["error_type"] == "AgentNotFoundError"


@pytest.mark.anyio
async def test_unknown_agent_name_is_not_retryable(runner: AgentRunner) -> None:
    """Retrying a misspelled name would fail identically."""
    registry = AgentRegistry()

    result = await runner.run_named("typo_here", registry=registry)

    assert result.is_retryable is False


@pytest.mark.anyio
async def test_unknown_agent_name_error_lists_available_agents(
    runner: AgentRunner,
) -> None:
    registry = AgentRegistry()
    registry.register(RecordingAgent())

    result = await runner.run_named("nonexistent", registry=registry)

    assert "recording" in result.error


@pytest.mark.anyio
async def test_registered_placeholder_fails_as_not_implemented(
    runner: AgentRunner,
) -> None:
    registry = AgentRegistry()
    registry.register_placeholder("researcher")

    result = await runner.run_named("researcher", registry=registry)

    assert result.status is AgentStatus.FAILED
    assert result.metadata["error_type"] == "AgentNotImplementedError"


@pytest.mark.anyio
async def test_placeholder_failure_is_retryable(runner: AgentRunner) -> None:
    """So a pipeline does not have to treat planned agents as fatal."""
    registry = AgentRegistry()
    registry.register_placeholder("scriptwriter")

    result = await runner.run_named("scriptwriter", registry=registry)

    assert result.is_retryable is True


# ---------- Sequential execution ----------


def test_runner_does_not_run_agents_concurrently() -> None:
    """Sequential execution is the runner's documented policy.

    A single 8B model on an i5 with integrated graphics is CPU-bound: two
    agents generating at once would contend for the same CPU and memory.
    The runner exposes no parallel execution entry point; a caller that
    starts two tasks opts into concurrency explicitly, which is the point
    of making the default visible rather than implicit.
    """
    runner = AgentRunner()

    assert runner.sequential is True
    public_methods = {
        name
        for name in dir(runner)
        if not name.startswith("_") and callable(getattr(runner, name))
    }

    assert not any(
        "parallel" in name or "gather" in name or "concurrent" in name
        for name in public_methods
    )


@pytest.mark.anyio
async def test_runner_runs_the_next_agent_after_the_first_finishes(
    runner: AgentRunner,
) -> None:
    """Awaits are chained: the second agent starts only once the first is done."""
    order: list[str] = []

    class OrderedAgent(Agent):
        version = "1.0.0"

        def __init__(self, label: str) -> None:
            self.name = label

        async def run(self, context, messages):  # noqa: ANN001, ANN201
            order.append(f"{self.name}:start")
            await anyio.sleep(0.01)
            order.append(f"{self.name}:end")
            return AgentOutput.ok(self.name)

    await runner.run(OrderedAgent("first"))
    await runner.run(OrderedAgent("second"))

    assert order == [
        "first:start",
        "first:end",
        "second:start",
        "second:end",
    ]

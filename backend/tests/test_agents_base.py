"""Tests for the agent contracts: context, output, result and status."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.agents.base import (
    Agent,
    AgentContext,
    AgentError,
    AgentNotFoundError,
    AgentNotImplementedError,
    AgentOutput,
    AgentResult,
    AgentStatus,
    Capability,
    PlaceholderAgent,
    new_execution_id,
    utc_now,
)
from app.services.ai.provider import Message


# ---------- Status ----------


def test_status_covers_the_full_lifecycle() -> None:
    assert {s.value for s in AgentStatus} == {
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
    }


@pytest.mark.parametrize(
    "status",
    [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED],
)
def test_terminal_statuses_are_terminal(status: AgentStatus) -> None:
    assert status.is_terminal


@pytest.mark.parametrize(
    "status", [AgentStatus.PENDING, AgentStatus.RUNNING]
)
def test_non_terminal_statuses(status: AgentStatus) -> None:
    assert not status.is_terminal


def test_status_serialises_as_its_value() -> None:
    """StrEnum, so the status can be used directly in JSON and logs."""
    assert AgentStatus.RUNNING == "running"
    assert f"{AgentStatus.FAILED}" == "failed"


# ---------- AgentContext ----------


def test_context_generates_a_unique_execution_id() -> None:
    assert new_execution_id() != new_execution_id()
    assert len(new_execution_id()) == 36


def test_context_defaults_to_a_fresh_execution_id() -> None:
    assert AgentContext().execution_id


def test_context_identifiers_are_optional() -> None:
    """An agent can run standalone, outside any project."""
    context = AgentContext()

    assert context.project_id is None
    assert context.channel_id is None
    assert context.user_id is None


def test_context_carries_identifiers_and_config() -> None:
    context = AgentContext(
        project_id="p1",
        channel_id="c1",
        user_id="u1",
        config={"temperature": 0.2},
    )

    assert context.project_id == "p1"
    assert context.channel_id == "c1"
    assert context.user_id == "u1"
    assert context.option("temperature") == 0.2


def test_context_option_falls_back_to_default() -> None:
    assert AgentContext().option("missing", "fallback") == "fallback"


def test_context_exposes_previous_stage_outputs() -> None:
    context = AgentContext(previous_stage_outputs={"research": {"topic": "T"}})

    assert context.previous_output("research") == {"topic": "T"}
    assert context.previous_output("script") is None


def test_context_available_files_default_to_empty() -> None:
    assert AgentContext().available_files == ()


def test_context_is_immutable() -> None:
    context = AgentContext(project_id="p1")

    with pytest.raises(AttributeError):
        context.project_id = "p2"  # type: ignore[misc]


def test_with_execution_id_returns_a_bound_copy() -> None:
    context = AgentContext(project_id="p1")

    bound = context.with_execution_id("exec-1")

    assert bound.execution_id == "exec-1"
    assert bound.project_id == "p1"
    assert context.execution_id != "exec-1"


# ---------- AgentOutput ----------


def test_output_ok_carries_data() -> None:
    output = AgentOutput.ok({"answer": 42})

    assert output.success is True
    assert output.data == {"answer": 42}
    assert output.error is None


def test_output_failed_carries_a_reason() -> None:
    output = AgentOutput.failed("no material supplied")

    assert output.success is False
    assert output.error == "no material supplied"
    assert output.data is None


def test_output_metadata_is_optional_on_both_paths() -> None:
    assert AgentOutput.ok().metadata == {}
    assert AgentOutput.failed("x").metadata == {}


# ---------- AgentResult ----------


def test_result_from_success_is_completed() -> None:
    result = AgentResult.from_success(
        agent_name="researcher",
        agent_version="1.0.0",
        execution_id="e1",
        data={"topic": "T"},
    )

    assert result.success is True
    assert result.status is AgentStatus.COMPLETED
    assert result.is_terminal
    assert result.data == {"topic": "T"}


def test_result_from_failure_is_failed() -> None:
    result = AgentResult.from_failure(
        agent_name="researcher",
        agent_version="1.0.0",
        execution_id="e1",
        error="boom",
    )

    assert result.success is False
    assert result.status is AgentStatus.FAILED
    assert result.error == "boom"


def test_result_from_failure_can_be_cancelled() -> None:
    result = AgentResult.from_failure(
        agent_name="researcher",
        agent_version="1.0.0",
        execution_id="e1",
        error="stopped",
        status=AgentStatus.CANCELLED,
    )

    assert result.status is AgentStatus.CANCELLED
    assert result.is_terminal


def test_result_records_timing() -> None:
    started = utc_now()
    completed = started + timedelta(milliseconds=250)

    result = AgentResult.from_success(
        agent_name="researcher",
        agent_version="1.0.0",
        execution_id="e1",
        started_at=started,
        completed_at=completed,
        duration_ms=250.0,
    )

    assert result.started_at == started
    assert result.completed_at == completed
    assert result.duration_ms == 250.0


def test_result_records_the_agent_version() -> None:
    result = AgentResult.from_success(
        agent_name="researcher", agent_version="1.2.0", execution_id="e1"
    )

    assert result.agent_version == "1.2.0"


def test_result_failure_reason_includes_the_error_type() -> None:
    result = AgentResult.from_failure(
        agent_name="researcher",
        agent_version="1.0.0",
        execution_id="e1",
        error="ValueError raised during agent execution.",
        metadata={"error_type": "ValueError"},
    )

    assert result.failure_reason() == (
        "ValueError: ValueError raised during agent execution."
    )


def test_failure_reason_is_none_for_successful_results() -> None:
    result = AgentResult.from_success(
        agent_name="researcher", agent_version="1.0.0", execution_id="e1"
    )

    assert result.failure_reason() is None


def test_retryability_is_read_from_metadata() -> None:
    retryable = AgentResult.from_failure(
        agent_name="a", agent_version="1", execution_id="e", error="x",
        metadata={"retryable": True},
    )
    not_retryable = AgentResult.from_failure(
        agent_name="a", agent_version="1", execution_id="e", error="x",
    )

    assert retryable.is_retryable is True
    assert not_retryable.is_retryable is False


def test_utc_now_is_timezone_aware() -> None:
    assert utc_now().tzinfo is UTC


def test_naive_and_aware_timestamps_can_be_compared() -> None:
    """Aware timestamps throughout means durations are unambiguous."""
    aware = utc_now()
    naive_equivalent = aware.astimezone(UTC).replace(tzinfo=UTC)

    assert isinstance(naive_equivalent, datetime)


# ---------- Agent errors ----------


def test_agent_errors_carry_codes() -> None:
    assert AgentNotFoundError.code == "agent_not_found"
    assert AgentNotImplementedError.code == "agent_not_implemented"


def test_not_implemented_is_retryable() -> None:
    """The logic is coming, so a retry may well succeed.

    This is what keeps a pipeline moving forward past a placeholder.
    """
    assert AgentNotImplementedError.retryable is True


def test_not_found_is_not_retryable() -> None:
    assert AgentNotFoundError.retryable is False


def test_agent_error_accepts_a_custom_code() -> None:
    error = AgentError("bad input", code="custom_code")

    assert error.code == "custom_code"
    assert error.message == "bad input"


# ---------- Placeholder agent ----------


def test_placeholder_agent_is_registerable() -> None:
    agent = PlaceholderAgent()
    agent.name = "researcher"
    agent.version = "1.0.0"

    assert agent.qualified_name == "researcher@1.0.0"
    assert agent.describe()["name"] == "researcher"


def test_placeholder_describe_lists_capabilities() -> None:
    agent = PlaceholderAgent()
    agent.name = "fact_checker"
    agent.capabilities = frozenset({Capability.REQUIRES_REASONING})

    described = agent.describe()

    assert described["capabilities"] == ["requires_reasoning"]
    assert described["version"]


@pytest.mark.anyio
async def test_placeholder_agent_raises_not_implemented() -> None:
    agent = PlaceholderAgent()
    agent.name = "researcher"

    with pytest.raises(AgentNotImplementedError):
        await agent.run(AgentContext(), [Message(role="user", content="go")])


def test_agent_requires_run_to_be_implemented() -> None:
    class Incomplete(Agent):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]

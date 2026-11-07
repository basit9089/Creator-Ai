"""Public response models for the research endpoint.

Kept separate from the domain models in
:mod:`app.agents.researcher.schemas` so the internal representation can
change without silently altering the API contract.

:class:`ResearchRequest` is deliberately *not* duplicated here. It is the
same validated model the agent consumes, so a request is parsed and
checked exactly once, at the boundary, and FastAPI's own validation
reports field errors rather than the route re-checking them.

Execution metadata is wrapped around the result rather than merged into
it. A caller needs to know which agent version produced the research and
whether the execution itself succeeded, independently of whether the
evidence turned out to be sufficient.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import AgentResult, AgentStatus
from app.agents.researcher.schemas import ResearchResult

#: Re-exported so a route can declare its body without importing the
#: agent package directly.
from app.agents.researcher.schemas import ResearchRequest  # noqa: F401


class ResearchExecution(BaseModel):
    """How the research run itself went.

    Separated from the research content because the two can disagree in an
    important way: ``status`` COMPLETED with ``succeeded`` False means the
    pipeline worked and the evidence was too thin, which is different from
    a run that failed outright.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str
    agent_version: str
    execution_id: str

    status: AgentStatus = Field(
        description="Execution status. COMPLETED or FAILED for a research run."
    )

    succeeded: bool = Field(
        description="Whether usable research was produced."
    )

    duration_ms: float | None = Field(
        default=None, description="Wall-clock duration of the execution."
    )

    error: str | None = Field(
        default=None, description="Client-safe reason, present when not succeeded."
    )

    code: str | None = Field(
        default=None,
        description=(
            "Stable machine-readable failure code, present when the run "
            "did not produce usable research."
        ),
    )

    retryable: bool = Field(
        default=False, description="Whether re-running this request unchanged could work."
    )

    @classmethod
    def from_result(cls, result: AgentResult) -> ResearchExecution:
        """Build execution metadata from a runner result.

        The failure code comes from the agent's metadata rather than being
        re-derived here, so the HTTP status and the reported code always
        describe the same decision.
        """
        return cls(
            agent_name=result.agent_name,
            agent_version=result.agent_version,
            execution_id=result.execution_id,
            status=result.status,
            succeeded=result.success,
            duration_ms=result.duration_ms,
            error=result.error,
            code=result.metadata.get("code"),
            retryable=bool(result.metadata.get("retryable", False)),
        )


class ResearchResponse(BaseModel):
    """The result of a research run.

    Returned with HTTP 200 for both a successful run and a completed run
    whose evidence was insufficient. The two are distinguished by
    :attr:`succeeded` and by ``result.sources_found``, which is more useful
    to a caller than a non-200 status for a normal research outcome.

    ``result`` is always present on a completed run, including a partial
    one: three retrieved sources with two recorded failures is real work,
    and discarding it would throw away the account of what went missing.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = True

    result: ResearchResult | None = Field(
        default=None,
        description="The grounded research, or None when the run failed.",
    )

    execution: ResearchExecution = Field(
        description="How the execution itself went."
    )

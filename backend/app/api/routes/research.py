"""Research execution endpoint.

One route, deliberately thin. It resolves the agent by name through the
:class:`~app.agents.registry.AgentRegistry`, runs it through the existing
:class:`~app.agents.runner.AgentRunner`, and translates the result into a
response. No retrieval, prompting or grounding logic lives here: the route
cannot drift from the agent because it contains none of it.

The agent is addressed as ``"researcher"``, a string. Nothing here
imports :class:`~app.agents.researcher.agent.ResearcherAgent`, so swapping
the implementation, or selecting a different one by configuration later,
is not a change to this file.

Status codes
------------
``200``
    The run completed. ``success`` is False when the evidence was too thin
    to analyse, which is a research verdict rather than an error, and the
    partial result is returned so the caller can see what was retrieved.
``4xx``
    The request was invalid, reported by FastAPI's validation.
``5xx``
    The execution itself failed, using the provider's own error envelope.
    Insufficient evidence is deliberately *not* a 5xx: nothing went wrong.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.base import AgentResult, AgentStatus
from app.agents.registry import AgentRegistry, get_registry
from app.agents.researcher.agent import REQUEST_CONTEXT_KEY
from app.agents.runner import AgentRunner
from app.core.logging import get_logger
from app.dependencies import get_agent_runner
from app.schemas.research import (
    ResearchExecution,
    ResearchRequest,
    ResearchResponse,
    ResearchResult,
)

logger = get_logger("api.research")

router = APIRouter(prefix="/api/research", tags=["research"])

#: The agent this endpoint runs. A name, resolved through the registry,
#: rather than a class: the API never chooses an implementation itself.
RESEARCHER_AGENT_NAME = "researcher"


@router.post(
    "",
    response_model=ResearchResponse,
    summary="Research a topic and return grounded, cited findings",
    responses={
        200: {"description": "The research run completed."},
        404: {"description": "The researcher agent is not registered."},
        422: {"description": "The request body failed validation."},
        500: {"description": "The run failed unexpectedly."},
        502: {"description": "The model returned unusable research output."},
        503: {"description": "No usable sources, or the AI provider was unavailable."},
        504: {"description": "The run exceeded its time limit."},
    },
)
async def research_topic(
    request: ResearchRequest,
    registry: AgentRegistry = Depends(get_registry),
    runner: AgentRunner = Depends(get_agent_runner),
) -> ResearchResponse:
    """Research ``request.topic`` from live web sources.

    Searches the web, retrieves and reads the sources it finds, and asks
    the configured AI provider to analyse them. The model does not search:
    it only ever sees text this pipeline retrieved, and every claim it
    produces is checked against the sources that were actually fetched.

    Slow by design. A standard run searches once, fetches several pages
    and then waits on a local model, so treat the client timeout as
    minutes rather than seconds.
    """
    agent = registry.get_agent(RESEARCHER_AGENT_NAME)

    # The validated request travels in the context, so it is parsed once
    # here rather than re-validated inside the agent. project_id is
    # supplied by a later phase, when persistence exists.
    context = _context_for(request)

    result = await runner.run(agent, context, [_user_message(request.topic)])

    return _to_response(result)


def _context_for(request: ResearchRequest):
    """Build the agent context for a research run."""
    from app.agents.base import AgentContext

    return AgentContext(
        config={REQUEST_CONTEXT_KEY: request},
        metadata={"topic": request.topic, "search_depth": request.search_depth.value},
    )


def _user_message(topic: str):
    """The incoming prompt for the run.

    The Researcher builds its own analysis prompt once it has evidence, so
    this message is not what reaches the model. It is supplied because the
    Phase 3 contract passes the assembled prompt to every agent, and the
    Researcher also accepts the topic from it as a fallback.
    """
    from app.services.ai.provider import Message

    return Message(role="user", content=f"Research the following topic: {topic}")


def _to_response(result: AgentResult) -> ResearchResponse:
    """Translate a runner result into an HTTP response or an error.

    The distinction the Phase 3 runner draws is preserved exactly:

    * ``FAILED`` - the execution broke. An error is raised so the client
      gets the standard error envelope with a meaningful status code.
    * ``COMPLETED`` with ``success=False`` - the run worked and the
      verdict was negative. A 200 carrying the partial result, because
      this is a legitimate research outcome rather than a fault.
    """
    execution = ResearchExecution.from_result(result)

    if result.status is AgentStatus.FAILED:
        logger.warning(
            "research_run_failed execution_id=%s code=%s error=%s",
            result.execution_id,
            execution.code,
            result.error,
        )
        # Raising AppError routes through the existing handler, so the
        # response shape matches every other failure in the API.
        from app.exceptions.handlers import AppError

        raise AppError(
            result.error or "The research run failed.",
            code=execution.code or "research_failed",
            status_code=_status_for(execution.code),
        )

    if result.status is AgentStatus.CANCELLED:
        from app.exceptions.handlers import AppError

        raise AppError(
            result.error or "The research run was cancelled.",
            code="research_cancelled",
            status_code=499,
        )

    payload = result.data if isinstance(result.data, ResearchResult) else None

    logger.info(
        "research_run_completed execution_id=%s duration_ms=%s succeeded=%s "
        "sources_found=%s sources_failed=%s",
        result.execution_id,
        result.duration_ms,
        result.success,
        payload.sources_found if payload else None,
        payload.sources_failed if payload else None,
    )

    return ResearchResponse(success=result.success, result=payload, execution=execution)


def _status_for(code: str | None) -> int:
    """Map a research failure code onto an HTTP status.

    Kept as a small table rather than read off the exception classes so
    that a code produced in agent metadata has a defined status even when
    the agent raised through the runner rather than directly.
    """
    return {
        "research_no_sources": 503,
        "research_insufficient_sources": 503,
        "research_search_unavailable": 503,
        "research_all_sources_failed": 503,
        "research_ai_unavailable": 503,
        "research_ai_output_invalid": 502,
        "research_invalid_request": 422,
    }.get(code or "", 500)

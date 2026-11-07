"""Fact check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.agents.base import AgentContext, AgentError, AgentResult, AgentStatus
from app.agents.fact_checker.agent import REQUEST_CONTEXT_KEY, FactCheckerAgent
from app.agents.fact_checker.exceptions import FactCheckError
from app.agents.fact_checker.schemas import FactCheckRequest, FactCheckResult
from app.agents.registry import AgentRegistry, get_registry
from app.agents.runner import AgentRunner
from app.dependencies import get_agent_runner
from app.core.logging import get_logger
from app.exceptions.handlers import AppError
from app.schemas.fact_check import FactCheckAPIResponse, FactCheckExecution

logger = get_logger("api.fact_check")

router = APIRouter()

FACT_CHECKER_AGENT_NAME = "fact_checker"


@router.post(
    "/api/fact-check",
    response_model=FactCheckAPIResponse,
    responses={
        200: {"description": "Fact check completed"},
        422: {"description": "Invalid request"},
        502: {"description": "AI output invalid"},
        503: {"description": "Service unavailable"},
    },
)
async def fact_check(
    request: FactCheckRequest,
    registry: AgentRegistry = Depends(get_registry),
    runner: AgentRunner = Depends(get_agent_runner),
) -> FactCheckAPIResponse | JSONResponse:
    agent = registry.get_agent(FACT_CHECKER_AGENT_NAME)

    context = AgentContext(config={REQUEST_CONTEXT_KEY: request})

    try:
        result = await runner.run(
            agent,
            context,
            [],  # no messages needed; request in context
        )
    except FactCheckError as exc:
        return _error_response_from_agent_error(exc)
    except AgentError as exc:
        return _error_response_from_agent_error(exc)
    except Exception as exc:
        logger.exception("fact_check_unexpected_error")
        raise AppError(
            "An unexpected error occurred during fact checking.",
            code="fact_check_internal",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    return _to_api_response(result)


def _to_api_response(agent_result: AgentResult) -> FactCheckAPIResponse:
    execution = FactCheckExecution(
        execution_id=agent_result.execution_id,
        agent_name=agent_result.agent_name,
        agent_version=agent_result.agent_version,
        status=agent_result.status,
        succeeded=agent_result.success,
        started_at=agent_result.started_at.isoformat() if agent_result.started_at else None,
        completed_at=agent_result.completed_at.isoformat() if agent_result.completed_at else None,
        duration_ms=agent_result.duration_ms,
        code=agent_result.metadata.get("code"),
        retryable=agent_result.metadata.get("retryable"),
        error=agent_result.error,
        metadata=agent_result.metadata,
    )

    result_data = agent_result.data if agent_result.data else None

    return FactCheckAPIResponse(
        success=agent_result.success,
        execution=execution,
        result=result_data if isinstance(result_data, FactCheckResult) else result_data,
    )


def _status_for(code: str | None) -> int:
    if code == "fact_check_ai_output_invalid":
        return status.HTTP_502_BAD_GATEWAY
    if code in ("fact_check_ai_unavailable",):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if code in ("fact_check_invalid_request", "fact_check_no_claims", "fact_check_missing_research_data"):
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _error_response_from_agent_error(exc: AgentError) -> JSONResponse:
    raise AppError(
        getattr(exc, "message", str(exc)),
        code=getattr(exc, "code", "fact_check_error"),
        status_code=_status_for(getattr(exc, "code", None)),
    )
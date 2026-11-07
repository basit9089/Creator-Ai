"""Script endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.agents.base import AgentContext, AgentError
from app.agents.scriptwriter.agent import REQUEST_KEY, ScriptwriterAgent
from app.agents.scriptwriter.schemas import ScriptRequest, ScriptResult
from app.agents.registry import AgentRegistry, get_registry
from app.agents.runner import AgentRunner
from app.dependencies import get_agent_runner
from app.exceptions.handlers import AppError
from app.core.logging import get_logger

logger = get_logger("api.script")

router = APIRouter()


@router.post(
    "/api/script",
    response_model=ScriptResult,
)
async def generate_script(
    request: ScriptRequest,
    registry: AgentRegistry = Depends(get_registry),
    runner: AgentRunner = Depends(get_agent_runner),
) -> ScriptResult | JSONResponse:
    agent = registry.get_agent("scriptwriter")
    context = AgentContext(config={REQUEST_KEY: request})
    try:
        result = await runner.run(agent, context, [])
    except AgentError as exc:
        raise AppError(getattr(exc, "message", str(exc)), code=getattr(exc, "code", "scriptwriter_error")) from exc
    except Exception as exc:
        logger.exception("script_generation_error")
        raise AppError("Script generation failed", code="scriptwriter_internal", status_code=500) from exc

    if result.data and isinstance(result.data, ScriptResult):
        return result.data
    raise AppError("Script generation returned no data", code="scriptwriter_no_data", status_code=502)
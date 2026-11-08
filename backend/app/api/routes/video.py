"""Video production endpoint.

``POST /api/video/produce`` plans and renders a video in one call.
The route stays thin: validation is Pydantic's, execution is the
AgentRunner's, and every decision lives in the agent/services.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.agents.base import AgentContext, AgentError
from app.agents.registry import AgentRegistry, get_registry
from app.agents.runner import AgentRunner
from app.agents.video_producer.agent import REQUEST_KEY
from app.agents.video_producer.schemas import (
    VideoProductionRequest,
    VideoProductionResult,
)
from app.core.logging import get_logger
from app.dependencies import get_agent_runner
from app.exceptions.handlers import AppError

logger = get_logger("api.video")

router = APIRouter()

#: Agent error codes that are the caller's fault rather than ours.
_CLIENT_ERROR_CODES = frozenset(
    {
        "video_producer_invalid_request",
        "video_producer_asset_error",
        "agent_not_found",
    }
)


def _status_for_code(code: str | None) -> int:
    """Map an agent error code to an HTTP status, leaking nothing."""
    if code in _CLIENT_ERROR_CODES:
        # Spelled numerically: Starlette renamed the constant and the
        # old name is deprecated (same convention as handlers.py).
        return 422
    return status.HTTP_500_INTERNAL_SERVER_ERROR


@router.post(
    "/api/video/produce",
    response_model=VideoProductionResult,
)
async def produce_video(
    request: VideoProductionRequest,
    registry: AgentRegistry = Depends(get_registry),
    runner: AgentRunner = Depends(get_agent_runner),
) -> VideoProductionResult:
    """Plan scenes, prepare assets and render the video.

    The HTTP contract has no partial success: either a probed, real
    video file exists and the result says ``completed``, or the call
    fails with an error. Nothing in between is ever reported as done.
    """
    try:
        agent = registry.get_agent("video_producer")
    except AgentError as exc:
        raise AppError(
            getattr(exc, "message", str(exc)),
            code=getattr(exc, "code", "agent_not_found"),
            status_code=status.HTTP_404_NOT_FOUND,
        ) from exc

    context = AgentContext(config={REQUEST_KEY: request})

    # The runner never raises for agent failures: they arrive as a
    # failed result carrying the agent's own error code in metadata.
    result = await runner.run(agent, context, [])

    if result.success and isinstance(result.data, VideoProductionResult):
        return result.data

    if isinstance(result.data, VideoProductionResult):
        # Completed partial work (e.g. a plan without a render) is
        # still reported as a failure: the endpoint promises a file.
        logger.warning(
            "video_production_incomplete status=%s errors=%s",
            result.data.status,
            list(result.data.errors),
        )

    code = str(result.metadata.get("code") or "video_producer_error")
    raise AppError(
        result.error or "Video production failed",
        code=code,
        status_code=_status_for_code(code),
    )

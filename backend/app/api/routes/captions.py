"""Captions + thumbnail + SEO metadata endpoint.

``POST /api/captions-thumbnail`` produces SRT/WebVTT caption files, a
real 1280x720 (configurable) PNG thumbnail and validated YouTube
metadata for a finished script in one call. The route stays thin:
validation is Pydantic's, execution is the AgentRunner's, and every
decision lives in the agent and its services.

HTTP contract — deliberate and documented:

* ``completed`` (every requested artifact exists, verified) → **200**
  with ``success: true``.
* ``partial`` (some artifacts verified, others failed) → **200** with
  ``success: false``, per-artifact ``status``/``error_code`` fields and
  an ``errors`` list. The 200 here means "the service executed and
  produced exactly the artifacts reported below"; the payload itself
  never claims success when an artifact failed, and each produced file
  is named with its real path and size.
* *failed* (nothing was produced) → structured error envelope
  (``{"success": false, "error": {"code", "message"}}``) with a mapped
  status: 422 for caller mistakes, 502/503/504 for AI provider
  problems, 500 otherwise. No HTTP 200 for a failed operation.
* Request validation failures → 422 from Pydantic, as everywhere else.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.agents.base import AgentContext, AgentError
from app.agents.captions_thumbnail.agent import REQUEST_KEY
from app.agents.captions_thumbnail.schemas import (
    CaptionsThumbnailRequest,
    CaptionsThumbnailResult,
    CaptionsThumbnailStatus,
)
from app.agents.registry import AgentRegistry, get_registry
from app.agents.runner import AgentRunner
from app.core.logging import get_logger
from app.dependencies import get_agent_runner
from app.exceptions.handlers import AppError

logger = get_logger("api.captions")

router = APIRouter()

#: Agent error codes that are the caller's fault rather than ours.
_CLIENT_ERROR_CODES = frozenset(
    {
        "captions_thumbnail_invalid_request",
        "captions_thumbnail_asset_error",
        "agent_not_found",
    }
)

#: Known AI-side codes get their provider-ish statuses instead of a
#: blanket 500, so a client can tell "model missing" from "model broke".
_AI_ERROR_STATUSES: dict[str, int] = {
    "ai_provider_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    "model_not_found": status.HTTP_503_SERVICE_UNAVAILABLE,
    "ai_provider_timeout": status.HTTP_504_GATEWAY_TIMEOUT,
    "ai_provider_error": status.HTTP_502_BAD_GATEWAY,
    "structured_output_invalid": status.HTTP_502_BAD_GATEWAY,
    "captions_thumbnail_ai_error": status.HTTP_502_BAD_GATEWAY,
}


def _status_for_code(code: str | None) -> int:
    """Map an agent error code to an HTTP status, leaking nothing."""
    if code in _CLIENT_ERROR_CODES:
        # Spelled numerically: Starlette renamed the constant and the
        # old name is deprecated (same convention as handlers.py).
        return 422
    if code in _AI_ERROR_STATUSES:
        return _AI_ERROR_STATUSES[code]
    return status.HTTP_500_INTERNAL_SERVER_ERROR


@router.post(
    "/api/captions-thumbnail",
    response_model=CaptionsThumbnailResult,
)
async def generate_captions_thumbnail(
    request: CaptionsThumbnailRequest,
    registry: AgentRegistry = Depends(get_registry),
    runner: AgentRunner = Depends(get_agent_runner),
) -> CaptionsThumbnailResult:
    """Generate captions, a thumbnail and SEO metadata for a script.

    See the module docstring for the full three-way HTTP contract
    (completed / partial / failed). On partial, the response carries
    the real paths, sizes and dimensions of what *was* produced plus
    per-artifact error codes for what was not.
    """
    try:
        agent = registry.get_agent("captions_thumbnail")
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

    if result.success and isinstance(result.data, CaptionsThumbnailResult):
        if result.data.status is not CaptionsThumbnailStatus.COMPLETED:
            # Partial outcomes are ok()'d by the agent so their verified
            # artifacts reach the caller; flag them for the logs too.
            logger.warning(
                "captions_thumbnail_partial status=%s errors=%s",
                result.data.status,
                list(result.data.errors),
            )
        return result.data

    if isinstance(result.data, CaptionsThumbnailResult):
        logger.warning(
            "captions_thumbnail_failed status=%s errors=%s",
            result.data.status,
            list(result.data.errors),
        )

    code = str(result.metadata.get("code") or "captions_thumbnail_error")
    raise AppError(
        result.error or "Captions, thumbnail and metadata generation failed",
        code=code,
        status_code=_status_for_code(code),
    )

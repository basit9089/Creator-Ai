"""Health and service status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.dependencies import get_ai_provider
from app.schemas.ai import AIHealthResponse, HealthResponse
from app.services.ai.provider import AIProvider, HealthStatus

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse, summary="Service information")
def root(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Identify the running service."""
    return HealthResponse(status="healthy", service=settings.app_name)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report that the API process is alive.

    This performs no external I/O, so it stays fast enough for a
    container or load-balancer probe.
    """
    return HealthResponse(status="healthy", service=settings.app_name)


@router.get(
    "/api/health/ai",
    response_model=AIHealthResponse,
    summary="AI provider health",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The AI provider is not ready.",
        }
    },
)
def ai_health(
    response: Response,
    provider: AIProvider = Depends(get_ai_provider),
) -> AIHealthResponse:
    """Check whether the AI provider is reachable and the model exists.

    Only a model listing is performed, so no tokens are generated.

    Returns HTTP 200 when healthy or degraded, and 503 when the provider
    cannot be reached at all.
    """
    health_result = provider.health_check()

    # The status code reflects reachability so that automated checks can
    # act on it without parsing the body.
    if health_result.status is HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return AIHealthResponse(
        status=health_result.status,
        provider=health_result.provider,
        model=health_result.model,
        reachable=health_result.reachable,
        model_available=health_result.model_available,
        detail=health_result.detail,
        latency_ms=health_result.latency_ms,
    )
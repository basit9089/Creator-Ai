"""Request and response models for the AI API.

These are the public contract. Validation happens here so that invalid
input is rejected before any provider work is done.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.ai.provider import HealthStatus

# Read once at import: this is a static deployment-level limit.
MAX_PROMPT_LENGTH = get_settings().max_prompt_length


class AIRequest(BaseModel):
    """Request payload for text generation.

    Field constraints prevent unbounded input reaching the model. The
    ``prompt`` length limit is read from configuration so it can be
    tuned without a code change.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="The user prompt to send to the model.",
        examples=["Explain quantum computing in simple terms."],
    )

    system_prompt: str | None = Field(
        default=None,
        max_length=MAX_PROMPT_LENGTH,
        description="Optional system instructions that frame the assistant's behaviour.",
        examples=["You are a concise, friendly explainer."],
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic.",
    )

    # Strict: Pydantic would otherwise accept "yes", "on" and 1 for a
    # bool. A caller who sends the wrong type wants a 422, not silent
    # coercion into a reasoning mode that takes 24x longer.
    think: bool | None = Field(
        default=None,
        strict=True,
        description=(
            "Reasoning mode for hybrid models such as Qwen3. "
            "Defaults to the configured setting, which is off because "
            "reasoning is roughly 24x slower on modest hardware."
        ),
        examples=[False, True],
    )

    model: str | None = Field(
        default=None,
        max_length=200,
        description="Optional model override. Defaults to the configured model.",
    )


class AIResponse(BaseModel):
    """Successful generation response.

    The ``response`` and ``model`` fields preserve the shape returned by
    the original working endpoint, so existing frontend code keeps
    working.
    """

    success: bool = True
    model: str
    response: str
    duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thinking: bool | None = Field(
        default=None,
        description="Whether reasoning mode was used for this generation.",
    )


class AIStreamChunk(BaseModel):
    """A single increment of a streamed generation."""

    chunk: str
    done: bool = False


class FieldError(BaseModel):
    """A single field-level validation problem."""

    field: str
    message: str


class ErrorDetail(BaseModel):
    """Client-safe error information.

    Contains no stack traces, file paths or internal identifiers.
    ``details`` is optional and only populated for validation failures,
    where identifying the offending field helps the caller.
    """

    code: str
    message: str
    details: list[FieldError] | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope used for every failure."""

    success: bool = False
    error: ErrorDetail


class AIHealthResponse(BaseModel):
    """Health of the configured AI provider."""

    status: HealthStatus
    provider: str
    model: str
    reachable: bool
    model_available: bool
    detail: str = ""
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    """Liveness of the API service itself."""

    status: str = "healthy"
    service: str
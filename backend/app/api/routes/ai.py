"""AI generation routes."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import anyio.to_thread
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.logging import get_logger
from app.dependencies import get_ai_provider
from app.schemas.ai import AIRequest, AIResponse, AIStreamChunk
from app.services.ai.provider import (
    AIProvider,
    GenerationOptions,
    GenerationResult,
    Message,
)

logger = get_logger("api.ai")

router = APIRouter(prefix="/api/ai", tags=["ai"])


def build_messages(request: AIRequest) -> list[Message]:
    """Convert a request into provider-neutral messages.

    A system prompt, when supplied, is placed first so it frames the
    whole exchange.
    """
    messages: list[Message] = []

    if request.system_prompt:
        messages.append(Message(role="system", content=request.system_prompt))

    messages.append(Message(role="user", content=request.prompt))
    return messages


def build_options(request: AIRequest) -> GenerationOptions:
    """Translate request fields into provider-neutral options.

    Anything left unset stays unset here; the provider merges in the
    deployment defaults. That is why omitting ``think`` does not silently
    re-enable reasoning.
    """
    return GenerationOptions(temperature=request.temperature, think=request.think)


@router.post(
    "/generate",
    response_model=AIResponse,
    summary="Generate an AI completion",
)
async def generate_ai(
    request: AIRequest,
    provider: AIProvider = Depends(get_ai_provider),
) -> AIResponse:
    """Generate a completion for a single prompt.

    Provider failures propagate to the registered handlers, which map them
    to the appropriate 5xx status code and a client-safe message.
    """
    messages = build_messages(request)
    options = build_options(request)

    # Resolve now so the response can report the reasoning mode that was
    # actually used, rather than leaving the caller to guess whether their
    # request overrode the configured default.
    resolved = provider.resolve_options(options)

    # Log the request start and the prompt's size, never its content.
    logger.info(
        "ai_request_started prompt_chars=%d system_prompt=%s temperature=%s thinking=%s",
        len(request.prompt),
        request.system_prompt is not None,
        resolved.temperature,
        resolved.think,
    )

    started = time.perf_counter()
    result: GenerationResult = await _run_generation(provider, messages, options)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    logger.info(
        "ai_request_completed model=%s duration_ms=%.2f prompt_tokens=%s completion_tokens=%s",
        result.model,
        duration_ms,
        result.prompt_tokens,
        result.completion_tokens,
    )

    return AIResponse(
        success=True,
        model=result.model,
        response=result.text,
        duration_ms=round(result.total_duration_ms, 2)
        if result.total_duration_ms is not None
        else duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        thinking=resolved.think,
    )


async def _run_generation(
    provider: AIProvider,
    messages: list[Message],
    options: GenerationOptions,
) -> GenerationResult:
    """Run generation off the event loop.

    The Ollama client is synchronous and CPU-bound, so calling it
    directly would block the event loop and stall every other request.
    """
    try:
        return await anyio.to_thread.run_sync(
            lambda: provider.generate(messages, options=options)
        )
    except Exception as exc:  # noqa: BLE001 - logged, then re-raised
        logger.warning("ai_request_failed error_type=%s", type(exc).__name__)
        raise


@router.post(
    "/generate/stream",
    summary="Stream a completion",
    response_class=StreamingResponse,
)
async def stream_ai(
    request: AIRequest,
    provider: AIProvider = Depends(get_ai_provider),
) -> StreamingResponse:
    """Stream a completion as newline-delimited JSON chunks.

    The streaming path is already defined by the provider interface, so
    this endpoint required no change to the abstraction.
    """
    messages = build_messages(request)
    options = build_options(request)

    logger.info("ai_stream_started prompt_chars=%d", len(request.prompt))

    def produce() -> Iterator[str]:
        try:
            for text in provider.stream(messages, options=options):
                yield json.dumps(
                    AIStreamChunk(chunk=text, done=False).model_dump(mode="json")
                ) + "\n"
        except Exception as exc:  # noqa: BLE001 - reported inside the stream
            # Headers are already sent, so the failure is signalled in
            # the final chunk rather than as an HTTP status.
            logger.warning("ai_stream_failed error_type=%s", type(exc).__name__)
            yield json.dumps(
                AIStreamChunk(
                    chunk="Generation failed. Please try again.", done=True
                ).model_dump(mode="json")
            ) + "\n"
            return

        yield json.dumps(
            AIStreamChunk(chunk="", done=True).model_dump(mode="json")
        ) + "\n"

    return StreamingResponse(produce(), media_type="application/x-ndjson")
"""Domain exceptions and the API error envelope.

Two distinct concerns are kept apart:

* :class:`AppError` — an expected failure whose message is safe to show
  a client.
* Anything else — an unexpected bug. These are logged with a full
  traceback server-side and reported to the client as a generic message.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger
from app.schemas.ai import ErrorDetail, ErrorResponse, FieldError
from app.services.ai.provider import AIProviderError

logger = get_logger("errors")

#: Generic message for unexpected failures. Never reveals internals.
INTERNAL_ERROR_MESSAGE = "An unexpected internal error occurred."

#: Generic message for unexpected HTTP-layer failures.
HTTP_ERROR_MESSAGE = "The request could not be completed."


class AppError(Exception):
    """An expected application error with a client-safe message."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "application_error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


def error_response(
    *,
    code: str,
    message: str,
    status_code: int,
    headers: dict[str, str] | None = None,
    details: list[FieldError] | None = None,
) -> JSONResponse:
    """Build a JSON error response using the standard envelope.

    ``details`` is optional field-level information for validation
    failures. It is included here rather than assigned to the response
    afterwards: ``JSONResponse`` computes ``content-length`` during
    construction, so mutating ``body`` later would leave the header and
    the payload out of sync and truncate the response.
    """
    payload = ErrorResponse(
        success=False,
        error=ErrorDetail(code=code, message=message, details=details),
    )

    # `details` is dropped when None so simple errors stay compact.
    content = payload.model_dump(mode="json", exclude_none=True)

    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers,
    )


async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    """Handle a deliberate application error."""
    logger.warning("app_error code=%s message=%s", exc.code, exc.message)
    return error_response(code=exc.code, message=exc.message, status_code=exc.status_code)


async def provider_error_handler(_: Request, exc: AIProviderError) -> JSONResponse:
    """Handle an AI provider failure with the correct HTTP status.

    Provider failures are genuine server-side problems, so they return
    5xx status codes. The original exception is logged with a traceback;
    only the safe message reaches the client.
    """
    logger.error(
        "ai_provider_error code=%s provider=%s message=%s",
        exc.code,
        exc.provider or "unknown",
        exc.message,
        exc_info=True,
    )
    return error_response(code=exc.code, message=exc.message, status_code=exc.status_code)


async def validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation failures.

    Field-level details are included because they help a legitimate
    caller correct their request. Stack traces and internal objects are
    never exposed.
    """
    logger.info("request_validation_failed errors=%d", len(exc.errors()))

    details = [
        FieldError(
            # loc[0] is the location ("body"/"query"); drop it so the
            # client sees "prompt" rather than "body.prompt".
            field=".".join(str(part) for part in error.get("loc", ())[1:]) or "body",
            message=error.get("msg", "Invalid value"),
        )
        for error in exc.errors()
    ]

    return error_response(
        code="validation_error",
        message="Request validation failed.",
        # 422 is spelled numerically: Starlette has renamed the constant
        # and the literal avoids a deprecation warning.
        status_code=422,
        details=details,
    )


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Convert framework HTTP errors into the standard envelope."""
    # FastAPI uses 404 for unknown routes; those messages are safe.
    message = exc.detail if isinstance(exc.detail, str) else HTTP_ERROR_MESSAGE

    logger.info("http_exception status=%s message=%s", exc.status_code, message)

    return error_response(
        code="http_error",
        message=message,
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected errors.

    The full traceback is logged server-side; the client receives a
    generic message and no implementation detail.
    """
    logger.critical(
        "unhandled_exception type=%s",
        type(exc).__name__,
        exc_info=True,
    )
    return error_response(
        code="internal_server_error",
        message=INTERNAL_ERROR_MESSAGE,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every exception handler to the application.

    Registration order matters: Starlette dispatches to the most
    specific handler, so provider errors are registered before the
    generic HTTP handler.
    """
    app.add_exception_handler(AppError, app_error_handler)  # pyright: ignore[reportArgumentType]
    app.add_exception_handler(AIProviderError, provider_error_handler)  # pyright: ignore[reportArgumentType]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # pyright: ignore[reportArgumentType]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # pyright: ignore[reportArgumentType]
    app.add_exception_handler(Exception, unhandled_exception_handler)
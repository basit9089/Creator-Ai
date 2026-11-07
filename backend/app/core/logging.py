"""Centralised logging configuration.

Provides a single place where log format and level are decided, plus a
request-logging middleware.

Privacy rules enforced here
---------------------------
* Prompt bodies, model output and credentials are never logged.
* Only non-sensitive metadata (route, status, duration, prompt size)
  is recorded.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Human-readable format. Kept readable for local development.
_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Format for log collectors: one JSON object per line.
_JSON_FORMAT = "%(message)s"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Standard extras are useful and safe to include.
        for key in ("request_id", "method", "path", "status_code", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        # Never leak internals through logging: if a caller passes exc_info
        # the traceback goes to the server log only, not to the payload.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure root and application loggers.

    Called once during application startup.
    """
    handler = logging.StreamHandler(sys.stdout)

    if json_output:
        handler.setFormatter(JsonFormatter(_JSON_FORMAT))
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn installs its own handlers; route them through ours instead so
    # that application and server logs share one format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # Access logs are replaced by RequestLoggingMiddleware, which adds
    # timing and avoids duplicating output.
    logging.getLogger("uvicorn.access").disabled = True

    # Third-party loggers are noisy at DEBUG and can leak request internals.
    for name in ("httpx", "httpcore", "ollama"):
        logging.getLogger(name).setLevel(max(logging.INFO, root.level))


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the application root."""
    return logging.getLogger(f"app.{name}")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status and duration for each request.

    Request and response bodies are deliberately excluded: they may
    contain user prompts or model output. Only the size of the prompt is
    recorded, which is enough to diagnose problems without storing
    content.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        log = get_logger("http")

        # 5xx is a server fault, 4xx is a client fault; colour the level
        # accordingly so real problems stand out in development.
        if response.status_code >= 500:
            log_level = logging.ERROR
        elif response.status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        log.log(
            log_level,
            "request_complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response
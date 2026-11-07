"""Tests for error handling, logging and CORS behaviour."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from app.core.logging import JsonFormatter, configure_logging, get_logger
from app.exceptions.handlers import INTERNAL_ERROR_MESSAGE


# ---------- Error envelope ----------


def test_errors_use_consistent_envelope(client: TestClient) -> None:
    """Every error shares the same structure."""
    response = client.post("/api/ai/generate", json={})

    body = response.json()
    assert set(body.keys()) == {"success", "error"}
    assert set(body["error"].keys()) >= {"code", "message"}


def test_validation_errors_include_field_details(client: TestClient) -> None:
    """Validation failures point at the offending field."""
    response = client.post("/api/ai/generate", json={"prompt": ""})

    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert any("prompt" in detail["field"] for detail in body["error"]["details"])


def test_error_body_is_not_truncated(client: TestClient) -> None:
    """The response body must match its declared content-length.

    ``TestClient`` does not enforce ``content-length``, so a mismatch is
    invisible here unless asserted directly. Mutating ``JSONResponse.body``
    after construction would silently truncate responses for real clients.
    """
    response = client.post("/api/ai/generate", json={"prompt": ""})

    declared = int(response.headers["content-length"])
    actual = len(response.content)

    assert declared == actual, (
        f"content-length {declared} does not match body length {actual}"
    )


def test_error_body_is_valid_json_with_details(client: TestClient) -> None:
    """Details are present in the serialised body, not dropped."""
    import json as json_module

    response = client.post("/api/ai/generate", json={"prompt": ""})

    # Parsing the raw bytes confirms the payload survived serialisation.
    parsed = json_module.loads(response.content.decode())

    assert parsed["error"]["code"] == "validation_error"
    assert len(parsed["error"]["details"]) >= 1


def test_internal_error_message_is_generic() -> None:
    """The generic message reveals nothing about internals."""
    assert "Traceback" not in INTERNAL_ERROR_MESSAGE
    assert "app/" not in INTERNAL_ERROR_MESSAGE


# ---------- CORS ----------


def test_cors_allows_configured_frontend_origin(client: TestClient) -> None:
    """A request from the dev server origin is permitted."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.headers.get("access-control-allow-origin") == (
        "http://localhost:5173"
    )


def test_cors_allows_loopback_frontend_origin(client: TestClient) -> None:
    """The 127.0.0.1 dev-server origin is permitted."""
    response = client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})

    assert response.headers.get("access-control-allow-origin") == (
        "http://127.0.0.1:5173"
    )


def test_cors_does_not_allow_unknown_origin(client: TestClient) -> None:
    """An unrelated origin is not granted access."""
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})

    # No allow-origin header means the browser will block the request.
    assert "access-control-allow-origin" not in response.headers


def test_cors_never_wildcards_origin(client: TestClient) -> None:
    """The wildcard origin is never emitted."""
    response = client.get("/health", headers={"Origin": "https://anything.example"})

    assert response.headers.get("access-control-allow-origin") != "*"


# ---------- Logging ----------


def test_json_formatter_emits_single_line_json() -> None:
    """JSON logging produces one parseable object per record."""
    formatter = JsonFormatter("%(message)s")
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something happened",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc123"

    output = formatter.format(record)

    assert "\n" not in output.strip()
    assert '"message": "something happened"' in output
    assert '"level": "INFO"' in output


def test_json_formatter_never_leaks_secrets(caplog) -> None:
    """Secrets are not emitted as top-level fields."""
    formatter = JsonFormatter("%(message)s")
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    # Simulate a secret being attached as an attribute.
    record.api_key = "super-secret-value"

    assert "super-secret-value" not in formatter.format(record)


def test_configure_logging_replaces_handlers() -> None:
    """Calling configure_logging twice does not duplicate output."""
    configure_logging(level="DEBUG", json_output=False)
    configure_logging(level="INFO", json_output=False)

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.INFO


def test_logger_is_namespaced_under_app() -> None:
    """Application loggers share a common prefix for filtering."""
    assert get_logger("test").name.startswith("app.")


def test_request_is_logged_without_prompt_content(client: TestClient) -> None:
    """The request log records metadata but not the prompt body."""
    secret_prompt = "SENSITIVE_USER_CONTENT_12345"

    with caplog_for_app() as records:
        client.post("/api/ai/generate", json={"prompt": secret_prompt})

    joined = " ".join(records)
    assert secret_prompt not in joined
    # The prompt's size is still recorded, which is enough to diagnose.
    assert "prompt_chars" in joined


class caplog_for_app:  # noqa: N801 - acts as a context manager
    """Capture records emitted by the ``app`` logger hierarchy."""

    def __enter__(self) -> list[str]:
        self._handler = _ListHandler()
        self._logger = logging.getLogger("app")
        self._previous_level = self._logger.level
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        return self._handler.records

    def __exit__(self, *_: object) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._previous_level)


class _ListHandler(logging.Handler):
    """Collect formatted messages into a list."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())
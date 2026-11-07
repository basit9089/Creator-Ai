"""Tests for the AI generation endpoint.

Provider behaviour is faked, so no real model inference runs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.services.ai.provider import (
    GenerationOptions,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from tests.conftest import FakeAIProvider


# ---------- Validation ----------


def test_generate_requires_prompt(client: TestClient) -> None:
    """A missing prompt is rejected."""
    response = client.post("/api/ai/generate", json={})

    assert response.status_code == 422
    assert response.json()["success"] is False


def test_generate_rejects_empty_prompt(client: TestClient) -> None:
    """An empty prompt is rejected."""
    response = client.post("/api/ai/generate", json={"prompt": ""})

    assert response.status_code == 422


def test_generate_rejects_oversized_prompt(client: TestClient) -> None:
    """An excessively large prompt is rejected before reaching the provider."""
    from app.core.config import get_settings

    oversized = "x" * (get_settings().max_prompt_length + 1)

    response = client.post("/api/ai/generate", json={"prompt": oversized})

    assert response.status_code == 422


def test_generate_rejects_non_string_prompt(client: TestClient) -> None:
    """A non-string prompt is rejected."""
    response = client.post("/api/ai/generate", json={"prompt": 12345})

    assert response.status_code == 422


def test_generate_rejects_out_of_range_temperature(client: TestClient) -> None:
    """Temperature outside the allowed range is rejected."""
    response = client.post(
        "/api/ai/generate", json={"prompt": "Hello", "temperature": 9.5}
    )

    assert response.status_code == 422


# ---------- Successful generation ----------


def test_generate_returns_response_and_model(client: TestClient) -> None:
    """A valid request returns the generated text and model name."""
    response = client.post("/api/ai/generate", json={"prompt": "Hello there"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["response"] == "Fake response"
    assert body["model"] == "fake-model"


def test_generate_preserves_legacy_response_fields(client: TestClient) -> None:
    """The response keeps the fields the original endpoint returned."""
    response = client.post("/api/ai/generate", json={"prompt": "Hi"})

    body = response.json()
    # These field names are relied upon by the existing frontend.
    assert "response" in body
    assert "model" in body
    assert "success" in body


def test_generate_passes_prompt_as_user_message(
    client: TestClient, fake_provider: FakeAIProvider
) -> None:
    """The prompt is forwarded as a user message."""
    client.post("/api/ai/generate", json={"prompt": "Explain gravity"})

    messages, _ = fake_provider.calls[0]
    assert messages[0].role == "user"
    assert messages[0].content == "Explain gravity"


def test_generate_prepends_system_prompt(
    client: TestClient, fake_provider: FakeAIProvider
) -> None:
    """A system prompt is placed before the user message."""
    client.post(
        "/api/ai/generate",
        json={"prompt": "Be brief", "system_prompt": "You are terse."},
    )

    messages, _ = fake_provider.calls[0]
    assert messages[0].role == "system"
    assert messages[0].content == "You are terse."
    assert messages[1].role == "user"


def test_generate_forwards_temperature(
    client: TestClient, fake_provider: FakeAIProvider
) -> None:
    """Temperature is forwarded to the provider as a neutral option."""
    client.post("/api/ai/generate", json={"prompt": "Hi", "temperature": 0.25})

    _, options = fake_provider.calls[0]
    assert isinstance(options, GenerationOptions)
    assert options.temperature == 0.25


# ---------- Error handling ----------


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (ProviderUnavailableError("unavailable"), 503, "ai_provider_unavailable"),
        (ProviderTimeoutError("timed out"), 504, "ai_provider_timeout"),
        (ModelNotFoundError("missing"), 503, "model_not_found"),
        (ProviderResponseError("bad response"), 502, "ai_provider_error"),
    ],
)
def test_provider_failures_map_to_http_status(
    client_factory, error: Exception, expected_status: int, expected_code: str
) -> None:
    """Provider failures return a 5xx status instead of HTTP 200."""
    provider = FakeAIProvider(error=error)

    with client_factory(provider) as client:
        response = client.post("/api/ai/generate", json={"prompt": "Hello"})

        assert response.status_code == expected_status
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == expected_code


def test_provider_error_message_is_returned_to_client(client_factory) -> None:
    """A provider error's own message is shown to the caller.

    Provider exceptions are constructed with client-safe messages by the
    provider layer (see ``OllamaProvider._translate_error``), which is the
    boundary that strips host, port and path detail. This test documents
    that contract rather than re-testing sanitisation, which is covered
    directly in ``test_provider.py``.
    """
    provider = FakeAIProvider(
        error=ProviderResponseError("The AI provider returned an error.")
    )

    with client_factory(provider) as client:
        response = client.post("/api/ai/generate", json={"prompt": "Hello"})

        body = response.json()
        assert body["error"]["message"] == "The AI provider returned an error."


def test_unexpected_error_returns_500_without_traceback(client_factory) -> None:
    """An unexpected exception becomes a generic 500."""
    provider = FakeAIProvider(error=RuntimeError("boom: internal detail"))

    with client_factory(provider) as client:
        response = client.post("/api/ai/generate", json={"prompt": "Hello"})

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_server_error"
        assert "Traceback" not in response.text
        assert "boom" not in response.text


# ---------- Streaming ----------


def test_stream_returns_chunked_output(client: TestClient) -> None:
    """The streaming endpoint yields incremental chunks."""
    response = client.post("/api/ai/generate/stream", json={"prompt": "Hello"})

    assert response.status_code == 200
    assert "Fake" in response.text
    assert '"done": true' in response.text
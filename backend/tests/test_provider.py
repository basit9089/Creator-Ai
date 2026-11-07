"""Tests for the AI provider abstraction and the Ollama implementation.

These tests use a stub client, so no Ollama server is required.
"""

from __future__ import annotations

import httpx
import pytest
from ollama import ResponseError

from app.core.config import Settings, get_settings
from app.services.ai.ollama_provider import OllamaProvider
from app.services.ai.provider import (
    GenerationOptions,
    Message,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

PROMPT = [Message(role="user", content="Hello")]


class StubMessage:
    """Minimal stand-in for the Ollama message object."""

    def __init__(self, content: str, thinking: str | None = None) -> None:
        self.content = content
        self.thinking = thinking


class StubChatResponse:
    """Minimal stand-in for the Ollama chat response."""

    def __init__(
        self,
        content: str = "Hello there",
        thinking: str | None = None,
        model: str = "qwen3:8b",
    ) -> None:
        self.model = model
        self.message = StubMessage(content, thinking)
        self.prompt_eval_count = 12
        self.eval_count = 8
        self.total_duration = 1_500_000_000  # nanoseconds


class StubModel:
    def __init__(self, name: str) -> None:
        self.model = name


class StubListResponse:
    def __init__(self, names: list[str]) -> None:
        self.models = [StubModel(n) for n in names]


class StubClient:
    """Stub Ollama client that records calls and returns canned data."""

    def __init__(
        self,
        *,
        chat_result: object | None = None,
        list_result: object | None = None,
        chat_error: Exception | None = None,
        list_error: Exception | None = None,
    ) -> None:
        self.chat_result = chat_result or StubChatResponse()
        self.list_result = list_result or StubListResponse(["qwen3:8b"])
        self.chat_error = chat_error
        self.list_error = list_error
        self.chat_calls: list[dict] = []
        self.closed = False

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.chat_error is not None:
            raise self.chat_error
        return self.chat_result

    def list(self):
        if self.list_error is not None:
            raise self.list_error
        return self.list_result

    def close(self) -> None:
        self.closed = True


def make_provider(stub: StubClient, **overrides) -> OllamaProvider:
    """Build an OllamaProvider backed by ``stub``."""
    settings = Settings(**overrides)
    provider = OllamaProvider(settings)
    provider._client = stub  # noqa: SLF001 - test seam
    return provider


# ---------- Configuration ----------


def test_provider_reads_host_and_model_from_settings() -> None:
    """The provider takes its host and model from configuration."""
    provider = OllamaProvider(
        Settings(ollama_host="http://ollama.internal:9999", ollama_model="llama3:8b")
    )

    assert provider.default_model == "llama3:8b"
    assert provider.name == "ollama"


def test_provider_does_not_hardcode_configuration() -> None:
    """No connection details are baked into the provider's executable code."""
    import ast
    import inspect

    import app.services.ai.ollama_provider as module

    source = inspect.getsource(module)

    # Comments and docstrings may mention example values for clarity, so
    # only executable string literals are inspected. This distinguishes
    # documentation from actual hardcoded configuration.
    tree = ast.parse(source)
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "http://127.0.0.1:11434" not in literals
    assert "qwen3:8b" not in literals


# ---------- Generation ----------


def test_generate_returns_text_and_usage() -> None:
    """A successful call returns text, model and token counts."""
    provider = make_provider(StubClient())

    result = provider.generate(PROMPT)

    assert result.text == "Hello there"
    assert result.model == "qwen3:8b"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 8
    assert result.total_tokens == 20


def test_generate_converts_nanoseconds_to_milliseconds() -> None:
    """Ollama reports durations in nanoseconds; the result uses milliseconds."""
    provider = make_provider(StubClient())

    result = provider.generate(PROMPT)

    assert result.total_duration_ms == 1500.0


def test_generate_captures_thinking_separately() -> None:
    """Reasoning content is kept out of the user-visible text."""
    provider = make_provider(
        StubClient(chat_result=StubChatResponse(content="Answer.", thinking="Reasoning..."))
    )

    result = provider.generate(PROMPT)

    assert result.text == "Answer."
    assert result.thinking == "Reasoning..."


def test_generate_forwards_options() -> None:
    """Options are translated into the shape ``Client.chat`` expects.

    Sampling settings live under ``options``; ``think`` is a separate
    top-level parameter. Passing ``temperature`` at the top level raises a
    ``TypeError`` in the real client, which is exactly the kind of mistake
    a stubbed unit test would not catch.
    """
    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(
        PROMPT,
        options=GenerationOptions(temperature=0.3, max_tokens=128, think=False),
    )

    call = stub.chat_calls[0]
    assert call["options"]["temperature"] == 0.3
    # The neutral output cap is `max_tokens`; Ollama calls it num_predict.
    assert call["options"]["num_predict"] == 128
    assert call["think"] is False
    assert call["stream"] is False


def test_generate_never_sends_neutral_option_names_to_ollama() -> None:
    """``max_tokens`` must be translated, not passed through."""
    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(PROMPT, options=GenerationOptions(max_tokens=64))

    call = stub.chat_calls[0]
    assert "max_tokens" not in call
    assert "max_tokens" not in call["options"]


def test_generate_sends_only_options_the_client_accepts() -> None:
    """Guards the exact failure mode of a ``TypeError`` from the real client."""
    import inspect

    from ollama import Client

    accepted = set(inspect.signature(Client.chat).parameters)

    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(
        PROMPT,
        options=GenerationOptions(
            temperature=0.5,
            top_p=0.8,
            max_tokens=32,
            seed=1,
            stop=("END",),
            think=True,
        ),
    )

    unexpected = set(stub.chat_calls[0]) - accepted
    assert not unexpected, f"unexpected chat arguments: {unexpected}"


def test_generate_applies_deployment_defaults() -> None:
    """Unset options fall back to configuration, not to the model's own default.

    This matters most for ``think``. Qwen3 enables reasoning on its own,
    and on the target hardware that costs roughly 86 seconds versus 3.6
    seconds, so the provider always states the configured value rather
    than leaving it to Ollama.
    """
    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(PROMPT)

    call = stub.chat_calls[0]
    settings = get_settings()
    assert call["options"]["temperature"] == settings.default_ai_temperature
    assert call["options"]["num_predict"] == settings.max_ai_output_tokens
    assert call["think"] is False


def test_generate_thinking_override_reaches_the_backend() -> None:
    """An agent may opt into reasoning for a single call."""
    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(PROMPT, options=GenerationOptions(think=True))

    assert stub.chat_calls[0]["think"] is True


def test_generate_keeps_other_defaults_when_overriding_thinking() -> None:
    """A partial override does not discard the rest of the defaults."""
    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(PROMPT, options=GenerationOptions(think=True))

    call = stub.chat_calls[0]
    assert call["think"] is True
    assert call["options"]["temperature"] == get_settings().default_ai_temperature


def test_generate_never_sends_seeded_randomness_by_default() -> None:
    """``seed`` stays unset unless asked for, keeping output reproducible
    only when a caller opts in."""
    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(PROMPT)

    assert "seed" not in stub.chat_calls[0]["options"]


def test_generate_sends_messages_as_mappings() -> None:
    """Messages are serialised to plain mappings."""
    stub = StubClient()
    provider = make_provider(stub)

    provider.generate(PROMPT)

    assert stub.chat_calls[0]["messages"] == [{"role": "user", "content": "Hello"}]


# ---------- Streaming ----------


def test_stream_yields_text() -> None:
    """Streaming yields incremental text fragments."""

    def stream_chunks():
        yield StubChatResponse(content="Hello ")
        yield StubChatResponse(content="there")
        yield StubChatResponse(content="")

    stub = StubClient()
    stub.chat_result = stream_chunks()
    provider = make_provider(stub)

    chunks = list(provider.stream(PROMPT))

    assert chunks == ["Hello ", "there"]


# ---------- Error translation ----------


def test_connection_error_maps_to_unavailable() -> None:
    """A refused connection becomes ProviderUnavailableError."""
    provider = make_provider(StubClient(chat_error=ConnectionError("refused")))

    with pytest.raises(ProviderUnavailableError):
        provider.generate(PROMPT)


def test_httpx_connect_error_maps_to_unavailable() -> None:
    """An httpx connect failure is classified as unavailable."""
    error = httpx.ConnectError("cannot connect")
    provider = make_provider(StubClient(chat_error=error))

    with pytest.raises(ProviderUnavailableError):
        provider.generate(PROMPT)


def test_timeout_maps_to_timeout_error() -> None:
    """A timeout is distinguished from a general failure."""
    provider = make_provider(StubClient(chat_error=httpx.ReadTimeout("slow")))

    with pytest.raises(ProviderTimeoutError):
        provider.generate(PROMPT)


def test_404_response_maps_to_model_not_found() -> None:
    """A 404 from Ollama means the model is not installed."""
    error = ResponseError("model not found", status_code=404)
    provider = make_provider(StubClient(chat_error=error))

    with pytest.raises(ModelNotFoundError):
        provider.generate(PROMPT)


def test_other_response_error_maps_to_provider_response_error() -> None:
    """A non-404 Ollama error maps to a generic provider error."""
    error = ResponseError("internal error", status_code=500)
    provider = make_provider(StubClient(chat_error=error))

    with pytest.raises(ProviderResponseError):
        provider.generate(PROMPT)


def test_unexpected_error_maps_to_provider_response_error() -> None:
    """An unrecognised failure still becomes a provider error."""
    provider = make_provider(StubClient(chat_error=RuntimeError("weird")))

    with pytest.raises(ProviderResponseError):
        provider.generate(PROMPT)


def test_errors_do_not_expose_internals() -> None:
    """Provider error messages are client-safe.

    This is the security boundary: raw backend exceptions carry host,
    port and path detail, so translation replaces them with a fixed,
    generic message. The error handler returns ``exc.message`` unchanged,
    which is only safe because this step happens here.
    """
    provider = make_provider(
        StubClient(
            chat_error=ConnectionError(
                "Failed to connect to http://127.0.0.1:11434/api/chat [internal path]"
            )
        )
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        provider.generate(PROMPT)

    message = str(exc_info.value)
    assert "11434" not in message
    assert "internal path" not in message
    assert "Traceback" not in message
    # The original exception is retained for server-side logging only.
    assert exc_info.value.__cause__ is not None


def test_model_not_found_message_is_safe() -> None:
    """The missing-model message names the model but leaks nothing else."""
    provider = make_provider(
        StubClient(chat_error=ResponseError("not found at /api/chat", status_code=404))
    )

    with pytest.raises(ModelNotFoundError) as exc_info:
        provider.generate(PROMPT)

    assert "/api/chat" not in str(exc_info.value)


# ---------- Health ----------


def test_health_healthy_when_model_installed() -> None:
    """Health is healthy when the model is present."""
    provider = make_provider(StubClient())

    health = provider.health_check()

    assert health.status == "healthy"
    assert health.reachable is True
    assert health.model_available is True


def test_health_degraded_when_model_missing() -> None:
    """A reachable server without the model is degraded, not unhealthy."""
    provider = make_provider(
        StubClient(list_result=StubListResponse(["llama3:8b"]))
    )

    health = provider.health_check()

    assert health.status == "degraded"
    assert health.reachable is True
    assert health.model_available is False
    assert "not installed" in health.detail


def test_health_unhealthy_when_unreachable() -> None:
    """An unreachable server is unhealthy."""
    provider = make_provider(StubClient(list_error=ConnectionError("refused")))

    health = provider.health_check()

    assert health.status == "unhealthy"
    assert health.reachable is False
    assert health.model_available is False


def test_health_matches_model_ignoring_tag() -> None:
    """A base-name match counts as installed, since tags vary by tool."""
    provider = make_provider(
        StubClient(list_result=StubListResponse(["qwen3"])),
    )

    health = provider.health_check()

    assert health.model_available is True


def test_health_check_does_not_generate() -> None:
    """The health check lists models only; it never calls chat."""
    stub = StubClient()
    provider = make_provider(stub)

    provider.health_check()

    assert stub.chat_calls == []


# ---------- Lifecycle ----------


def test_close_is_safe_and_idempotent() -> None:
    """Closing twice does not raise."""
    stub = StubClient()
    provider = make_provider(stub)

    provider.close()
    provider.close()

    assert stub.closed is True


# ---------- Structured output ----------


def test_generate_json_parses_plain_json() -> None:
    """Plain JSON output is parsed."""
    provider = make_provider(
        StubClient(chat_result=StubChatResponse(content='{"answer": "42"}'))
    )

    assert provider.generate_json(PROMPT) == {"answer": "42"}


def test_generate_json_handles_code_fences() -> None:
    """Markdown-fenced JSON is handled."""
    provider = make_provider(
        StubClient(chat_result=StubChatResponse(content='```json\n{"answer": "42"}\n```'))
    )

    assert provider.generate_json(PROMPT) == {"answer": "42"}


def test_generate_json_raises_on_invalid_output() -> None:
    """Non-JSON output raises a provider response error."""
    provider = make_provider(StubClient(chat_result=StubChatResponse(content="not json")))

    with pytest.raises(ProviderResponseError):
        provider.generate_json(PROMPT)
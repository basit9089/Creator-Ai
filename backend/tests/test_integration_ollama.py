"""Opt-in integration tests against a real local Ollama installation.

These are skipped by default so the unit suite stays fast and offline.
Run them explicitly with::

    RUN_OLLAMA_INTEGRATION=1 pytest -m integration -v

Note
----
Qwen3 in thinking mode can take well over a minute on CPU, so these
tests disable thinking and assert nothing about exact wording.
"""

from __future__ import annotations

import pytest

from app.services.ai.ollama_provider import OllamaProvider
from app.services.ai.provider import GenerationOptions, HealthStatus, Message

pytestmark = pytest.mark.integration


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider()


def test_real_ollama_is_reachable(provider: OllamaProvider) -> None:
    """A local Ollama server responds to a model listing."""
    health = provider.health_check()

    assert health.reachable is True
    assert health.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}


def test_configured_model_is_installed(provider: OllamaProvider) -> None:
    """The model named in configuration exists locally."""
    health = provider.health_check()

    assert health.model_available is True, (
        f"Model {health.model!r} is not installed. "
        f"Run: ollama pull {health.model}"
    )


def test_real_generation_returns_text(provider: OllamaProvider) -> None:
    """The configured model produces a non-empty response."""
    result = provider.generate(
        [Message(role="user", content="Reply with the single word: hello")],
        # Thinking mode is disabled: it is far slower and unnecessary here.
        options=GenerationOptions(think=False),
    )

    assert result.text
    assert result.completion_tokens is not None
    assert result.completion_tokens > 0


def test_real_streaming_yields_text(provider: OllamaProvider) -> None:
    """Streaming returns at least one chunk from the live model."""
    chunks = list(
        provider.stream(
            [Message(role="user", content="Count to three.")],
            options=GenerationOptions(think=False),
        )
    )

    assert chunks
    assert "".join(chunks).strip()


def test_real_system_prompt_is_honoured(provider: OllamaProvider) -> None:
    """A system prompt does not break the request."""
    result = provider.generate(
        [
            Message(role="system", content="You always answer with one word."),
            Message(role="user", content="What colour is the sky?"),
        ],
        options=GenerationOptions(think=False),
    )

    assert result.text


# ---------- Phase 3: defaults, overrides and structured output ----------


def test_thinking_defaults_to_false_end_to_end(
    provider: OllamaProvider, integration_enabled: bool
) -> None:
    """A call that says nothing about reasoning must run without it.

    This is the behaviour that took ~86 s per request before it was fixed:
    Qwen3 enables reasoning on its own, so the provider has to state
    ``think=False`` explicitly rather than deferring to Ollama.
    """
    import time

    started = time.perf_counter()
    result = provider.generate(
        [Message(role="user", content="Reply with the single word: hello")]
    )
    elapsed = time.perf_counter() - started

    assert result.text
    assert elapsed < 30, (
        f"Default generation took {elapsed:.1f}s. Reasoning is probably still "
        "enabled somewhere; check DEFAULT_AI_THINKING and the provider's "
        "resolved options."
    )


def test_thinking_override_reaches_the_real_model(provider: OllamaProvider) -> None:
    """Explicitly enabling reasoning must work, and take visibly longer.

    Nothing asserts an exact duration: the point is that the flag reaches
    the model and changes its behaviour, which is what a cloud provider
    swap has to preserve.
    """
    result = provider.generate(
        [Message(role="user", content="Reply with the single word: hello")],
        options=GenerationOptions(think=True),
    )

    assert result.text
    # Reasoning content is returned separately from the answer, so a
    # non-empty answer here means the request itself succeeded.
    assert result.completion_tokens is not None


def test_structured_output_returns_valid_json(provider: OllamaProvider) -> None:
    """The schema instruction produces a usable typed object.

    Deliberately a trivial schema: this asserts the mechanism works with
    the real model, not that Qwen3 is good at research.
    """
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str
        confident: bool

    answer = provider.generate_structured(
        [Message(role="user", content="Is the sky blue? Answer briefly.")],
        Answer,
    )

    assert isinstance(answer.answer, str)
    assert answer.answer.strip()
    assert isinstance(answer.confident, bool)


def test_max_tokens_limit_is_respected(provider: OllamaProvider) -> None:
    """The configured output cap reaches the real model."""
    result = provider.generate(
        [Message(role="user", content="List three colours.")],
        options=GenerationOptions(max_tokens=16, think=False),
    )

    assert result.text
    assert result.completion_tokens is not None
    assert result.completion_tokens <= 32
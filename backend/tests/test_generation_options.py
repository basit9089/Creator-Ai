"""Tests for GenerationOptions and deployment-default merging."""

from __future__ import annotations

import pytest

from app.core.config import Settings, get_settings, reset_settings_cache
from app.services.ai.provider import (
    AIProvider,
    GenerationOptions,
    GenerationResult,
    Message,
    ProviderHealth,
)


class _EchoProvider(AIProvider):
    """Minimal provider that records the options it was asked for."""

    name = "echo"

    def __init__(self, defaults: GenerationOptions | None = None) -> None:
        self._defaults = defaults
        self.received: GenerationOptions | None = None

    def default_options(self) -> GenerationOptions:
        return self._defaults or super().default_options()

    def generate(self, messages, *, options=None) -> GenerationResult:  # noqa: ANN001, ANN201
        self.received = self.resolve_options(options)
        return GenerationResult(text="ok", model="echo")

    def stream(self, messages, *, options=None):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def health_check(self) -> ProviderHealth:
        raise NotImplementedError


def test_to_dict_omits_unset_options() -> None:
    options = GenerationOptions(temperature=0.1)

    assert options.to_dict() == {"temperature": 0.1}


def test_to_dict_includes_every_set_option() -> None:
    options = GenerationOptions(
        temperature=0.2,
        top_p=0.8,
        max_tokens=128,
        seed=7,
        stop=("END",),
    )

    assert options.to_dict() == {
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 128,
        "seed": 7,
        "stop": ["END"],
    }


def test_to_dict_never_includes_thinking() -> None:
    """``think`` is a top-level provider parameter, not a model option.

    Including it in the options mapping would be silently ignored, or
    worse, rejected by providers that validate that mapping.
    """
    assert "think" not in GenerationOptions(think=True).to_dict()


def test_merge_keeps_defaults_for_unset_fields() -> None:
    defaults = GenerationOptions(temperature=0.7, top_p=0.9, max_tokens=2048, think=False)

    merged = defaults.merge(GenerationOptions(temperature=0.1))

    assert merged.temperature == 0.1
    assert merged.top_p == 0.9
    assert merged.max_tokens == 2048
    assert merged.think is False


def test_merge_allows_overriding_thinking() -> None:
    defaults = GenerationOptions(temperature=0.7, think=False)

    merged = defaults.merge(GenerationOptions(think=True))

    assert merged.think is True
    assert merged.temperature == 0.7


def test_merge_with_none_returns_equivalent_options() -> None:
    defaults = GenerationOptions(temperature=0.7, think=False)

    assert defaults.merge(None) == defaults


def test_merge_does_not_mutate_either_operand() -> None:
    defaults = GenerationOptions(temperature=0.7)
    overrides = GenerationOptions(temperature=0.1, think=True)

    defaults.merge(overrides)

    assert defaults.temperature == 0.7
    assert defaults.think is None
    assert overrides.think is True


def test_with_thinking_preserves_other_options() -> None:
    base = GenerationOptions(temperature=0.5, max_tokens=64)

    enabled = base.with_thinking(True)
    disabled = base.with_thinking(False)

    assert enabled.think is True
    assert disabled.think is False
    assert enabled.max_tokens == 64
    assert enabled.temperature == 0.5


def test_with_thinking_does_not_mutate_base() -> None:
    base = GenerationOptions(think=False)

    base.with_thinking(True)

    assert base.think is False


# ---------- Deployment defaults ----------


def test_settings_thinking_defaults_to_false() -> None:
    """Reasoning is off by default.

    Measured on the target hardware: ~86 s with reasoning on versus ~3.6 s
    off for the same prompt.
    """
    assert get_settings().default_ai_thinking is False


def test_default_generation_options_carry_configured_thinking() -> None:
    options = get_settings().default_generation_options()

    assert options.think is False
    assert options.temperature == get_settings().default_ai_temperature
    assert options.max_tokens == get_settings().max_ai_output_tokens


def test_settings_can_configure_thinking_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_AI_THINKING", "true")
    reset_settings_cache()

    settings = get_settings()

    assert settings.default_ai_thinking is True
    assert settings.default_generation_options().think is True


def test_provider_resolve_options_defaults_thinking_to_false() -> None:
    provider = _EchoProvider()

    resolved = provider.resolve_options(None)

    assert resolved.think is False


def test_provider_resolve_options_honours_explicit_thinking() -> None:
    provider = _EchoProvider()

    assert provider.resolve_options(GenerationOptions(think=True)).think is True


def test_generate_applies_configured_thinking_default() -> None:
    provider = _EchoProvider()

    provider.generate([Message(role="user", content="hi")])

    assert provider.received is not None
    assert provider.received.think is False


def test_generate_honours_thinking_override() -> None:
    provider = _EchoProvider()

    provider.generate([Message(role="user", content="hi")], options=GenerationOptions(think=True))

    assert provider.received is not None
    assert provider.received.think is True


def test_provider_can_supply_its_own_defaults() -> None:
    """A cloud provider with different defaults overrides ``default_options``."""
    provider = _EchoProvider(defaults=GenerationOptions(temperature=0.2, max_tokens=8192))

    resolved = provider.resolve_options(GenerationOptions(think=True))

    assert resolved.temperature == 0.2
    assert resolved.max_tokens == 8192
    assert resolved.think is True


def test_settings_default_generation_options_is_reflecting_custom_settings() -> None:
    settings = Settings(
        default_ai_temperature=0.1,
        default_ai_top_p=0.5,
        default_ai_thinking=True,
        max_ai_output_tokens=32,
    )

    options = settings.default_generation_options()

    assert options.temperature == 0.1
    assert options.top_p == 0.5
    assert options.think is True
    assert options.max_tokens == 32

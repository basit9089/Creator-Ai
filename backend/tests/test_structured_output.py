"""Tests for structured generation.

The point of the structured-output capability is that agents ask for typed
data instead of parsing prose. These tests use a scripted provider so the
model's behaviour, including its mistakes, can be exercised deterministically.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from app.services.ai.provider import (
    AIProvider,
    GenerationResult,
    Message,
    StructuredOutputError,
)


class Research(BaseModel):
    """Stand-in for the researcher's eventual output shape."""

    topic: str = Field(description="The researched topic.")
    summary: str
    key_points: list[str] = []


class Verdict(BaseModel):
    verdict: Literal["supported", "unsupported", "partial"]
    reason: str


class ScriptedProvider(AIProvider):
    """Returns pre-arranged text and records what it was asked."""

    name = "scripted"

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.messages: list[Sequence[Message]] = []
        self.options: list[object] = []

    def generate(
        self, messages: Sequence[Message], *, options=None
    ) -> GenerationResult:
        self.messages.append(messages)
        self.options.append(options)
        return GenerationResult(
            text=self._responses.pop(0) if self._responses else "{}",
            model="scripted",
        )

    def stream(self, messages, *, options=None) -> Iterator[str]:
        raise NotImplementedError

    def health_check(self):
        raise NotImplementedError


# ---------- Happy path ----------


def test_generate_structured_returns_validated_model() -> None:
    payload = {"topic": "Recursion", "summary": "A function calling itself.", "key_points": ["base case"]}
    provider = ScriptedProvider(json.dumps(payload))

    result = provider.generate_structured(
        [Message(role="user", content="Research recursion")], Research
    )

    assert isinstance(result, Research)
    assert result.topic == "Recursion"
    assert result.key_points == ["base case"]


def test_generate_structured_applies_schema_defaults() -> None:
    """A model that omits an optional field still validates."""
    provider = ScriptedProvider(json.dumps({"topic": "T", "summary": "S"}))

    result = provider.generate_structured(
        [Message(role="user", content="go")], Research
    )

    assert result.key_points == []


def test_generate_structured_forwards_options_unchanged() -> None:
    provider = ScriptedProvider(json.dumps({"verdict": "supported", "reason": "stated"}))

    provider.generate_structured(
        [Message(role="user", content="check")], Verdict, options=None
    )

    assert provider.options == [None]


# ---------- Schema instruction ----------


def test_schema_instruction_is_appended_as_the_final_user_message() -> None:
    """The instruction lands closest to generation, which small models follow
    far more reliably."""
    provider = ScriptedProvider(json.dumps({"topic": "T", "summary": "S"}))

    provider.generate_structured([Message(role="user", content="Research recursion")], Research)

    sent = provider.messages[0]
    assert sent[0].content == "Research recursion"
    assert sent[-1].role == "user"


def test_schema_instruction_includes_field_names() -> None:
    provider = ScriptedProvider(json.dumps({"topic": "T", "summary": "S"}))

    provider.generate_structured([Message(role="user", content="go")], Research)

    instruction = provider.messages[0][-1].content
    assert "topic" in instruction
    assert "summary" in instruction
    assert "key_points" in instruction


def test_schema_instruction_forbids_markdown_fences() -> None:
    """Fenced JSON is the most common failure mode and must be asked away."""
    provider = ScriptedProvider(json.dumps({"topic": "T", "summary": "S"}))

    provider.generate_structured([Message(role="user", content="go")], Research)

    assert "fence" in provider.messages[0][-1].content.lower()


def test_schema_instruction_preserves_existing_messages() -> None:
    provider = ScriptedProvider(json.dumps({"topic": "T", "summary": "S"}))

    provider.generate_structured(
        [
            Message(role="system", content="You are terse."),
            Message(role="user", content="Research recursion"),
        ],
        Research,
    )

    sent = provider.messages[0]
    assert [m.role for m in sent] == ["system", "user", "user"]


def test_schema_instruction_does_not_mutate_caller_messages() -> None:
    provider = ScriptedProvider(json.dumps({"topic": "T", "summary": "S"}))
    original = [Message(role="user", content="Research recursion")]

    provider.generate_structured(original, Research)

    assert len(original) == 1


# ---------- Failure modes ----------


def test_invalid_json_raises_structured_output_error() -> None:
    provider = ScriptedProvider("Sure! Here is the JSON you asked for:")

    with pytest.raises(StructuredOutputError):
        provider.generate_structured([Message(role="user", content="go")], Research)


def test_fenced_json_is_tolerated() -> None:
    """Markdown fences are stripped before parsing.

    Small local models wrap JSON in fences even when told not to. Being
    strict here would fail otherwise-correct answers, which would make the
    abstraction look broken rather than the model.
    """
    fenced = "```json\n{\"topic\": \"T\", \"summary\": \"S\"}\n```"
    provider = ScriptedProvider(fenced)

    result = provider.generate_structured([Message(role="user", content="go")], Research)

    assert result.topic == "T"


def test_schema_mismatch_raises_structured_output_error() -> None:
    provider = ScriptedProvider(json.dumps({"topic": "T", "summary": "S", "extra": 1}))

    with pytest.raises(StructuredOutputError):
        provider.generate_structured([Message(role="user", content="go")], Verdict)


def test_non_object_json_raises_structured_output_error() -> None:
    provider = ScriptedProvider(json.dumps(["a", "list"]))

    with pytest.raises(StructuredOutputError):
        provider.generate_structured([Message(role="user", content="go")], Research)


def test_validation_error_names_only_failing_fields() -> None:
    """The message stays short enough to log or return.

    A full Pydantic error report can run to hundreds of lines and is not
    useful to a caller.
    """
    provider = ScriptedProvider(json.dumps({"topic": "T"}))

    with pytest.raises(StructuredOutputError) as excinfo:
        provider.generate_structured([Message(role="user", content="go")], Research)

    message = str(excinfo.value)
    assert "summary" in message
    assert len(message) < 300


def test_validation_error_preserves_original_exception_as_cause() -> None:
    """Full detail stays available server-side through ``__cause__``."""
    import pydantic

    provider = ScriptedProvider(json.dumps({"topic": "T"}))

    with pytest.raises(StructuredOutputError) as excinfo:
        provider.generate_structured([Message(role="user", content="go")], Research)

    assert isinstance(excinfo.value.__cause__, pydantic.ValidationError)


def test_structured_output_error_is_retryable_in_principle() -> None:
    """It is a provider response error, so it maps to a 5xx, not a 4xx.

    The caller did nothing wrong; the model produced unusable output.
    """
    assert issubclass(StructuredOutputError, Exception)
    assert StructuredOutputError.status_code == 502
    assert StructuredOutputError.code == "structured_output_invalid"

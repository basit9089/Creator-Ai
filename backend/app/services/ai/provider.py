"""Provider-agnostic AI interface.

This module defines the contract that every AI backend must satisfy.
Agents (Researcher, Fact Checker, Scriptwriter, ...) depend on
:class:`AIProvider` only, never on a concrete backend. That way a cloud
provider can be introduced later without touching agent code.

Design goals
------------
* **Backend independence.** Agents import :class:`AIProvider`.
* **Streaming-ready.** :meth:`AIProvider.stream` is part of the contract
  from day one, so adding token streaming later does not change callers.
* **Structured-output-ready.** :meth:`AIProvider.generate_json` is
  defined alongside text generation.
* **No backend types leak.** Messages, options and results are plain
  dataclasses defined here, so provider SDK objects never escape this
  layer.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

# Roles supported by the message model. Deliberately a closed set so that
# invalid roles fail fast instead of reaching a provider.
MessageRole = Literal["system", "user", "assistant", "tool"]

#: Bound used by :meth:`AIProvider.generate_structured` to return a
#: caller-specified Pydantic model without importing typing generics at
#: every call site.
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversation message.

    Uses ``slots=True`` and ``frozen=True`` because messages are small,
    numerous, and should never be mutated after construction.
    """

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to the plain mapping shape used by provider SDKs."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Optional parameters for a generation request.

    Every field is optional: providers apply their own defaults when a
    value is ``None``. This keeps the abstraction neutral and avoids
    baking one backend's tuning vocabulary into the interface.

    Deliberately provider-neutral. Agents set these; they never set
    backend-specific parameters such as Ollama's ``num_ctx`` or a cloud
    vendor's own tuning fields.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] | None = None

    #: Reasoning mode for hybrid models such as Qwen3.
    #:
    #: ``None`` means "let the provider decide". In practice agents should
    #: pass an explicit value: on modest hardware, enabling reasoning took
    #: roughly 86 seconds versus 3.6 seconds for the same prompt, so it is
    #: opt-in per call rather than a blanket default.
    think: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a mapping, omitting unset options.

        Providers receive only what was explicitly requested, so their
        own defaults remain in effect for everything else.
        """
        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop is not None:
            options["stop"] = list(self.stop)
        return options

    def merge(self, overrides: GenerationOptions | None) -> GenerationOptions:
        """Return these options with ``overrides`` applied on top.

        Only fields explicitly set on ``overrides`` take effect, so a
        partial override never discards the deployment defaults. This is
        how an agent opts into reasoning for one call without restating
        every other setting.
        """
        if overrides is None:
            return self

        defaults: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": self.stop,
            "think": self.think,
        }

        for field_name in defaults:
            value = getattr(overrides, field_name)
            if value is not None:
                defaults[field_name] = value

        return GenerationOptions(**defaults)

    def with_thinking(self, enabled: bool) -> GenerationOptions:
        """Return a copy with reasoning mode set explicitly."""
        return self.merge(GenerationOptions(think=enabled))


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The outcome of a single non-streaming generation."""

    text: str
    model: str

    #: Reasoning content, when the provider exposes it separately.
    #: Empty when the model does not produce one.
    thinking: str = ""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ms: float | None = None

    @property
    def total_tokens(self) -> int | None:
        """Combined token count, when the provider reports both parts."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class HealthStatus(StrEnum):
    """Overall availability of an AI provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Result of a provider health check.

    Carries no secrets: only status, reachability and the configured
    model name, which is already public information.
    """

    status: HealthStatus
    provider: str
    model: str
    reachable: bool
    model_available: bool

    #: Short, client-safe explanation. Never contains stack traces.
    detail: str = ""

    #: Latency of the check itself, useful for spotting a slow host.
    latency_ms: float | None = None


class AIProviderError(Exception):
    """Base class for all provider failures.

    Carries a stable machine-readable ``code`` plus a client-safe
    ``message``. The original exception is attached as ``__cause__`` for
    logging, but never rendered into an HTTP response.
    """

    code = "ai_provider_error"
    status_code = 500

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider


class ProviderUnavailableError(AIProviderError):
    """The provider could not be reached (host down, connection refused)."""

    code = "ai_provider_unavailable"
    status_code = 503


class ProviderTimeoutError(AIProviderError):
    """The provider did not respond within the configured timeout."""

    code = "ai_provider_timeout"
    status_code = 504


class ModelNotFoundError(AIProviderError):
    """The requested model is not installed on the provider."""

    code = "model_not_found"
    status_code = 503


class ProviderResponseError(AIProviderError):
    """The provider responded, but with an error or unusable output."""

    code = "ai_provider_error"
    status_code = 502


class StructuredOutputError(ProviderResponseError):
    """The model returned JSON that does not match the requested schema.

    Kept distinct from a transport failure: the provider worked fine, but
    the output was unusable. Agents may reasonably retry this with a
    corrected prompt.
    """

    code = "structured_output_invalid"
    status_code = 502


class AIProvider(ABC):
    """Abstract base class for AI backends.

    Implementations must translate backend-specific failures into the
    exceptions defined in this module, so that callers can react to
    stable error types without importing backend SDKs.
    """

    #: Short identifier, for example ``"ollama"``.
    name: str = "unknown"

    #: Model this provider uses by default.
    default_model: str = ""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        """Generate a complete response and return it once finished.

        Implementations must apply the deployment defaults supplied in
        ``settings.default_generation_options()`` to any option the
        caller left unset. Callers therefore only need to specify what
        differs from the defaults.
        """

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
    ) -> Iterator[str]:
        """Yield response text incrementally.

        Streaming is part of the interface now so that adding a
        streaming endpoint later requires no change to callers.
        """

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Report reachability and model availability.

        Must be cheap: it should never trigger a real generation.
        """

    def generate_json(
        self,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
    ) -> Any:
        """Generate a response constrained to valid JSON.

        The base implementation generates text and parses it. Providers
        with native structured-output support should override this.
        """
        result = self.generate(messages, options=options)
        return _parse_json(result.text)

    def default_options(self) -> GenerationOptions:
        """Deployment-level defaults applied to every generation.

        Read from configuration, so changing ``DEFAULT_AI_THINKING`` or
        ``DEFAULT_AI_TEMPERATURE`` changes behaviour without touching
        agent code. A provider with its own sensible defaults may
        override this.
        """
        from app.core.config import get_settings

        return get_settings().default_generation_options()

    def resolve_options(self, options: GenerationOptions | None = None) -> GenerationOptions:
        """Combine caller options with the deployment defaults.

        Values the caller set win; anything left unset falls back to
        :meth:`default_options`. This is what makes reasoning mode
        opt-in per call: an agent that does not mention ``think``
        inherits the configured value rather than the model's own
        default, which for Qwen3 would be reasoning *on*.
        """
        return self.default_options().merge(options)

    def generate_structured(
        self,
        messages: Sequence[Message],
        schema: type[BaseModelT],
        *,
        options: GenerationOptions | None = None,
    ) -> BaseModelT:
        """Generate a response validated against a Pydantic model.

        Agents need typed data, not prose. This centralises the
        "ask the model for JSON, then validate it" dance in one place so
        no agent has to hand-roll it or parse text itself.

        The default implementation works with any backend: it asks for
        JSON using the schema as a guide, parses the response and
        validates it. A provider with native schema enforcement (such as
        a cloud API with a structured-output mode) can override this to
        pass the schema to the backend instead.

        Raises:
            StructuredOutputError: the model returned JSON that does not
                satisfy the schema.
        """
        request_messages = _with_schema_instruction(messages, schema)
        result = self.generate(request_messages, options=options)

        try:
            payload = _parse_json(result.text)
        except ProviderResponseError as exc:
            # Narrower than a generic response error: the model returned
            # something that is not JSON at all, which an agent may be able
            # to fix by rephrasing.
            raise StructuredOutputError(
                f"Response was not valid JSON for {schema.__name__}."
            ) from exc

        return _validate_structured(payload, schema)

    def close(self) -> None:
        """Release any underlying resources. Safe to call repeatedly."""
        return None


def _with_schema_instruction(
    messages: Sequence[Message], schema: type[BaseModel]
) -> list[Message]:
    """Append an explicit JSON Schema instruction to the message list.

    Small local models follow a concrete schema far more reliably than
    prose such as "return a topic and a summary". Field names and types
    are emitted; no values or examples are included.

    The instruction goes last so it is closest to the generation point.
    """
    try:
        json_schema = schema.model_json_schema()
    except Exception:  # noqa: BLE001 - a schema we cannot describe is not fatal
        return list(messages)

    instruction = (
        "Respond with a single JSON object and nothing else. "
        "Do not wrap it in Markdown code fences. "
        "Do not include any commentary before or after the JSON. "
        "The object must conform to this JSON Schema:\n"
        f"{json.dumps(json_schema, indent=2)}"
    )

    return [*messages, Message(role="user", content=instruction)]


def _validate_structured(payload: Any, schema: type[BaseModelT]) -> BaseModelT:
    """Validate parsed JSON against ``schema``.

    Error details are summarised rather than dumped: a validation report
    can be enormous and is not useful to an API client. The full report is
    available through the exception's ``__cause__`` for server logs.
    """
    if not isinstance(payload, dict):
        raise StructuredOutputError(
            f"Expected a JSON object matching {schema.__name__}, "
            f"received {type(payload).__name__}."
        )

    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        # List only the field paths that failed, not the full report.
        fields = ", ".join(
            ".".join(str(part) for part in error.get("loc", ())) or "<root>"
            for error in exc.errors()
        )
        raise StructuredOutputError(
            f"Response did not match {schema.__name__}. Invalid fields: {fields}."
        ) from exc


def _parse_json(text: str) -> Any:
    """Parse JSON from model output, tolerating markdown code fences.

    Models frequently wrap JSON in ```json fences even when asked not to,
    so the fences are stripped before parsing.
    """
    candidate = text.strip()

    if candidate.startswith("```"):
        # Drop the opening fence and any language tag, then the closing fence.
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
        candidate = candidate.rsplit("```", 1)[0].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            "The AI response was not valid JSON."
        ) from exc
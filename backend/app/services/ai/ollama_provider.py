"""Ollama-backed implementation of :class:`AIProvider`.

All connection details come from application settings; no host, model
name or timeout is hardcoded in this module.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
from ollama import Client, ResponseError

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.ai.provider import (
    AIProvider,
    GenerationOptions,
    GenerationResult,
    HealthStatus,
    Message,
    ModelNotFoundError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = get_logger("ai.ollama")


class OllamaProvider(AIProvider):
    """Generate text with a local or remote Ollama server.

    The client is created once per provider instance. Constructing an
    ``ollama.Client`` is cheap and does not perform I/O, so this does not
    delay application startup.
    """

    name = "ollama"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.ollama_model

        # Deployment defaults (temperature, top_p, output cap, reasoning
        # mode), resolved once. Caller options are merged over these by
        # AIProvider.resolve_options, so an agent only states what it
        # wants to change and cannot accidentally revert the configured
        # reasoning mode by omitting it.
        self._defaults = self._settings.default_generation_options()

        self._client = Client(
            host=self._settings.ollama_host,
            timeout=self._settings.ollama_timeout,
        )

        logger.info(
            "ollama_provider_initialised host=%s model=%s timeout=%ss thinking_default=%s",
            self._settings.ollama_host,
            self._model,
            self._settings.ollama_timeout,
            self._defaults.think,
        )

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def default_options(self) -> GenerationOptions:
        """Deployment-level defaults for this provider."""
        return self._defaults

    def resolve_options(self, options: GenerationOptions | None = None) -> GenerationOptions:
        """Combine caller options with the cached deployment defaults.

        Overrides the base implementation only to reuse the defaults
        resolved at construction time, since reading settings on every
        call would rebuild the options object needlessly.
        """
        return self._defaults.merge(options)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        """Generate a complete response.

        Raises:
            ProviderUnavailableError: the server is unreachable.
            ProviderTimeoutError: the request exceeded the timeout.
            ModelNotFoundError: the model is not installed.
            ProviderResponseError: the server returned an error.
        """
        request = self._build_request(options)
        payload = [message.to_dict() for message in messages]

        logger.debug(
            "ollama_generate model=%s messages=%d think=%s num_predict=%s",
            self._model,
            len(payload),
            request.get("think"),
            (request.get("options") or {}).get("num_predict"),
        )

        try:
            response = self._client.chat(
                model=self._model,
                messages=payload,
                stream=False,
                **request,
            )
        except Exception as exc:  # noqa: BLE001 - translated below
            raise self._translate_error(exc) from exc

        return GenerationResult(
            text=self._extract_text(response, strip=True),
            model=getattr(response, "model", None) or self._model,
            thinking=getattr(getattr(response, "message", None), "thinking", None) or "",
            prompt_tokens=getattr(response, "prompt_eval_count", None),
            completion_tokens=getattr(response, "eval_count", None),
            total_duration_ms=self._to_ms(getattr(response, "total_duration", None)),
        )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
    ) -> Iterator[str]:
        """Yield response text as it is produced.

        The underlying connection is opened lazily so that connection
        errors surface on the first iteration.
        """
        request = self._build_request(options)
        payload = [message.to_dict() for message in messages]

        try:
            stream = self._client.chat(
                model=self._model,
                messages=payload,
                stream=True,
                **request,
            )
        except Exception as exc:  # noqa: BLE001 - translated below
            raise self._translate_error(exc) from exc

        try:
            for chunk in stream:
                # Not stripped: whitespace between tokens is significant.
                text = self._extract_text(chunk, strip=False)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001 - translated below
            # Mid-stream failures still need translating; the client may
            # already have emitted partial text by this point.
            raise self._translate_error(exc) from exc

    def health_check(self) -> ProviderHealth:
        """Check reachability and that the configured model is installed.

        Uses the models listing only. No generation is performed, so this
        stays fast even when the model is cold.
        """
        started = time.perf_counter()

        try:
            listing = self._client.list()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            translated = self._translate_error(exc)
            latency_ms = self._elapsed_ms(started)

            logger.warning("ollama_health_check_failed reason=%s", translated.code)

            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                provider=self.name,
                model=self._model,
                reachable=False,
                model_available=False,
                detail=translated.message,
                latency_ms=latency_ms,
            )

        latency_ms = self._elapsed_ms(started)
        installed = self._installed_model_names(listing)
        model_available = self._is_model_installed(installed)

        # Reachable but missing the model is a degraded state rather than
        # a hard failure: the server is fine, the deployment is not ready.
        status = HealthStatus.HEALTHY if model_available else HealthStatus.DEGRADED
        detail = (
            "Model available."
            if model_available
            else f"Configured model {self._model!r} is not installed."
        )

        logger.info(
            "ollama_health_check status=%s model_available=%s latency_ms=%.1f",
            status.value,
            model_available,
            latency_ms,
        )

        return ProviderHealth(
            status=status,
            provider=self.name,
            model=self._model,
            reachable=True,
            model_available=model_available,
            detail=detail,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            logger.debug("ollama_client_close_failed", exc_info=True)

    # ---------- Internals ----------

    def _build_request(self, options: GenerationOptions | None) -> dict[str, Any]:
        """Translate neutral options into ``Client.chat`` keyword arguments.

        Two shapes have to be respected. The Ollama client takes sampling
        settings nested under ``options`` (``temperature``, ``top_p``,
        ``num_predict``, ...) and takes ``think`` as a top-level
        parameter. Passing ``temperature`` at the top level raises a
        ``TypeError``, so the split matters.

        Caller options are merged over the deployment defaults first, so
        ``think`` is always explicit rather than depending on Ollama's own
        default. That matters here: Qwen3 enables reasoning by default,
        which took ~86 seconds per request on modest hardware versus ~3.6
        seconds with it disabled.
        """
        resolved = self.resolve_options(options)
        model_options = resolved.to_dict()

        # The neutral name for an output cap is ``max_tokens``; Ollama
        # calls it ``num_predict``.
        max_tokens = model_options.pop("max_tokens", None)
        if max_tokens is not None:
            model_options["num_predict"] = max_tokens

        request: dict[str, Any] = {}
        if model_options:
            request["options"] = model_options

        if resolved.think is not None:
            request["think"] = resolved.think

        return request

    def _translate_error(self, exc: Exception) -> Exception:
        """Map a backend exception onto a stable provider exception.

        The Ollama client raises plain ``ConnectionError`` for
        connection failures, and its own ``ResponseError`` for HTTP
        errors, so all of these shapes are handled explicitly.
        """
        # httpx timeouts are checked first: a connect timeout is also a
        # connection error, but deserves a more specific classification.
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(
                "The AI provider timed out while generating a response."
            )

        if isinstance(exc, httpx.ConnectError | ConnectionError):
            return ProviderUnavailableError(
                "The AI provider is currently unavailable."
            )

        if isinstance(exc, ResponseError):
            # The Ollama client reports a missing model as HTTP 404.
            if getattr(exc, "status_code", None) == 404:
                return ModelNotFoundError(
                    f"Model {self._model!r} is not available on the AI provider."
                )
            return ProviderResponseError("The AI provider returned an error.")

        if isinstance(exc, httpx.HTTPError):
            return ProviderUnavailableError(
                "The AI provider could not be reached."
            )

        return ProviderResponseError("The AI provider returned an unexpected error.")

    def _extract_text(self, response: Any, *, strip: bool) -> str:
        """Pull the assistant text out of a chat response.

        Attribute access is used rather than dictionary indexing: recent
        Ollama clients return Pydantic models, and relying on the legacy
        mapping interface would be fragile.

        ``strip`` is disabled while streaming. Trimming each chunk would
        silently discard the spaces and newlines that separate tokens,
        corrupting the reassembled output.
        """
        message = getattr(response, "message", None)
        if message is None:
            return ""

        content = getattr(message, "content", None)
        if content is None:
            return ""

        text = str(content)
        return text.strip() if strip else text

    def _installed_model_names(self, listing: Any) -> set[str]:
        """Collect model names from a listing response.

        Handles both the Pydantic model shape (``.model`` attribute) and
        the older mapping shape, so the code keeps working if the client
        representation changes.
        """
        models: Any = getattr(listing, "models", None)
        if models is None and isinstance(listing, Mapping):
            models = listing.get("models")

        names: set[str] = set()
        if not models:
            return names

        for entry in models:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if name is None and isinstance(entry, Mapping):
                name = entry.get("model") or entry.get("name")
            if name:
                names.add(str(name))

        return names

    def _is_model_installed(self, installed: set[str]) -> bool:
        """Check whether the configured model is present.

        Tries progressively looser matches, because tag handling differs
        between tools: ``qwen3:8b`` may be listed as ``qwen3:8b`` or
        simply ``qwen3``.
        """
        wanted = self._model
        if wanted in installed:
            return True

        # Compare without the tag: "qwen3:8b" matches an installed "qwen3".
        if ":" in wanted:
            base = wanted.split(":", 1)[0]
            if base in installed:
                return True
            if any(name.split(":", 1)[0] == base for name in installed):
                return True

        return False

    @staticmethod
    def _to_ms(nanoseconds: int | None) -> float | None:
        """Convert an Ollama nanosecond duration to milliseconds."""
        if nanoseconds is None:
            return None
        return round(nanoseconds / 1_000_000, 2)

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
"""AI provider abstraction and implementations.

Agents must depend on :class:`~app.services.ai.provider.AIProvider`, not
on a concrete backend such as
:class:`~app.services.ai.ollama_provider.OllamaProvider`.

:class:`OllamaProvider` is exposed lazily through the module-level
``__getattr__`` below. Importing it eagerly would pull the ``ollama`` SDK
into every process that merely touches the abstraction — including agent
code, which must stay runnable when no local backend is installed at all.
That is what makes a cloud provider a drop-in replacement rather than a
rewrite.
"""

from typing import TYPE_CHECKING, Any

from app.services.ai.provider import (
    AIProvider,
    AIProviderError,
    GenerationOptions,
    GenerationResult,
    HealthStatus,
    Message,
    ModelNotFoundError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
)

if TYPE_CHECKING:
    from app.services.ai.ollama_provider import OllamaProvider

_LAZY_PROVIDERS = {"OllamaProvider": "app.services.ai.ollama_provider"}


def __getattr__(name: str) -> Any:
    """Import a concrete provider only when it is actually requested."""
    module_path = _LAZY_PROVIDERS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_PROVIDERS})

__all__ = [
    "AIProvider",
    "AIProviderError",
    "GenerationOptions",
    "GenerationResult",
    "HealthStatus",
    "Message",
    "ModelNotFoundError",
    "OllamaProvider",
    "ProviderHealth",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "StructuredOutputError",
]
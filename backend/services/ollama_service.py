"""Compatibility shim for the original module location.

The Ollama integration now lives in
:mod:`app.services.ai.ollama_provider`, which reads its configuration
from the application settings instead of hardcoded values.

This shim preserves the previous import path so that existing code
continues to work:

    from services.ollama_service import generate_response

New code should depend on :class:`app.services.ai.provider.AIProvider`
instead, so that agents are not coupled to a specific backend.

.. deprecated::
    Use :func:`app.dependencies.get_ai_provider` and the
    :class:`app.services.ai.provider.AIProvider` interface.
"""

from __future__ import annotations

import warnings

from app.dependencies import get_ai_provider
from app.services.ai.provider import Message


def generate_response(prompt: str) -> str:
    """Generate a completion for ``prompt``.

    .. deprecated::
        Retained for backwards compatibility. Prefer injecting an
        ``AIProvider`` via :func:`app.dependencies.get_ai_provider`.
    """
    warnings.warn(
        "services.ollama_service.generate_response is deprecated; "
        "use app.dependencies.get_ai_provider() and the AIProvider "
        "interface instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    provider = get_ai_provider()
    result = provider.generate([Message(role="user", content=prompt)])
    return result.text


__all__ = ["generate_response"]
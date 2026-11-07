"""Package marker for the ``services`` namespace.

Note
----
This package intentionally contains no modules. The original
``services/ollama_service.py`` implementation has moved to
:mod:`app.services.ai.ollama_provider`, and this file exists so that any
stray imports of ``services`` still resolve rather than raising
``ModuleNotFoundError``.
"""
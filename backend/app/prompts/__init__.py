"""Prompt composition for agents.

Prompts live outside agent code so that wording can be revised without
touching agent logic, and so channel-specific instructions have an
obvious home.

Scope for this phase is intentionally small: a :class:`PromptTemplate`
with ``{placeholder}`` substitution, plus :class:`PromptBuilder` for
assembling system instructions, context and the user turn. Storing
prompts in files or a database can be added later without changing
either API.
"""

from app.prompts.builder import PromptBuilder, PromptTemplate
from app.prompts.library import PromptLibrary, get_prompt_library

__all__ = [
    "PromptBuilder",
    "PromptLibrary",
    "PromptTemplate",
    "get_prompt_library",
]

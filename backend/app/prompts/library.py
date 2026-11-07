"""A small library of shared prompt templates.

These are the pieces every Creator AI agent needs. Actual agent-specific
wording is deliberately absent: no agent logic is implemented in this
phase, so the library holds only shared framing and the contracts agents
will build on.

Storing templates in code rather than files keeps them versioned with the
source and reviewable in a diff. A file- or database-backed store can be
introduced later behind the same :class:`PromptLibrary` interface.
"""

from __future__ import annotations

from functools import lru_cache

from app.prompts.template import PromptTemplate

#: Keys are stable identifiers used by agents. Values are the wording.
_LIBRARY_TEMPLATES: dict[str, str] = {
    "json_output": (
        "Respond with a single valid JSON object and nothing else. "
        "Do not use Markdown code fences. Do not add commentary before or "
        "after the JSON."
    ),
    "fact_checker_grounding": (
        "Assess only the claims present in the supplied material. "
        "If the material does not support a claim, say so rather than "
        "substituting outside knowledge. State what each verdict is based on."
    ),
    "scriptwriter_structure": (
        "Write for a spoken video format: short sentences, no bullet "
        "symbols, no stage directions. Target roughly {target_seconds} "
        "seconds of narration."
    ),
    "researcher_summary": (
        "Summarise the topic in {max_sentences} sentences, using only "
        "the supplied material."
    ),
    "channel_tone": (
        "Write in a {tone} tone suitable for {channel}."
    ),
}


class PromptLibrary:
    """Read-only access to named prompt templates."""

    __slots__ = ("_templates",)

    def __init__(self, templates: dict[str, str]) -> None:
        self._templates = {
            name: PromptTemplate(name, text) for name, text in templates.items()
        }

    def __contains__(self, name: object) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)

    def names(self) -> tuple[str, ...]:
        """Available template names, sorted."""
        return tuple(sorted(self._templates))

    def get(self, name: str) -> PromptTemplate:
        """Return a template by name.

        Raises:
            KeyError: the name is not registered.
        """
        try:
            return self._templates[name]
        except KeyError:
            available = ", ".join(self.names())
            raise KeyError(
                f"Unknown prompt template {name!r}. Available: {available}"
            ) from None

    def with_template(self, name: str, text: str) -> PromptLibrary:
        """Return a new library with one template replaced or added.

        Used to layer deployment-specific wording over the defaults
        without mutating the shared library.
        """
        return PromptLibrary({**self._as_dict(), name: text})

    def _as_dict(self) -> dict[str, str]:
        return {name: template.text for name, template in self._templates.items()}


@lru_cache(maxsize=1)
def get_prompt_library() -> PromptLibrary:
    """Return the shared, cached prompt library."""
    return PromptLibrary(_LIBRARY_TEMPLATES)

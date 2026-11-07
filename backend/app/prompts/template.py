"""Prompt templates and safe placeholder substitution.

:class:`PromptTemplate` mirrors :meth:`str.format` semantics but records
which placeholders were supplied. That distinction matters when building
prompts programmatically: a template referring to a value the caller
forgot is a bug worth surfacing rather than a blank that reaches the
model and produces confidently wrong output.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

#: Standard-library format parser. Used instead of a regular expression so
#: ``{{`` and ``}}`` escapes are honoured: ``"{{literal}}"`` is not a
#: placeholder, and a naive pattern would report ``literal`` as one.
_FORMATTER = string.Formatter()


class MissingPlaceholderError(KeyError):
    """A template was rendered without a required placeholder value."""

    def __init__(self, template_name: str, missing: tuple[str, ...]) -> None:
        self.template_name = template_name
        self.missing = missing
        super().__init__(
            f"Prompt template {template_name!r} is missing values for: "
            f"{', '.join(missing)}"
        )

    def __str__(self) -> str:
        # KeyError's default str() wraps the message in quotes, which reads
        # poorly in logs.
        return self.args[0]


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A template rendered with specific values."""

    text: str
    template_name: str
    values: dict[str, str] = field(default_factory=dict)


class PromptTemplate:
    """A named prompt with named placeholders.

    Example:
        >>> template = PromptTemplate(
        ...     "brief",
        ...     "Write a {tone} intro about {topic}.",
        ... )
        >>> template.render({"tone": "friendly", "topic": "recursion"}).text
        'Write a friendly intro about recursion.'
    """

    __slots__ = ("_name", "_text", "_placeholders")

    def __init__(self, name: str, text: str) -> None:
        if not name.strip():
            raise ValueError("Prompt template name must not be empty")
        if not text.strip():
            raise ValueError("Prompt template text must not be empty")

        self._name = name
        self._text = text
        self._placeholders = _collect_placeholders(text)

    @property
    def name(self) -> str:
        return self._name

    @property
    def text(self) -> str:
        return self._text

    @property
    def placeholders(self) -> tuple[str, ...]:
        """Placeholder names, in order of first appearance."""
        return self._placeholders

    @property
    def required_placeholders(self) -> tuple[str, ...]:
        """Alias for :attr:`placeholders`, for readability at call sites."""
        return self._placeholders

    def render(self, values: dict[str, str] | None = None, **extra: str) -> RenderedPrompt:
        """Substitute ``values`` and ``extra`` into the template.

        Raises:
            MissingPlaceholderError: a referenced placeholder has no value.
        """
        supplied: dict[str, str] = {**(values or {}), **extra}

        missing = tuple(name for name in self._placeholders if name not in supplied)
        if missing:
            raise MissingPlaceholderError(self._name, missing)

        # Only pass placeholders the template actually uses, so extra
        # caller context does not confuse str.format.
        used = {name: str(supplied[name]) for name in self._placeholders}
        return RenderedPrompt(
            text=self._text.format(**used),
            template_name=self._name,
            values=used,
        )

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self._name!r}, placeholders={self._placeholders})"


def _collect_placeholders(text: str) -> tuple[str, ...]:
    """Return unique placeholder names in order of appearance.

    Only the root of a field name is used, so ``{topic}`` and
    ``{topic.title}`` both register as ``topic``.
    """
    seen: dict[str, None] = {}
    for _, field_name, _, _ in _FORMATTER.parse(text):
        if not field_name:
            continue
        root = field_name.split(".")[0].split("[")[0].strip()
        if root.isidentifier():
            seen.setdefault(root, None)
    return tuple(seen)

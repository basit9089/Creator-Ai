"""Assembles provider messages from prompt pieces.

An agent needs to send more than one string to a model: framing
instructions, channel-specific guidance, retrieved context, then the
actual request. :class:`PromptBuilder` keeps that assembly in one place
so agents describe *what* they want rather than assembling message dicts
by hand.

The builder produces :class:`~app.services.ai.provider.Message` objects,
so it stays provider-independent: nothing here knows Ollama exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.prompts.template import PromptTemplate
from app.services.ai.provider import GenerationOptions, Message

#: Heading used for injected context, kept consistent so prompts that
#: refer to "the context" behave predictably.
CONTEXT_HEADING = "Context:"


@dataclass(slots=True)
class PromptBuilder:
    """Builds an ordered message list for a single generation call.

    Parts accumulate in the order they are added, which keeps the final
    message the most recent instruction — usually what the model should
    attend to last.
    """

    system_parts: list[str] = field(default_factory=list)
    context_parts: list[str] = field(default_factory=list)
    user_parts: list[str] = field(default_factory=list)

    #: Per-call generation overrides, for example enabling reasoning.
    options: GenerationOptions | None = None

    # ---------- System instructions ----------

    def with_system(self, text: str) -> PromptBuilder:
        """Append a system instruction."""
        cleaned = text.strip()
        if cleaned:
            self.system_parts.append(cleaned)
        return self

    def with_system_template(
        self, template: PromptTemplate, values: dict[str, str] | None = None, **extra: str
    ) -> PromptBuilder:
        """Render a template and add it as a system instruction.

        Returns the builder so template and literal instructions can be
        chained together.
        """
        self.system_parts.append(template.render(values, **extra).text)
        return self

    # ---------- Context ----------

    def with_context(self, text: str) -> PromptBuilder:
        """Append background material the model should consider."""
        cleaned = text.strip()
        if cleaned:
            self.context_parts.append(cleaned)
        return self

    def with_context_items(self, items: Iterable[str]) -> PromptBuilder:
        """Append several context items."""
        for item in items:
            self.with_context(item)
        return self

    # ---------- User turn ----------

    def with_user(self, text: str) -> PromptBuilder:
        """Append to the user turn."""
        cleaned = text.strip()
        if cleaned:
            self.user_parts.append(cleaned)
        return self

    def with_user_template(
        self, template: PromptTemplate, values: dict[str, str] | None = None, **extra: str
    ) -> PromptBuilder:
        """Render a template and add it to the user turn."""
        self.user_parts.append(template.render(values, **extra).text)
        return self

    # ---------- Options ----------

    def with_options(self, options: GenerationOptions) -> PromptBuilder:
        """Set per-call generation overrides."""
        self.options = options
        return self

    def with_reasoning(self, enabled: bool) -> PromptBuilder:
        """Enable or disable reasoning for this call only.

        Deliberately an argument rather than a per-agent constant: whether
        reasoning is worth its cost depends on the task, and on hardware
        that measured ~86 seconds with it on versus ~3.6 seconds off.
        """
        base = self.options or GenerationOptions()
        self.options = base.with_thinking(enabled)
        return self

    # ---------- Output ----------

    @property
    def system_prompt(self) -> str:
        """Combined system instructions, or an empty string."""
        return "\n\n".join(self.system_parts)

    @property
    def user_prompt(self) -> str:
        """Combined user text, or an empty string."""
        return "\n\n".join(self.user_parts)

    def build(self) -> list[Message]:
        """Produce the message list to send to the provider.

        Context is appended to the user turn rather than given its own
        message: providers do not agree on how many non-system messages
        they handle well, and folding context into a single user turn is
        the most predictable across backends.
        """
        messages: list[Message] = []

        system = self.system_prompt
        if system:
            messages.append(Message(role="system", content=system))

        parts: list[str] = []

        if self.context_parts:
            context_block = "\n\n".join(self.context_parts)
            parts.append(f"{CONTEXT_HEADING}\n{context_block}")

        user = self.user_prompt
        if user:
            parts.append(user)

        combined = "\n\n".join(parts)
        if combined:
            messages.append(Message(role="user", content=combined))

        return messages

    def build_with_options(self) -> tuple[Sequence[Message], GenerationOptions | None]:
        """Return both the messages and their generation options."""
        return self.build(), self.options

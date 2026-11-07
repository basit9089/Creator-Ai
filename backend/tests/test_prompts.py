"""Tests for the prompt abstraction.

Prompts are separated from agent logic so wording can change without code
changes. These tests cover substitution, error reporting and the message
shapes the builder produces.
"""

from __future__ import annotations

import pytest

from app.prompts.builder import CONTEXT_HEADING, PromptBuilder
from app.prompts.library import PromptLibrary, get_prompt_library
from app.prompts.template import MissingPlaceholderError, PromptTemplate
from app.services.ai.provider import GenerationOptions


# ---------- PromptTemplate ----------


def test_template_substitutes_placeholders() -> None:
    template = PromptTemplate("intro", "Write a {tone} intro about {topic}.")

    rendered = template.render({"tone": "friendly", "topic": "recursion"})

    assert rendered.text == "Write a friendly intro about recursion."


def test_template_accepts_keyword_values() -> None:
    template = PromptTemplate("intro", "Write about {topic}.")

    assert template.render(topic="recursion").text == "Write about recursion."


def test_template_coerces_non_string_values() -> None:
    template = PromptTemplate("len", "Write about {seconds} seconds.")

    assert template.render(seconds=60).text == "Write about 60 seconds."


def test_template_reports_missing_values() -> None:
    """A blank reaching the model produces confidently wrong output, so a
    forgotten value must be an error rather than an empty string."""
    template = PromptTemplate("intro", "Write about {topic} in a {tone} tone.")

    with pytest.raises(MissingPlaceholderError) as excinfo:
        template.render({"tone": "friendly"})

    assert "topic" in excinfo.value.missing
    assert "tone" not in excinfo.value.missing


def test_missing_placeholder_message_names_the_template() -> None:
    template = PromptTemplate("researcher.summary", "Summarise {topic}.")

    with pytest.raises(MissingPlaceholderError, match="researcher.summary"):
        template.render()


def test_missing_placeholder_error_reads_cleanly() -> None:
    """KeyError's default str() adds quotes, which reads badly in logs."""
    error = MissingPlaceholderError("researcher.summary", ("topic",))

    assert str(error) == (
        "Prompt template 'researcher.summary' is missing values for: topic"
    )


def test_template_reports_its_placeholders_in_order() -> None:
    template = PromptTemplate("t", "{tone} then {topic} then {tone}")

    assert template.placeholders == ("tone", "topic")


def test_template_without_placeholders_renders_unchanged() -> None:
    template = PromptTemplate("static", "Always answer in one sentence.")

    assert template.render().text == "Always answer in one sentence."


def test_extra_caller_values_are_ignored() -> None:
    """Agents pass broad context; the template should not choke on it."""
    template = PromptTemplate("t", "Write about {topic}.")

    rendered = template.render({"topic": "recursion", "unrelated": "value"})

    assert rendered.text == "Write about recursion."


def test_rendered_prompt_records_the_values_used() -> None:
    """Recorded for logging and debugging without storing the prompt itself."""
    template = PromptTemplate("t", "Write about {topic}.")

    rendered = template.render({"topic": "recursion"})

    assert rendered.values == {"topic": "recursion"}
    assert rendered.template_name == "t"


def test_template_escapes_are_not_treated_as_placeholders() -> None:
    template = PromptTemplate("t", "Literal {{braces}} and {topic}")

    assert template.placeholders == ("topic",)


def test_empty_template_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        PromptTemplate("  ", "text")


def test_empty_template_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        PromptTemplate("name", "   ")


def test_template_repr_is_informative() -> None:
    template = PromptTemplate("intro", "About {topic}")

    assert "intro" in repr(template)
    assert "topic" in repr(template)


# ---------- PromptBuilder ----------


def test_builder_produces_system_then_user() -> None:
    messages = PromptBuilder().with_system("Be terse.").with_user("Research recursion.").build()

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == "Be terse."
    assert messages[1].content == "Research recursion."


def test_builder_omits_empty_sections() -> None:
    messages = PromptBuilder().with_user("Research recursion.").build()

    assert [m.role for m in messages] == ["user"]


def test_builder_skips_blank_parts() -> None:
    """Empty strings should not create empty message sections."""
    messages = PromptBuilder().with_system("   ").with_user("  ").build()

    assert messages == []


def test_builder_joins_multiple_system_instructions() -> None:
    builder = PromptBuilder().with_system("Be terse.").with_system("Use British English.")

    assert builder.system_prompt == "Be terse.\n\nUse British English."


def test_builder_folds_context_into_the_user_turn() -> None:
    """One user turn is the most predictable shape across backends."""
    builder = PromptBuilder().with_context("Source material.").with_user("Summarise it.")

    messages = builder.build()

    assert [m.role for m in messages] == ["user"]
    assert CONTEXT_HEADING in messages[0].content
    assert "Source material." in messages[0].content
    assert messages[0].content.index("Source material.") < messages[0].content.index(
        "Summarise it."
    )


def test_builder_places_the_request_last() -> None:
    """The most recent instruction is what a model should attend to last."""
    builder = PromptBuilder().with_context("Context.").with_user("The actual request.")

    content = builder.build()[0].content

    assert content.endswith("The actual request.")


def test_builder_appends_multiple_context_items() -> None:
    builder = PromptBuilder().with_context_items(["First.", "Second."]).with_user("Go.")

    content = builder.build()[0].content

    assert "First." in content
    assert "Second." in content


def test_builder_accepts_a_system_template() -> None:
    template = PromptTemplate("tone", "Write in a {tone} tone.")

    builder = PromptBuilder().with_system_template(template, tone="serious").with_user("Go.")

    assert builder.system_prompt == "Write in a serious tone."


def test_builder_accepts_a_user_template() -> None:
    template = PromptTemplate("topic", "Research {topic}.")

    builder = PromptBuilder().with_user_template(template, topic="recursion")

    assert builder.user_prompt == "Research recursion."


def test_builder_template_can_be_chained() -> None:
    """Template and literal instructions compose in one expression."""
    template = PromptTemplate("tone", "Write in a {tone} tone.")

    builder = (
        PromptBuilder()
        .with_system_template(template, tone="serious")
        .with_user("Go.")
    )

    assert builder.system_prompt == "Write in a serious tone."


def test_builder_produces_no_user_message_without_content() -> None:
    builder = PromptBuilder().with_system("Be terse.")

    assert [m.role for m in builder.build()] == ["system"]


def test_builder_messages_have_the_expected_shape() -> None:
    """Provider-neutral: no backend-specific keys leak into a message."""
    message = PromptBuilder().with_user("Go.").build()[0]

    assert message.to_dict() == {"role": "user", "content": "Go."}


# ---------- Per-call options ----------


def test_builder_carries_generation_options() -> None:
    builder = PromptBuilder().with_user("Go.").with_options(GenerationOptions(temperature=0.2))

    _, options = builder.build_with_options()

    assert options is not None
    assert options.temperature == 0.2


def test_builder_has_no_options_by_default() -> None:
    """No options means the provider's deployment defaults apply."""
    _, options = PromptBuilder().with_user("Go.").build_with_options()

    assert options is None


def test_builder_can_enable_reasoning_per_call() -> None:
    builder = PromptBuilder().with_user("Go.").with_reasoning(True)

    _, options = builder.build_with_options()

    assert options is not None
    assert options.think is True


def test_builder_can_disable_reasoning_per_call() -> None:
    builder = PromptBuilder().with_user("Go.").with_reasoning(False)

    _, options = builder.build_with_options()

    assert options is not None
    assert options.think is False


def test_builder_reasoning_preserves_other_options() -> None:
    builder = (
        PromptBuilder()
        .with_user("Go.")
        .with_options(GenerationOptions(temperature=0.3, max_tokens=64))
        .with_reasoning(True)
    )

    _, options = builder.build_with_options()

    assert options is not None
    assert options.temperature == 0.3
    assert options.max_tokens == 64
    assert options.think is True


def test_builder_chaining_returns_self() -> None:
    """Chaining is the documented style; it must actually work."""
    builder = PromptBuilder()

    assert builder.with_system("a").with_user("b").with_context("c") is builder


# ---------- Prompt library ----------


def test_library_exposes_shared_templates() -> None:
    library = get_prompt_library()

    assert len(library) > 0
    assert "json_output" in library


def test_library_get_returns_a_usable_template() -> None:
    template = get_prompt_library().get("scriptwriter_structure")

    assert template.render(target_seconds=60).text.count("60") == 1


def test_library_names_are_sorted() -> None:
    names = get_prompt_library().names()

    assert list(names) == sorted(names)


def test_unknown_template_error_lists_alternatives() -> None:
    library = get_prompt_library()

    with pytest.raises(KeyError) as excinfo:
        library.get("nonexistent")

    assert "json_output" in str(excinfo.value)


def test_library_is_shared() -> None:
    assert get_prompt_library() is get_prompt_library()


def test_library_template_can_be_layered_without_mutating_the_default() -> None:
    """Deployment wording can override the default safely."""
    custom = get_prompt_library().with_template("json_output", "Return JSON only.")

    assert custom.get("json_output").text == "Return JSON only."
    assert get_prompt_library().get("json_output").text != "Return JSON only."


def test_library_layering_keeps_other_templates() -> None:
    library = get_prompt_library()

    assert len(library.with_template("json_output", "Return JSON only.")) == len(library)


def test_library_can_be_built_from_scratch() -> None:
    library = PromptLibrary({"only": "The only template."})

    assert library.names() == ("only",)


# ---------- Prompts are not scattered in agent code ----------


def test_prompt_wording_lives_outside_the_agent_modules() -> None:
    """Agent modules must not embed large prompt strings.

    Wording belongs to the prompt layer so it can be revised without
    touching agent logic. Docstrings are excluded, since explaining the
    design in place is the point of the comments.
    """
    import ast
    import textwrap

    import app.agents.base
    import app.agents.registry
    import app.agents.runner

    modules = (app.agents.base, app.agents.registry, app.agents.runner)
    docstring_nodes = set()

    for module in modules:
        source = module.__file__
        assert source is not None
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        tree = ast.parse(text)

        # Collect docstrings by node identity, then check every other string
        # literal for prompt-like length.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node, clean=False)
                if docstring is not None:
                    docstring_nodes.add((node.body[0].lineno, node.body[0].col_offset))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if (node.lineno, node.col_offset) in docstring_nodes:
                continue
            assert len(node.value) < 200, (
                f"{module.__name__}:{node.lineno} contains a {len(node.value)}-character "
                "literal; prompt wording belongs in app.prompts"
            )

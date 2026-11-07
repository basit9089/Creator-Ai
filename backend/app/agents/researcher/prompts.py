"""Prompts for the Researcher.

Kept out of the agent module, in line with the Phase 3 rule that wording
lives in :mod:`app.prompts` and not inside agent logic. The wording here
is doing real work rather than being decorative, so each part is
explained.

The evidence block
------------------
``format_evidence_block`` is the most consequential function in the
file. It is the *only* place where retrieved text reaches the model, and
it renders each source with an explicit identifier::

    [source_01] Title — https://example.com/article
    <text>

That pairing is what makes grounding checkable. A claim citing
``source_01`` can be traced to a URL because the model was shown the two
together, and the agent later verifies the reference against the material
it actually retrieved.

Truncation is explicit. A source cut at the character cap says so, because
a model shown a silently truncated page may treat an early sentence as
the whole article.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.prompts.template import PromptTemplate

#: Identifier format used throughout. Kept as a constant because the
#: prompt, the schemas and the grounding checks must all agree on it.
SOURCE_ID_TEMPLATE = "source_{index:02d}"

#: Headings a model can rely on being present, so the instructions can
#: refer to them without the model having to infer the structure.
EVIDENCE_HEADING = "RETRIEVED SOURCES"
EVIDENCE_RULES = """\
Evidence rules you must follow without exception:

1. Use ONLY the material in the RETRIEVED SOURCES section. Do not use \
knowledge from your training, and do not rely on recollection of a topic.
2. Every factual claim must cite one or more source identifiers, exactly \
as written, for example ["source_01", "source_03"].
3. Never invent a source identifier. Only cite identifiers that appear in \
the RETRIEVED SOURCES section. If you cannot support a statement from a \
retrieved source, either omit it or list it under uncertainties.
4. If the sources disagree about something, record it as a conflict. Do \
not silently pick the account you find more plausible.
5. Distinguish what is established from what is inferred. Anything you \
cannot attribute to a retrieved source belongs under uncertainties.
6. If a source is marked as partially read, do not treat it as complete \
evidence for claims it may not have covered.
7. Do not include facts that merely sound plausible. If the sources are \
thin, say so in the summary rather than filling the gap."""


#: The system prompt. Sets the role, the evidence contract and the
#: instruction to return JSON.
RESEARCH_SYSTEM_PROMPT = PromptTemplate(
    "researcher.system",
    f"""You are a research analyst. You summarise and analyse material \
that has already been retrieved from the web for you.

You do not search the web, you cannot open links, and you have no \
information beyond what is provided below. That is a deliberate \
restriction: your value is careful reading of real sources, not recall.

{EVIDENCE_RULES}

Output rules:

- Return a single JSON object matching the required schema. No Markdown \
code fences, no commentary before or after it.
- The "topic" field must echo the requested topic.
- The "summary" is prose for a human reviewer: what the evidence supports, \
and where it falls short.
- Put dates, names and terminology in the summary and the key points \
where they matter; do not invent dates.
- Report figures in "statistics" only when a retrieved source states \
them, and cite the source.
""",
)

#: The user turn template. Describes the task and where the evidence goes.
#:
#: Built as an f-string like the system prompt, so the shared evidence
#: headings are interpolated at import time and the only placeholders left
#: for ``render`` are the four per-run values. Substituting them any other
#: way would leave ``{EVIDENCE_HEADING}`` in the text as a literal
#: placeholder, or raise when rendering.
RESEARCH_USER_PROMPT = PromptTemplate(
    "researcher.user",
    f"""\
Research the following topic using only the retrieved sources below.

Topic: {{topic}}

Requested depth: {{depth}}
Sources retrieved: {{source_count}}

{EVIDENCE_HEADING}

{{evidence}}

{EVIDENCE_RULES}

Now produce the JSON object. Every claim must cite its sources.""",
)

#: Used when retrieval produced nothing usable. Asking the model for JSON
#: in this situation invites it to fill the gap from memory, which is the
#: exact failure this pipeline exists to avoid, so the model is not called
#: at all and this prompt documents why.
NO_EVIDENCE_EXPLANATION = (
    "No sources could be retrieved for this topic. The analysis stage is "
    "skipped deliberately: a model asked to analyse nothing will answer "
    "from memory, and a confident unsourced answer is worse than an "
    "empty result."
)


def format_evidence_block(
    sources: Sequence[tuple[str, str, str]],
    *,
    max_characters: int,
) -> str:
    """Render retrieved sources for the prompt.

    Args:
        sources: ``(identifier, url, text)`` triples in the order they
            should appear. The identifier is generated by the agent from
            the retrieved material, never by the model.
        max_characters: total character budget across all sources, shared
            rather than per-source. A shared budget is what stops eight
            long pages from producing a prompt the model cannot attend to
            properly; each source also gets its own cap derived from it.

    Returns:
        A plain-text block. Returns an empty string when there is nothing
        to show, so the caller can detect the no-evidence case without a
        separate length check.
    """
    if not sources or max_characters <= 0:
        return ""

    # Space for the headers, so the body budget is what actually fits.
    per_source = max(500, max_characters // max(1, len(sources)))
    blocks: list[str] = []

    for source_id, url, text in sources:
        body = text.strip()
        if len(body) > per_source:
            body = body[:per_source].rstrip() + "\n[... source truncated ...]"
        if not body:
            continue
        blocks.append(f"[{source_id}] {url}\n{body}")

    return "\n\n".join(blocks)

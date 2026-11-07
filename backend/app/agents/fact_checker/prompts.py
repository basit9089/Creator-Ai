"""Prompts for the Fact Checker."""

from __future__ import annotations

from app.prompts.template import PromptTemplate

SYSTEM_PROMPT = PromptTemplate(
    "fact_checker.system",
    """You are a fact-checking analyst. Your job is to evaluate claims against
supplied evidence, NOT to create new facts or browse the internet.

Rules (strict):
- Use ONLY the provided research sources and their extracted content.
- Do NOT invent sources, URLs, citations, or evidence.
- Do NOT assume the Researcher's confidence is correct. Evaluate independently.
- Base every verification decision on actual text in the supplied sources.
- Distinguish support from contradiction. Preserve conflicts; do not silently pick sides.
- If evidence is missing or insufficient, say so explicitly.
- Return ONLY valid JSON matching the requested schema.
- For each claim, cite supporting and/or contradicting source IDs that exist in the supplied sources.
""",
)

USER_PROMPT = PromptTemplate(
    "fact_checker.user",
    """Verify the claims below against the supplied research evidence.

Topic: {topic}

{research_sources}

Claims to verify:
{claims}

Instructions:
- For each claim, produce:
  - claim_id
  - statement
  - status (one of: verified, partially_verified, contradicted, insufficient_evidence, unsupported)
  - confidence (evidence quality: high, medium, low, unknown)
  - supporting_source_ids (array of source IDs that support)
  - contradicting_source_ids (array of source IDs that contradict)
  - evidence (array of objects with keys source_id, url, domain, excerpt)
  - explanation (why this status was chosen, citing specific evidence)
  - original_researcher_confidence (if present)
  - issues (array of notes)

- Only reference source IDs that appear in the supplied sources.
- If no evidence supports, mark as unsupported or insufficient_evidence as appropriate.
- If sources disagree, mark conflicts and include contradicting_source_ids.
- Do not invent facts.
- Return JSON in the exact structure required.
""",
)

FACT_CHECK_SCHEMA = {
    "type": "object",
    "required": ["topic", "checked_claims"],
    "properties": {
        "topic": {"type": "string"},
        "overall_status": {
            "type": "string",
            "enum": [
                "verified",
                "partially_verified",
                "contradicted",
                "insufficient_evidence",
                "unsupported",
            ],
        },
        "overall_summary": {"type": "string"},
        "checked_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_id", "statement", "status"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "statement": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "verified",
                            "partially_verified",
                            "contradicted",
                            "insufficient_evidence",
                            "unsupported",
                        ],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low", "unknown"],
                    },
                    "supporting_source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contradicting_source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["source_id", "url", "domain"],
                            "properties": {
                                "source_id": {"type": "string"},
                                "url": {"type": "string"},
                                "domain": {"type": "string"},
                                "excerpt": {"type": "string"},
                            },
                        },
                    },
                    "explanation": {"type": "string"},
                    "original_researcher_confidence": {
                        "type": ["string", "null"],
                        "enum": ["high", "medium", "low", "unsupported", None],
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "descriptions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

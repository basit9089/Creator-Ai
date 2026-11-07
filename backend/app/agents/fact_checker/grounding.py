"""Grounding for fact check results.

Ensures all source IDs referenced by the Fact Checker actually
exist in the ResearchResult and preserves the evidence chain.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.agents.fact_checker.schemas import (
    ClaimVerification,
    ConflictDetails,
    EvidenceEntry,
    EvidenceQuality,
    FactCheckResult,
    VerificationStatus,
)
from app.agents.researcher.schemas import ResearchResult


def _clean_source_ids(
    source_ids: Iterable[str],
    allowed: set[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for sid in source_ids:
        if sid in allowed and sid not in seen:
            seen.add(sid)
            result.append(sid)
    return tuple(result)


def _clean_evidence(
    evidence: Iterable[EvidenceEntry],
    allowed: set[str],
    source_map: dict[str, object],
) -> tuple[EvidenceEntry, ...]:
    seen: set[str] = set()
    cleaned: list[EvidenceEntry] = []
    for entry in evidence:
        if entry.source_id in allowed and entry.source_id not in seen:
            seen.add(entry.source_id)
            cleaned.append(entry)
    return tuple(cleaned)


def ground_fact_check(
    analysis: FactCheckResult,
    research: ResearchResult,
) -> FactCheckResult:
    """Ground fact check analysis against the ResearchResult."""
    allowed = {s.id for s in research.sources}
    source_map = {s.id: s for s in research.sources}

    cleaned_claims: list[ClaimVerification] = []
    for claim in analysis.checked_claims:
        supporting = _clean_source_ids(claim.supporting_source_ids, allowed)
        contradicting = _clean_source_ids(claim.contradicting_source_ids, allowed)
        evidence = _clean_evidence(claim.evidence, allowed, source_map)

        status = claim.status
        if not supporting and not contradicting and status == VerificationStatus.VERIFIED:
            status = VerificationStatus.INSUFFICIENT_EVIDENCE

        confidence = claim.confidence
        if status == VerificationStatus.VERIFIED and confidence == EvidenceQuality.UNKNOWN:
            confidence = EvidenceQuality.HIGH
        elif status == VerificationStatus.CONTRADICTED and confidence == EvidenceQuality.UNKNOWN:
            confidence = EvidenceQuality.MEDIUM
        elif status in (VerificationStatus.INSUFFICIENT_EVIDENCE, VerificationStatus.UNSUPPORTED) and confidence == EvidenceQuality.UNKNOWN:
            confidence = EvidenceQuality.LOW

        cleaned_claims.append(
            claim.model_copy(
                update={
                    "supporting_source_ids": supporting,
                    "contradicting_source_ids": contradicting,
                    "evidence": evidence,
                    "status": status,
                    "confidence": confidence,
                }
            )
        )

    conflicts = tuple(
        ConflictDetails(
            topic=c.topic,
            source_ids=_clean_source_ids(c.source_ids, allowed),
            descriptions=c.descriptions,
        )
        for c in analysis.conflicts
    )

    return analysis.model_copy(
        update={
            "checked_claims": tuple(cleaned_claims),
            "conflicts": conflicts,
            "topic": research.topic or analysis.topic,
        }
    )

    conflicts = tuple(
        ConflictDetails(
            topic=c.topic,
            source_ids=_clean_source_ids(c.source_ids, allowed),
            descriptions=c.descriptions,
        )
        for c in analysis.conflicts
    )

    return analysis.model_copy(
        update={
            "checked_claims": tuple(cleaned_claims),
            "conflicts": conflicts,
            "topic": research.topic or analysis.topic,
        }
    )


def verify_references(
    research: ResearchResult,
    claims: Iterable[ClaimVerification],
) -> tuple[str, ...]:
    """Verify all referenced source IDs exist."""
    allowed = {s.id for s in research.sources}
    dangling: set[str] = set()

    for claim in claims:
        for sid in claim.supporting_source_ids:
            if sid not in allowed:
                dangling.add(sid)
        for sid in claim.contradicting_source_ids:
            if sid not in allowed:
                dangling.add(sid)
        for ev in claim.evidence:
            if ev.source_id not in allowed:
                dangling.add(ev.source_id)

    return tuple(sorted(dangling))

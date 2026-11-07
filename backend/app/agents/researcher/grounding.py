"""Reconciling the model's analysis with the material actually retrieved.

This is where source grounding is enforced. It is a separate module
because it is the part of the Researcher most worth testing exhaustively
and the part least worth reimplementing later.

What grounding does
-------------------
The model returns claims that reference source identifiers. Those
identifiers are checked against the sources that were genuinely
retrieved, and then four things happen.

1. **Unknown references are removed.** A claim citing ``source_09`` when
   only six sources exist is not trusted to have meant one of them. The
   dangling reference is dropped rather than guessed at.
2. **Confidence is recomputed from evidence.** The grade is a function of
   how many distinct publisher domains support the claim. The model's own
   confidence is retained for comparison but can only lower the computed
   grade, never raise it.
3. **Unsupported claims are surfaced, not deleted.** A claim that ends up
   with no supporting source stays in the result, graded
   ``unsupported`` and counted, because hiding it would defeat the point
   of collecting structured research.
4. **Every reference is verified.** The finished result is checked so
   that no dangling identifier survives anywhere, including in conflicts
   and statistics.

Why confidence is a heuristic
-----------------------------
``high``/``medium``/``low`` mean "backed by many/several/one independent
publisher". They are not probabilities and must not be presented as
reliability scores. What they are good for is separating a
multiply-sourced claim from a single anonymous blog post, which is the
distinction a human reviewer or the Fact Checker needs first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

from pydantic import BaseModel

from app.agents.researcher.schemas import (
    Confidence,
    ResearchAnalysis,
    ResearchClaim,
    ResearchConflict,
    ResearchKeyPoint,
    ResearchSource,
    ResearchStatistic,
    ResearchUncertainty,
)
from app.core.logging import get_logger

logger = get_logger("agents.researcher.grounding")

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _with(model: _ModelT, **changes: object) -> _ModelT:
    """Return a copy of ``model`` with ``changes`` applied.

    Pydantic's ``model_copy`` rather than ``dataclasses.replace``: these
    are Pydantic models, and ``dataclasses.replace`` raises ``TypeError``
    on anything that is not a dataclass.

    Validation is skipped by design. Every value passed here is derived
    from an already-validated field — a cleaned tuple of source
    identifiers, a computed :class:`Confidence`, an assigned identifier —
    so re-validating would buy nothing, and the copy is frozen exactly
    like its original.
    """
    return model.model_copy(update=changes)

#: Ordering used when the model's confidence may only lower the grade.
#: ``None`` means the model expressed no opinion, which changes nothing.
_CONFIDENCE_SEVERITY: dict[Confidence, int] = {
    Confidence.UNSUPPORTED: 0,
    Confidence.LOW: 1,
    Confidence.MEDIUM: 2,
    Confidence.HIGH: 3,
}

#: Distinct publisher domains needed for each grade.
DOMAINS_FOR_HIGH = 3
DOMAINS_FOR_MEDIUM = 2


def grade_claim(
    claim: ResearchClaim,
    sources: dict[str, ResearchSource],
    *,
    model_confidence: Confidence | None = None,
) -> ResearchClaim:
    """Return ``claim`` with its references cleaned and its grade recomputed.

    Args:
        claim: the claim as the model produced it.
        sources: the retrieved sources, keyed by identifier.
        model_confidence: the grade the model reported, if any. Advisory
            only: it can lower the computed grade, never raise it.

    Returns:
        A new claim. ``ResearchClaim`` is frozen, so nothing is mutated.
    """
    known = _clean_references(claim.source_ids, sources)

    computed = _confidence_for(known, sources)
    final = _apply_model_pessimism(computed, model_confidence or claim.model_confidence)

    return _with(
        claim,
        source_ids=known,
        confidence=final,
        model_confidence=model_confidence or claim.model_confidence,
    )


def confidence_for_references(
    source_ids: Iterable[str], sources: dict[str, ResearchSource]
) -> Confidence:
    """Grade a set of source references from the evidence alone."""
    return _confidence_for(tuple(source_ids), sources)


def ground_analysis(
    analysis: ResearchAnalysis,
    sources: Sequence[ResearchSource],
    *,
    research_topic: str,
) -> ResearchAnalysis:
    """Return ``analysis`` with every reference checked and every grade set.

    Claims, key points, conflicts, uncertainties and statistics are all
    processed, because a dangling identifier in any of them would make the
    result unresolvable for a reader trying to follow a citation.

    Identifiers are also made unique. A small model will happily emit two
    ``claim_01`` entries; since the result is consumed as a mapping, the
    agent assigns final identifiers itself rather than trusting the model's.
    """
    index = {source.id: source for source in sources}

    # One counter for the whole list: a fresh one per claim would hand
    # every claim the identifier "claim_01".
    claim_ids = _ClaimIds()
    claims = tuple(
        _renumber(grade_claim(claim, index), used=claim_ids)
        for claim in analysis.claims
    )

    key_points = tuple(
        _with(
            point,
            source_ids=_clean_references(point.source_ids, index),
        )
        for point in analysis.key_points
    )

    conflicts = tuple(
        _ground_conflict(conflict, index)
        for conflict in analysis.conflicts
    )

    uncertainties = tuple(
        _with(
            uncertainty,
            source_ids=_clean_references(uncertainty.source_ids, index),
        )
        for uncertainty in analysis.uncertainties
    )

    statistics = tuple(
        _with(
            statistic,
            source_ids=_clean_references(statistic.source_ids, index),
        )
        for statistic in analysis.statistics
    )

    # The model's echo of the topic is advisory: a topic that came from the
    # request and was never verified should not silently change.
    topic = research_topic or analysis.topic

    grounded = ResearchAnalysis(
        topic=topic,
        summary=analysis.summary,
        key_points=list(key_points),
        claims=list(claims),
        conflicts=list(conflicts),
        uncertainties=list(uncertainties),
        statistics=list(statistics),
    )

    dropped = _count_dropped_references(analysis, index)
    if dropped:
        logger.warning(
            "research_grounding_dropped_references dropped=%d source_count=%d",
            dropped,
            len(index),
        )

    logger.info(
        "research_grounding_complete claims=%d unsupported=%d sources=%d",
        len(claims),
        sum(1 for claim in claims if claim.confidence is Confidence.UNSUPPORTED),
        len(index),
    )

    return grounded


def confidence_counts(claims: Iterable[ResearchClaim]) -> dict[str, int]:
    """Count claims per confidence grade, with every grade present.

    Including the zero entries keeps the shape stable for a client
    building a chart from the first response onward.
    """
    counts = {grade.value: 0 for grade in Confidence}
    for claim in claims:
        counts[claim.confidence.value] += 1
    return counts


def verify_references(
    sources: Sequence[ResearchSource], *collections: Sequence[object]
) -> tuple[str, ...]:
    """Return any source identifier referenced but never retrieved.

    A final safety net. Grounding already filters references, so a
    non-empty result here means a new field was added to a schema and
    grounding was not updated to cover it. Surfacing the leftover
    identifiers makes that omission obvious instead of letting a dangling
    citation ship.
    """
    known = {source.id for source in sources}
    dangling: list[str] = []

    for collection in collections:
        for item in collection:
            for value in getattr(item, "source_ids", ()) or ():
                if value not in known:
                    dangling.append(value)
            for value in getattr(item, "contradicting_source_ids", ()) or ():
                if value not in known:
                    dangling.append(value)

    if dangling:
        logger.error(
            "research_grounding_dangling_references found=%d samples=%s",
            len(dangling),
            ",".join(sorted(set(dangling))[:5]),
        )

    return tuple(sorted(set(dangling)))


# ---------- Internals ----------


class _ClaimIds:
    """Hands out unique claim identifiers.

    A small model emitting two ``claim_01`` entries would otherwise make
    the result ambiguous for anything that looks claims up by id.
    """

    def __init__(self) -> None:
        self._count = 0

    def next(self) -> str:
        self._count += 1
        return f"claim_{self._count:02d}"


def _renumber(claim: ResearchClaim, *, used: _ClaimIds) -> ResearchClaim:
    """Give ``claim`` a fresh identifier from the agent's own sequence."""
    return _with(claim, id=used.next())


def _clean_references(
    source_ids: Iterable[str], sources: dict[str, ResearchSource]
) -> tuple[str, ...]:
    """Keep only identifiers that match a retrieved source, in order.

    Guessing at an intended reference is deliberately avoided. If a model
    cites ``source_07`` out of a range that ends at six, the honest reading
    is that it produced the number rather than retrieved it.
    """
    cleaned: list[str] = []
    for source_id in source_ids:
        if source_id in sources and source_id not in cleaned:
            cleaned.append(source_id)
    return tuple(cleaned)


def _confidence_for(
    source_ids: Sequence[str], sources: dict[str, ResearchSource]
) -> Confidence:
    """Grade from how many independent publishers support a reference set."""
    if not source_ids:
        return Confidence.UNSUPPORTED

    domains = {
        sources[source_id].domain
        for source_id in source_ids
        if source_id in sources and sources[source_id].domain
    }

    if len(domains) >= DOMAINS_FOR_HIGH:
        return Confidence.HIGH
    if len(domains) >= DOMAINS_FOR_MEDIUM:
        return Confidence.MEDIUM
    return Confidence.LOW


def _apply_model_pessimism(
    computed: Confidence, model_confidence: Confidence | None
) -> Confidence:
    """Let the model's own confidence lower the grade, never raise it.

    A model that is unsure about a claim is a signal worth honouring. A
    model that is *confident* is not, because that is exactly what a local
    model does with something it invented. So the comparison is one-way by
    construction.
    """
    if model_confidence is None:
        return computed

    if _CONFIDENCE_SEVERITY[model_confidence] < _CONFIDENCE_SEVERITY[computed]:
        return model_confidence

    return computed


def _ground_conflict(
    conflict: ResearchConflict, sources: dict[str, ResearchSource]
) -> ResearchConflict:
    """Clean both sides of a conflict and check it still is one.

    A conflict whose two sides collapse onto the same source, or whose only
    remaining support is on one side, is no longer a disagreement. It is
    kept rather than deleted, because discarding it would be exactly the
    silent resolution this module exists to prevent; the emptied
    ``contradicting_source_ids`` is the visible signal that no second
    account survived, so the Fact Checker is not sent chasing a phantom.

    Dropping malformed conflicts here rather than in the schema is
    deliberate. One nonsense entry should not fail an otherwise sound
    analysis and lose the whole run's evidence.
    """
    supporting = _clean_references(conflict.source_ids, sources)
    contradicting = _clean_references(
        conflict.contradicting_source_ids, sources
    )

    # A reference on both sides is not evidence of disagreement.
    contradicting = tuple(
        source_id for source_id in contradicting if source_id not in supporting
    )

    return _with(
        conflict,
        source_ids=supporting,
        contradicting_source_ids=contradicting,
    )


def _count_dropped_references(
    analysis: ResearchAnalysis, sources: dict[str, ResearchSource]
) -> int:
    """How many references the model produced that do not exist.

    Logged rather than raised. One hallucinated identifier is a normal
    failure mode for a local model and is handled by dropping it; the
    count is worth recording so a rising trend would be visible.
    """
    dropped = 0

    for collection in (
        analysis.key_points,
        analysis.claims,
        analysis.uncertainties,
        analysis.statistics,
    ):
        for item in collection:
            dropped += sum(
                1
                for source_id in item.source_ids
                if source_id not in sources
            )

    for conflict in analysis.conflicts:
        for group in (conflict.source_ids, conflict.contradicting_source_ids):
            dropped += sum(1 for source_id in group if source_id not in sources)

    return dropped

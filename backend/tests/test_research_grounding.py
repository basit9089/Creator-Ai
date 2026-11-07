"""Grounding: the checks that make a citation mean something.

This is the module that decides whether a claim in the result is actually
backed by a page that was retrieved. A failure here is not a crash; it is a
fabricated citation shipped to a reviewer who will act on it. So the tests
below are deliberately adversarial: they play the part of a model that
confidently cites a source it never saw.
"""

from __future__ import annotations

import pytest

from app.agents.researcher.grounding import (
    confidence_counts,
    confidence_for_references,
    grade_claim,
    ground_analysis,
    verify_references,
)
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

# ---------- Fixtures ----------


def make_sources() -> dict[str, ResearchSource]:
    """Three publishers, four sources. ``source_02`` is a second voice on
    the same site as ``source_01``, so citing both is still one publisher."""
    return {
        "source_01": ResearchSource(
            id="source_01", url="https://example.com/a", domain="example.com"
        ),
        "source_02": ResearchSource(
            id="source_02", url="https://example.com/b", domain="example.com"
        ),
        "source_03": ResearchSource(
            id="source_03", url="https://other.com/c", domain="other.com"
        ),
        "source_04": ResearchSource(
            id="source_04", url="https://third.com/d", domain="third.com"
        ),
    }


def claim(*source_ids: str, statement: str = "A claim.") -> ResearchClaim:
    return ResearchClaim(id="claim_01", statement=statement, source_ids=source_ids)


def graded(*source_ids: str) -> ResearchClaim:
    """A claim that has been through :func:`grade_claim`.

    Grading is what assigns the confidence grade, so anything that is
    going to be counted must be graded first. An ungraded claim carries
    the field's default, which is ``unsupported``, and would otherwise be
    miscounted as genuine evidence.
    """
    return grade_claim(claim(*source_ids), make_sources())


# ---------- Confidence from evidence ----------


def test_a_claim_with_no_sources_is_unsupported() -> None:
    sources = make_sources()

    assert grade_claim(claim(), sources).confidence is Confidence.UNSUPPORTED


def test_one_publisher_gives_low_confidence() -> None:
    sources = make_sources()

    assert grade_claim(claim("source_01"), sources).confidence is Confidence.LOW


def test_two_publishers_give_medium_confidence() -> None:
    sources = make_sources()

    graded = grade_claim(claim("source_01", "source_03"), sources)

    assert graded.confidence is Confidence.MEDIUM


def test_three_publishers_give_high_confidence() -> None:
    sources = make_sources()

    graded = grade_claim(claim("source_01", "source_03", "source_04"), sources)

    assert graded.confidence is Confidence.HIGH


def test_several_pages_on_one_site_still_count_as_one_publisher() -> None:
    """The failure this grading exists to catch.

    A model that cites four pages of one blog looks well-sourced. Ten
    articles on one site are one voice, so the grade stays ``low``.
    """
    sources = make_sources()

    graded = grade_claim(claim("source_01", "source_02"), sources)

    assert graded.confidence is Confidence.LOW


def test_a_source_with_no_domain_cannot_inflate_the_grade() -> None:
    sources = {
        "source_01": ResearchSource(
            id="source_01", url="https://example.com/a", domain=""
        ),
        "source_02": ResearchSource(
            id="source_02", url="https://example.com/b", domain=""
        ),
        "source_03": ResearchSource(
            id="source_03", url="https://other.com/c", domain=""
        ),
    }

    assert grade_claim(claim("source_01", "source_02", "source_03"), sources).confidence is (
        Confidence.LOW
    )


def test_confidence_can_be_computed_for_a_reference_set_directly() -> None:
    sources = make_sources()

    assert confidence_for_references(["source_03"], sources) is Confidence.LOW
    assert confidence_for_references([], sources) is Confidence.UNSUPPORTED


# ---------- The model's own opinion ----------


def test_the_model_cannot_raise_the_computed_grade() -> None:
    """A model certain about something backed by one blog post is exactly
    the failure mode being defended against."""
    sources = make_sources()

    graded = grade_claim(
        claim("source_01"), sources, model_confidence=Confidence.HIGH
    )

    assert graded.confidence is Confidence.LOW
    assert graded.model_confidence is Confidence.HIGH


def test_the_model_can_lower_the_computed_grade() -> None:
    """The reverse is worth honouring: a model hedging is a real signal."""
    sources = make_sources()

    graded = grade_claim(
        claim("source_01", "source_03", "source_04"),
        sources,
        model_confidence=Confidence.LOW,
    )

    assert graded.confidence is Confidence.LOW
    assert graded.model_confidence is Confidence.LOW


def test_a_grade_the_model_put_on_the_claim_is_overwritten() -> None:
    """A model may set ``confidence`` on the claim, and it is ignored.

    The field exists on the model output because a model will fill it in
    whether or not it is asked. Grading is the agent's job alone: reading
    the model's own copy would be the whole vulnerability. The claimed
    value is only advisory when passed explicitly as ``model_confidence``,
    where it can lower the grade and never raise it.
    """
    sources = make_sources()
    self_graded = ResearchClaim(
        id="claim_01",
        statement="A claim.",
        source_ids=("source_01", "source_03", "source_04"),
        confidence=Confidence.LOW,
    )

    graded = grade_claim(self_graded, sources)

    assert graded.confidence is Confidence.HIGH


def test_a_model_that_claims_high_confidence_about_one_blog_gets_low() -> None:
    """The scenario the whole mechanism exists for, stated end to end."""
    sources = make_sources()
    overconfident = ResearchClaim(
        id="claim_01",
        statement="YouTube was founded in 2005.",
        source_ids=("source_01",),
        confidence=Confidence.HIGH,
        model_confidence=Confidence.HIGH,
    )

    graded = grade_claim(overconfident, sources)

    assert graded.confidence is Confidence.LOW
    assert graded.model_confidence is Confidence.HIGH


# ---------- Fabricated references ----------


def test_a_reference_to_a_source_that_was_never_retrieved_is_dropped() -> None:
    sources = make_sources()

    graded = grade_claim(claim("source_01", "source_99"), sources)

    assert graded.source_ids == ("source_01",)


def test_a_claim_citing_only_a_fabricated_source_becomes_unsupported() -> None:
    """Not deleted, not quietly trusted: kept and marked for the checker."""
    sources = make_sources()

    graded = grade_claim(claim("source_42"), sources)

    assert graded.source_ids == ()
    assert graded.confidence is Confidence.UNSUPPORTED
    assert graded.statement == "A claim."


def test_repeated_references_are_collapsed() -> None:
    sources = make_sources()

    graded = grade_claim(claim("source_01", "source_01", "source_01"), sources)

    assert graded.source_ids == ("source_01",)


def test_reference_order_is_preserved() -> None:
    """The model's ordering carries no meaning, but stability between
    identical runs does."""
    sources = make_sources()

    graded = grade_claim(claim("source_04", "source_01", "source_03"), sources)

    assert graded.source_ids == ("source_04", "source_01", "source_03")


def test_grading_does_not_mutate_the_input_claim() -> None:
    """The claim is part of the frozen analysis; grading returns a new one."""
    sources = make_sources()
    original = claim("source_01", "source_99")

    grade_claim(original, sources)

    assert original.source_ids == ("source_01", "source_99")
    assert original.confidence is Confidence.UNSUPPORTED


# ---------- Whole-analysis grounding ----------


def analysis_with(**kwargs) -> ResearchAnalysis:
    return ResearchAnalysis.model_validate(kwargs)


def test_grounding_cleans_every_kind_of_reference() -> None:
    """A dangling identifier anywhere makes the result unfollowable, so
    every collection is processed, not just the claims."""
    grounded = ground_analysis(
        analysis_with(
            topic="History of YouTube",
            key_points=[{"text": "Point", "source_ids": ["source_01", "source_99"]}],
            claims=[{"id": "claim_01", "statement": "C", "source_ids": ["source_99"]}],
            conflicts=[
                {
                    "id": "conflict_01",
                    "statement": "Disagreement",
                    "source_ids": ["source_01", "source_88"],
                    "contradicting_source_ids": ["source_03", "source_77"],
                }
            ],
            uncertainties=[
                {"description": "Unclear", "source_ids": ["source_03", "nope"]}
            ],
            statistics=[
                {"id": "stat_01", "label": "Year", "value": "2005", "source_ids": ["fake"]}
            ],
        ),
        tuple(make_sources().values()),
        research_topic="History of YouTube",
    )

    assert grounded.key_points[0].source_ids == ("source_01",)
    assert grounded.claims[0].source_ids == ()
    assert grounded.conflicts[0].source_ids == ("source_01",)
    assert grounded.conflicts[0].contradicting_source_ids == ("source_03",)
    assert grounded.uncertainties[0].source_ids == ("source_03",)
    assert grounded.statistics[0].source_ids == ()


def test_claim_identifiers_are_assigned_by_the_agent() -> None:
    """A model that emits ``claim_01`` twice would make lookups ambiguous."""
    grounded = ground_analysis(
        analysis_with(
            topic="x",
            claims=[
                {"id": "claim_01", "statement": "First", "source_ids": ["source_01"]},
                {"id": "claim_01", "statement": "Second", "source_ids": ["source_01"]},
                {"id": "whatever", "statement": "Third", "source_ids": ["source_01"]},
            ],
        ),
        tuple(make_sources().values()),
        research_topic="x",
    )

    assert [c.id for c in grounded.claims] == ["claim_01", "claim_02", "claim_03"]


def test_the_requested_topic_wins_over_the_models_echo() -> None:
    """The model's restatement was never verified and must not silently
    rename the research."""
    grounded = ground_analysis(
        analysis_with(topic="something else entirely", summary="S"),
        (),
        research_topic="History of YouTube",
    )

    assert grounded.topic == "History of YouTube"


def test_grounding_an_analysis_with_no_sources_at_all() -> None:
    """Every claim survives, graded unsupported. Hiding them would defeat
    the purpose of collecting structured research."""
    grounded = ground_analysis(
        analysis_with(
            topic="x",
            claims=[{"id": "claim_01", "statement": "C", "source_ids": ["source_01"]}],
        ),
        (),
        research_topic="x",
    )

    assert len(grounded.claims) == 1
    assert grounded.claims[0].confidence is Confidence.UNSUPPORTED


# ---------- Conflicts ----------


def test_a_conflict_whose_sides_collapse_is_preserved_but_emptied() -> None:
    """One source cannot disagree with itself, but the disagreement is not
    deleted either: an empty second side is the visible signal."""
    sources = make_sources()
    grounded = ground_analysis(
        analysis_with(
            topic="x",
            conflicts=[
                {
                    "id": "conflict_01",
                    "statement": "They disagree",
                    "source_ids": ["source_01"],
                    "contradicting_source_ids": ["source_01"],
                }
            ],
        ),
        tuple(sources.values()),
        research_topic="x",
    )

    conflict = grounded.conflicts[0]
    assert conflict.source_ids == ("source_01",)
    assert conflict.contradicting_source_ids == ()


def test_a_real_conflict_keeps_both_sides() -> None:
    sources = make_sources()
    grounded = ground_analysis(
        analysis_with(
            topic="x",
            conflicts=[
                {
                    "id": "conflict_01",
                    "statement": "They disagree",
                    "source_ids": ["source_01"],
                    "contradicting_source_ids": ["source_03"],
                    "note": "Different founding dates.",
                }
            ],
        ),
        tuple(sources.values()),
        research_topic="x",
    )

    assert grounded.conflicts[0].contradicting_source_ids == ("source_03",)
    assert grounded.conflicts[0].note == "Different founding dates."


def test_conflicting_sources_are_not_silently_merged() -> None:
    """The Researcher has no basis for deciding which account is right."""
    sources = make_sources()
    grounded = ground_analysis(
        analysis_with(
            topic="x",
            conflicts=[
                {
                    "id": "conflict_01",
                    "statement": "Disagreement",
                    "source_ids": ["source_01", "source_02"],
                    "contradicting_source_ids": ["source_03"],
                }
            ],
        ),
        tuple(sources.values()),
        research_topic="x",
    )

    conflict = grounded.conflicts[0]
    assert len(conflict.source_ids) == 2
    assert len(conflict.contradicting_source_ids) == 1


# ---------- The final safety net ----------


def test_verification_passes_on_a_grounded_analysis() -> None:
    sources = tuple(make_sources().values())
    grounded = ground_analysis(
        analysis_with(
            topic="x",
            claims=[{"id": "claim_01", "statement": "C", "source_ids": ["source_01"]}],
            conflicts=[
                {
                    "id": "conflict_01",
                    "statement": "D",
                    "source_ids": ["source_01"],
                    "contradicting_source_ids": ["source_03"],
                }
            ],
        ),
        sources,
        research_topic="x",
    )

    dangling = verify_references(
        sources, grounded.claims, grounded.conflicts, grounded.statistics
    )

    assert dangling == ()


def test_verification_catches_a_dangling_reference_anywhere() -> None:
    """If a new field is added to a schema and grounding forgets it, this
    is what makes the omission obvious instead of shipping the citation."""
    sources = tuple(make_sources().values())
    orphan = ResearchKeyPoint(text="Point", source_ids=("source_99",))

    dangling = verify_references(sources, [orphan])

    assert dangling == ("source_99",)


def test_verification_checks_the_second_side_of_a_conflict() -> None:
    sources = tuple(make_sources().values())
    orphan = ResearchConflict(
        id="conflict_01",
        statement="D",
        source_ids=("source_01",),
        contradicting_source_ids=("source_99",),
    )

    assert verify_references(sources, [orphan]) == ("source_99",)


def test_verification_tolerates_a_model_with_no_opinion() -> None:
    sources = tuple(make_sources().values())

    assert verify_references(sources, [ResearchUncertainty(description="U")]) == ()


# ---------- Counts ----------


def test_counts_include_zero_grades() -> None:
    """A client charting the first response should not have to handle a
    missing key that appears in the second."""
    counts = confidence_counts([graded("source_01")])

    assert set(counts) == {"high", "medium", "low", "unsupported"}
    assert counts["low"] == 1
    assert counts["high"] == 0


def test_counting_no_claims_gives_all_zeros() -> None:
    assert confidence_counts([]) == {
        "high": 0,
        "medium": 0,
        "low": 0,
        "unsupported": 0,
    }


def test_counting_several_grades() -> None:
    """Counts describe *grounded* claims.

    Grading is the agent's step, so an ungraded claim is not
    ``unsupported`` by default in any meaningful sense: it simply has not
    been assessed. Counting raw model output here would understate how
    much support the research actually found.
    """
    counts = confidence_counts(
        [
            graded("source_01"),
            graded("source_01", "source_03"),
            graded(),
        ]
    )

    assert counts == {"high": 0, "medium": 1, "low": 1, "unsupported": 1}


# ---------- Statistics ----------


def test_a_statistic_citing_an_invented_source_is_left_unreferenced() -> None:
    sources = make_sources()
    grounded = ground_analysis(
        analysis_with(
            topic="x",
            statistics=[
                {"id": "stat_01", "label": "Users", "value": "2 billion", "source_ids": ["source_42"]}
            ],
        ),
        tuple(sources.values()),
        research_topic="x",
    )

    statistic = grounded.statistics[0]
    assert isinstance(statistic, ResearchStatistic)
    assert statistic.source_ids == ()
    assert statistic.value == "2 billion"
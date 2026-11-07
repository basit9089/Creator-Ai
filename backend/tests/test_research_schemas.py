"""The request and result contracts.

The request is validated at the API boundary by FastAPI, so a mistake in
these models surfaces as a field error rather than as a research run that
fails an hour of downloading later. These tests therefore care about the
bounds that protect a laptop: how long a topic may be, how many pages may
be fetched, and what happens when the caller sends nonsense.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.agents.researcher.schemas import (
    MAX_REQUESTED_SOURCES,
    MAX_STATEMENT_LENGTH,
    MAX_TOPIC_LENGTH,
    Confidence,
    ResearchAnalysis,
    ResearchClaim,
    ResearchConflict,
    ResearchErrorCode,
    ResearchKeyPoint,
    ResearchRequest,
    ResearchResult,
    ResearchSource,
    ResearchStatistic,
    ResearchUncertainty,
    SourceFailure,
    SourceOutcome,
)
from app.services.research.search.base import SearchDepth

# ---------- Topic ----------


def test_a_minimal_request_needs_only_a_topic() -> None:
    request = ResearchRequest(topic="History of YouTube")

    assert request.topic == "History of YouTube"
    assert request.search_depth is SearchDepth.STANDARD
    assert request.max_sources == 6
    assert request.language == "en"


def test_a_topic_is_trimmed() -> None:
    """A topic padded with newlines would otherwise be a bad search query."""
    assert ResearchRequest(topic="  History of YouTube \n").topic == "History of YouTube"


def test_an_empty_topic_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="")


def test_a_whitespace_only_topic_is_rejected() -> None:
    """``min_length=1`` accepts ``"   "``, which is not a research question."""
    with pytest.raises(ValidationError):
        ResearchRequest(topic="   ")


def test_an_overlong_topic_is_rejected() -> None:
    """Research is a topic, not a pasted document."""
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x" * (MAX_TOPIC_LENGTH + 1))


def test_a_topic_at_the_maximum_length_is_accepted() -> None:
    assert len(ResearchRequest(topic="x" * MAX_TOPIC_LENGTH).topic) == MAX_TOPIC_LENGTH


# ---------- Source budget ----------


def test_max_sources_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", max_sources=0)


def test_max_sources_is_capped_per_request() -> None:
    """A caller must not be able to ask for an unbounded crawl."""
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", max_sources=MAX_REQUESTED_SOURCES + 1)


def test_max_sources_at_the_cap_is_accepted() -> None:
    request = ResearchRequest(topic="x", max_sources=MAX_REQUESTED_SOURCES)

    assert request.max_sources == MAX_REQUESTED_SOURCES


def test_max_sources_must_be_an_integer() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", max_sources="six")


def test_an_unknown_depth_is_rejected() -> None:
    """A typo must not silently become the default depth."""
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", search_depth="exhaustive")


@pytest.mark.parametrize("depth", list(SearchDepth))
def test_every_depth_is_accepted(depth: SearchDepth) -> None:
    assert ResearchRequest(topic="x", search_depth=depth).search_depth is depth


# ---------- Depth versus requested count ----------


@pytest.mark.parametrize(
    ("depth", "max_sources", "expected"),
    [
        (SearchDepth.QUICK, 2, 2),
        (SearchDepth.QUICK, 3, 3),
        (SearchDepth.QUICK, 6, 5),
        (SearchDepth.QUICK, 12, 5),
        (SearchDepth.STANDARD, 2, 2),
        (SearchDepth.STANDARD, 6, 6),
        (SearchDepth.STANDARD, 12, 8),
        (SearchDepth.DEEP, 8, 8),
        (SearchDepth.DEEP, 12, 12),
    ],
)
def test_the_smaller_of_depth_and_count_wins(
    depth: SearchDepth, max_sources: int, expected: int
) -> None:
    """The depth is the instruction and the count is the ceiling.

    Asking for twelve sources at ``quick`` depth gets five: running a deep
    crawl because a client ignored the depth field is exactly the
    surprising behaviour the two bounds exist to prevent. The count is
    never raised to reach the depth's floor, because a caller asking for
    two sources has asked for two.
    """
    request = ResearchRequest(
        topic="x", search_depth=depth, max_sources=max_sources
    )

    assert request.effective_source_limit == expected


def test_the_depth_budgets_match_the_documented_ranges() -> None:
    assert SearchDepth.QUICK.source_budget == (3, 5)
    assert SearchDepth.STANDARD.source_budget == (5, 8)
    assert SearchDepth.DEEP.source_budget == (8, 12)


# ---------- Language ----------


@pytest.mark.parametrize("language", ["en", "EN", "en-GB", "pt-br", "de"])
def test_language_tags_are_normalised(language: str) -> None:
    request = ResearchRequest(topic="x", language=language)

    assert request.language == language.lower()


@pytest.mark.parametrize("language", ["e", "english-language-code-too-long", "en_GB", "12"])
def test_a_malformed_language_tag_is_rejected(language: str) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", language=language)


# ---------- Forward-compatible options ----------


def test_domain_lists_are_normalised() -> None:
    request = ResearchRequest(
        topic="x",
        preferred_domains=["Wikipedia.org", ".britannica.com", "wikipedia.org"],
        excluded_domains=["Spam.example"],
    )

    assert request.preferred_domains == ("britannica.com", "wikipedia.org")
    assert request.excluded_domains == ("spam.example",)


def test_blank_domain_entries_are_dropped() -> None:
    request = ResearchRequest(topic="x", preferred_domains=["  ", "", "example.com"])

    assert request.preferred_domains == ("example.com",)


def test_too_many_domains_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", preferred_domains=[f"site{i}.com" for i in range(21)])


def test_future_options_default_to_absent() -> None:
    """Accepted today and ignored, so adding support later is not a contract change."""
    request = ResearchRequest(topic="x")

    assert request.date_range is None
    assert request.region is None
    assert request.preferred_domains == ()
    assert request.excluded_domains == ()


def test_a_malformed_region_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", region="not a region!")


def test_an_overlong_date_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="x", date_range="x" * 41)


# ---------- Round-tripping ----------


def test_a_request_serialises_and_parses_unchanged() -> None:
    """It is the body of a POST, so the JSON form has to survive a round trip."""
    original = ResearchRequest(
        topic="History of YouTube",
        search_depth=SearchDepth.DEEP,
        max_sources=10,
        language="en-GB",
        preferred_domains=("blog.youtube",),
    )

    assert ResearchRequest.model_validate(original.model_dump()) == original


# ---------- The model's output ----------


def test_the_analysis_schema_has_no_field_for_sources() -> None:
    """The structural guarantee against fabrication.

    The model is never shown a URL it did not receive, and it has nowhere
    to put an invented one. Grounding would then be a matter of tidying
    rather than of catching anything.
    """
    assert "sources" not in ResearchAnalysis.model_fields
    assert "failed_sources" not in ResearchAnalysis.model_fields


def test_the_analysis_schema_defaults_to_an_empty_analysis() -> None:
    """Every field is optional, so a terse model yields a thin analysis
    rather than a validation failure the run cannot recover from."""
    analysis = ResearchAnalysis()

    assert analysis.summary == ""
    assert analysis.claims == []


def test_the_analysis_schema_ignores_unknown_fields() -> None:
    """Models add fields unbidden; that must not fail an otherwise good run."""
    analysis = ResearchAnalysis.model_validate(
        {"summary": "ok", "invented_sources": [{"url": "https://nowhere.example"}]}
    )

    assert analysis.summary == "ok"


def test_claim_statements_are_bounded() -> None:
    with pytest.raises(ValidationError):
        ResearchClaim(id="claim_01", statement="x" * (MAX_STATEMENT_LENGTH + 1))


def test_a_claim_needs_a_statement() -> None:
    with pytest.raises(ValidationError):
        ResearchClaim(id="claim_01", statement="")


def test_a_conflict_may_arrive_with_one_side_and_is_kept() -> None:
    """Malformed conflicts are tolerated here and cleaned by grounding.

    Rejecting one at the schema would fail the whole analysis and throw
    away every good claim with it, so an empty side is accepted and the
    grounding pass empties it visibly instead.
    """
    conflict = ResearchConflict(
        id="conflict_01",
        statement="They disagree",
        source_ids=("source_01",),
        contradicting_source_ids=(),
    )

    assert conflict.contradicting_source_ids == ()


# ---------- The result ----------


def source(source_id: str, domain: str = "example.com") -> ResearchSource:
    return ResearchSource(id=source_id, url=f"https://{domain}/{source_id}", domain=domain)


def test_the_result_is_immutable() -> None:
    """A result is handed to the API layer and to later stages; it must not
    be edited after grounding has been done to it."""
    result = ResearchResult(topic="x")

    with pytest.raises(ValidationError):
        result.topic = "y"


def test_a_result_can_be_looked_up_by_source_id() -> None:
    result = ResearchResult(topic="x", sources=(source("source_01"),))

    assert result.source("source_01") is not None
    assert result.source("source_99") is None


def test_a_result_lists_the_claims_citing_a_source() -> None:
    result = ResearchResult(
        topic="x",
        sources=(source("source_01"), source("source_02")),
        claims=(
            ResearchClaim(id="claim_01", statement="A", source_ids=("source_01",)),
            ResearchClaim(id="claim_02", statement="B", source_ids=("source_02",)),
            ResearchClaim(id="claim_03", statement="C", source_ids=("source_01", "source_02")),
        ),
    )

    assert [c.id for c in result.claims_for("source_01")] == ["claim_01", "claim_03"]
    assert result.claims_for("source_03") == ()


def test_the_result_counts_distinct_publishers() -> None:
    """Ten articles on one site are one voice, not ten."""
    result = ResearchResult(
        topic="x",
        sources=(
            source("source_01", "example.com"),
            source("source_02", "example.com"),
            source("source_03", "other.com"),
        ),
    )

    assert result.domain_count() == 2
    assert result.distinct_domains == 0  # not populated by the constructor alone


def test_source_failures_carry_a_code_and_a_safe_reason() -> None:
    failure = SourceFailure(url="https://example.com/a", reason="Blocked by the site.")

    assert failure.code == "fetch_error"
    assert failure.outcome is SourceOutcome.FAILED


@pytest.mark.parametrize(
    "outcome", [SourceOutcome.USED, SourceOutcome.SKIPPED, SourceOutcome.FAILED, SourceOutcome.EMPTY]
)
def test_every_source_outcome_exists(outcome: SourceOutcome) -> None:
    assert outcome.value


@pytest.mark.parametrize(
    "grade", [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW, Confidence.UNSUPPORTED]
)
def test_every_confidence_grade_exists(grade: Confidence) -> None:
    assert grade.value in {"high", "medium", "low", "unsupported"}


def test_error_codes_are_stable_strings() -> None:
    """A client may branch on these, so they are values rather than names."""
    assert ResearchErrorCode.NO_SOURCES == "research_no_sources"
    assert ResearchErrorCode.SEARCH_UNAVAILABLE == "research_search_unavailable"
    assert ResearchErrorCode.AI_UNAVAILABLE == "research_ai_unavailable"
    assert ResearchErrorCode.INVALID_REQUEST == "research_invalid_request"


def test_a_result_serialises_to_json() -> None:
    """It crosses the API boundary, and later stages read it back."""
    result = ResearchResult(
        topic="History of YouTube",
        summary="It began in 2005.",
        generated_at=datetime.now().astimezone(),
        sources=(source("source_01"),),
        key_points=(ResearchKeyPoint(text="Founded in 2005", source_ids=("source_01",)),),
        uncertainties=(ResearchUncertainty(description="Original funding unclear"),),
        statistics=(ResearchStatistic(id="stat_01", label="Founding year", value="2005"),),
    )

    restored = ResearchResult.model_validate_json(result.model_dump_json())

    assert restored == result


# ---------- Immutability of the leaf models ----------


@pytest.mark.parametrize(
    "model",
    [ResearchSource, SourceFailure, ResearchClaim, ResearchConflict, ResearchUncertainty,
     ResearchStatistic, ResearchKeyPoint],
)
def test_leaf_models_are_frozen(model: type) -> None:
    """Only the agent constructs these. A caller mutating one after grounding
    could reintroduce exactly the unverified reference grounding removed."""
    assert model.model_config.get("frozen") is True


def test_optional_model_output_models_are_not_frozen() -> None:
    """The analysis models are parsed, not authored, and Pydantic builds them."""
    assert ResearchAnalysis.model_config.get("frozen") is not True
"""Typed contracts for research: the request, the evidence and the result.

Three shapes, deliberately separate
-----------------------------------
:class:`ResearchRequest`
    What the caller asked for. Validated at the API boundary.
:class:`ResearchAnalysis`
    What the *model* returns. It has no sources: the model is not allowed
    to invent them, and it never sees a URL it did not have supplied.
:class:`ResearchResult`
    What the agent returns, after the model's analysis has been grounded
    against the material actually retrieved.

Keeping the model's output separate from the final result is what makes
grounding enforceable. Because :class:`ResearchAnalysis` has no field for
sources, there is nothing for the model to fill with a plausible-looking
fabrication: it can only reference source identifiers, which the agent
checks against the real ones and re-grades.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.research.search.base import SearchDepth

#: Long enough for a specific subject, short enough to reject a pasted
#: document. Research is a topic, not an essay.
MAX_TOPIC_LENGTH = 500

#: Ceiling on sources per request, independent of the deployment-wide
#: ``MAX_RESEARCH_SOURCES``. Both apply; the lower one wins.
MAX_REQUESTED_SOURCES = 12

#: Upper bound on text copied from one page into a research note.
MAX_STATEMENT_LENGTH = 1200


class ResearchErrorCode(StrEnum):
    """Stable failure codes a client may branch on."""

    INVALID_REQUEST = "research_invalid_request"
    NO_SOURCES = "research_no_sources"
    INSUFFICIENT_SOURCES = "research_insufficient_sources"
    SEARCH_UNAVAILABLE = "research_search_unavailable"
    ALL_SOURCES_FAILED = "research_all_sources_failed"
    AI_UNAVAILABLE = "research_ai_unavailable"
    AI_OUTPUT_INVALID = "research_ai_output_invalid"
    INTERNAL = "research_internal"


class Confidence(StrEnum):
    """How well the retrieved evidence supports a claim.

    This is an *evidence* grade, not the model's own certainty. It is
    computed from how many independent sources were found, and the model's
    self-assessment can only lower it, never raise it. A model that says
    "I am very confident" about something backed by one blog post is
    exactly the failure this grading exists to catch.

    The mapping is a deliberate heuristic, not a measurement:

    ``high``
        Three or more distinct publisher domains retrieved the claim.
    ``medium``
        Two distinct domains.
    ``low``
        A single domain. Usable, but corroboration is missing.
    ``unsupported``
        No retrieved source backs the claim at all.

    Distinct *domains* rather than distinct sources: ten articles on one
    site are one voice. Documented plainly so nobody later mistakes it for
    a calculated reliability score.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSUPPORTED = "unsupported"


class SourceOutcome(StrEnum):
    """Why a candidate source did or did not make it into the evidence."""

    #: Retrieved, extracted and usable.
    USED = "used"

    #: Rejected before fetching: unsafe URL, duplicate, unsupported type.
    SKIPPED = "skipped"

    #: A fetch or extraction was attempted and failed.
    FAILED = "failed"

    #: Fetched, but the page held too little text to be worth using.
    EMPTY = "empty"


# ---------- Request ----------


class ResearchRequest(BaseModel):
    """A research request.

    Used directly as the body of ``POST /api/research``, so every
    constraint here is enforced by FastAPI and reported as a field error
    rather than becoming a runtime surprise.

    Validation intent
    -----------------
    A topic must be present and bounded, and ``max_sources`` must be
    capped. Without those bounds the endpoint is a way to ask the server
    to spend minutes and megabytes fetching arbitrary numbers of pages on
    a laptop that is already running a local language model.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    topic: str = Field(
        min_length=1,
        max_length=MAX_TOPIC_LENGTH,
        description="The subject to research.",
        examples=["The history of artificial intelligence"],
    )

    search_depth: SearchDepth = Field(
        default=SearchDepth.STANDARD,
        description=(
            "How much research to do: 'quick', 'standard' or 'deep'. "
            "Maps to a source budget, which configuration still caps."
        ),
    )

    max_sources: int = Field(
        default=6,
        ge=1,
        le=MAX_REQUESTED_SOURCES,
        description=(
            "Maximum sources to retrieve. Further capped by the "
            "deployment-wide maximum."
        ),
        examples=[8],
    )

    language: str = Field(
        default="en",
        min_length=2,
        max_length=8,
        pattern=r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{2,8})*$",
        description="Language for the summary and the search results.",
        examples=["en"],
    )

    # Accepted now so callers can start sending them, and so adding real
    # support later is a change of behaviour rather than a change of
    # contract. Each is recorded in the result's warnings rather than
    # silently dropped.
    date_range: str | None = Field(
        default=None,
        max_length=40,
        description="Optional publication-date restriction, such as 'past_year'.",
    )

    region: str | None = Field(
        default=None,
        max_length=16,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Optional region code used to bias search results.",
    )

    preferred_domains: tuple[str, ...] = Field(
        default=(),
        max_length=20,
        description="Optional domains to favour when ranking sources.",
    )

    excluded_domains: tuple[str, ...] = Field(
        default=(),
        max_length=20,
        description="Optional domains to leave out entirely.",
    )

    @field_validator("topic")
    @classmethod
    def _reject_blank_topic(cls, value: str) -> str:
        """Reject a topic that is only whitespace.

        ``min_length=1`` accepts ``"   "``, which would then be used as a
        search query. Silently searching for whitespace is worse than
        refusing it.
        """
        if not value.strip():
            raise ValueError("Topic must not be blank.")
        return value.strip()

    @field_validator("language")
    @classmethod
    def _normalise_language(cls, value: str) -> str:
        """Lowercase the language tag so lookups are consistent."""
        return value.strip().lower()

    @field_validator("preferred_domains", "excluded_domains")
    @classmethod
    def _normalise_domains(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Lowercase, strip and de-duplicate domain entries."""
        normalised = {item.strip().lower().lstrip(".") for item in value if item.strip()}
        return tuple(sorted(normalised))

    @property
    def effective_source_limit(self) -> int:
        """Sources this request may use before the deployment cap.

        The lower of the requested count and the depth's upper bound. A
        caller asking for ten sources at ``quick`` depth gets the smaller
        figure: the depth is the instruction, the count is the ceiling.
        """
        _, depth_maximum = self.search_depth.source_budget
        return min(self.max_sources, depth_maximum)


# ---------- Evidence ----------


class ResearchSource(BaseModel):
    """One retrieved source, as cited by claims.

    ``id`` is what a claim references, so it is generated by the agent
    from the retrieved material and is never chosen by the model.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable identifier, e.g. 'source_01'.")

    title: str = Field(default="", max_length=MAX_STATEMENT_LENGTH)

    url: str = Field(description="URL that was actually retrieved.")

    domain: str = Field(default="", description="Publisher domain.")

    snippet: str = Field(
        default="",
        max_length=600,
        description="Search-engine snippet for this source.",
    )

    retrieved_at: datetime | None = Field(
        default=None, description="When the page was fetched."
    )

    #: Characters of text actually used as evidence. Recorded so a claim
    #: can be weighed against how much material actually backed it.
    text_length: int = Field(default=0, ge=0)

    truncated: bool = Field(
        default=False, description="Whether only part of the page was read."
    )


class SourceFailure(BaseModel):
    """A source that was tried and could not be used.

    Kept, rather than discarded, because the shape of the failure is part
    of the result: "this page was blocked" tells a reader something about
    the evidence that "this source was missing" does not.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    domain: str = ""
    reason: str = Field(description="Human-readable, client-safe reason.")
    code: str = Field(default="fetch_error", description="Stable machine-readable code.")
    outcome: SourceOutcome = SourceOutcome.FAILED


class ResearchClaim(BaseModel):
    """A factual statement tied to the sources that support it.

    ``source_ids`` is the whole point. A claim with no sources is not
    deleted: it is kept, graded ``unsupported`` and surfaced for the Fact
    Checker, because "the model said this and nothing backs it" is
    precisely the thing that stage exists to catch.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable identifier, e.g. 'claim_01'.")

    statement: str = Field(min_length=1, max_length=MAX_STATEMENT_LENGTH)

    source_ids: tuple[str, ...] = Field(
        default=(),
        description="Identifiers of the sources supporting this claim.",
    )

    confidence: Confidence = Field(
        default=Confidence.UNSUPPORTED,
        description=(
            "Evidence grade computed by the agent from the supporting "
            "sources, not self-reported by the model."
        ),
    )

    #: The grade the model asked for. Retained so a disagreement between
    #: the model's judgement and the evidence can be inspected.
    model_confidence: Confidence | None = Field(
        default=None, description="Confidence as reported by the model."
    )


class ResearchConflict(BaseModel):
    """Two sources that disagree, preserved rather than resolved.

    Silently choosing one is the failure mode this prevents. The
    Researcher has no basis for deciding which source is right, so it
    reports both and leaves resolution to the Fact Checker.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable identifier, e.g. 'conflict_01'.")

    statement: str = Field(
        min_length=1,
        max_length=MAX_STATEMENT_LENGTH,
        description="The point on which the sources disagree.",
    )

    source_ids: tuple[str, ...] = Field(
        description="Sources supporting the first account."
    )

    contradicting_source_ids: tuple[str, ...] = Field(
        description="Sources supporting the conflicting account."
    )

    note: str = Field(
        default="",
        max_length=MAX_STATEMENT_LENGTH,
        description="How the accounts differ, in the model's words.",
    )


class ResearchUncertainty(BaseModel):
    """Something the material does not settle."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(min_length=1, max_length=MAX_STATEMENT_LENGTH)

    source_ids: tuple[str, ...] = Field(
        default=(), description="Sources that raised the uncertainty."
    )


class ResearchStatistic(BaseModel):
    """A figure worth citing, kept with the source it came from."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable identifier, e.g. 'stat_01'.")

    label: str = Field(min_length=1, max_length=300, description="What is measured.")

    value: str = Field(min_length=1, max_length=120, description="The figure.")

    source_ids: tuple[str, ...] = Field(default=())
    unit: str = Field(default="", max_length=40)


class ResearchKeyPoint(BaseModel):
    """A takeaway from the research, with its support."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=MAX_STATEMENT_LENGTH)
    source_ids: tuple[str, ...] = Field(default=())


# ---------- Model output ----------


class ResearchAnalysis(BaseModel):
    """What the model is asked to return.

    This is the schema handed to
    :meth:`~app.services.ai.provider.AIProvider.generate_structured`, so
    every field must be something a language model can plausibly produce
    from the evidence it was given.

    Two fields are conspicuously absent. There is no ``sources`` field,
    because the model must not choose what was retrieved, and no
    ``confidence`` the agent will trust, because the grade is recomputed
    from the real evidence. ``model_confidence`` is kept as an advisory
    signal and can only ever lower the computed grade.
    """

    model_config = ConfigDict(extra="ignore")

    topic: str = Field(
        default="",
        max_length=MAX_TOPIC_LENGTH,
        description="The topic as the model understood it.",
    )

    summary: str = Field(
        default="",
        max_length=8000,
        description="A prose summary of what the evidence establishes.",
    )

    key_points: list[ResearchKeyPoint] = Field(
        default_factory=list,
        max_length=20,
        description="The main takeaways, each citing its sources.",
    )

    claims: list[ResearchClaim] = Field(
        default_factory=list,
        max_length=40,
        description=(
            "Factual statements with the source identifiers that support "
            "them. Claims without support must still be listed; they are "
            "graded unsupported rather than dropped."
        ),
    )

    conflicts: list[ResearchConflict] = Field(
        default_factory=list,
        max_length=15,
        description="Points where the sources disagree.",
    )

    uncertainties: list[ResearchUncertainty] = Field(
        default_factory=list,
        max_length=20,
        description="Questions the material leaves open.",
    )

    statistics: list[ResearchStatistic] = Field(
        default_factory=list,
        max_length=25,
        description="Figures worth citing.",
    )


# ---------- Result ----------


class ResearchResult(BaseModel):
    """Grounded research, ready for human review and later fact checking.

    Produced by the agent after the model's analysis has been reconciled
    with the material that was actually retrieved. Every ``source_ids``
    anywhere in this object resolves to an entry in :attr:`sources`.
    """

    model_config = ConfigDict(frozen=True)

    topic: str
    summary: str = ""

    key_points: tuple[ResearchKeyPoint, ...] = ()
    claims: tuple[ResearchClaim, ...] = ()
    sources: tuple[ResearchSource, ...] = ()
    conflicts: tuple[ResearchConflict, ...] = ()
    uncertainties: tuple[ResearchUncertainty, ...] = ()
    statistics: tuple[ResearchStatistic, ...] = ()

    # ---------- Partial success ----------

    sources_requested: int = Field(
        default=0, ge=0, description="Sources the request aimed for."
    )
    sources_found: int = Field(
        default=0, ge=0, description="Sources retrieved and usable."
    )
    sources_failed: int = Field(
        default=0, ge=0, description="Sources attempted but unusable."
    )
    failed_sources: tuple[SourceFailure, ...] = ()

    # ---------- Provenance ----------

    search_query: str = Field(default="", description="Query actually issued.")
    search_provider: str = Field(default="", description="Search backend used.")
    search_depth: SearchDepth = SearchDepth.STANDARD
    language: str = "en"

    #: Count of distinct publisher domains among the sources. The single
    #: most useful number for judging how corroborated the research is.
    distinct_domains: int = Field(default=0, ge=0)

    #: Claim counts per confidence grade, so a caller can see the shape of
    #: the evidence without scanning every claim.
    confidence_counts: dict[str, int] = Field(default_factory=dict)

    unsupported_claims: int = Field(
        default=0,
        ge=0,
        description="Claims with no supporting source. For the Fact Checker.",
    )

    warnings: tuple[str, ...] = Field(
        default=(),
        description="Client-safe notes, such as ignored request options.",
    )

    #: Whether the evidence was sufficient to be worth reviewing. ``False``
    #: with status COMPLETED means the run worked but the material was
    #: too thin, which is different from the run failing.
    sufficient: bool = True

    generated_at: datetime | None = None

    def source(self, source_id: str) -> ResearchSource | None:
        """Look up a source by identifier."""
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def claims_for(self, source_id: str) -> tuple[ResearchClaim, ...]:
        """Every claim citing ``source_id``."""
        return tuple(claim for claim in self.claims if source_id in claim.source_ids)

    def domain_count(self) -> int:
        """Distinct publisher domains among the sources."""
        return len({source.domain for source in self.sources if source.domain})

"""Typed contracts for fact checking: request, evaluation, and result.

Separates Researcher output from verification decisions so the
model's opinion cannot be conflated with evidence-based verification.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.agents.researcher.schemas import Confidence, ResearchResult


class FactCheckErrorCode(StrEnum):
    """Stable failure codes for the Fact Checker."""

    INVALID_REQUEST = "fact_check_invalid_request"
    NO_CLAIMS_TO_VERIFY = "fact_check_no_claims"
    MISSING_RESEARCH_DATA = "fact_check_missing_research_data"
    AI_UNAVAILABLE = "fact_check_ai_unavailable"
    AI_OUTPUT_INVALID = "fact_check_ai_output_invalid"
    INTERNAL = "fact_check_internal"


class VerificationStatus(StrEnum):
    """Evidence-based verification status."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"


class EvidenceQuality(StrEnum):
    """Quality of evidence supporting a claim."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class FactCheckRequest(BaseModel):
    """Request to verify Researcher results."""

    model_config = ConfigDict(frozen=True)

    research_result: ResearchResult = Field(
        description="Researcher output containing claims and sources to verify."
    )
    claims: tuple[str, ...] = Field(
        default=(),
        description="Optional subset of claim IDs to verify; empty means all claims.",
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=8,
        pattern=r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{2,8})*$",
    )

    @property
    def claim_id_set(self) -> set[str]:
        return set(self.claims)


class EvidenceEntry(BaseModel):
    """Evidence supporting or contradicting a claim."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Source identifier from ResearchResult.")
    url: str = Field(description="URL of the source.")
    domain: str = Field(description="Domain of the source.")
    excerpt: str = Field(default="", description="Relevant excerpt from the source.")


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    statement: str = Field(min_length=1)

    status: VerificationStatus
    confidence: EvidenceQuality = Field(default=EvidenceQuality.UNKNOWN)

    supporting_source_ids: tuple[str, ...] = ()
    contradicting_source_ids: tuple[str, ...] = ()

    evidence: tuple[EvidenceEntry, ...] = ()
    explanation: str = Field(default="")

    original_researcher_confidence: Confidence | None = None
    issues: tuple[str, ...] = ()


class ConflictDetails(BaseModel):
    """Details of a detected contradiction."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(default="")
    source_ids: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()


class FactCheckAnalysis(BaseModel):
    """Analysis returned by the AI model."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(default="")
    overall_status: VerificationStatus | None = None
    overall_summary: str = Field(default="")

    checked_claims: tuple[ClaimVerification, ...] = ()
    conflicts: tuple[ConflictDetails, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class FactCheckResult(BaseModel):
    """Complete fact check result."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(default="")
    overall_status: VerificationStatus = VerificationStatus.INSUFFICIENT_EVIDENCE
    overall_summary: str = Field(default="")

    checked_claims: tuple[ClaimVerification, ...] = ()
    conflicts: tuple[ConflictDetails, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    statistics: tuple[dict, ...] = ()

    sources_considered: int = 0
    claims_checked: int = 0
    verified_claims: int = 0
    partially_verified_claims: int = 0
    contradicted_claims: int = 0
    insufficient_claims: int = 0
    unsupported_claims_count: int = 0

    generated_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())

    @property
    def has_contradictions(self) -> bool:
        return self.contradicted_claims > 0
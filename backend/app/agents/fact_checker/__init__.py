"""Fact Checker Agent package.

Consumes Researcher output, independently evaluates claims against
supplied source evidence, detects contradictions, and produces a
structured FactCheckResult.
"""

from app.agents.fact_checker.agent import REQUEST_CONTEXT_KEY, FactCheckerAgent
from app.agents.fact_checker.schemas import (
    ClaimVerification,
    EvidenceEntry,
    EvidenceQuality,
    FactCheckRequest,
    FactCheckResult,
    VerificationStatus,
)

__all__ = [
    "ClaimVerification",
    "EvidenceEntry",
    "EvidenceQuality",
    "FactCheckRequest",
    "FactCheckResult",
    "FactCheckerAgent",
    "REQUEST_CONTEXT_KEY",
    "VerificationStatus",
]
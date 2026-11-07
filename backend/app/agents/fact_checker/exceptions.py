"""Exceptions for the Fact Checker Agent."""

from __future__ import annotations

from app.agents.base import AgentError
from app.agents.fact_checker.schemas import FactCheckErrorCode


class FactCheckError(AgentError):
    """Base exception for Fact Checker failures."""

    code = FactCheckErrorCode.INTERNAL.value
    retryable = False


class FactCheckAIError(FactCheckError):
    """AI provider failed during verification."""

    code = FactCheckErrorCode.AI_UNAVAILABLE.value
    retryable = True


class FactCheckGroundingError(FactCheckError):
    """Model output could not be grounded to supplied evidence."""

    code = FactCheckErrorCode.AI_OUTPUT_INVALID.value
    retryable = False


class FactCheckInputError(FactCheckError):
    """Invalid request to Fact Checker."""

    code = FactCheckErrorCode.INVALID_REQUEST.value
    retryable = False


class FactCheckNoClaimsError(FactCheckError):
    """No claims available to verify."""

    code = FactCheckErrorCode.NO_CLAIMS_TO_VERIFY.value
    retryable = False
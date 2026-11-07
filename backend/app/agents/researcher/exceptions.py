"""Errors raised by the Researcher.

Two kinds, and the split matters.

:class:`ResearchError`
    An expected, reportable failure: the request was unusable, no source
    could be found, the AI provider was unreachable. Each carries a stable
    ``code`` so a client can branch on it, a client-safe ``message``, and
    a ``status_code`` for the HTTP layer.

:class:`ResearchGroundingError`
    The model returned output that could not be reconciled with the
    retrieved evidence. Kept separate because it is a distinct failure
    with a distinct cause: the transport worked, the retrieval worked, and
    the analysis did not land.

Detailed causes stay on ``__cause__`` for server-side logs and are never
rendered into a response.

Why they derive from :class:`~app.agents.base.AgentError`
--------------------------------------------------------
The Phase 3 runner recognises failures by that base class: anything else
is treated as an unexpected crash, which loses the ``code``, the
``retryable`` flag and the client-safe ``message`` and replaces them with
"ResearchAIError raised during agent execution". That is exactly the
information the API layer needs to choose a status code. Deriving from it
is what makes the Researcher's failures reach the client intact.
"""

from __future__ import annotations

from app.agents.base import AgentError
from app.agents.researcher.schemas import ResearchErrorCode


class ResearchError(AgentError):
    """An expected research failure with a safe message.

    ``retryable`` mirrors the Phase 3 agent contract: infrastructure
    problems are worth another attempt, bad input usually is not.
    ``status_code`` is a hint for callers that have the exception to hand;
    the API layer maps the ``code`` instead, so the status and the reported
    code always describe the same decision.
    """

    code: ResearchErrorCode = ResearchErrorCode.INTERNAL
    status_code: int = 500
    retryable: bool = False

    def __init__(self, message: str, *, code: ResearchErrorCode | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ResearchInputError(ResearchError):
    """The request itself is unusable."""

    code = ResearchErrorCode.INVALID_REQUEST
    status_code = 422
    retryable = False


class NoSourcesFoundError(ResearchError):
    """Search returned nothing usable.

    Not a crash: the pipeline worked and there was simply no material.
    Worth retrying, since a search backend can return an empty page
    transiently.
    """

    code = ResearchErrorCode.NO_SOURCES
    status_code = 503
    retryable = True


class InsufficientSourcesError(ResearchError):
    """Too few sources survived retrieval to support the research.

    Distinct from :class:`NoSourcesFoundError` so a caller can tell "the
    search found nothing" from "everything it found was unreachable".
    """

    code = ResearchErrorCode.INSUFFICIENT_SOURCES
    status_code = 503
    retryable = True


class SearchUnavailableError(ResearchError):
    """No configured search backend could answer."""

    code = ResearchErrorCode.SEARCH_UNAVAILABLE
    status_code = 503
    retryable = True


class ResearchAIError(ResearchError):
    """The AI provider failed while analysing the evidence."""

    code = ResearchErrorCode.AI_UNAVAILABLE
    status_code = 503
    retryable = True


class ResearchGroundingError(ResearchError):
    """The model's output could not be reconciled with the evidence.

    Raised when the analysis is unusable rather than merely thin. The
    agent prefers to return an incomplete result over failing here, so
    this is reserved for output that cannot be repaired by grounding.
    """

    code = ResearchErrorCode.AI_OUTPUT_INVALID
    status_code = 502
    retryable = True

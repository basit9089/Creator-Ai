"""The Researcher agent.

The first real Creator AI agent, and the reference implementation of the
Phase 3 agent contract.

Structure
---------
``schemas``
    :class:`ResearchRequest`, :class:`ResearchAnalysis` (what the model
    returns) and :class:`ResearchResult` (what the agent returns).
``exceptions``
    Expected failures with stable codes and client-safe messages.
``grounding``
    Reconciles the model's claims with the material actually retrieved.
``prompts``
    The wording, kept out of :mod:`agent` on purpose.
``agent``
    :class:`ResearcherAgent`, which orchestrates the stages.

The separation is load-bearing. Grounding and prompting are the two parts
that must be right for the output to be trustworthy, so they are isolated
and tested on their own rather than being entangled with the fetch loop.
"""

from app.agents.researcher.agent import REQUEST_CONTEXT_KEY, ResearcherAgent
from app.agents.researcher.exceptions import (
    InsufficientSourcesError,
    NoSourcesFoundError,
    ResearchAIError,
    ResearchError,
    ResearchGroundingError,
    ResearchInputError,
    SearchUnavailableError,
)
from app.agents.researcher.grounding import confidence_counts, ground_analysis
from app.agents.researcher.schemas import (
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

__all__ = [
    "REQUEST_CONTEXT_KEY",
    "Confidence",
    "InsufficientSourcesError",
    "NoSourcesFoundError",
    "ResearchAIError",
    "ResearchAnalysis",
    "ResearchClaim",
    "ResearchConflict",
    "ResearchError",
    "ResearchErrorCode",
    "ResearchGroundingError",
    "ResearchInputError",
    "ResearchKeyPoint",
    "ResearchRequest",
    "ResearchResult",
    "ResearchSource",
    "ResearchStatistic",
    "ResearchUncertainty",
    "ResearcherAgent",
    "SearchUnavailableError",
    "SourceFailure",
    "SourceOutcome",
    "confidence_counts",
    "ground_analysis",
]

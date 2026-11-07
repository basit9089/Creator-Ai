"""The contract for finding candidate sources.

The Researcher does not search the web itself and neither does the model.
It asks a :class:`SearchProvider` for candidate documents; that
abstraction is what makes the search engine replaceable. Swapping
DuckDuckGo for Brave, Tavily or a self-hosted SearXNG later is a new
class, not a change to the agent.

The interface is deliberately narrow: one method, returning structured
results. Search engines disagree wildly about ranking, pagination and
result shape, and normalising all of that into one call is what keeps the
agent from growing engine-specific branches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class SearchError(Exception):
    """Base class for search failures.

    Carries a machine-readable ``code`` and a client-safe ``message``.
    The underlying transport exception stays on ``__cause__`` for logs.
    """

    code = "search_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class SearchUnavailableError(SearchError):
    """The search backend could not be reached at all."""

    code = "search_unavailable"


class SearchTimeoutError(SearchError):
    """The search backend did not answer within the timeout."""

    code = "search_timeout"


class SearchRateLimitedError(SearchError):
    """The search backend rejected the request, usually for rate limiting.

    Kept separate because it is transient and worth retrying after a
    pause, unlike an unreachable host.
    """

    code = "search_rate_limited"


class SearchDepth(StrEnum):
    """How much research the caller is asking for.

    A depth maps to a source budget. Keeping it an explicit enum rather
    than a free-form number means "as much as possible" cannot be
    requested: the budgets are bounded by configuration.
    """

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

    @property
    def source_budget(self) -> tuple[int, int]:
        """Inclusive range of sources this depth normally aims for.

        The upper bound is a preference, not a permission. Configuration
        still caps the request, so a generous depth cannot outrun
        ``MAX_RESEARCH_SOURCES``.
        """
        match self:
            case SearchDepth.QUICK:
                return (3, 5)
            case SearchDepth.STANDARD:
                return (5, 8)
            case SearchDepth.DEEP:
                return (8, 12)
        return (5, 8)  # pragma: no cover - unreachable, match is total


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """One search to run.

    ``terms`` is deliberately separate from the agent's topic: a future
    version can expand one topic into several angles (definitions,
    history, criticism) without the agent knowing anything about how a
    given backend wants to receive them.
    """

    terms: str
    depth: SearchDepth = SearchDepth.STANDARD

    #: Preferred content language as an ISO 639-1 code. Backends that
    #: cannot honour it ignore it rather than failing: fewer results in
    #: the right language still beats no results at all.
    language: str = "en"

    #: Optional domain restrictions, honoured when the backend supports
    #: them and ignored otherwise. Recorded on the result either way, so
    #: an ignored restriction is visible rather than silent.
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One candidate source returned by a search backend.

    ``rank`` is the backend's ordering, 1-based, preserved so the agent
    can prefer better-ranked candidates when it has to drop some. It is
    not a quality score and is not treated as one.
    """

    title: str
    url: str
    snippet: str = ""
    domain: str = ""
    rank: int = 0
    published_at: datetime | None = None

    #: Backend-specific extras, kept opaque so a richer backend can add
    #: fields without changing this contract. Never trusted by the agent.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain:
            object.__setattr__(self, "domain", _domain_of(self.url))

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping, for logging and tests. Not an API payload."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
            "rank": self.rank,
        }


class SearchProvider(ABC):
    """Finds candidate documents for a query.

    Implementations must translate their backend's failures into the
    :class:`SearchError` subclasses above, so the agent can react to a
    stable set of error types without importing any search SDK.
    """

    #: Short identifier used in logs and metadata, e.g. ``"duckduckgo"``.
    name: str = "unknown"

    #: True when the backend returns genuine web results rather than an
    #: answer from its own corpus. Useful in reports: an encyclopaedia
    #: search is weaker evidence than a web search.
    returns_web_results: bool = True

    @abstractmethod
    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        """Return up to ``limit`` candidate results, best first.

        Returns an empty list when the backend simply has nothing. Raises
        a :class:`SearchError` subclass for anything else; the agent
        decides whether that is fatal or merely reduces the evidence
        available.
        """

    async def close(self) -> None:
        """Release any resources. Safe to call repeatedly."""
        return None


def _domain_of(url: str) -> str:
    from app.services.research.urls import domain_of

    return domain_of(url)

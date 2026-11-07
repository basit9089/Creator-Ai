"""Search backends behind :class:`SearchProvider`.

* :class:`DuckDuckGoSearchProvider` — free web search, no API key.
* :class:`WikipediaSearchProvider` — free encyclopaedia search, no key.
* :class:`CompositeSearchProvider` — tries several in order.

Concrete implementations are imported here rather than from the package
root, because they pull in ``httpx`` and code that only needs the
:class:`SearchProvider` contract should not pay for it.
"""

from app.services.research.search.base import (
    SearchDepth,
    SearchError,
    SearchProvider,
    SearchQuery,
    SearchRateLimitedError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
)
from app.services.research.search.composite import CompositeSearchProvider
from app.services.research.search.duckduckgo import DuckDuckGoSearchProvider
from app.services.research.search.wikipedia import WikipediaSearchProvider

__all__ = [
    "CompositeSearchProvider",
    "DuckDuckGoSearchProvider",
    "SearchDepth",
    "SearchError",
    "SearchProvider",
    "SearchQuery",
    "SearchRateLimitedError",
    "SearchResult",
    "SearchTimeoutError",
    "SearchUnavailableError",
    "WikipediaSearchProvider",
]

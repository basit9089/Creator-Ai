"""Retrieval services the Researcher depends on.

Layering
--------
The Researcher never opens a network connection and never searches. It
asks these abstractions for material:

* :class:`~app.services.research.search.base.SearchProvider` finds
  candidate documents.
* :class:`~app.services.research.fetching.base.SourceFetcher`
  retrieves one document.
* :class:`~app.services.research.extraction.base.ContentExtractor`
  turns a document into readable text.
* :mod:`app.services.research.urls` decides which URLs may be fetched at
  all.
* :mod:`app.services.research.dedupe` reduces candidates to distinct
  sources.

Each abstraction has more than one plausible implementation, which is the
point: a different search engine, a headless-browser fetcher or a
readability-scoring extractor can be added later without the Researcher
noticing.

Importing this package pulls in the abstractions only. Concrete
implementations that depend on ``httpx`` live in the subpackages and are
imported explicitly, so a module that merely needs the contracts does not
pay for an HTTP client it will never use.
"""

from app.services.research.dedupe import DeduplicationReport, deduplicate
from app.services.research.extraction.base import (
    ContentExtractor,
    ExtractedContent,
    ExtractionError,
)
from app.services.research.fetching.base import (
    FetchedDocument,
    FetchError,
    SourceFetcher,
)
from app.services.research.search.base import (
    SearchDepth,
    SearchError,
    SearchProvider,
    SearchQuery,
    SearchResult,
)

__all__ = [
    "ContentExtractor",
    "DeduplicationReport",
    "ExtractedContent",
    "ExtractionError",
    "FetchError",
    "FetchedDocument",
    "SearchDepth",
    "SearchError",
    "SearchProvider",
    "SearchQuery",
    "SearchResult",
    "SourceFetcher",
    "deduplicate",
]

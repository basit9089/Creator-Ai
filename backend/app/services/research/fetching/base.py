"""The contract for retrieving a source document.

Searching finds candidates; fetching retrieves them. Splitting the two
keeps the Researcher honest: a search result is a title and a snippet
from a third party, which is not evidence, while a fetched document is
the actual published text.

Every failure mode is a distinct exception so the Researcher can record
*why* a source was unusable. "This page was blocked" and "this page timed
out" are different facts about the evidence, and losing that distinction
would make a partial result impossible to interpret.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class FetchError(Exception):
    """Base class for retrieval failures.

    ``code`` is stable and machine-readable; ``message`` is safe to log
    and to show a user. The underlying transport exception stays on
    ``__cause__``.
    """

    code = "fetch_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class UnsafeUrlFetchError(FetchError):
    """The URL failed a safety check, so it was never requested.

    Almost always a search result pointing at the local network. Kept
    distinct from an HTTP failure because nothing was attempted: no
    connection was made and no request was sent.
    """

    code = "unsafe_url"


class FetchTimeoutError(FetchError):
    """The source did not respond within the timeout."""

    code = "fetch_timeout"


class SourceTooLargeError(FetchError):
    """The response exceeded ``MAX_SOURCE_SIZE`` and was abandoned.

    Raised while streaming, so the connection is dropped before the whole
    body is transferred rather than after.
    """

    code = "source_too_large"


class UnsupportedContentTypeError(FetchError):
    """The response was not a document type the extractor can read."""

    code = "unsupported_content_type"


class HttpStatusError(FetchError):
    """The server answered with an error status.

    ``status_code`` is kept for reporting. The response body is
    deliberately discarded: an error page is not evidence and may be very
    large.
    """

    code = "http_error"

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class TooManyRedirectsError(FetchError):
    """The URL redirected more times than allowed."""

    code = "too_many_redirects"


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    """A retrieved document, before text extraction.

    ``final_url`` differs from ``url`` after redirects. Claims must cite
    the URL actually read, since that is the one a reader can open.
    """

    url: str
    final_url: str
    status_code: int
    content_type: str
    charset: str | None
    text: str
    size_bytes: int
    retrieved_at: datetime

    #: True when the body hit the size cap and was cut short. Surfaced so
    #: a truncated page is never mistaken for a complete one.
    truncated: bool = False

    #: True when the final URL came from at least one redirect.
    redirected: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def media_type(self) -> str:
        """Content type with parameters removed, lowercased.

        ``text/html; charset=utf-8`` becomes ``text/html``, which is what
        extractor dispatch should match on.
        """
        return self.content_type.split(";", 1)[0].strip().lower()


class SourceFetcher(ABC):
    """Retrieves one document by URL."""

    #: Short identifier used in logs and failure metadata.
    name: str = "unknown"

    @abstractmethod
    async def fetch(self, url: str) -> FetchedDocument:
        """Retrieve ``url`` and return the raw document.

        Raises a :class:`FetchError` subclass on any failure. Never
        returns a partially-read document as if it were complete: the
        :attr:`FetchedDocument.truncated` flag is how truncation is
        reported instead.
        """

    async def close(self) -> None:
        """Release any resources. Safe to call repeatedly."""
        return None

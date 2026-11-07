"""The contract for turning a fetched document into readable text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ExtractionError(Exception):
    """A document could not be turned into text."""

    code = "extraction_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class UnsupportedDocumentError(ExtractionError):
    """No extractor handles this content type."""

    code = "unsupported_document"


@dataclass(frozen=True, slots=True)
class ExtractedContent:
    """Readable text pulled from a document.

    ``text`` is what the model is given as evidence, so it is capped by
    the caller and :attr:`truncated` records when that happened. Keeping
    the flag means a cut-off document is never mistaken for a complete
    one, which matters because a truncated page can silently omit the
    answer to the question being researched.
    """

    title: str = ""
    text: str = ""
    meta_description: str = ""
    language: str | None = None

    #: Character count of the extracted text, before any capping.
    original_length: int = 0

    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """Whether there is enough text to be worth sending to the model.

        A page that yields a few dozen characters is usually a cookie
        banner, a paywall stub or an error page dressed as an article.
        Counting those as sources would inflate the evidence count while
        adding nothing.
        """
        return len(self.text.strip()) >= MIN_USABLE_TEXT_LENGTH


#: Below this many characters an extracted page is treated as unusable.
#: Calibrated against real pages: a paywall stub is typically under 500
#: characters, while the shortest useful article is well over 1,000.
MIN_USABLE_TEXT_LENGTH = 400


class ContentExtractor(ABC):
    """Extracts readable text from a fetched document."""

    #: Short identifier used in logs.
    name: str = "unknown"

    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """Whether this extractor handles ``content_type``.

        The full header is passed, including parameters, so an extractor
        can look at a charset when it needs to.
        """

    @abstractmethod
    def extract(self, document: Any) -> ExtractedContent:
        """Pull text out of ``document``.

        Args:
            document: a
                :class:`~app.services.research.fetching.base.FetchedDocument`.

        Raises:
            UnsupportedDocumentError: the document type is not handled.
        """

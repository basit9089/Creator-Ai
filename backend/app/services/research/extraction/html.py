"""Text extraction from HTML using the standard library.

Scope, and why it is this small
-------------------------------
Real article extraction needs to score candidate regions, drop boilerplate
and keep reading order. Doing that properly means either a large scraping
dependency or headless browser automation. Neither is justified yet: this
Researcher's job is to give the model readable evidence to analyse, and a
tag-filtering pass does that well enough at a fraction of the cost.

What it does
------------
* Drops elements that never contain article text: ``script``, ``style``,
  ``noscript``, ``template``, ``svg``, ``iframe``, ``form`` and
  ``canvas``.
* Drops page furniture: ``nav``, ``header``, ``footer``, ``aside``.
  Comments, cookie banners and "related articles" lists live here, and
  keeping them injects noise that competes with the actual content.
* Prefers the densest content container when one is obviously present
  (``article``, ``main``), which removes most of the remaining chrome
  without any scoring.
* Collects text with paragraph structure preserved, since paragraph
  boundaries are how the model tells one point from the next.
* Reads ``og:title``, ``description`` and ``lang`` where present.

The acknowledged cost is that ``header`` and ``footer`` are stripped even
on pages that use them to wrap the whole article. That is a deliberate
trade: it errs towards losing some text over drowning the evidence in
navigation. A future phase can add readability scoring behind the same
:class:`~app.services.research.extraction.base.ContentExtractor`
interface without touching anything that calls it.
"""

from __future__ import annotations

from html.parser import HTMLParser

from app.core.logging import get_logger
from app.services.research.extraction.base import (
    ContentExtractor,
    ExtractedContent,
    UnsupportedDocumentError,
)
from app.services.research.fetching.base import FetchedDocument

logger = get_logger("research.extraction.html")

#: Media types this extractor handles.
SUPPORTED_MEDIA_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "application/xml", "text/xml"}
)

#: Elements whose entire subtree is discarded: code, not article text.
_SKIPPED_ELEMENTS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "math",
        "iframe",
        "object",
        "embed",
        "form",
        "button",
        "select",
        "textarea",
        "audio",
        "video",
    }
)

#: Page furniture rather than content. Discarded by tag name.
_CHROME_ELEMENTS = frozenset({"nav", "header", "footer", "aside"})

#: Containers that hold the article when a page marks it clearly. Text
#: inside one of these is preferred over the rest of the body, which is
#: how most navigation and sidebar text gets dropped without scoring.
_ARTICLE_ELEMENTS = frozenset({"article", "main"})

#: Elements that end a line in the extracted text. Preserving paragraph
#: boundaries matters: the model uses them to separate distinct points,
#: and a single run-on block makes separate claims look like one.
_BLOCK_ELEMENTS = frozenset(
    {
        "address", "article", "blockquote", "dd", "div", "dl", "dt",
        "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5",
        "h6", "header", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "td", "th", "tr", "ul",
    }
)

#: Headings are separated more strongly, so a title never runs into the
#: paragraph beneath it.
_HEADING_ELEMENTS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

#: Never opened and never closed.
_VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    }
)

#: The ``article``/``main`` region is preferred when it holds at least
#: this fraction of the page's text. See :func:`_select_body`.
_ARTICLE_PREFERENCE_RATIO = 0.5


class HtmlContentExtractor(ContentExtractor):
    """Extracts readable text from an HTML document."""

    name = "html"

    def __init__(self, *, max_text: int = 20_000) -> None:
        self._max_text = max_text

    def supports(self, content_type: str) -> bool:
        media_type = content_type.split(";", 1)[0].strip().lower()
        return media_type in SUPPORTED_MEDIA_TYPES

    def extract(self, document: FetchedDocument) -> ExtractedContent:
        """Return readable text from ``document``.

        Raises:
            UnsupportedDocumentError: the document is not HTML or XML.
        """
        if not self.supports(document.content_type):
            raise UnsupportedDocumentError(
                f"No extractor handles {document.media_type!r}."
            )

        parser = _HtmlTextParser()
        # The response was already decoded to text by the fetcher, so this
        # parse deals in ``str`` and never has to guess an encoding.
        parser.feed(document.text)
        parser.close()

        text = _select_body(parser)

        capped = text[: self._max_text]
        stripped = capped.rstrip()
        # Cutting mid-word leaves a dangling fragment that reads as a
        # truncation artefact rather than as content.
        if len(text) > len(stripped) and " " in stripped[-80:]:
            stripped = stripped[: stripped.rfind(" ")]
        stripped = stripped.strip()

        if not stripped:
            raise UnsupportedDocumentError(
                "The page contained no readable text."
            )

        title = _clean(parser.title) or parser.og_title

        return ExtractedContent(
            title=title,
            text=stripped,
            meta_description=parser.meta_description,
            language=parser.language,
            original_length=len(text),
            truncated=len(text) > len(stripped),
        )


def _select_body(parser: _HtmlTextParser) -> str:
    """Choose between the whole page body and its article region.

    The article region wins when it carries *most* of the page's text. A
    ratio is used rather than an absolute character count because it is
    scale-invariant: a 600-character news item and a 40,000-character
    investigation are both judged on their own merits, and a fixed
    threshold would quietly disable the preference for every short page.

    It also distinguishes the two cases that matter. A page whose
    ``<main>`` holds the article keeps only the article, dropping sidebar
    and boilerplate. A page whose ``<main>`` holds a login prompt loses
    nothing, because the prompt is a small share of the body and the body
    wins.
    """
    article = parser.article_text
    body = parser.body_text

    if not article:
        return body
    if not body:
        return article

    if len(article) >= len(body) * _ARTICLE_PREFERENCE_RATIO:
        return article

    return body


def _clean(value: str) -> str:
    """Collapse whitespace in a short piece of text."""
    return " ".join(value.split())


class _HtmlTextParser(HTMLParser):
    """Collect article text, title and metadata from an HTML document.

    Two text buffers are maintained: one for the whole body and one for
    the ``article``/``main`` region. Keeping both costs little and lets the
    caller choose, which avoids discarding a page because it marked its
    content region badly.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.body_chunks: list[str] = []
        self.article_chunks: list[str] = []

        self.title = ""
        self.og_title = ""
        self.meta_description = ""
        self.language: str | None = None

        #: Depth counters, because the regions can nest.
        self._skip_depth = 0
        self._article_depth = 0
        self._in_title = False
        self._in_head = False
        self._title_chunks: list[str] = []

    @property
    def body_text(self) -> str:
        return _join(self.body_chunks)

    @property
    def article_text(self) -> str:
        return _join(self.article_chunks)

    # -- HTMLParser hooks --

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: (value or "").strip() for key, value in attrs}

        if tag == "html" and attributes.get("lang"):
            self.language = attributes["lang"][:16]

        # ``meta`` is a void element, so it returns below with the rest of
        # them. It is read here first because that early return would
        # otherwise skip it entirely and the page's description and
        # Open Graph title would be silently lost.
        if tag == "meta":
            self._handle_meta(attributes)
            return

        if tag in _VOID_ELEMENTS:
            return

        if tag == "head":
            self._in_head = True

        elif tag == "title":
            # The head's own <title> is the page title; a <title> outside
            # the head is rare enough not to matter.
            self._in_title = True
            self._title_chunks = []

        elif tag in _SKIPPED_ELEMENTS or tag in _CHROME_ELEMENTS:
            # A depth counter rather than a boolean: these elements nest,
            # and a boolean would un-skip at the first inner close tag.
            self._skip_depth += 1
            return

        elif tag in _ARTICLE_ELEMENTS:
            self._article_depth += 1

        self._emit_break()

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_ELEMENTS:
            return

        if tag == "title" and self._in_title:
            self._in_title = False
            self.title = _clean("".join(self._title_chunks))

        elif tag == "head":
            self._in_head = False

        elif tag in _SKIPPED_ELEMENTS or tag in _CHROME_ELEMENTS:
            self._skip_depth = max(0, self._skip_depth - 1)

        elif tag in _ARTICLE_ELEMENTS:
            self._article_depth = max(0, self._article_depth - 1)

        self._emit_break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # A self-closing tag such as <meta ... /> carries no text. Handled
        # separately because HTMLParser routes it here rather than through
        # handle_starttag/handle_endtag.
        if tag != "meta":
            self._emit_break()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_chunks.append(data)
            return

        if self._skip_depth or not data.strip():
            # Whitespace-only data is dropped rather than emitted: HTML is
            # indented for humans, and preserving it would pad every line.
            return

        # Runs of whitespace *inside* one data chunk are source formatting,
        # not paragraph structure, so they collapse to single spaces here.
        # The blank lines between paragraphs come from ``_emit_break``
        # instead, which is what keeps the two distinguishable.
        cleaned = _clean(data)
        if not cleaned:
            return

        self.body_chunks.append(cleaned)

        if self._article_depth:
            self.article_chunks.append(cleaned)

    # -- Internals --

    def _handle_meta(self, attributes: dict[str, str]) -> None:
        """Read the handful of metadata fields worth having."""
        name = (attributes.get("name") or attributes.get("property") or "").lower()
        content = attributes.get("content", "")

        if not content:
            return

        if name in {"description", "og:description"} and not self.meta_description:
            self.meta_description = _clean(content)[:600]
        elif name == "og:title" and not self.og_title:
            self.og_title = _clean(content)[:300]

    def _emit_break(self) -> None:
        """Append a line break marker between blocks.

        A sentinel rather than a real newline, so the joins below can tell
        a paragraph boundary from incidental whitespace.
        """
        if self._skip_depth:
            return
        self.body_chunks.append("\n")
        if self._article_depth:
            self.article_chunks.append("\n")


def _join(chunks: list[str]) -> str:
    """Turn collected chunks into tidy paragraph-separated text.

    Blank-line runs collapse to a single break, so a heavily indented
    document does not produce a page of empty lines.
    """
    text = "".join(chunks)
    lines = [_clean(line) for line in text.split("\n")]
    return "\n\n".join(line for line in lines if line)

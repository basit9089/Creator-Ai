"""Turning HTML into the readable text the model is given.

Extraction is where evidence quality is decided. A page whose navigation,
cookie banner and related-articles list survive into the prompt will have
the model answering from links to other pages instead of from the article,
so these tests care as much about what is *absent* as what is present.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.services.research.extraction.base import (
    MIN_USABLE_TEXT_LENGTH,
    ContentExtractor,
    ExtractedContent,
    ExtractionError,
    UnsupportedDocumentError,
)
from app.services.research.extraction.html import HtmlContentExtractor
from app.services.research.fetching.base import FetchedDocument

ARTICLE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <title>History of YouTube</title>
  <meta name="description" content="How YouTube began and grew.">
  <script>window.analytics = {track: function() {}};</script>
  <style>body { color: red; }</style>
</head>
<body>
  <nav><a href="/home">Home</a><a href="/about">About</a></nav>
  <main>
    <h1>History of YouTube</h1>
    <p>YouTube was founded in February 2005 by three former PayPal employees.</p>
    <p>The site grew quickly and was acquired by Google in October 2006.</p>
  </main>
  <footer><p>Copyright 2026</p></footer>
</body>
</html>
"""


def document_for(html: str, content_type: str = "text/html; charset=utf-8") -> FetchedDocument:
    """Build a fetched document, as the fetcher would hand one over."""
    return FetchedDocument(
        url="https://example.com/a",
        final_url="https://example.com/a",
        status_code=200,
        content_type=content_type,
        charset="utf-8",
        text=html,
        size_bytes=len(html),
        retrieved_at=datetime.now().astimezone(),
    )


# ---------- Title and metadata ----------


def test_extract_reads_the_page_title() -> None:
    content = HtmlContentExtractor().extract(document_for(ARTICLE))

    assert content.title == "History of YouTube"


def test_extract_falls_back_to_the_open_graph_title() -> None:
    html = """<html><head>
      <title></title>
      <meta property="og:title" content="A Shared Headline">
    </head><body><p>Body text that is long enough to be usable here.</p></body></html>"""

    assert HtmlContentExtractor().extract(document_for(html)).title == "A Shared Headline"


def test_extract_reads_the_meta_description() -> None:
    content = HtmlContentExtractor().extract(document_for(ARTICLE))

    assert content.meta_description == "How YouTube began and grew."

def test_extract_reads_the_document_language() -> None:
    assert HtmlContentExtractor().extract(document_for(ARTICLE)).language == "en"


def test_extract_survives_a_page_with_no_head_metadata() -> None:
    html = "<html><body><p>Just a paragraph of body text, nothing else at all.</p></body></html>"

    content = HtmlContentExtractor().extract(document_for(html))

    assert content.text
    assert content.title == ""


# ---------- What must be removed ----------


def test_extract_removes_scripts_and_styles() -> None:
    content = HtmlContentExtractor().extract(document_for(ARTICLE))

    assert "analytics" not in content.text
    assert "color: red" not in content.text
    assert "track: function" not in content.text


def test_extract_removes_navigation_and_footer_chrome() -> None:
    """Page furniture competes with the article for the model's attention."""
    content = HtmlContentExtractor().extract(document_for(ARTICLE))

    assert "Home" not in content.text
    assert "About" not in content.text
    assert "Copyright" not in content.text


def test_extract_removes_a_cookie_banner() -> None:
    html = """<html><body>
      <div id="cookie-banner">
        <p>We use cookies to improve your experience on this website.</p>
      </div>
      <article>
        <p>The substantive article text that should be extracted instead. It
        runs to a realistic length, because on a real page the article is
        overwhelmingly the bulk of the text and the banner is not.</p>
      </article>
    </body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "cookies" not in content.text
    assert "substantive article text" in content.text


@pytest.mark.parametrize(
    "tag", ["script", "style", "noscript", "template", "svg", "iframe", "form", "button"]
)
def test_extract_removes_every_kind_of_code_element(tag: str) -> None:
    html = f"""<html><body>
      <article><p>The real article content lives here.</p></article>
      <{tag}>SHOULD_NOT_APPEAR</{tag}>
    </body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "SHOULD_NOT_APPEAR" not in content.text


def test_extract_handles_nested_skipped_elements() -> None:
    """A depth counter is needed: a boolean would un-skip too early.

    ``<form><div>...</div></form>`` is common, and a boolean flag would end
    skipping at the inner ``</div>`` and leak the rest of the form.
    """
    html = """<html><body><article>
      <p>The article text comes first and must be kept.</p>
      <form><div><span>LEAKED_FORM_CONTENT</span></div>tail of the form</form>
      <p>More article text afterwards.</p>
    </article></body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "LEAKED_FORM_CONTENT" not in content.text
    assert "tail of the form" not in content.text
    assert "More article text afterwards." in content.text


def test_extract_ignores_comments_and_doctype() -> None:
    html = """<!DOCTYPE html><html><body><article>
      <!-- COMMENT_SHOULD_NOT_APPEAR -->
      <p>The article body text that matters most.</p>
    </article></body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "COMMENT" not in content.text
    assert "DOCTYPE" not in content.text


def test_extract_decodes_html_entities() -> None:
    html = """<html><body><article>
      <p>R&amp;D spending &mdash; what &quot;growth&quot; meant in 2005.</p>
    </article></body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "R&D" in content.text
    assert "growth" in content.text


# ---------- Region selection ----------


def test_extract_prefers_the_article_region() -> None:
    """Body text that is not in the article is dropped when one exists.

    The proportions are realistic: on a real article page the sidebar and
    boilerplate are a small share of the text, and the article region
    dominates.
    """
    article_body = (
        "<p>The actual article paragraph, kept as evidence. It continues for "
        "some length so that it plainly dominates the surrounding chrome, "
        "which is how a real article page is laid out.</p>"
    )
    sidebar_body = "<p>SIDEBAR_NOISE that should not reach the model.</p>"
    html = f"""<html><body>
      <div class="sidebar">{sidebar_body}</div>
      <article>{article_body}</article>
    </body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "SIDEBAR_NOISE" not in content.text
    assert "actual article paragraph" in content.text


def test_extract_keeps_body_text_the_article_region_does_not_cover() -> None:
    """A small article region must not discard the rest of the page.

    A page can mark a summary in ``<main>`` and carry the substance in
    plain ``<div>``s. Requiring only a share of the body, rather than an
    absolute count, is what keeps both layouts working.
    """
    html = """<html><body>
      <main><p>Log in to continue.</p></main>
      <p>Substantive content that lives outside the main element entirely.</p>
    </body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "Substantive content" in content.text


def test_extract_prefers_a_thin_article_when_it_is_the_whole_page() -> None:
    """With no body text outside the region, the region is all there is."""
    html = "<html><body><article><p>Short but complete.</p></article></body></html>"

    assert "Short but complete." in HtmlContentExtractor().extract(document_for(html)).text


# ---------- Structure ----------


def test_extract_preserves_paragraph_boundaries() -> None:
    """Paragraph breaks are how the model tells one point from the next."""
    content = HtmlContentExtractor().extract(document_for(ARTICLE))

    assert "\n\n" in content.text
    paragraphs = [p for p in content.text.split("\n\n") if p]
    assert any("founded in February 2005" in p for p in paragraphs)
    assert any("acquired by Google" in p for p in paragraphs)


def test_extract_collapses_heavy_indentation() -> None:
    """HTML is indented for humans; preserving it would pad every line.

    Blank lines *within* one paragraph are source formatting, not
    structure, so they must collapse. Blank lines *between* paragraphs are
    structure and are produced deliberately by the block handling.
    """
    html = "<html><body><article><p>\n\n\n      spaced   out\n\n   text\n\n</p></article></body></html>"

    content = HtmlContentExtractor().extract(document_for(html))

    assert content.text == "spaced out text"


def test_extract_survives_void_elements() -> None:
    """``<br>`` and ``<img>`` are never closed and must not break the parse."""
    html = """<html><body><article>
      <p>First line<br>second line<img src="x.png"> after the image.</p>
    </article></body></html>"""

    content = HtmlContentExtractor().extract(document_for(html))

    assert "First line" in content.text
    assert "after the image." in content.text


# ---------- Bounds and failure ----------


def test_extract_truncates_at_the_configured_cap_and_flags_it() -> None:
    """A cut-off page must never be mistaken for a complete one."""
    body = "<p>" + ("word " * 500) + "</p>"
    html = f"<html><body><article>{body}</article></body></html>"

    content = HtmlContentExtractor(max_text=200).extract(document_for(html))

    assert len(content.text) <= 200
    assert content.truncated is True
    assert content.original_length > 200


def test_extract_does_not_flag_a_complete_document_as_truncated() -> None:
    content = HtmlContentExtractor(max_text=10_000).extract(document_for(ARTICLE))

    assert content.truncated is False
    assert content.original_length == len(content.text)


def test_extract_rejects_a_page_with_no_readable_text() -> None:
    """An empty extraction is a failure, not an empty source.

    Returning one would let a navigation-only page count as evidence.
    """
    html = "<html><body><nav><a href='/x'>Link</a></nav></body></html>"

    with pytest.raises(UnsupportedDocumentError):
        HtmlContentExtractor().extract(document_for(html))


def test_extract_rejects_an_unsupported_content_type() -> None:
    with pytest.raises(UnsupportedDocumentError, match="application/pdf"):
        HtmlContentExtractor().extract(document_for("data", content_type="application/pdf"))


def test_extract_is_usable_only_with_enough_text() -> None:
    """A paywall stub is not evidence, however successfully it parsed."""
    thin = ExtractedContent(text="Subscribe to continue reading.")
    thick = ExtractedContent(text="x" * (MIN_USABLE_TEXT_LENGTH + 1))

    assert thin.is_usable is False
    assert thick.is_usable is True


# ---------- The contract ----------


@pytest.mark.parametrize(
    ("content_type", "supported"),
    [
        ("text/html", True),
        ("text/html; charset=utf-8", True),
        ("TEXT/HTML", True),
        ("application/xhtml+xml", True),
        ("application/xml", True),
        ("text/plain", False),
        ("application/pdf", False),
        ("", False),
    ],
)
def test_supports_only_handles_document_types(content_type: str, supported: bool) -> None:
    assert HtmlContentExtractor().supports(content_type) is supported


def test_the_extractor_satisfies_the_abstract_contract() -> None:
    assert isinstance(HtmlContentExtractor(), ContentExtractor)


def test_extraction_errors_carry_stable_codes() -> None:
    assert ExtractionError("x").code == "extraction_error"
    assert UnsupportedDocumentError("x").code == "unsupported_document"
    assert issubclass(UnsupportedDocumentError, ExtractionError)


def test_the_extractor_is_reusable_across_documents() -> None:
    """It holds no per-document state, so one instance serves every source."""
    extractor = HtmlContentExtractor()

    first = extractor.extract(document_for(ARTICLE))
    second = extractor.extract(document_for("<html><body><p>Different page body here.</p></body></html>"))

    assert first.title == "History of YouTube"
    assert second.title == ""
    assert "Different page body here." in second.text
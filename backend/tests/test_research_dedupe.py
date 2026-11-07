"""Deduplication of search candidates.

Deduping is not a tidiness concern here. Ten copies of one article are one
source, and counting them as independent would inflate every confidence
grade computed downstream.
"""

from __future__ import annotations

import pytest

from app.services.research.dedupe import deduplicate, group_by_domain
from app.services.research.search.base import SearchResult


def result(url: str, title: str = "") -> SearchResult:
    return SearchResult(title=title or f"Title for {url}", url=url, rank=1)


def untitled(url: str) -> SearchResult:
    """A result whose title the backend genuinely did not supply."""
    return SearchResult(title="", url=url, rank=1)


# ---------- Identical URLs ----------


def test_exact_duplicate_urls_collapse_to_one() -> None:
    report = deduplicate(
        [
            result("https://example.com/a", "First"),
            result("https://example.com/a", "Second"),
        ]
    )

    assert len(report.kept) == 1
    assert len(report.dropped) == 1
    assert report.kept[0].title == "First"


def test_urls_differing_only_by_tracking_parameters_collapse() -> None:
    """The same article arriving from two campaigns is one source."""
    report = deduplicate(
        [
            result("https://example.com/a?utm_source=newsletter", "Article A"),
            result("https://example.com/a?fbclid=xyz", "Article A"),
        ]
    )

    assert report.kept_count == 1


def test_urls_differing_only_by_scheme_case_and_www_collapse() -> None:
    report = deduplicate(
        [
            result("https://www.Example.com/Path/", "Article"),
            result("https://example.com/Path", "Article"),
        ]
    )

    assert report.kept_count == 1


def test_urls_differing_by_a_meaningful_query_do_not_collapse() -> None:
    """A query parameter that filters results is not tracking."""
    report = deduplicate(
        [
            result("https://example.com/search?q=youtube", "YouTube"),
            result("https://example.com/search?q=vimeo", "Vimeo"),
        ]
    )

    assert report.kept_count == 2


# ---------- Titles ----------


def test_repeated_titles_on_one_domain_collapse() -> None:
    """Two pages with one headline on one site are the same article."""
    report = deduplicate(
        [
            result("https://example.com/a", "History of YouTube"),
            result("https://example.com/b", "History of YouTube"),
        ]
    )

    assert report.kept_count == 1


def test_repeated_titles_across_domains_collapse() -> None:
    """Syndicated content is not independent corroboration.

    Treating a republished article as a second voice would hand the model
    what looks like agreement and grade it ``high`` on the strength of one
    press release.
    """
    report = deduplicate(
        [
            result("https://example.com/a", "History of YouTube"),
            result("https://other.com/b", "History of YouTube"),
        ]
    )

    assert report.kept_count == 1
    assert report.dropped_count == 1


LONG_LEFT = (
    "YouTube began in February 2005 as a video sharing site created by three "
    "former PayPal employees who wanted to share videos with friends and family online"
)
LONG_RIGHT = LONG_LEFT.replace("employees", "engineers")


def test_near_identical_titles_collapse_by_similarity() -> None:
    """Long headlines differing by one word clear the 0.90 threshold.

    Token Jaccard of 25 shared tokens against a 26-token union is 0.926.
    A short pair cannot demonstrate this: 3 shared tokens out of 4 scores
    0.75, which is deliberately treated as a different article.
    """
    report = deduplicate(
        [
            result("https://example.com/a", LONG_LEFT),
            result("https://other.com/b", LONG_RIGHT),
        ]
    )

    assert report.kept_count == 1
    assert report.kept[0].url == "https://example.com/a"


def test_distinct_titles_are_kept() -> None:
    """Deduplication must not start eating genuinely different sources."""
    report = deduplicate(
        [
            result("https://example.com/a", "History of YouTube"),
            result("https://other.com/b", "How YouTube Makes Money"),
        ]
    )

    assert report.kept_count == 2


def test_the_similarity_threshold_is_a_real_knob() -> None:
    """At 1.0 only exact title matches collapse; at 0.80 this pair does.

    The two titles share most of their tokens but are not identical, so
    the threshold alone decides the outcome.
    """
    strict = deduplicate(
        [
            result("https://example.com/a", LONG_LEFT),
            result("https://other.com/b", LONG_RIGHT),
        ],
        similarity_threshold=1.0,
    )
    loose = deduplicate(
        [
            result("https://example.com/a", LONG_LEFT),
            result("https://other.com/b", LONG_RIGHT),
        ],
        similarity_threshold=0.80,
    )

    assert strict.kept_count == 2
    assert loose.kept_count == 1


def test_untitled_results_are_not_collapsed_together() -> None:
    """Empty titles share their emptiness; that is not a duplicate."""
    report = deduplicate(
        [
            untitled("https://example.com/a"),
            untitled("https://other.com/b"),
        ]
    )

    assert report.kept_count == 2


# ---------- Unsafe URLs ----------


def test_unsafe_urls_are_rejected_before_deduplication() -> None:
    report = deduplicate(
        [
            result("https://example.com/good", "Good article"),
            result("file:///etc/passwd", "Local file"),
            result("http://127.0.0.1:11434/", "Local server"),
        ]
    )

    assert report.kept_count == 1
    assert report.rejected_unsafe is not None
    assert len(report.rejected_unsafe) == 2


def test_rejected_urls_are_reported_separately_from_duplicates() -> None:
    """Different reasons need different counts.

    A duplicate was redundant; an unsafe URL was refused. Reporting them as
    one number would hide a security rejection behind a tidiness statistic.
    """
    report = deduplicate(
        [
            result("https://example.com/a", "Article"),
            result("https://example.com/a", "Article"),
            result("javascript:alert(1)", "Script"),
        ]
    )

    assert report.kept_count == 1
    assert report.dropped_count == 1
    assert len(report.rejected_unsafe) == 1


# ---------- Limits and ordering ----------


def test_the_limit_is_applied_after_deduplication() -> None:
    """The cap must not decide which of two duplicates survives."""
    report = deduplicate(
        [
            result("https://example.com/a", "Article A"),
            result("https://example.com/a", "Article A duplicate"),
            result("https://example.com/b", "Article B"),
            result("https://example.com/c", "Article C"),
        ],
        limit=2,
    )

    assert report.kept_count == 2
    assert [r.url for r in report.kept] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_input_order_is_preserved() -> None:
    """Search rank decides what survives; reordering would change results."""
    report = deduplicate(
        [
            result("https://c.example/3", "Third"),
            result("https://a.example/1", "First"),
            result("https://b.example/2", "Second"),
        ]
    )

    assert [r.url for r in report.kept] == [
        "https://c.example/3",
        "https://a.example/1",
        "https://b.example/2",
    ]


def test_kept_results_carry_the_normalised_url() -> None:
    """The URL that is fetched is the URL that gets cited."""
    report = deduplicate([result("https://www.example.com/a/?utm_source=x#frag", "A")])

    assert report.kept[0].url == "https://example.com/a"


def test_deduplicating_nothing_is_harmless() -> None:
    report = deduplicate([])

    assert report.kept_count == 0
    assert report.dropped_count == 0


def test_a_trailing_site_name_does_not_defeat_the_comparison() -> None:
    """``History of YouTube - Wikipedia`` and ``History of YouTube`` agree.

    Search engines append the site name, so without suffix stripping every
    syndicated copy would look distinct.
    """
    report = deduplicate(
        [
            result("https://wikipedia.org/a", "History of YouTube"),
            result("https://britannica.com/b", "History of YouTube | Britannica"),
        ]
    )

    assert report.kept_count == 1
    assert report.kept[0].url == "https://wikipedia.org/a"


def test_a_subtitle_suffix_also_collapses_and_the_limit_is_documented() -> None:
    """Suffix stripping cannot distinguish a site name from a subtitle.

    This pins the known bias rather than hiding it: ``Part 2`` is treated
    as a site name and merged away. The earlier-ranked copy survives, so
    the topic is still covered.
    """
    report = deduplicate(
        [
            result("https://example.com/a", "History of YouTube"),
            result("https://other.com/b", "History of YouTube - Part 2"),
        ]
    )

    assert report.kept_count == 1
    assert report.dropped_count == 1


# ---------- Grouping ----------


def test_group_by_domain_collects_results_per_publisher() -> None:
    grouped = group_by_domain(
        [
            result("https://wikipedia.org/1"),
            result("https://britannica.com/2"),
            result("https://blog.youtube/3"),
        ]
    )

    assert set(grouped) == {"wikipedia.org", "britannica.com", "blog.youtube"}
    assert all(len(items) == 1 for items in grouped.values())


def test_group_by_domain_uses_the_registrable_domain() -> None:
    """``a.example.com`` and ``example.com`` are one publisher.

    Grouping by full hostname would present one site as several
    independent voices, which is the exact illusion confidence grading
    exists to prevent.
    """
    grouped = group_by_domain(
        [
            result("https://a.example.com/1"),
            result("https://example.com/2"),
        ]
    )

    assert set(grouped) == {"example.com"}
    assert len(grouped["example.com"]) == 2


def test_the_report_exposes_counts() -> None:
    report = deduplicate([result("https://example.com/a", "A")])

    assert report.kept_count == 1
    assert report.dropped_count == 0
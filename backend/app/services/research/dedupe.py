"""Removing duplicate candidates before any page is fetched.

Search engines repeat themselves. The same article can appear twice with
different tracking parameters, the same site can return several near
identical pages, and a piece can be syndicated across sites under one
title. Each of those wastes a fetch and, worse, inflates the evidence
count: ten copies of one article are one source, not ten, and counting
them as independent would corrupt every confidence grade downstream.

Why cross-domain title matching is safe
---------------------------------------
Two pages on different domains with the same title are usually the same
content republished, not corroboration. Treating them as independent
would hand the model what looks like agreement and grade it ``high``.
Dropping the later copy is therefore the conservative choice, and the
dropped entry is returned alongside the kept ones so the decision is
visible in logs rather than silent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.logging import get_logger
from app.services.research.search.base import SearchResult
from app.services.research.urls import (
    UnsafeURLError,
    assert_structurally_safe_url,
    domain_of,
    normalize_url,
    registrable_domain,
)

logger = get_logger("research.dedupe")

#: Characters stripped before comparing titles, so "YouTube's History" and
#: "Youtubes history" compare equal.
_PUNCTUATION_RE = re.compile(r"[^\w\s]+", re.UNICODE)

#: Trailing site names that carry no topical meaning. "History of YouTube
#: - Wikipedia" and "History of YouTube | Britannica" are the same
#: headline with a different suffix.
#:
#: This cannot tell a site name from a subtitle, so "History of YouTube -
#: Part 2" also collapses onto "History of YouTube". That over-merge is
#: deliberate rather than a bug to be tuned away: the earlier-ranked copy
#: is always the one kept, the topic stays covered, and the alternative
#: (no stripping at all) would leave the common site-name case undetected.
#: The dropped entries are returned so the loss is visible.
_SITE_SUFFIX_RE = re.compile(
    r"\s*[|\-–—:]\s*[^-|\u2013\u2014:]{1,40}$",
    re.UNICODE,
)

#: Similarity above which two titles are treated as the same content.
#: High on purpose: a false merge removes real evidence, while a missed
#: merge merely wastes one fetch.
TITLE_SIMILARITY_THRESHOLD = 0.90


@dataclass(frozen=True, slots=True)
class DeduplicationReport:
    """What deduplication kept, dropped and why.

    Returned rather than only logged, because "why did my sixth source
    disappear" is a reasonable question and answering it from a return
    value beats answering it from a log line.
    """

    kept: tuple[SearchResult, ...]
    dropped: tuple[SearchResult, ...]
    rejected_unsafe: tuple[SearchResult, ...]

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def deduplicate(
    results: Sequence[SearchResult],
    *,
    limit: int | None = None,
    similarity_threshold: float = TITLE_SIMILARITY_THRESHOLD,
) -> DeduplicationReport:
    """Reduce ``results`` to distinct sources, preserving rank order.

    Three filters run in sequence:

    1. URLs that fail the structural safety check are rejected outright.
       These are the non-http schemes, local host names and loopback
       addresses, none of which may ever be fetched. The check is
       deliberately the DNS-free one: resolving every candidate here
       would put a blocking ``getaddrinfo`` on the event loop for a
       tidiness step. A host name that passes is still unverified, which
       is why the fetcher re-checks it with resolution before requesting
       anything.
    2. Identical normalised URLs collapse to the highest-ranked copy.
    3. Titles that match an already-kept source, whether exactly after
       normalisation or closely by similarity, collapse onto it.

    Args:
        results: candidates in the order the search backend ranked them.
        limit: maximum sources to keep. Applied last, so the cap never
            influences which of two duplicates survives.

    Returns:
        A :class:`DeduplicationReport`.
    """
    seen_urls: dict[str, SearchResult] = {}
    kept: list[SearchResult] = []
    dropped: list[SearchResult] = []
    rejected: list[SearchResult] = []

    for result in results:
        try:
            normalised_url = assert_structurally_safe_url(result.url)
        except UnsafeURLError:
            rejected.append(result)
            continue

        if normalised_url in seen_urls:
            dropped.append(result)
            continue

        title_key = _title_key(result.title)
        if _matches_kept_title(kept, title_key, result, similarity_threshold):
            dropped.append(result)
            continue

        candidate = _with_normalised_url(result, normalised_url)
        seen_urls[normalised_url] = candidate
        kept.append(candidate)

        if limit is not None and len(kept) >= limit:
            break

    if dropped or rejected:
        logger.info(
            "research_deduplicated kept=%d dropped=%d rejected_unsafe=%d",
            len(kept),
            len(dropped),
            len(rejected),
        )

    return DeduplicationReport(
        kept=tuple(kept),
        dropped=tuple(dropped),
        rejected_unsafe=tuple(rejected),
    )


def _matches_kept_title(
    kept: Sequence[SearchResult],
    title_key: str,
    candidate: SearchResult,
    threshold: float,
) -> bool:
    """Whether ``candidate`` repeats a title that has already been kept.

    An empty title key never matches, so untitled candidates are kept
    rather than collapsed together on their shared emptiness.
    """
    if not title_key:
        return False

    keys = [(_title_key(existing.title), existing) for existing in kept]

    # An exact match on the normalised title is decisive. Two pages with
    # one headline on the same site are the same article; across sites,
    # syndicated content is not independent corroboration either.
    if any(key == title_key for key, _ in keys):
        return True

    if threshold >= 1.0:
        return False

    for key, existing in keys:
        if not key:
            continue
        score = _similarity(title_key, key)
        if score >= threshold:
            logger.info(
                "research_dedupe_title_similarity candidate_domain=%s kept_domain=%s score=%.2f",
                domain_of(candidate.url),
                existing.domain,
                score,
            )
            return True

    return False


def _title_key(title: str) -> str:
    """Normalise a title for comparison.

    Lowercases, strips punctuation, removes a trailing site name and
    collapses whitespace. The result is only ever compared, never shown.
    """
    if not title:
        return ""

    text = title.strip().lower()
    # Order matters. The site suffix is stripped first, while the "|",
    # "-" or ":" separator is still present; removing punctuation first
    # would erase the very separator the suffix rule splits on, silently
    # reducing this to plain lowercasing.
    text = _SITE_SUFFIX_RE.sub("", text)
    text = _PUNCTUATION_RE.sub(" ", text)
    return " ".join(text.split())


def _similarity(left: str, right: str) -> float:
    """Similarity of two normalised titles, in ``0.0..1.0``.

    Token-based rather than character-based: headlines reword the same
    facts ("YouTube launched in 2005" versus "2005: the year YouTube
    launched"), which a character comparison scores as unrelated.

    Returns 0.0 when either side has no tokens, so an empty title never
    looks like a match.
    """
    if not left or not right:
        return 0.0

    left_tokens = set(left.split())
    right_tokens = set(right.split())

    if not left_tokens or not right_tokens:
        return 0.0

    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union


def _with_normalised_url(result: SearchResult, normalised_url: str) -> SearchResult:
    """Return ``result`` with its URL replaced by the normalised form.

    The normalised URL is what gets fetched and what the citation must
    show, so a reader following a claim lands on a working address rather
    than one carrying a click-tracking suffix.
    """
    from dataclasses import replace

    return replace(result, url=normalised_url, domain=domain_of(normalised_url))


def group_by_domain(results: Sequence[SearchResult]) -> dict[str, list[SearchResult]]:
    """Group results by registrable domain.

    Used when reporting how the evidence was spread across publishers,
    which is often the first thing worth knowing about a research result.
    """
    grouped: dict[str, list[SearchResult]] = {}
    for result in results:
        grouped.setdefault(registrable_domain(result.url), []).append(result)
    return grouped

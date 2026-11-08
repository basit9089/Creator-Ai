"""Metadata construction: normalization, local heuristics, grounding.

Everything here is deterministic post-processing around one AI call.
The model proposes; this module *verifies and normalises*:

* titles are length-checked and selected by a **documented local
  heuristic** (no external ranking system exists or is pretended);
* tags, keywords and phrases are deduplicated, whitespace-normalised
  and dropped unless they are traceable to words in the script
  (unsupported-entity protection);
* the description is scrubbed of any URL the script does not source,
  then given a real ``Sources:`` section built from the script's own
  references;
* claim/source grounding is carried through from the script unchanged.
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from app.agents.captions_thumbnail.schemas import (
    ClaimRef,
    GroundingInfo,
    MetadataInfo,
    SourceRef,
)
from app.agents.scriptwriter.schemas import ScriptResult

#: YouTube's hard title limit. Candidates longer than this are dropped.
TITLE_MAX_CHARS = 100

#: YouTube's description limit.
DESCRIPTION_MAX_CHARS = 5000

#: Caps that keep tags useful rather than spammy. 15 tags stay well
#: inside YouTube's 500-character tag budget once joined by commas.
MAX_TAGS = 15
MAX_TAG_CHARS = 40
MAX_SECONDARY_KEYWORDS = 10
MAX_PHRASES = 10
MAX_PHRASE_CHARS = 60
MAX_DESCRIPTION_KEYWORDS = 10

#: Patterns the local heuristic treats as clickbait markers. Short and
#: explicit: this is a lint list, not a mind-reading exercise.
CLICKBAIT_MARKERS = (
    "you won't believe",
    "you wont believe",
    "shocking",
    "shocked",
    "mind-blowing",
    "mind blowing",
    "secret trick",
    "you need to see",
    "!!!",
)

#: Any URL-looking token. The AI body must contain none of these; real
#: source URLs are appended afterwards from the script itself.
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Light stopword list for keyword derivation (not for tag filtering —
#: tags come from the model, this only picks sensible fallbacks).
_STOPWORDS = frozenset(
    {
        "about", "after", "again", "against", "also", "because", "before",
        "being", "between", "both", "could", "does", "doing", "during",
        "each", "from", "further", "have", "having", "into", "itself",
        "more", "most", "other", "ought", "ours", "outside", "over",
        "same", "should", "some", "such", "than", "that", "their", "theirs",
        "them", "then", "there", "these", "they", "this", "those",
        "through", "under", "until", "very", "were", "what", "when",
        "where", "which", "while", "with", "would", "your", "yours",
        "just", "like", "really", "into", "onto", "upon", "video",
        "watch", "here", "much", "many", "well", "even", "still",
    }
)


class MetadataValidationError(Exception):
    """The model's proposal was unusable even after normalization."""


class YouTubeMetadataDraft(BaseModel):
    """AI-facing schema: what the model must propose, nothing more."""

    model_config = ConfigDict(frozen=True)

    title_candidates: list[str] = Field(min_length=2, max_length=8)
    description: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list, max_length=40)
    primary_keyword: str = Field(min_length=1)
    secondary_keywords: list[str] = Field(default_factory=list, max_length=25)
    search_phrases: list[str] = Field(default_factory=list, max_length=20)
    content_category: str = ""


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def collapse_ws(text: str) -> str:
    """Trim and collapse every whitespace run to a single space."""
    return " ".join(text.split())


def script_vocabulary(script: ScriptResult) -> set[str]:
    """Lowercased words of every narratable part of the script."""
    parts = [
        script.title,
        script.hook,
        script.introduction,
        script.narration,
        script.conclusion,
        script.cta,
        *(section.narration for section in script.sections),
        *(section.heading for section in script.sections),
    ]
    words: set[str] = set()
    for part in parts:
        words.update(word.lower() for word in _WORD_RE.findall(part))
    return words


def is_grounded(term: str, vocabulary: set[str]) -> bool:
    """Whether ``term`` is traceable to words actually in the script.

    Grounded when the whole term appears as a script word, or when at
    least one of its (4+ character) content words does. This is the
    unsupported-entity guard: a hallucinated person, place or concept
    shares no word with the script and is dropped.
    """
    tokens = [token.lower() for token in _WORD_RE.findall(term)]
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] in vocabulary
    if all(token in vocabulary for token in tokens):
        return True
    return any(len(token) >= 4 and token in vocabulary for token in tokens)


def most_frequent_content_word(text: str, vocabulary: set[str]) -> str:
    """Most frequent non-stopword of 4+ chars; ties break by first use."""
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    counts = Counter(
        token for token in tokens if len(token) >= 4 and token not in _STOPWORDS
    )
    if not counts:
        return ""
    first_index = {token: tokens.index(token) for token in counts}
    return min(counts, key=lambda token: (-counts[token], first_index[token], token))


def score_title(title: str, primary_keyword: str) -> int:
    """Local heuristic score in [0, 100] for one title candidate.

    Factors (documented, deterministic, entirely local):
      * length 30–70 chars (search-result friendly): 50; other lengths
        inside YouTube's 100-char cap: 30; extreme lengths: 10;
      * contains the primary keyword: +30;
      * a specificity digit (years, counts): +10;
      * a listed clickbait marker: −25;
      * shouting in ALL CAPS: −20.

    This is **not** an external SEO ranking — nothing here contacts a
    ranking service or estimates real search performance.
    """
    score = 0
    length = len(title)
    if 30 <= length <= 70:
        score += 50
    elif 20 <= length <= 100:
        score += 30
    else:
        score += 10
    lowered = title.lower()
    if primary_keyword and primary_keyword.lower() in lowered:
        score += 30
    if any(character.isdigit() for character in title):
        score += 10
    if any(marker in lowered for marker in CLICKBAIT_MARKERS):
        score -= 25
    letters = [character for character in title if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        score -= 20
    return max(0, min(100, score))


# --------------------------------------------------------------------------
# Normalizers
# --------------------------------------------------------------------------


def normalize_titles(candidates: list[str], warnings: list[str]) -> list[str]:
    """Clean, dedupe and length-check title candidates."""
    cleaned: list[str] = []
    seen: set[str] = set()
    dropped_overlong = 0
    for candidate in candidates:
        title = collapse_ws(candidate)
        if not title:
            continue
        if len(title) > TITLE_MAX_CHARS:
            dropped_overlong += 1
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(title)
    if dropped_overlong:
        warnings.append(
            f"Dropped {dropped_overlong} title candidate(s) longer than "
            f"{TITLE_MAX_CHARS} characters."
        )
    return cleaned


def select_title(candidates: list[str], primary_keyword: str) -> tuple[str, int]:
    """Pick the best-scoring candidate; earlier candidates win ties."""
    if not candidates:
        raise MetadataValidationError("no usable title candidate (all were empty or too long)")
    scores = [score_title(title, primary_keyword) for title in candidates]
    best_index = max(range(len(candidates)), key=lambda i: (scores[i], -i))
    return candidates[best_index], scores[best_index]


def normalize_terms(
    items: list[str],
    *,
    vocabulary: set[str],
    max_items: int,
    max_chars: int,
    label: str,
    warnings: list[str],
) -> tuple[str, ...]:
    """Whiten, dedupe, ground-check and cap a list of terms.

    Unsupported entries (no word shared with the script) are removed —
    the model cannot smuggle entities into the metadata.
    """
    result: list[str] = []
    seen: set[str] = set()
    dropped_unsupported = 0
    dropped_overlong = 0
    for item in items:
        term = collapse_ws(item)
        if not term:
            continue
        if len(term) > max_chars:
            dropped_overlong += 1
            continue
        key = term.lower()
        if key in seen:
            continue
        if not is_grounded(term, vocabulary):
            dropped_unsupported += 1
            continue
        seen.add(key)
        result.append(term)
        if len(result) >= max_items:
            break
    if dropped_unsupported:
        warnings.append(
            f"Dropped {dropped_unsupported} {label} not traceable to the script."
        )
    if dropped_overlong:
        warnings.append(
            f"Dropped {dropped_overlong} overlong {label} (limit {max_chars} chars)."
        )
    return tuple(result)


def build_description(
    ai_body: str, script: ScriptResult, warnings: list[str]
) -> str:
    """Scrub unsourced URLs, append the real source list, enforce length.

    URLs written by the model are removed (the model never sees source
    URLs, only domains, so anything URL-shaped it produces is unsourced).
    The ``Sources:`` section contains exactly the URLs the script
    carries from the Researcher/Fact Checker stages.
    """
    body = ai_body.replace("\r\n", "\n").replace("\r", "\n").strip()
    removed = len(_URL_RE.findall(body))
    if removed:
        body = _URL_RE.sub("", body)
        body = re.sub(r"[ \t]{2,}", " ", body)
        warnings.append(
            f"Removed {removed} URL(s) from the description: only the "
            "script's own sources may be linked."
        )

    source_urls: list[str] = []
    seen_urls: set[str] = set()
    for source in script.sources:
        url = str(source.get("url", "")).strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            source_urls.append(url)
    sources_block = ""
    if source_urls:
        sources_block = "\n\nSources:\n" + "\n".join(f"- {url}" for url in source_urls)

    combined = body + sources_block
    if len(combined) > DESCRIPTION_MAX_CHARS:
        # Cut the model's prose at a word boundary; the sources section
        # is real attribution, so it goes first in the trimming order.
        room = DESCRIPTION_MAX_CHARS - len(sources_block)
        if room < 0:
            combined = _truncate_at_word(combined, DESCRIPTION_MAX_CHARS)
            warnings.append(
                f"Description trimmed to YouTube's {DESCRIPTION_MAX_CHARS}-"
                "character limit (source list is very large)."
            )
        else:
            trimmed = _truncate_at_word(body, max(0, room))
            combined = trimmed + sources_block
            if len(combined) > DESCRIPTION_MAX_CHARS:
                combined = combined[:DESCRIPTION_MAX_CHARS]
            warnings.append(
                f"Description body trimmed to fit YouTube's "
                f"{DESCRIPTION_MAX_CHARS}-character limit."
            )
    return combined


def _truncate_at_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut.rstrip()


def build_grounding(script: ScriptResult) -> GroundingInfo:
    """Carry the script's claim/source references through unchanged.

    Sections map a claim back to the headings that use it, so the chain
    ``metadata → script section → claim → source`` is walkable whenever
    the underlying data supports it.
    """
    sections_by_claim: dict[str, list[str]] = {}
    extra_refs: dict[str, tuple[str, ...]] = {}
    for section in script.sections:
        for reference in section.claim_references:
            claim_id = str(reference.get("claim_id", "")) if isinstance(reference, dict) else ""
            if not claim_id:
                continue
            headings = sections_by_claim.setdefault(claim_id, [])
            if section.heading not in headings:
                headings.append(section.heading)
            if isinstance(reference, dict):
                extra_refs.setdefault(
                    claim_id, tuple(str(sid) for sid in reference.get("source_ids", ()) )
                )

    claim_refs: list[ClaimRef] = []
    seen: set[str] = set()
    for claim in script.claims_used:
        seen.add(claim.claim_id)
        claim_refs.append(
            ClaimRef(
                claim_id=claim.claim_id,
                source_ids=claim.source_ids,
                sections=tuple(sections_by_claim.get(claim.claim_id, ())),
            )
        )
    for claim_id, source_ids in extra_refs.items():
        if claim_id in seen:
            continue
        claim_refs.append(
            ClaimRef(
                claim_id=claim_id,
                source_ids=source_ids,
                sections=tuple(sections_by_claim.get(claim_id, ())),
            )
        )

    source_refs = tuple(
        SourceRef(
            id=str(source.get("id", "")),
            url=str(source.get("url", "")),
            domain=str(source.get("domain", "")),
        )
        for source in script.sources
    )
    return GroundingInfo(claim_refs=tuple(claim_refs), source_refs=source_refs)


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def compose_metadata(
    draft: YouTubeMetadataDraft,
    *,
    script: ScriptResult,
    language: str,
    audience: str,
) -> tuple[MetadataInfo, tuple[str, ...]]:
    """Turn the model's proposal into validated, grounded metadata.

    Returns the populated :class:`MetadataInfo` (status ``completed``)
    plus any warnings the normalisation produced.

    Raises:
        MetadataValidationError: nothing usable survived normalisation.
    """
    warnings: list[str] = []
    vocabulary = script_vocabulary(script)
    narration = " ".join(
        part
        for part in (
            script.hook,
            script.introduction,
            script.narration,
            script.conclusion,
            script.cta,
            *(section.narration for section in script.sections),
        )
        if part
    )

    titles = normalize_titles(draft.title_candidates, warnings)

    primary = collapse_ws(draft.primary_keyword)
    if not primary or not is_grounded(primary, vocabulary):
        fallback = most_frequent_content_word(narration or script.title, vocabulary)
        if fallback:
            warnings.append(
                f"Primary keyword {primary!r} was not traceable to the "
                f"script; using {fallback!r} instead."
            )
            primary = fallback
        elif primary:
            warnings.append(
                f"Primary keyword {primary!r} is not traceable to the script."
            )

    final_title, title_score = select_title(titles, primary)
    alternates = tuple(title for title in titles if title.lower() != final_title.lower())

    tags = normalize_terms(
        draft.tags,
        vocabulary=vocabulary,
        max_items=MAX_TAGS,
        max_chars=MAX_TAG_CHARS,
        label="tag(s)",
        warnings=warnings,
    )
    secondary = normalize_terms(
        draft.secondary_keywords,
        vocabulary=vocabulary,
        max_items=MAX_SECONDARY_KEYWORDS,
        max_chars=MAX_TAG_CHARS,
        label="keyword(s)",
        warnings=warnings,
    )
    secondary = tuple(
        keyword for keyword in secondary if keyword.lower() != primary.lower()
    )
    phrases = normalize_terms(
        draft.search_phrases,
        vocabulary=vocabulary,
        max_items=MAX_PHRASES,
        max_chars=MAX_PHRASE_CHARS,
        label="search phrase(s)",
        warnings=warnings,
    )

    description = build_description(draft.description, script, warnings)

    lowered_description = description.lower()
    description_keywords = [
        keyword
        for keyword in (primary, *secondary)
        if keyword and keyword.lower() in lowered_description
    ]
    seen_keywords: set[str] = set()
    ordered_keywords = []
    for keyword in description_keywords:
        key = keyword.lower()
        if key not in seen_keywords:
            seen_keywords.add(key)
            ordered_keywords.append(keyword)
    description_keywords = ordered_keywords[:MAX_DESCRIPTION_KEYWORDS]

    info = MetadataInfo(
        title=final_title,
        alternate_titles=alternates,
        description=description,
        tags=tags,
        primary_keyword=primary,
        secondary_keywords=secondary,
        search_phrases=phrases,
        title_score=title_score,
        description_keywords=tuple(description_keywords),
        content_category=collapse_ws(draft.content_category)[:60],
        language=language,
        audience=audience,
    )
    return info, tuple(warnings)

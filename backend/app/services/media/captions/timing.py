"""Deterministic caption timing and chunking.

Two sources of truth, in priority order:

1. A supplied :class:`VideoProductionResult` — its scene narration and
   timeline windows *are* the production timing, so captions follow the
   rendered video exactly.
2. The script alone — timing falls back to the project's own
   convention: narration length at 150 WPM (or the script's
   ``words_per_minute``), the same deterministic rule the scene planner
   uses.

Text comes from narration verbatim: cues split it at sentence/clause
boundaries but never rewrite a word. Within a segment, cue time is
allocated in proportion to each cue's share of the words, so no cue gets
a fake one-second stamp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence

from app.agents.scriptwriter.schemas import ScriptResult
from app.agents.video_producer.scene_planner import WORDS_PER_MINUTE_DEFAULT, plan_scenes
from app.agents.video_producer.schemas import VideoProductionResult
from app.services.media.captions.base import CaptionCue, CaptionError, validate_cues

#: A cue holds at most two lines of ~42 characters — the classic
#: subtitle comfort zone. Oversized sentences are split to fit this.
MAX_CUE_CHARS = 84

#: Characters per display line when wrapping a cue.
LINE_CHARS = 42

#: Sentence enders. ``…`` is included because scripts use ellipses.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")

#: Clause boundaries for a sentence that is still too long. Lookbehind
#: keeps the punctuation with the left chunk, so splitting never drops a
#: character of narration.
_CLAUSE_RE = re.compile(r"(?<=[,;:\-])\s+|(?<=[—–])\s+")

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Segment:
    """Narration text bound to a half-open time window ``[start_ms, end_ms)``."""

    text: str
    start_ms: int
    end_ms: int
    label: str = ""


def _words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def validate_segments(segments: Sequence[Segment]) -> None:
    """Reject windows that could produce negative or overlapping cues.

    Raises:
        CaptionError: a segment has no duration, a negative start, or
            overlaps its predecessor.
    """
    previous_end = 0
    for position, segment in enumerate(segments):
        if segment.start_ms < 0:
            raise CaptionError(
                f"segment {position} ({segment.label or 'unnamed'}): "
                f"negative start {segment.start_ms}"
            )
        if segment.start_ms >= segment.end_ms:
            raise CaptionError(
                f"segment {position} ({segment.label or 'unnamed'}): "
                f"start {segment.start_ms} is not before end {segment.end_ms}"
            )
        if segment.start_ms < previous_end:
            raise CaptionError(
                f"segment {position} ({segment.label or 'unnamed'}): "
                f"overlaps the previous segment"
            )
        previous_end = segment.end_ms


def segments_from_video(video: VideoProductionResult) -> list[Segment]:
    """Build caption segments from a Phase 7 production result.

    Uses the real timeline window per scene when a timeline exists;
    otherwise falls back to each scene's own ``duration_seconds``,
    accumulated in scene order. Narration comes from the scenes
    unchanged.
    """
    if not video.scenes:
        raise CaptionError("video production result contains no scenes")

    timeline_by_id = {}
    if video.timeline is not None:
        timeline_by_id = {ts.scene_id: ts for ts in video.timeline.scenes}

    segments: list[Segment] = []
    cursor_ms = 0
    for scene in video.scenes:
        window = timeline_by_id.get(scene.scene_id)
        if window is not None:
            start_ms = round(window.start_time * 1000)
            end_ms = round(window.end_time * 1000)
        else:
            start_ms = cursor_ms
            end_ms = cursor_ms + round(scene.duration_seconds * 1000)
        cursor_ms = end_ms

        text = scene.narration.strip()
        if not text:
            # Empty scenes produce no cues; they also never block the
            # non-overlap check because no window is kept for them.
            continue
        segments.append(
            Segment(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                label=scene.heading or scene.scene_id,
            )
        )

    if not segments:
        raise CaptionError("video production result contains no narratable scenes")
    validate_segments(segments)
    return segments


def segments_from_script(script: ScriptResult) -> list[Segment]:
    """Build caption segments with the project's 150 WPM convention.

    Reuses the scene planner's deterministic durations, so captions
    spoken without a rendered video still match what the Video Producer
    would produce for the same script.
    """
    scenes = plan_scenes(script)
    if not scenes:
        raise CaptionError("script produced no caption segments")

    segments: list[Segment] = []
    cursor_ms = 0
    for scene in scenes:
        text = scene.narration.strip()
        duration_ms = round(scene.duration_seconds * 1000)
        if text:
            segments.append(
                Segment(
                    text=text,
                    start_ms=cursor_ms,
                    end_ms=cursor_ms + duration_ms,
                    label=scene.heading or scene.scene_id,
                )
            )
        cursor_ms += duration_ms

    if not segments:
        raise CaptionError("script contains no narratable text")
    validate_segments(segments)
    return segments


def split_narration(text: str, *, max_chars: int = MAX_CUE_CHARS) -> list[str]:
    """Split narration into caption-sized chunks without altering words.

    Strategy, in order: sentences first; a sentence longer than
    ``max_chars`` is cut at clause boundaries; anything still too long
    is packed word-by-word. Concatenating the chunks reproduces the
    input's words exactly (whitespace runs collapse to single spaces).
    """
    normalised = " ".join(text.split())
    if not normalised:
        return []

    chunks: list[str] = []
    for sentence in _SENTENCE_RE.split(normalised):
        if len(sentence) <= max_chars:
            if sentence:
                chunks.append(sentence)
            continue
        for clause in _CLAUSE_RE.split(sentence):
            if len(clause) <= max_chars:
                if clause:
                    chunks.append(clause)
                continue
            chunks.extend(_pack_words(clause, max_chars))
    return chunks


def _pack_words(text: str, max_chars: int) -> list[str]:
    """Greedily pack words into chunks of at most ``max_chars``.

    A single word longer than the limit becomes its own chunk: text is
    never broken or dropped to satisfy a limit.
    """
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def wrap_cue_text(text: str, *, line_chars: int = LINE_CHARS) -> str:
    """Wrap one cue into display lines of at most ``line_chars`` chars.

    Wrapping only inserts line breaks: every character of ``text``
    survives in order. A single word longer than ``line_chars`` sits on
    its own overlong line rather than being truncated.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= line_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines) if lines else text.strip()


def _allocate(
    weights: Sequence[int], *, start_ms: int, end_ms: int
) -> list[tuple[int, int]]:
    """Split ``[start_ms, end_ms)`` proportionally to ``weights``.

    Boundaries are rounded to whole milliseconds, the final boundary is
    pinned to ``end_ms``, and every span is forced to be strictly
    positive. If the window is too small to give each cue at least one
    millisecond the input is impossible, and we say so rather than emit
    overlapping timestamps.
    """
    total_weight = sum(weights)
    if not weights or total_weight <= 0:
        raise CaptionError("cannot allocate time across zero-weight cues")

    span = end_ms - start_ms
    if span < len(weights):
        raise CaptionError(
            f"window of {span}ms is too short for {len(weights)} captions"
        )

    # Raw proportional boundaries. The weights are positive, so the raw
    # values are non-decreasing and the ends are exact.
    bounds = [start_ms]
    cumulative = 0
    for weight in weights:
        cumulative += weight
        bounds.append(start_ms + round(span * cumulative / total_weight))
    bounds[-1] = end_ms

    # Rounding can tie neighbouring boundaries. Walk backwards forcing
    # each bound at least one millisecond below its successor: after the
    # pass the sequence is strictly increasing whenever the window has
    # room (``span >= cue count``), and the failure mode for a genuinely
    # impossible window is a clear error rather than an overlap.
    for i in range(len(bounds) - 2, -1, -1):
        if bounds[i] >= bounds[i + 1]:
            bounds[i] = bounds[i + 1] - 1
    if bounds[0] < start_ms:
        raise CaptionError(
            f"window of {span}ms is too short for {len(weights)} captions"
        )

    # Two views over the same list, so they necessarily differ by one.
    return list(zip(bounds, bounds[1:]))  # noqa: B905 - intentional pairing


def build_cues(segments: Sequence[Segment]) -> list[CaptionCue]:
    """Turn segments into a validated, ordered cue list.

    Deterministic: identical inputs yield identical cues. Empty
    narration is skipped; a segment whose text cannot fit its window
    raises rather than producing overlapping times.
    """
    validate_segments(segments)

    cues: list[CaptionCue] = []
    for segment in segments:
        chunks = split_narration(segment.text)
        if not chunks:
            continue
        windows = _allocate(
            [_words(chunk) for chunk in chunks],
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
        )
        for (start_ms, end_ms), chunk in zip(windows, chunks, strict=True):
            cues.append(
                CaptionCue(
                    index=len(cues) + 1,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=wrap_cue_text(chunk),
                )
            )

    if not cues:
        raise CaptionError("narration produced no captions")
    validate_cues(cues)
    return cues

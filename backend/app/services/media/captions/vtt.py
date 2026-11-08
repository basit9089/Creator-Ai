"""WebVTT (``.vtt``) formatting and parsing.

Canonical layout, LF line endings (per the WebVTT specification)::

    WEBVTT

    1
    00:00:00.000 --> 00:00:01.200
    First caption line

The parser mirrors :func:`app.services.media.captions.srt.parse_srt`:
strict enough to prove a written file round-trips, tolerant of LF/CRLF.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.services.media.captions.base import (
    CaptionCue,
    CaptionFormatError,
    CaptionFormatter,
    format_timestamp,
    validate_cues,
)

_HEADER = "WEBVTT"

_TIMESTAMP_RE = re.compile(
    r"^(\d{2,}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2,}):(\d{2}):(\d{2})\.(\d{3})$"
)


class WebVTTFormatter(CaptionFormatter):
    """Render validated cues as a WebVTT document with LF endings."""

    name = "vtt"
    extension = "vtt"
    newline = "\n"
    timestamp_separator = "."

    def format(self, cues: Sequence[CaptionCue]) -> str:
        validate_cues(cues)
        parts: list[str] = [_HEADER, ""]
        for cue in cues:
            start = format_timestamp(cue.start_ms, self.timestamp_separator)
            end = format_timestamp(cue.end_ms, self.timestamp_separator)
            body = cue.text.replace("\r\n", "\n").replace("\n", self.newline)
            parts.append(f"{cue.index}")
            parts.append(f"{start} --> {end}")
            parts.append(body)
            parts.append("")
        return self.newline.join(parts)


def _to_ms(match: re.Match[str], offset: int) -> int:
    hours = int(match.group(offset))
    minutes = int(match.group(offset + 1))
    seconds = int(match.group(offset + 2))
    millis = int(match.group(offset + 3))
    if minutes > 59 or seconds > 59:
        raise CaptionFormatError(
            f"invalid timestamp component in {match.group(0)!r}"
        )
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_vtt(text: str) -> list[CaptionCue]:
    """Parse WebVTT text back into cues.

    Raises:
        CaptionFormatError: the ``WEBVTT`` header is missing or a cue
            block is malformed.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    blocks = [block for block in normalised.split("\n\n") if block.strip()]
    if not blocks or blocks[0].split("\n", 1)[0].strip() != _HEADER:
        raise CaptionFormatError("WebVTT header is missing")

    cues: list[CaptionCue] = []
    for block_number, block in enumerate(blocks[1:], start=1):
        lines = block.split("\n")
        if len(lines) < 2:
            raise CaptionFormatError(
                f"WebVTT block {block_number} has fewer than 2 lines"
            )
        # A cue identifier line is optional; accept both shapes.
        timestamp_line_index = 0
        try:
            index = int(lines[0].strip())
            timestamp_line_index = 1
        except ValueError:
            index = block_number
        if timestamp_line_index >= len(lines):
            raise CaptionFormatError(
                f"WebVTT block {block_number} has no timestamp line"
            )
        match = _TIMESTAMP_RE.match(lines[timestamp_line_index].strip())
        if match is None:
            raise CaptionFormatError(
                f"WebVTT block {block_number} has an invalid timestamp line "
                f"{lines[timestamp_line_index]!r}"
            )
        body_lines = lines[timestamp_line_index + 1 :]
        if not body_lines or not "".join(body_lines).strip():
            raise CaptionFormatError(
                f"WebVTT block {block_number} has no caption text"
            )
        cues.append(
            CaptionCue(
                index=index,
                start_ms=_to_ms(match, 1),
                end_ms=_to_ms(match, 5),
                text="\n".join(body_lines),
            )
        )
    validate_cues(cues)
    return cues

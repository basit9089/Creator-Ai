"""SubRip (``.srt``) formatting and parsing.

Canonical layout, CRLF line endings (the de-facto Windows convention,
and what most editors expect from ``.srt``)::

    1
    00:00:00,000 --> 00:00:01,200
    First caption line
    Second line

    2
    ...

The parser is strict: it exists so the caption service can re-read what
it wrote and prove the file on disk matches the cue list, rather than
trusting that a write succeeded.
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

_TIMESTAMP_RE = re.compile(
    r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2,}):(\d{2}):(\d{2}),(\d{3})$"
)


class SRTFormatter(CaptionFormatter):
    """Render validated cues as SubRip text with CRLF line endings."""

    name = "srt"
    extension = "srt"
    newline = "\r\n"
    timestamp_separator = ","

    def format(self, cues: Sequence[CaptionCue]) -> str:
        validate_cues(cues)
        blocks: list[str] = []
        for cue in cues:
            start = format_timestamp(cue.start_ms, self.timestamp_separator)
            end = format_timestamp(cue.end_ms, self.timestamp_separator)
            body = cue.text.replace("\r\n", "\n").replace("\n", self.newline)
            blocks.append(
                f"{cue.index}{self.newline}{start} --> {end}{self.newline}{body}"
            )
        if not blocks:
            return ""
        # Blocks are separated by a blank line, and the last block ends
        # with one too.
        separator = self.newline + self.newline
        return separator.join(blocks) + self.newline + self.newline


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


def parse_srt(text: str) -> list[CaptionCue]:
    """Parse SubRip text back into cues.

    Tolerates CRLF or LF line endings (files may have been edited by
    hand); rejects malformed blocks instead of guessing.

    Raises:
        CaptionFormatError: the text is not valid SubRip.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block for block in normalised.split("\n\n") if block.strip()]
    cues: list[CaptionCue] = []
    for block_number, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 3:
            raise CaptionFormatError(
                f"SubRip block {block_number} has fewer than 3 lines"
            )
        try:
            index = int(lines[0].strip())
        except ValueError as exc:
            raise CaptionFormatError(
                f"SubRip block {block_number} has a non-numeric index {lines[0]!r}"
            ) from exc
        match = _TIMESTAMP_RE.match(lines[1].strip())
        if match is None:
            raise CaptionFormatError(
                f"SubRip block {block_number} has an invalid timestamp line "
                f"{lines[1]!r}"
            )
        cues.append(
            CaptionCue(
                index=index,
                start_ms=_to_ms(match, 1),
                end_ms=_to_ms(match, 5),
                text="\n".join(lines[2:]),
            )
        )
    validate_cues(cues)
    return cues

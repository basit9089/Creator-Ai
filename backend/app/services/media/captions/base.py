"""Caption primitives: the cue model, validation and the formatter ABC.

A *cue* is one on-screen caption: a sequence number, a half-open time
range ``[start_ms, end_ms)`` and the text to show. Everything downstream
(SRT, WebVTT, verification) works on validated cues, so a formatting bug
can never quietly invent overlapping or negative timestamps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class CaptionError(Exception):
    """Base class for caption failures (generation, formatting, I/O)."""


class CaptionFormatError(CaptionError):
    """Cues or formatted output violate the caption contract."""


class CaptionCue(BaseModel):
    """One caption cue with millisecond precision.

    ``index`` is 1-based and follows presentation order. ``text`` holds
    display lines joined by ``"\\n"``; it is never empty.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=1)
    text: str


def validate_cues(cues: Sequence[CaptionCue]) -> None:
    """Enforce the invariants every caption format needs.

    Rules:
      * indices are sequential from 1 (the format's own numbering);
      * ``0 <= start < end`` for every cue (no negative or zero-length);
      * cues never overlap: cue *i* ends no later than cue *i+1* starts;
      * text is present after stripping.

    Raises:
        CaptionFormatError: on the first violated rule, naming the cue.
    """
    previous_end = 0
    for position, cue in enumerate(cues):
        expected_index = position + 1
        if cue.index != expected_index:
            raise CaptionFormatError(
                f"cue {expected_index}: index is {cue.index}, expected {expected_index}"
            )
        if cue.start_ms < 0:
            raise CaptionFormatError(f"cue {cue.index}: negative start {cue.start_ms}")
        if cue.start_ms >= cue.end_ms:
            raise CaptionFormatError(
                f"cue {cue.index}: start {cue.start_ms} is not before end {cue.end_ms}"
            )
        if cue.start_ms < previous_end:
            raise CaptionFormatError(
                f"cue {cue.index}: overlaps the previous cue "
                f"(starts at {cue.start_ms}, previous ends at {previous_end})"
            )
        if not cue.text.strip():
            raise CaptionFormatError(f"cue {cue.index}: empty text")
        previous_end = cue.end_ms


def format_timestamp(ms: int, separator: str) -> str:
    """Render ``ms`` as ``HH:MM:SS<sep>mmm``.

    Hours are zero-padded to at least two digits (they grow naturally
    past 99 for very long videos). SRT uses ``,`` and WebVTT uses ``.``.
    """
    if ms < 0:
        raise CaptionFormatError(f"negative timestamp: {ms}")
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"
    )


class CaptionFormatter(ABC):
    """Renders a validated cue list to one caption file's text."""

    #: Short format identifier, e.g. ``"srt"``.
    name: str = ""

    #: File extension without the dot.
    extension: str = ""

    #: Line ending the format is written with. SRT conventionally uses
    #: CRLF on Windows; WebVTT is LF by specification. Files are written
    #: with newline translation disabled, so what we emit is what lands
    #: on disk.
    newline: str = "\n"

    #: Timestamp millisecond separator for this format.
    timestamp_separator: str = "."

    @abstractmethod
    def format(self, cues: Sequence[CaptionCue]) -> str:
        """Serialise ``cues`` (validated first) to file text."""

    def _checked(self, cues: Sequence[CaptionCue]) -> Sequence[CaptionCue]:
        validate_cues(cues)
        return cues

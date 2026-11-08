"""Voiceover provider contract.

Provider-neutral on purpose: Phase 7 ships no TTS engine (downloading
a heavyweight model would violate the hardware budget), so the default
provider reports ``unavailable`` and renders stay silent. A real local
or cloud TTS implementation can be dropped in later without the Video
Producer changing a line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceoverResult:
    """Outcome of a voiceover attempt — never a lie.

    ``status`` is one of ``available`` (a real audio file exists),
    ``pending`` (no provider configured) or ``failed``. ``audio_path``
    is empty unless a file was actually produced.
    """

    status: str
    audio_path: str = ""
    duration_seconds: float | None = None
    detail: str = ""


class VoiceProvider(ABC):
    """Synthesises narration audio for a script."""

    name: str = "voice-provider"

    @abstractmethod
    def synthesize(self, text: str, *, output_path: Path | None = None) -> VoiceoverResult:
        """Turn ``text`` into an audio file.

        Implementations must not fabricate success: if no engine is
        configured, return ``status="pending"`` rather than writing a
        placeholder and calling it speech.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Whether this provider could actually produce audio now."""

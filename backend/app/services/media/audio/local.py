"""Local voiceover support.

Two responsibilities, no TTS:

* validate an *existing* local audio file (the user's own recording),
* report honestly when no voice track is available.

This keeps the voiceover seam real — the Video Producer calls the same
``synthesize``/``health_check`` interface it will call when a genuine
TTS provider arrives — without downloading a model in Phase 7.
"""

from __future__ import annotations

from pathlib import Path

from app.agents.video_producer.grounding import validate_path
from app.services.media.audio.base import VoiceProvider, VoiceoverResult
from app.services.media.assets.local import media_type_for

#: Audio container types accepted as a supplied voice track.
_ACCEPTED = {"audio", }


class LocalVoiceProvider(VoiceProvider):
    """Voice provider backed by local files only."""

    name = "local-file"

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)

    def synthesize(self, text: str, *, output_path: Path | None = None) -> VoiceoverResult:
        """No synthesis engine ships in Phase 7.

        If ``output_path`` points at an existing, valid audio file it
        is used as the voice track; otherwise the result is ``pending``.
        """
        if output_path is not None:
            validated = self.validate_audio_file(output_path)
            if validated is not None:
                return VoiceoverResult(
                    status="available",
                    audio_path=str(validated),
                    detail="Using supplied local audio file.",
                )
        return VoiceoverResult(
            status="pending",
            detail="No TTS provider configured; voiceover pending.",
        )

    def validate_audio_file(self, candidate: str | Path) -> Path | None:
        """Return the validated path, or ``None`` if unusable.

        Checks containment in the media root, existence, and that the
        extension is an audio type. Raises nothing: an unusable voice
        file downgrades the render to silent, it does not fail it.
        """
        try:
            resolved = validate_path(candidate, self._base)
        except Exception:  # noqa: BLE001 - traversal simply means "not usable"
            return None
        if not resolved.is_file():
            return None
        if media_type_for(resolved) not in _ACCEPTED:
            return None
        return resolved

    def health_check(self) -> bool:
        """Local files alone cannot synthesise, so this is ``False``.

        Reported honestly: a caller seeing ``False`` knows to expect a
        silent render instead of discovering a fake voice track later.
        """
        return False

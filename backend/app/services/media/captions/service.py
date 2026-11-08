"""Writes caption files into a project and verifies what landed on disk.

The service is deliberately paranoid: after writing, each file is
re-read and re-parsed, and the parsed cues must equal the cues we
intended to write. ``CaptionFiles`` is only ever returned for files
that exist, are non-empty and round-trip — so a caller cannot be told a
caption file exists when it does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.services.media.captions.base import (
    CaptionCue,
    CaptionError,
    CaptionFormatError,
    CaptionFormatter,
    validate_cues,
)
from app.services.media.captions.srt import SRTFormatter, parse_srt
from app.services.media.captions.vtt import WebVTTFormatter, parse_vtt
from app.services.media.paths import safe_filename, validate_path


@dataclass(frozen=True, slots=True)
class CaptionFiles:
    """Verified facts about caption files that were actually written."""

    srt_path: str = ""
    vtt_path: str = ""
    formats: tuple[str, ...] = ()
    caption_count: int = 0
    duration_seconds: float = 0.0
    size_bytes: dict[str, int] = field(default_factory=dict)


class CaptionService:
    """Format cues and persist them under ``projects/<id>/captions/``.

    Args:
        projects_dir: the configured projects root; every output path
            must resolve inside it.
    """

    def __init__(
        self,
        projects_dir: Path,
        *,
        formatters: dict[str, CaptionFormatter] | None = None,
    ) -> None:
        self._projects_dir = Path(projects_dir)
        self._formatters: dict[str, CaptionFormatter] = formatters or {
            "srt": SRTFormatter(),
            "vtt": WebVTTFormatter(),
        }

    def available_formats(self) -> tuple[str, ...]:
        return tuple(sorted(self._formatters))

    def write(
        self,
        cues: Sequence[CaptionCue],
        *,
        project_id: str,
        formats: Sequence[str] = ("srt", "vtt"),
        stem: str = "captions",
    ) -> CaptionFiles:
        """Write ``cues`` in each requested format and verify the result.

        Raises:
            CaptionFormatError: the cue list is invalid or a written
                file does not parse back to exactly these cues.
            MediaPathError: the project id or file stem would escape the
                projects root (or is not a plain name).
            CaptionError: a file could not be written or came back
                empty.
        """
        validate_cues(cues)

        unknown = [fmt for fmt in formats if fmt not in self._formatters]
        if unknown:
            raise CaptionError(
                f"unsupported caption format(s): {', '.join(unknown)}; "
                f"available: {', '.join(sorted(self._formatters))}"
            )
        if not formats:
            raise CaptionError("no caption formats requested")

        safe_filename(stem)
        output_dir = validate_path(Path(project_id) / "captions", self._projects_dir)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CaptionError(
                f"cannot create caption directory {output_dir}: {exc}"
            ) from exc

        written: dict[str, int] = {}
        paths: dict[str, str] = {}
        for fmt in formats:
            formatter = self._formatters[fmt]
            content = formatter.format(cues)
            target = output_dir / f"{stem}.{formatter.extension}"
            try:
                with open(target, "w", encoding="utf-8", newline="") as handle:
                    handle.write(content)
            except OSError as exc:
                raise CaptionError(f"cannot write {target}: {exc}") from exc
            paths[fmt] = str(target)
            written[fmt] = self._verify(target, cues, fmt)

        last_end_ms = max((cue.end_ms for cue in cues), default=0)
        return CaptionFiles(
            srt_path=paths.get("srt", ""),
            vtt_path=paths.get("vtt", ""),
            formats=tuple(formats),
            caption_count=len(cues),
            duration_seconds=round(last_end_ms / 1000, 3),
            size_bytes=written,
        )

    def _verify(self, path: Path, cues: Sequence[CaptionCue], fmt: str) -> int:
        """Re-read, re-parse and compare a written file.

        Returns the file's size in bytes.
        """
        if not path.is_file():
            raise CaptionError(f"caption file was not created: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise CaptionError(f"caption file is empty: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CaptionError(f"cannot re-read {path}: {exc}") from exc

        parser = parse_srt if fmt == "srt" else parse_vtt
        try:
            parsed = parser(text)
        except CaptionFormatError as exc:
            raise CaptionError(
                f"written caption file {path} does not parse: {exc}"
            ) from exc
        if list(parsed) != list(cues):
            raise CaptionError(
                f"written caption file {path} does not match the cue list"
            )
        return size

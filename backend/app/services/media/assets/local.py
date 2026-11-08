"""Local-filesystem asset provider.

Supports images, video and audio files already on disk. Internet
search and AI generation are deliberately out of scope for Phase 7:
this provider's only job is to tell the truth about files that do or
do not exist, safely.
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from app.agents.video_producer.exceptions import VideoProducerAssetError
from app.agents.video_producer.grounding import validate_path
from app.agents.video_producer.schemas import AssetStatus, MediaAsset
from app.services.media.assets.base import AssetProvider

#: Extension → media type. Unsupported files are rejected, not guessed.
_MEDIA_TYPES: dict[str, str] = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".webm": "video",
    ".avi": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".flac": "audio",
}


def media_type_for(path: Path) -> str | None:
    """Return the media type for a file extension, or ``None``."""
    return _MEDIA_TYPES.get(path.suffix.lower())


def checksum_of(path: Path, *, limit_bytes: int = 64 * 1024 * 1024) -> str:
    """SHA-256 of a file, read in chunks.

    Files above ``limit_bytes`` hash their first and last megabyte plus
    the size, which is enough to detect swaps without reading a
    multi-gigabyte video end to end on a laptop.
    """
    digest = hashlib.sha256()
    size = path.stat().st_size
    if size > limit_bytes:
        with path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            handle.seek(max(size - 1024 * 1024, 0))
            digest.update(handle.read(1024 * 1024))
        digest.update(str(size).encode())
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class LocalAssetProvider(AssetProvider):
    """Resolves asset references confined to ``base_dir``."""

    name = "local"

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)

    def root(self) -> Path:
        return self._base

    def resolve(self, reference: str) -> MediaAsset:
        """Resolve ``reference`` (relative to the media root).

        Raises:
            VideoProducerAssetError: traversal attempt or unsupported
                file type. A *missing* file is not an error: it comes
                back with ``status=MISSING``.
        """
        if not reference or not reference.strip():
            raise VideoProducerAssetError("Empty asset reference")

        candidate = validate_path(reference, self._base)
        media_type = media_type_for(candidate)
        if media_type is None:
            raise VideoProducerAssetError(
                f"Unsupported asset type: {Path(reference).suffix!r}"
            )

        asset_id = candidate.stem
        if not candidate.is_file():
            return MediaAsset(
                asset_id=asset_id,
                filename=candidate.name,
                path="",
                media_type=media_type,
                status=AssetStatus.MISSING,
            )

        try:
            size = candidate.stat().st_size
            checksum = checksum_of(candidate)
        except OSError as exc:
            raise VideoProducerAssetError(f"Unreadable asset: {reference}") from exc

        return MediaAsset(
            asset_id=asset_id,
            filename=candidate.name,
            path=str(candidate),
            media_type=media_type,
            mime_type=mimetypes.guess_type(candidate.name)[0] or "",
            size_bytes=size,
            source="local",
            checksum=checksum,
            status=AssetStatus.AVAILABLE,
        )

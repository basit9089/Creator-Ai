"""Path and content safety helpers for the Video Producer.

Every filesystem path that reaches FFmpeg or the asset manager passes
through here first. The rule is simple: paths are resolved with their
symlinks and then must remain inside an allowed root. Anything else —
``..``, absolute paths, drive letters, symlink escapes — raises rather
than being "cleaned up", because silently rewriting a suspicious path
hides attacks instead of stopping them.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.video_producer.exceptions import VideoProducerAssetError


def validate_path(candidate: str | os.PathLike[str], base_dir: str | os.PathLike[str]) -> Path:
    """Resolve ``candidate`` and require it to live inside ``base_dir``.

    Args:
        candidate: path supplied by a caller (relative or absolute).
        base_dir: the only root the result may live under.

    Returns:
        The resolved absolute path.

    Raises:
        VideoProducerAssetError: the path escapes ``base_dir`` or its
            base cannot be resolved.
    """
    base = Path(base_dir)
    try:
        base_resolved = base.resolve(strict=False)
    except OSError as exc:  # pragma: no cover - unusual FS failure
        raise VideoProducerAssetError(f"Cannot resolve base directory {base!s}") from exc

    # Resolve without requiring existence so a not-yet-created output
    # file can still be validated before FFmpeg writes it.
    try:
        resolved = (base_resolved / candidate).resolve(strict=False) if not os.path.isabs(
            os.fspath(candidate)
        ) else Path(os.fspath(candidate)).resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise VideoProducerAssetError(f"Invalid path: {candidate!s}") from exc

    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise VideoProducerAssetError(f"Path escapes its base directory: {candidate!s}")
    return resolved


def ensure_within(candidate: str | os.PathLike[str], *bases: str | os.PathLike[str]) -> Path:
    """Validate ``candidate`` against any one of several allowed roots.

    Used where both the project directory and the shared media root are
    legitimate homes for a file.
    """
    last_error: VideoProducerAssetError | None = None
    for base in bases:
        try:
            return validate_path(candidate, base)
        except VideoProducerAssetError as exc:
            last_error = exc
    raise last_error or VideoProducerAssetError("No base directory supplied")


def safe_filename(name: str) -> str:
    """Return ``name`` if it is a plain filename, else raise.

    Prevents ``output_filename`` style inputs from carrying separators
    or traversal into the renderer.
    """
    if not name or name != os.path.basename(name) or name in {".", ".."}:
        raise VideoProducerAssetError(f"Unsafe filename: {name!r}")
    return name

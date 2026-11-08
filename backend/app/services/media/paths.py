"""Path-safety helpers shared by media-producing services.

Same rule as the Video Producer's grounding module: paths are resolved
(with their symlinks) and must remain inside an allowed root. Anything
else — ``..``, absolute paths, drive letters, symlink escapes — raises
rather than being "cleaned up", because silently rewriting a suspicious
path hides attacks instead of stopping them.

This is the canonical home for the helpers used by new services
(captions, thumbnails, future media stages). The Video Producer keeps
its own equivalent module so an approved phase is not refactored
underneath it.
"""

from __future__ import annotations

import os
from pathlib import Path


class MediaPathError(Exception):
    """A path escaped its allowed root or is not a safe name."""


def validate_path(candidate: str | os.PathLike[str], base_dir: str | os.PathLike[str]) -> Path:
    """Resolve ``candidate`` and require it to live inside ``base_dir``.

    Args:
        candidate: path supplied by a caller (relative or absolute).
        base_dir: the only root the result may live under.

    Returns:
        The resolved absolute path. The file does not need to exist yet,
        so a not-yet-written output can be validated before writing.

    Raises:
        MediaPathError: the path escapes ``base_dir`` or cannot be
            resolved.
    """
    base = Path(base_dir)
    try:
        base_resolved = base.resolve(strict=False)
    except OSError as exc:  # pragma: no cover - unusual FS failure
        raise MediaPathError(f"Cannot resolve base directory {base!s}") from exc

    try:
        raw = os.fspath(candidate)
        resolved = (
            (base_resolved / raw).resolve(strict=False)
            if not os.path.isabs(raw)
            else Path(raw).resolve(strict=False)
        )
    except (OSError, ValueError) as exc:
        raise MediaPathError(f"Invalid path: {candidate!s}") from exc

    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise MediaPathError(f"Path escapes its base directory: {candidate!s}")
    return resolved


def ensure_within(candidate: str | os.PathLike[str], *bases: str | os.PathLike[str]) -> Path:
    """Validate ``candidate`` against any one of several allowed roots.

    Used where both a project directory and the shared media root are
    legitimate homes for a file.
    """
    last_error: MediaPathError | None = None
    for base in bases:
        try:
            return validate_path(candidate, base)
        except MediaPathError as exc:
            last_error = exc
    raise last_error or MediaPathError("No base directory supplied")


def safe_filename(name: str) -> str:
    """Return ``name`` if it is a plain filename, else raise.

    Prevents caller-supplied names from carrying separators or traversal
    into the filesystem.
    """
    if not name or name != os.path.basename(name) or name in {".", ".."}:
        raise MediaPathError(f"Unsafe filename: {name!r}")
    if any(sep in name for sep in ("/", "\\")) or (len(name) > 1 and name[1] == ":"):
        raise MediaPathError(f"Unsafe filename: {name!r}")
    return name

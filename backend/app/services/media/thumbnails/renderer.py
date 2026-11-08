"""Text layout helpers for the local thumbnail renderer.

Pure functions over "font-like" and "draw-like" objects (anything
implementing ``getlength``/``getmetrics``), so layout logic is testable
without importing Pillow and reusable by any future renderer.

Every helper is bounded: text either fits the box or is explicitly
reported as truncated. Nothing here draws outside the caller's box.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: Default candidate TrueType fonts, most preferred first. Windows is
#: the primary target; the DejaVu/Arial paths cover common Linux/macOS
#: installs. All are small system fonts — no model download involved.
FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)

#: Optional bold faces, preferred for headlines when present.
BOLD_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\seguisb.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)

#: Line height multiplier relative to the font's own metrics.
LINE_SPACING = 1.18

#: Most lines a headline may occupy before the font shrinks further.
MAX_TITLE_LINES = 4

#: Ellipsis used when a title genuinely cannot fit.
ELLIPSIS = "…"


def find_font(candidates: tuple[str, ...] | list[str]) -> str | None:
    """Return the first candidate that exists, else ``None``."""
    from pathlib import Path

    for candidate in candidates:
        try:
            if Path(candidate).is_file():
                return candidate
        except OSError:  # pragma: no cover - unreadable font directory
            continue
    return None


def wrap_text(text: str, font: Any, max_width: float) -> list[str]:
    """Greedy word wrap measured in pixels with ``font``.

    Never drops characters: a single word wider than ``max_width`` sits
    alone on an overlong line for the caller to handle. Existing
    newlines in ``text`` are honoured as hard breaks.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}" if current else word
            if not current or font.getlength(candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [text.strip()]


def line_height(font: Any) -> int:
    """Rendered height of one line for ``font``, including spacing."""
    ascent, descent = font.getmetrics()
    return int((ascent + descent) * LINE_SPACING)


def truncate_to_width(text: str, font: Any, max_width: float) -> str:
    """Shorten ``text`` with an ellipsis until it measures within width."""
    if font.getlength(text) <= max_width:
        return text
    trimmed = text
    while trimmed and font.getlength(trimmed + ELLIPSIS) > max_width:
        trimmed = trimmed[:-1]
        if trimmed.endswith(" "):
            trimmed = trimmed.rstrip(" ")
    return (trimmed.rstrip() + ELLIPSIS) if trimmed else ELLIPSIS


def fit_text(
    text: str,
    *,
    load_font: Callable[[int], Any],
    max_width: float,
    max_height: float,
    max_size: int,
    min_size: int,
) -> tuple[Any, list[str], bool]:
    """Find the largest font size at which ``text`` fits the box.

    Args:
        text: the title to fit.
        load_font: callable ``load_font(size) -> font``.
        max_width/max_height: the box in pixels.
        max_size/min_size: font size search bounds (``max >= min``).

    Returns:
        ``(font, lines, truncated)``. ``truncated`` is True only when
        even the minimum size could not fit the text and content had to
        be cut — callers surface that honestly as a warning.
    """
    size = max_size
    while True:
        font = load_font(size)
        lines = wrap_text(text, font, max_width)
        widest = max((font.getlength(line) for line in lines), default=0.0)
        height = line_height(font) * len(lines)
        if widest <= max_width and height <= max_height and len(lines) <= MAX_TITLE_LINES:
            return font, lines, False
        if size <= min_size:
            break
        size = max(min_size, int(size * 0.92))

    # The text does not fit even at the minimum size: cut lines, then
    # characters, until the box is respected.
    font = load_font(min_size)
    lines = wrap_text(text, font, max_width)
    height = line_height(font)
    max_lines = max(1, int(max_height // height))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    lines = [truncate_to_width(line, font, max_width) for line in lines]
    # Signal the cut on the final line (single ellipsis, never doubled).
    last = lines[-1].removesuffix(ELLIPSIS).rstrip()
    lines[-1] = truncate_to_width(last + ELLIPSIS, font, max_width)
    return font, lines, True

"""Local, deterministic thumbnail renderer (Pillow only).

Honest description of what this is — and is not:

* It composes a **real PNG** from a background (flat gradient or an
  optional local image), accent shapes, wrapped title text and optional
  branding.
* It is **deterministic**: the same spec produces the same pixels.
* It is **not** an AI image generator. No model is downloaded, no GPU is
  used, and no artwork is "imagined" — typography over a gradient is
  what renders, and reports are worded accordingly.
"""

from __future__ import annotations

from pathlib import Path

from app.services.media.thumbnails.base import (
    RenderedThumbnail,
    ThumbnailError,
    ThumbnailRenderError,
    ThumbnailRenderer,
    ThumbnailSpec,
    ThumbnailUnavailableError,
)
from app.services.media.thumbnails.renderer import (
    BOLD_FONT_CANDIDATES,
    FONT_CANDIDATES,
    fit_text,
    find_font,
    line_height,
)

try:  # Pillow is a declared dependency; degrade clearly if it is absent.
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - only when the dependency is missing
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


class LocalThumbnailRenderer(ThumbnailRenderer):
    """Composes thumbnails with Pillow and system fonts.

    Args:
        font_candidates: override font search paths (used by tests to
            simulate a machine without fonts).
        font_bold_candidates: override bold font search paths.
    """

    name = "local"

    def __init__(
        self,
        *,
        font_candidates: tuple[str, ...] | list[str] | None = None,
        font_bold_candidates: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self._font_candidates = tuple(
            FONT_CANDIDATES if font_candidates is None else font_candidates
        )
        self._font_bold_candidates = tuple(
            BOLD_FONT_CANDIDATES if font_bold_candidates is None else font_bold_candidates
        )

    def health_check(self) -> bool:
        """True when Pillow imports and at least one usable font exists."""
        if not PIL_AVAILABLE:
            return False
        return find_font(self._font_candidates) is not None

    def unavailable_reason(self) -> str:
        """Human-readable reason :meth:`health_check` returns False."""
        if not PIL_AVAILABLE:
            return "Pillow is not installed; thumbnail rendering is unavailable"
        return (
            "no usable TrueType font found (searched "
            f"{len(self._font_candidates)} candidate paths)"
        )

    def render(self, spec: ThumbnailSpec, output_path: Path) -> RenderedThumbnail:
        """Render ``spec`` to ``output_path`` and verify the PNG."""
        if not PIL_AVAILABLE:
            raise ThumbnailUnavailableError(self.unavailable_reason())
        regular = find_font(self._font_candidates)
        if regular is None:
            raise ThumbnailUnavailableError(self.unavailable_reason())
        bold = find_font(self._font_bold_candidates) or regular

        width, height = spec.width, spec.height
        top_color = _hex_to_rgb(spec.background)
        bottom_color = _lerp(top_color, (0, 0, 0), 0.45)

        try:
            image = self._backdrop(spec, width, height, top_color, bottom_color)
        except ThumbnailError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised into a render error
            raise ThumbnailRenderError(f"could not build the backdrop: {exc}") from exc

        canvas = image.convert("RGBA")
        # A scrim keeps text legible over the backdrop. It must be
        # *composited*, not drawn in place: a semi-transparent rectangle
        # drawn onto the canvas would have its alpha channel discarded by
        # the final RGBA -> RGB save, blacking the backdrop out entirely.
        scrim = Image.new("RGBA", canvas.size, (0, 0, 0, 100))
        canvas = Image.alpha_composite(canvas, scrim)
        draw = ImageDraw.Draw(canvas)

        # Accent bar on the left edge.
        bar_width = max(6, width // 48)
        draw.rectangle((0, 0, bar_width, height), fill=(*_hex_to_rgb(spec.accent), 255))

        # Title: largest font that fits the box, never drawn outside it.
        box_left = bar_width + int(width * 0.045)
        box_right = int(width * 0.95)
        box_top = int(height * 0.10)
        box_bottom = int(height * 0.84)
        box_width = max(1, box_right - box_left)
        box_height = max(1, box_bottom - box_top)

        def load_font(size: int):
            return ImageFont.truetype(bold, size)

        font, lines, truncated = fit_text(
            spec.title,
            load_font=load_font,
            max_width=box_width,
            max_height=box_height,
            max_size=max(16, int(height * 0.17)),
            min_size=max(14, int(height * 0.05)),
        )
        stroke = max(1, font.size // 16)
        block_height = line_height(font) * len(lines)
        y = box_top + max(0, (box_height - block_height) // 2)
        for line in lines:
            draw.text(
                (box_left, y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=stroke,
                stroke_fill=(0, 0, 0, 255),
                anchor="la",
            )
            y += line_height(font)

        if spec.branding.strip():
            brand_font = ImageFont.truetype(regular, max(14, int(height * 0.04)))
            draw.text(
                (width - int(width * 0.03), height - int(height * 0.05)),
                spec.branding.strip(),
                font=brand_font,
                fill=(*_hex_to_rgb(spec.accent), 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0, 255),
                anchor="rb",
            )

        return self._save_and_verify(canvas, spec, output_path, truncated_title=truncated)

    # ---------- Internals ----------

    def _backdrop(
        self,
        spec: ThumbnailSpec,
        width: int,
        height: int,
        top_color: tuple[int, int, int],
        bottom_color: tuple[int, int, int],
    ):
        """Build the base layer: the supplied image, else a gradient."""
        if spec.image_path:
            source = Path(spec.image_path)
            if not source.is_file():
                raise ThumbnailRenderError(
                    f"thumbnail image does not exist: {source}"
                )
            with Image.open(source) as opened:
                opened.load()
                photo = opened.convert("RGB")
            # Cover-fit: scale up, then centre-crop to the canvas.
            scale = max(width / photo.width, height / photo.height)
            resized = photo.resize(
                (max(1, round(photo.width * scale)), max(1, round(photo.height * scale))),
                Image.LANCZOS,
            )
            left = max(0, (resized.width - width) // 2)
            top = max(0, (resized.height - height) // 2)
            return resized.crop((left, top, left + width, top + height))

        # Deterministic vertical gradient built row by row.
        rows: list[bytes] = []
        span = max(1, height - 1)
        for y in range(height):
            color = _lerp(top_color, bottom_color, y / span)
            rows.append(bytes(color) * width)
        return Image.frombytes("RGB", (width, height), b"".join(rows))

    def _save_and_verify(
        self, canvas: object, spec: ThumbnailSpec, output_path: Path, *, truncated_title: bool
    ) -> RenderedThumbnail:
        """Write the PNG, re-open it, and only then report success."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ThumbnailRenderError(
                f"cannot create thumbnail directory {output_path.parent}: {exc}"
            ) from exc

        try:
            rgba = canvas
            rgba.convert("RGB").save(output_path, format="PNG")
        except OSError as exc:
            raise ThumbnailRenderError(f"cannot write thumbnail {output_path}: {exc}") from exc

        # Verification: the file must exist, be non-empty, re-open as a
        # PNG and report the requested geometry.
        if not output_path.is_file():
            raise ThumbnailRenderError(f"thumbnail was not created: {output_path}")
        size_bytes = output_path.stat().st_size
        if size_bytes <= 0:
            raise ThumbnailRenderError(f"thumbnail file is empty: {output_path}")
        try:
            with Image.open(output_path) as check:
                check.load()
                actual_format = check.format
                actual_size = check.size
        except Exception as exc:  # noqa: BLE001 - corrupt output is a render failure
            raise ThumbnailRenderError(
                f"written thumbnail cannot be re-opened: {exc}"
            ) from exc
        if actual_format != "PNG":
            raise ThumbnailRenderError(
                f"written file is {actual_format}, not PNG"
            )
        if actual_size != (spec.width, spec.height):
            raise ThumbnailRenderError(
                f"written thumbnail is {actual_size[0]}x{actual_size[1]}, "
                f"expected {spec.width}x{spec.height}"
            )

        return RenderedThumbnail(
            path=str(output_path),
            filename=output_path.name,
            mime_type="image/png",
            width=actual_size[0],
            height=actual_size[1],
            size_bytes=size_bytes,
            renderer=self.name,
            truncated_title=truncated_title,
        )

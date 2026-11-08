"""Thumbnail rendering tests: real PNGs, deterministic bytes, honest failures.

No AI image generation exists in this phase, so what is under test is
the local Pillow renderer: it must produce a genuine 8-bit RGB PNG at
the requested geometry, deterministically, with the backdrop actually
visible behind the scrim, and it must fail *clearly* (never quietly)
when a required resource — Pillow, a font, a backdrop image — is
missing or broken.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image, ImageFont
from pydantic import ValidationError

from app.services.media.thumbnails import (
    LocalThumbnailRenderer,
    ThumbnailRenderError,
    ThumbnailSpec,
    ThumbnailUnavailableError,
)
from app.services.media.thumbnails import local as local_module
from app.services.media.thumbnails.renderer import (
    FONT_CANDIDATES,
    find_font,
    fit_text,
    line_height,
)


def render(
    renderer: LocalThumbnailRenderer, tmp_path: Path, spec: ThumbnailSpec, name: str = "thumb.png"
):
    output = tmp_path / name
    return renderer.render(spec, output), output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------


class TestThumbnailSpec:
    def test_defaults_are_youtube_shaped(self) -> None:
        spec = ThumbnailSpec(title="A title")
        assert (spec.width, spec.height) == (1280, 720)
        assert spec.background == "#0f172a"
        assert spec.accent == "#f59e0b"

    @pytest.mark.parametrize("width,height", [(50, 720), (1280, 50), (99999, 720), (1280, 99999)])
    def test_absurd_geometry_is_rejected(self, width: int, height: int) -> None:
        with pytest.raises(ValidationError):
            ThumbnailSpec(title="A title", width=width, height=height)

    @pytest.mark.parametrize("color", ["red", "#ff", "rgb(0,0,0)", "", "#ffff"])
    def test_non_hex_colors_are_rejected(self, color: str) -> None:
        with pytest.raises(ValidationError):
            ThumbnailSpec(title="A title", background=color)

    @pytest.mark.parametrize("title", ["", "   ", "\n\t"])
    def test_blank_title_is_rejected(self, title: str) -> None:
        with pytest.raises(ValidationError):
            ThumbnailSpec(title=title)

    def test_title_is_trimmed_and_colors_normalised(self) -> None:
        spec = ThumbnailSpec(title="  Spaced out  ", background="#ABCDEF")
        assert spec.title == "Spaced out"
        assert spec.background == "#abcdef"

    def test_overlong_branding_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThumbnailSpec(title="A title", branding="x" * 65)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


class TestLocalRenderer:
    def test_renders_a_real_png_at_the_default_geometry(self, tmp_path: Path) -> None:
        rendered, output = render(LocalThumbnailRenderer(), tmp_path, ThumbnailSpec(title="How YouTube Started"))

        assert output.is_file()
        assert output.stat().st_size > 0
        with Image.open(output) as opened:
            assert opened.format == "PNG"
            assert opened.mode == "RGB"
            assert opened.size == (1280, 720)

        assert rendered.width == 1280 and rendered.height == 720
        assert rendered.mime_type == "image/png"
        assert rendered.renderer == "local"
        assert rendered.filename == "thumb.png"
        assert Path(rendered.path).is_file()
        assert rendered.size_bytes == output.stat().st_size

    @pytest.mark.parametrize("size", [(640, 360), (1920, 1080), (160, 160)])
    def test_renders_the_requested_resolution(self, tmp_path: Path, size: tuple[int, int]) -> None:
        width, height = size
        rendered, output = render(
            LocalThumbnailRenderer(), tmp_path, ThumbnailSpec(title="Sized", width=width, height=height)
        )
        with Image.open(output) as opened:
            assert opened.size == (width, height)
        assert (rendered.width, rendered.height) == size

    def test_renders_are_byte_deterministic(self, tmp_path: Path) -> None:
        spec = ThumbnailSpec(title="How YouTube Started in 2005")
        renderer = LocalThumbnailRenderer()

        _, first = render(renderer, tmp_path, spec, "first.png")
        _, second = render(renderer, tmp_path, spec, "second.png")

        assert sha256(first) == sha256(second)

    def test_the_backdrop_is_visible_behind_the_scrim(self, tmp_path: Path) -> None:
        """Regression: the scrim must darken the gradient, not erase it."""
        _, output = render(
            LocalThumbnailRenderer(), tmp_path, ThumbnailSpec(title="Backdrop check")
        )
        with Image.open(output) as opened:
            image = opened.convert("RGB")
            top_right = image.getpixel((image.width - 10, 10))
            bottom_right = image.getpixel((image.width - 10, image.height - 10))

        assert top_right != (0, 0, 0), "scrim blacked out the backdrop"
        assert bottom_right != (0, 0, 0)
        assert top_right != bottom_right, "gradient is not visible"

    def test_accent_bar_is_painted(self, tmp_path: Path) -> None:
        _, output = render(LocalThumbnailRenderer(), tmp_path, ThumbnailSpec(title="Accent"))
        with Image.open(output) as opened:
            image = opened.convert("RGB")
            assert image.getpixel((1, image.height // 2)) == (245, 158, 11)

    def test_a_long_title_is_fitted_and_flagged(self, tmp_path: Path) -> None:
        long_title = ("extremely long title " * 30)[:500]
        spec = ThumbnailSpec(title=long_title, width=160, height=160)
        rendered, output = render(LocalThumbnailRenderer(), tmp_path, spec)

        assert rendered.truncated_title is True
        with Image.open(output) as opened:
            assert opened.size == (160, 160)

    def test_a_short_title_is_not_flagged(self, tmp_path: Path) -> None:
        rendered, _ = render(LocalThumbnailRenderer(), tmp_path, ThumbnailSpec(title="Short"))
        assert rendered.truncated_title is False

    def test_unicode_titles_render(self, tmp_path: Path) -> None:
        spec = ThumbnailSpec(title="Ünïcode — Héllo, ça va?")
        renderer = LocalThumbnailRenderer()
        _, first = render(renderer, tmp_path, spec, "first.png")
        _, second = render(renderer, tmp_path, spec, "second.png")
        assert sha256(first) == sha256(second)

    def test_branding_renders(self, tmp_path: Path) -> None:
        rendered, _ = render(
            LocalThumbnailRenderer(),
            tmp_path,
            ThumbnailSpec(title="Branded", branding="CREATOR AI"),
        )
        assert rendered.width == 1280

    def test_nested_output_directories_are_created(self, tmp_path: Path) -> None:
        output = tmp_path / "deep" / "nested" / "thumb.png"
        LocalThumbnailRenderer().render(ThumbnailSpec(title="Nested"), output)
        assert output.is_file()

    def test_a_photographic_backdrop_is_cover_fitted(self, tmp_path: Path) -> None:
        source = tmp_path / "source.png"
        Image.new("RGB", (300, 200), (255, 0, 0)).save(source)

        rendered, output = render(
            LocalThumbnailRenderer(),
            tmp_path,
            ThumbnailSpec(title="Photo", image_path=str(source)),
            "photo.png",
        )
        with Image.open(output) as opened:
            assert opened.size == (1280, 720)
            # Far right, away from the text box: the red photo must show
            # through the scrim rather than being replaced by it.
            red_pixel = opened.convert("RGB").getpixel((opened.width - 10, opened.height // 2))
        assert red_pixel[0] > red_pixel[1] and red_pixel[0] > red_pixel[2]

        # A photo backdrop produces different pixels than the gradient.
        _, gradient = render(LocalThumbnailRenderer(), tmp_path, ThumbnailSpec(title="Photo"), "gradient.png")
        assert sha256(output) != sha256(gradient)
        assert rendered.mime_type == "image/png"


# --------------------------------------------------------------------------
# Text fitting (pure, with the real font)
# --------------------------------------------------------------------------


class TestFitText:
    @staticmethod
    def loader(size: int):
        font_path = find_font(FONT_CANDIDATES)
        assert font_path is not None, "test machine has no usable TrueType font"
        return ImageFont.truetype(font_path, size)

    def test_text_that_fits_is_returned_untouched(self) -> None:
        font, lines, truncated = fit_text(
            "Short title",
            load_font=self.loader,
            max_width=600,
            max_height=200,
            max_size=72,
            min_size=24,
        )
        assert truncated is False
        assert "Short title" in " ".join(lines)
        assert font is not None

    def test_text_that_never_fits_stays_inside_the_box(self) -> None:
        font, lines, truncated = fit_text(
            "word " * 100,
            load_font=self.loader,
            max_width=300,
            max_height=120,
            max_size=60,
            min_size=16,
        )
        assert truncated is True
        assert lines
        assert all(font.getlength(line) <= 300 for line in lines)
        # The fitted block is inside the height budget of the box.
        assert line_height(font) * len(lines) <= 120


# --------------------------------------------------------------------------
# Honest failure modes
# --------------------------------------------------------------------------


class TestRendererFailures:
    def test_missing_fonts_fail_clearly(self, tmp_path: Path) -> None:
        renderer = LocalThumbnailRenderer(font_candidates=(), font_bold_candidates=())
        assert renderer.health_check() is False
        assert "font" in renderer.unavailable_reason()
        with pytest.raises(ThumbnailUnavailableError, match="font"):
            renderer.render(ThumbnailSpec(title="No fonts"), tmp_path / "thumb.png")
        assert not (tmp_path / "thumb.png").exists()

    def test_missing_pillow_fails_clearly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_module, "PIL_AVAILABLE", False)
        renderer = LocalThumbnailRenderer()
        assert renderer.health_check() is False
        assert "Pillow" in renderer.unavailable_reason()
        with pytest.raises(ThumbnailUnavailableError, match="Pillow"):
            renderer.render(ThumbnailSpec(title="No Pillow"), tmp_path / "thumb.png")

    def test_a_missing_backdrop_image_fails_the_render(self, tmp_path: Path) -> None:
        spec = ThumbnailSpec(title="Missing", image_path=str(tmp_path / "nope.png"))
        with pytest.raises(ThumbnailRenderError, match="does not exist"):
            LocalThumbnailRenderer().render(spec, tmp_path / "thumb.png")
        assert not (tmp_path / "thumb.png").exists()

    def test_a_corrupt_backdrop_image_fails_the_render(self, tmp_path: Path) -> None:
        source = tmp_path / "corrupt.png"
        source.write_bytes(b"this is not a png at all")
        spec = ThumbnailSpec(title="Corrupt", image_path=str(source))
        with pytest.raises(ThumbnailRenderError, match="backdrop"):
            LocalThumbnailRenderer().render(spec, tmp_path / "thumb.png")
        assert not (tmp_path / "thumb.png").exists()

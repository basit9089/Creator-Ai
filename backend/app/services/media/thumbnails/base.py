"""Thumbnail contracts: spec, result, renderer ABC.

The renderer is an interface so the local, deterministic Pillow
renderer of this phase can later be joined by other implementations
(cloud or AI image providers) without the agent changing. No AI image
generation exists in this phase — see :mod:`.local` for what actually
renders.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Thumbnail sizes are bounded so a typo cannot ask Pillow to allocate
#: an absurd canvas on a laptop.
_MIN_DIM = 160
_MAX_DIM = 7680

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class ThumbnailError(Exception):
    """Base class for thumbnail failures."""


class ThumbnailUnavailableError(ThumbnailError):
    """A required rendering resource (Pillow or a font) is missing."""


class ThumbnailRenderError(ThumbnailError):
    """Rendering or writing the PNG failed."""


class ThumbnailSpec(BaseModel):
    """Everything the local renderer needs to compose one thumbnail.

    Purely descriptive: the spec never touches the filesystem except
    through the optional ``image_path``, which callers must resolve and
    validate before the renderer sees it.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1, max_length=500)
    width: int = Field(default=1280, ge=_MIN_DIM, le=_MAX_DIM)
    height: int = Field(default=720, ge=_MIN_DIM, le=_MAX_DIM)
    background: str = "#0f172a"
    accent: str = "#f59e0b"
    branding: str = Field(default="", max_length=64)
    #: Optional existing local image to use as the backdrop. The caller
    #: validates containment and existence first.
    image_path: str | None = None

    @field_validator("background", "accent")
    @classmethod
    def _valid_hex(cls, value: str) -> str:
        if not _HEX_COLOR_RE.match(value):
            raise ValueError(f"color must look like #rrggbb, got {value!r}")
        return value.lower()

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must contain non-whitespace characters")
        return value.strip()


class RenderedThumbnail(BaseModel):
    """Verified facts about a PNG that actually exists on disk."""

    model_config = ConfigDict(frozen=True)

    path: str
    filename: str
    mime_type: str = "image/png"
    width: int
    height: int
    size_bytes: int
    renderer: str = ""
    truncated_title: bool = False


class ThumbnailRenderer(ABC):
    """Renders a :class:`ThumbnailSpec` to a real image file."""

    #: Short identifier reported in results, e.g. ``"local"``.
    name: str = "unknown"

    @abstractmethod
    def health_check(self) -> bool:
        """Whether the resources this renderer needs are available."""

    @abstractmethod
    def render(self, spec: ThumbnailSpec, output_path: Path) -> RenderedThumbnail:
        """Render ``spec`` into ``output_path`` and verify the result.

        Raises:
            ThumbnailUnavailableError: required resources are missing.
            ThumbnailRenderError: rendering or verification failed.
        """

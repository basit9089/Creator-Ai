"""Schemas for the Video Producer.

The Video Producer consumes a :class:`ScriptResult` from Phase 6 and
produces a structured plan (scenes, timeline, assets) plus, when FFmpeg
is available, a real rendered MP4. Nothing here performs I/O: every
model is a plain, validated description of what production *should*
look like, which is what lets the future frontend render progress,
missing assets and source attribution without guessing.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.scriptwriter.schemas import ScriptResult

#: Resolution strings look like ``1920x1080``. Bounded so a typo cannot
#: ask FFmpeg to allocate an absurd frame on a laptop.
_RESOLUTION_RE = re.compile(r"^([1-9]\d{0,4})x([1-9]\d{0,4})$")

#: Project identifiers become directory names, so they accept only a
#: conservative character set that cannot escape the storage root.
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

#: Output filenames are user-influenced; separators and traversal are
#: rejected outright rather than sanitised into something surprising.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,119}\.(mp4|mov|mkv)$")


class ProductionStatus(StrEnum):
    """Lifecycle of one video production run.

    Deliberately separate from :class:`~app.agents.base.AgentStatus`:
    the agent execution can be ``COMPLETED`` while the media pipeline
    still waits for assets (``ASSET_PENDING``).
    """

    DRAFT = "draft"
    PLANNING = "planning"
    ASSET_PENDING = "asset_pending"
    READY_TO_RENDER = "ready_to_render"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class VisualType(StrEnum):
    """What a scene needs visually. Descriptions only — never generated."""

    BACKGROUND = "background"
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    IMAGE_SEQUENCE = "image_sequence"
    NONE = "none"


class AssetStatus(StrEnum):
    """Honest status of an asset reference.

    ``MISSING`` means "no such file exists"; nothing in this phase
    ever reports an asset as produced when it was not.
    """

    PLANNED = "planned"
    AVAILABLE = "available"
    MISSING = "missing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MediaAsset(BaseModel):
    """Metadata for a local media file, existing or not.

    Fields mirror what the frontend will need for previews and
    attribution. ``status`` tells the truth: a planned-but-absent image
    is ``MISSING`` with an empty ``path``.
    """

    model_config = ConfigDict(frozen=True)

    asset_id: str
    filename: str = ""
    path: str = ""
    media_type: str = ""  # image | video | audio
    mime_type: str = ""
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    size_bytes: int = 0
    source: str = "local"
    attribution: str = ""
    checksum: str = ""
    status: AssetStatus = AssetStatus.PLANNED


class Scene(BaseModel):
    """One scene of the planned video.

    Kept intentionally data-heavy and behaviour-free: Phase 8 (captions,
    thumbnails) and the frontend both read these fields, so claim and
    source references travel with the narration they support.
    """

    model_config = ConfigDict(frozen=True)

    scene_id: str
    order: int = 0
    heading: str = ""
    narration: str = ""
    duration_seconds: float = 0.0
    visual_type: VisualType = VisualType.BACKGROUND
    visual_prompt: str = ""
    asset_requirements: tuple[dict, ...] = ()
    asset_references: tuple[MediaAsset, ...] = ()
    transition: str = ""
    background: str = "#101010"
    claim_references: tuple[dict, ...] = ()
    source_references: tuple[dict, ...] = ()


class VoiceoverSettings(BaseModel):
    """Voiceover configuration for a production run.

    ``provider`` is informational in Phase 7: no TTS engine ships yet,
    so an absent voice track is reported as pending rather than faked.
    """

    model_config = ConfigDict(frozen=True)

    provider: str = "none"  # none | local_file
    voice: str = "default"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    #: Optional existing local audio file (absolute path inside the
    #: configured media root). Absent means "silent render".
    audio_file: str | None = None


class BackgroundSettings(BaseModel):
    """Default background for scenes that do not request anything else."""

    model_config = ConfigDict(frozen=True)

    color: str = "#101010"
    image_path: str | None = None


class BrandingSettings(BaseModel):
    """Optional branding overlay. Text only; no asset generation."""

    model_config = ConfigDict(frozen=True)

    watermark: str = ""


class VideoProductionRequest(BaseModel):
    """Input to ``POST /api/video/produce``.

    Defaults are YouTube-shaped (1080p, 30 fps, 16:9) but live here
    once, so no other module hard-codes them.
    """

    model_config = ConfigDict(frozen=True)

    script: ScriptResult
    project_id: str = "default"
    output_format: str = "mp4"
    resolution: str = "1920x1080"
    fps: int = Field(default=30, ge=1, le=120)
    aspect_ratio: str = "16:9"
    include_subtitles: bool = False
    output_filename: str | None = None
    asset_directory: str | None = None
    voiceover: VoiceoverSettings = Field(default_factory=VoiceoverSettings)
    background: BackgroundSettings = Field(default_factory=BackgroundSettings)
    branding: BrandingSettings = Field(default_factory=BrandingSettings)

    @field_validator("script")
    @classmethod
    def _script_not_empty(cls, value: ScriptResult) -> ScriptResult:
        if not (value.sections or value.narration or value.hook):
            raise ValueError("script contains no narratable content")
        return value

    @field_validator("resolution")
    @classmethod
    def _valid_resolution(cls, value: str) -> str:
        match = _RESOLUTION_RE.match(value.strip())
        if not match:
            raise ValueError(f"resolution must look like 1920x1080, got {value!r}")
        width, height = int(match.group(1)), int(match.group(2))
        if width > 7680 or height > 4320:
            raise ValueError("resolution exceeds 7680x4320")
        return f"{width}x{height}"

    @field_validator("output_format")
    @classmethod
    def _valid_format(cls, value: str) -> str:
        allowed = {"mp4", "mov", "mkv"}
        normalised = value.strip().lower()
        if normalised not in allowed:
            raise ValueError(f"output_format must be one of {sorted(allowed)}")
        return normalised

    @field_validator("aspect_ratio")
    @classmethod
    def _valid_aspect_ratio(cls, value: str) -> str:
        allowed = {"16:9", "9:16", "1:1", "4:3"}
        if value.strip() not in allowed:
            raise ValueError(f"aspect_ratio must be one of {sorted(allowed)}")
        return value.strip()

    @field_validator("project_id")
    @classmethod
    def _valid_project_id(cls, value: str) -> str:
        if not _PROJECT_ID_RE.match(value):
            raise ValueError(
                "project_id must be 1-64 characters of letters, digits, "
                "'-' or '_' and start with a letter or digit"
            )
        return value

    @field_validator("output_filename")
    @classmethod
    def _valid_filename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _FILENAME_RE.match(value):
            raise ValueError(
                "output_filename must be a simple name ending in .mp4, "
                ".mov or .mkv without path separators"
            )
        return value

    @field_validator("asset_directory")
    @classmethod
    def _valid_asset_directory(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Reject traversal syntactically: resolved containment is also
        # checked at use time, but a suspicious path should fail early.
        if ".." in value or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
            raise ValueError("asset_directory must be a relative path inside the media root")
        return value

    @property
    def resolution_size(self) -> tuple[int, int]:
        """Parsed ``(width, height)`` — always valid after validation."""
        match = _RESOLUTION_RE.match(self.resolution)
        assert match is not None  # noqa: S101 - guaranteed by validator
        return int(match.group(1)), int(match.group(2))


class TimelineScene(BaseModel):
    """A scene placed on the timeline with deterministic times."""

    model_config = ConfigDict(frozen=True)

    scene_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    visual: dict = Field(default_factory=dict)
    audio: dict = Field(default_factory=dict)
    overlays: tuple[dict, ...] = ()


class Timeline(BaseModel):
    """The full schedule for one render.

    Times are rounded to milliseconds by the builder so repeated runs
    over the same script produce byte-identical schedules.
    """

    model_config = ConfigDict(frozen=True)

    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration_seconds: float = 0.0
    scenes: tuple[TimelineScene, ...] = ()


class VideoProductionResult(BaseModel):
    """Everything a caller (or the future frontend) needs to display.

    Every status field is honest: ``asset_status`` reflects whether
    files actually exist, ``audio_status`` whether a real audio stream
    was produced, and ``COMPLETED`` is only ever set after the output
    was probed successfully.
    """

    model_config = ConfigDict(frozen=True)

    project_id: str = "default"
    status: ProductionStatus = ProductionStatus.DRAFT
    scenes: tuple[Scene, ...] = ()
    timeline: Timeline | None = None
    output_path: str = ""
    output_filename: str = ""
    duration_seconds: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    file_size: int = 0
    codec: str = ""
    audio_status: str = "none"  # none | silent | present | pending
    asset_status: str = "unknown"  # none | available | missing
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    generated_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())

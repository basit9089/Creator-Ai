"""Schemas for the Captions / Thumbnail / SEO agent.

The request consumes Phase 6 and Phase 7 outputs; the result is shaped
for the future frontend: every artifact carries a real path (or none),
measurable dimensions/sizes, an honest per-artifact status and an error
code when that artifact failed. Nothing in here fabricates metadata —
``title_score`` is documented as a *local heuristic*, and image fields
stay empty until a real PNG exists.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.scriptwriter.schemas import ScriptResult
from app.agents.video_producer.schemas import VideoProductionResult

#: Same conservative shape the Video Producer uses: identifiers become
#: directory names, so only a safe character set is accepted.
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

#: ``1280x720``-style strings. Bounded so a typo cannot request an
#: absurd canvas; the thumbnail's own default is YouTube's 1280x720.
_RESOLUTION_RE = re.compile(r"^([1-9]\d{0,4})x([1-9]\d{0,4})$")

#: Caption formats this phase implements.
_ALLOWED_CAPTION_FORMATS = frozenset({"srt", "vtt"})


class ArtifactStatus(StrEnum):
    """Honest status of one artifact within the run."""

    COMPLETED = "completed"
    FAILED = "failed"


class CaptionsThumbnailStatus(StrEnum):
    """Overall outcome.

    ``PARTIAL`` means some artifacts were produced and verified while
    others failed — the result says *exactly* which is which.
    """

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class CaptionsInfo(BaseModel):
    """Caption files that exist on disk, or an honest failure."""

    model_config = ConfigDict(frozen=True)

    status: ArtifactStatus = ArtifactStatus.FAILED
    srt_path: str = ""
    vtt_path: str = ""
    formats: tuple[str, ...] = ()
    caption_count: int = 0
    duration_seconds: float = 0.0
    #: Where timing came from: ``video_timeline``,
    #: ``video_scene_durations`` or ``script_wpm``.
    timing_source: str = ""
    #: Bytes per written file, keyed by format (``{"srt": 249, ...}``).
    size_bytes: dict[str, int] = Field(default_factory=dict)
    error_code: str = ""


class ThumbnailInfo(BaseModel):
    """A real PNG and its measurable properties, or an honest failure."""

    model_config = ConfigDict(frozen=True)

    status: ArtifactStatus = ArtifactStatus.FAILED
    asset_id: str = ""
    path: str = ""
    filename: str = ""
    mime_type: str = ""
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    #: Which renderer produced it (``local`` in this phase — there is
    #: no AI image generation).
    renderer: str = ""
    error_code: str = ""


class MetadataInfo(BaseModel):
    """YouTube title/description/tags/SEO built from the script only."""

    model_config = ConfigDict(frozen=True)

    status: ArtifactStatus = ArtifactStatus.FAILED
    title: str = ""
    alternate_titles: tuple[str, ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()
    primary_keyword: str = ""
    secondary_keywords: tuple[str, ...] = ()
    search_phrases: tuple[str, ...] = ()
    #: Local heuristic in [0, 100] — length, keyword presence and
    #: clickbait markers. Not an external ranking system; see
    #: ``app.agents.captions_thumbnail.metadata``.
    title_score: int | None = None
    description_keywords: tuple[str, ...] = ()
    content_category: str = ""
    language: str = "en"
    audience: str = "general"
    #: Where the metadata JSON was persisted ("" if not written).
    path: str = ""
    error_code: str = ""


class ClaimRef(BaseModel):
    """Traceability: a claim, its sources, and the sections using it."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    source_ids: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()


class SourceRef(BaseModel):
    """A source exactly as the script carries it."""

    model_config = ConfigDict(frozen=True)

    id: str
    url: str
    domain: str = ""


class GroundingInfo(BaseModel):
    """Links metadata back through script → claim → source.

    Preserved from the Scriptwriter's own references: nothing here is
    newly researched or upgraded.
    """

    model_config = ConfigDict(frozen=True)

    claim_refs: tuple[ClaimRef, ...] = ()
    source_refs: tuple[SourceRef, ...] = ()


class CaptionsThumbnailRequest(BaseModel):
    """Input to ``POST /api/captions-thumbnail``."""

    model_config = ConfigDict(frozen=True)

    script: ScriptResult
    #: Optional Phase 7 result. When present its scene timing drives
    #: the captions; otherwise 150 WPM script timing is used.
    video: VideoProductionResult | None = None
    project_id: str = "default"
    target_audience: str = "general"
    language: str = Field(default="en", min_length=2, max_length=8)
    caption_formats: tuple[str, ...] = ("srt", "vtt")
    thumbnail_resolution: str = "1280x720"
    #: Optional existing image under the media root (relative path).
    thumbnail_image: str | None = None
    #: Optional small branding text drawn into the thumbnail.
    watermark: str = Field(default="", max_length=64)

    @field_validator("script")
    @classmethod
    def _script_not_empty(cls, value: ScriptResult) -> ScriptResult:
        if not (value.sections or value.narration or value.hook):
            raise ValueError("script contains no narratable content")
        return value

    @field_validator("project_id")
    @classmethod
    def _valid_project_id(cls, value: str) -> str:
        if not _PROJECT_ID_RE.match(value):
            raise ValueError(
                "project_id must be 1-64 characters of letters, digits, "
                "'-' or '_' and start with a letter or digit"
            )
        return value

    @field_validator("target_audience")
    @classmethod
    def _valid_audience(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("target_audience must not be blank")
        return stripped[:120]

    @field_validator("caption_formats")
    @classmethod
    def _valid_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("caption_formats must request at least one format")
        unknown = [fmt for fmt in value if fmt not in _ALLOWED_CAPTION_FORMATS]
        if unknown:
            raise ValueError(
                f"unsupported caption format(s): {unknown}; "
                f"allowed: {sorted(_ALLOWED_CAPTION_FORMATS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError("caption_formats contains duplicates")
        return tuple(fmt.strip().lower() for fmt in value)

    @field_validator("thumbnail_resolution")
    @classmethod
    def _valid_resolution(cls, value: str) -> str:
        match = _RESOLUTION_RE.match(value.strip())
        if not match:
            raise ValueError(
                f"thumbnail_resolution must look like 1280x720, got {value!r}"
            )
        width, height = int(match.group(1)), int(match.group(2))
        if width < 160 or height < 160:
            raise ValueError("thumbnail_resolution must be at least 160x160")
        if width > 7680 or height > 4320:
            raise ValueError("thumbnail_resolution exceeds 7680x4320")
        return f"{width}x{height}"

    @field_validator("thumbnail_image")
    @classmethod
    def _valid_image(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Reject traversal syntactically here; resolved containment is
        # checked again at use time.
        if ".." in value or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
            raise ValueError(
                "thumbnail_image must be a relative path inside the media root"
            )
        return value

    @property
    def thumbnail_size(self) -> tuple[int, int]:
        """Parsed ``(width, height)`` — always valid after validation."""
        match = _RESOLUTION_RE.match(self.thumbnail_resolution)
        assert match is not None  # noqa: S101 - guaranteed by validator
        return int(match.group(1)), int(match.group(2))


class CaptionsThumbnailResult(BaseModel):
    """Everything produced, honestly reported.

    ``success`` is true only when *every* requested artifact completed.
    A ``partial`` result still carries the real paths, sizes and
    dimensions of what did complete — plus ``errors`` naming exactly
    what failed and why.
    """

    model_config = ConfigDict(frozen=True)

    project_id: str = "default"
    status: CaptionsThumbnailStatus = CaptionsThumbnailStatus.FAILED
    success: bool = False
    captions: CaptionsInfo = Field(default_factory=CaptionsInfo)
    thumbnail: ThumbnailInfo = Field(default_factory=ThumbnailInfo)
    metadata: MetadataInfo = Field(default_factory=MetadataInfo)
    grounding: GroundingInfo = Field(default_factory=GroundingInfo)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    generated_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())

"""Captions / Thumbnail / SEO Agent — the fifth production stage.

Pipeline::

    ScriptResult (+ optional VideoProductionResult)
        ├─ captions:  segments → cues → SRT/WebVTT files   (deterministic)
        ├─ metadata:  AIProvider → validated + grounded    (one AI call)
        └─ thumbnail: local Pillow renderer → real PNG     (deterministic)

The agent orchestrates; the caption service writes files; the renderer
renders. AI is used *only* for the YouTube metadata half (title,
description, tags, SEO) and only through the ``AIProvider`` abstraction
— there is no Ollama import here, and ``DEFAULT_AI_THINKING=false``
continues to apply.

Honesty rules, same as the Video Producer: an artifact is ``completed``
only when its file exists, is non-empty and was re-read/verified. When
some artifacts succeed and others fail the result is ``partial`` with
per-artifact statuses and error codes; when *nothing* was produced the
agent fails outright.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agents.base import Agent, AgentContext, AgentOutput, Capability
from app.agents.captions_thumbnail.exceptions import (
    CaptionsThumbnailAIError,
    CaptionsThumbnailAssetError,
    CaptionsThumbnailCaptionError,
    CaptionsThumbnailError,
    CaptionsThumbnailInputError,
    CaptionsThumbnailOutputError,
    CaptionsThumbnailRenderError,
)
from app.agents.captions_thumbnail.metadata import (
    MetadataValidationError,
    YouTubeMetadataDraft,
    build_grounding,
    compose_metadata,
)
from app.agents.captions_thumbnail.prompts import SYSTEM_PROMPT, USER_PROMPT
from app.agents.captions_thumbnail.schemas import (
    ArtifactStatus,
    CaptionsInfo,
    CaptionsThumbnailRequest,
    CaptionsThumbnailResult,
    CaptionsThumbnailStatus,
    GroundingInfo,
    MetadataInfo,
    ThumbnailInfo,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.prompts.builder import PromptBuilder
from app.services.ai.provider import (
    AIProvider,
    AIProviderError,
    GenerationOptions,
    Message,
    StructuredOutputError,
)
from app.services.media.captions import (
    CaptionError,
    CaptionService,
    Segment,
    build_cues,
    segments_from_script,
    segments_from_video,
)
from app.services.media.paths import MediaPathError, validate_path
from app.services.media.thumbnails import (
    LocalThumbnailRenderer,
    ThumbnailError,
    ThumbnailRenderer,
    ThumbnailSpec,
)

logger = get_logger("agents.captions_thumbnail")

#: Key under which the caller places the request in the agent context.
REQUEST_KEY = "captions_thumbnail_request"

#: Cap on narration passed into the metadata prompt, so a very long
#: script cannot blow the model's context. Output stays bounded by the
#: schema limits regardless.
MAX_PROMPT_NARRATION_CHARS = 8000

#: Conservative generation settings for metadata (see PART 12).
_METADATA_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 1500,
}


@dataclass
class _Outcome:
    """Result of producing one artifact, whatever happened."""

    info: CaptionsInfo | ThumbnailInfo | MetadataInfo
    warnings: list[str] = field(default_factory=list)
    #: Client-safe entry for the result's ``errors`` tuple.
    error_entry: str | None = None
    #: Typed error to raise if the *whole* run fails.
    error: CaptionsThumbnailError | None = None


def _code_for(exc: BaseException, *, default: str = "captions_thumbnail_error") -> str:
    """Best machine-readable code for an exception (never invented)."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(exc, MediaPathError):
        return "captions_thumbnail_asset_error"
    if isinstance(exc, ThumbnailError):
        # Service-level renderer failures (a crashing renderer double,
        # an unwritable canvas) are render failures, not generic ones.
        return "captions_thumbnail_render_error"
    if isinstance(exc, CaptionError):
        return "captions_thumbnail_caption_error"
    if isinstance(exc, MetadataValidationError):
        return "structured_output_invalid"
    if isinstance(exc, (CaptionsThumbnailRenderError, CaptionsThumbnailAssetError)):
        return exc.code
    return default


class CaptionsThumbnailAgent(Agent):
    """Produces captions, a local thumbnail and SEO metadata."""

    name = "captions_thumbnail"
    version = "1.0.0"
    description = "Generates captions, a local thumbnail and SEO metadata for a finished script."
    capabilities = frozenset({Capability.STRUCTURED_OUTPUT, Capability.RETRYABLE})

    def __init__(
        self,
        *,
        provider: AIProvider | None = None,
        renderer: ThumbnailRenderer | None = None,
        caption_service: CaptionService | None = None,
        settings: Settings | None = None,
        thinking: bool = False,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._renderer = renderer
        self._caption_service = caption_service
        self._settings = settings
        self._thinking_default = thinking

    # ---------- Collaborators (injectable for tests) ----------

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def provider(self) -> AIProvider:
        """The configured provider, resolved lazily and typed as the abstraction."""
        if self._provider is None:
            from app.dependencies import get_ai_provider

            self._provider = get_ai_provider()
        return self._provider

    @property
    def renderer(self) -> ThumbnailRenderer:
        if self._renderer is None:
            self._renderer = LocalThumbnailRenderer()
        return self._renderer

    @property
    def caption_service(self) -> CaptionService:
        if self._caption_service is None:
            self._caption_service = CaptionService(self.settings.projects_dir)
        return self._caption_service

    # ---------- Execution ----------

    async def run(self, context: AgentContext, messages: Sequence[Message]) -> AgentOutput:
        request = self._load_request(context)
        warnings: list[str] = []
        errors: list[str] = []

        grounding = build_grounding(request.script)
        segments, timing_source = self._caption_segments(request)

        logger.info(
            "captions_thumbnail_start project_id=%s video=%s formats=%s",
            request.project_id,
            "yes" if request.video else "no",
            ",".join(request.caption_formats),
        )

        captions = self._produce_captions(request, segments, timing_source)
        # Metadata before the thumbnail: the thumbnail headlines the
        # final, selected title when it exists.
        metadata = self._produce_metadata(request, context, grounding)
        thumbnail = self._produce_thumbnail(request, metadata.info)

        for outcome in (captions, metadata, thumbnail):
            warnings.extend(outcome.warnings)
            if outcome.error_entry:
                errors.append(outcome.error_entry)

        outcomes = (captions, metadata, thumbnail)
        completed = sum(
            1 for outcome in outcomes if outcome.info.status is ArtifactStatus.COMPLETED
        )
        if completed == len(outcomes):
            status, success = CaptionsThumbnailStatus.COMPLETED, True
        elif completed:
            status, success = CaptionsThumbnailStatus.PARTIAL, False
            logger.warning(
                "captions_thumbnail_partial project_id=%s errors=%s",
                request.project_id,
                errors,
            )
        else:
            first_error = next(
                (outcome.error for outcome in outcomes if outcome.error is not None),
                CaptionsThumbnailOutputError("unknown failure"),
            )
            summary = "; ".join(errors) or "no details"
            raise type(first_error)(f"No artifacts were produced. {summary}")

        result = CaptionsThumbnailResult(
            project_id=request.project_id,
            status=status,
            success=success,
            captions=captions.info,  # type: ignore[arg-type]
            thumbnail=thumbnail.info,  # type: ignore[arg-type]
            metadata=metadata.info,  # type: ignore[arg-type]
            grounding=grounding,
            warnings=tuple(dict.fromkeys(warnings)),  # dedupe, keep order
            errors=tuple(errors),
        )
        logger.info(
            "captions_thumbnail_completed project_id=%s status=%s captions=%d "
            "thumbnail_bytes=%d title=%r",
            result.project_id,
            result.status,
            result.captions.caption_count,
            result.thumbnail.size_bytes,
            result.metadata.title,
        )
        return AgentOutput.ok(data=result)

    # ---------- Artifact producers ----------

    def _caption_segments(
        self, request: CaptionsThumbnailRequest
    ) -> tuple[list[Segment], str]:
        """Resolve caption timing: video timing when present, else 150 WPM."""
        try:
            if request.video is not None:
                segments = segments_from_video(request.video)
                timing_source = (
                    "video_timeline"
                    if request.video.timeline is not None
                    else "video_scene_durations"
                )
            else:
                segments = segments_from_script(request.script)
                timing_source = "script_wpm"
        except CaptionError as exc:
            raise CaptionsThumbnailInputError(
                f"Cannot derive caption timing: {exc}"
            ) from exc
        return segments, timing_source

    def _produce_captions(
        self,
        request: CaptionsThumbnailRequest,
        segments: Sequence[Segment],
        timing_source: str,
    ) -> _Outcome:
        try:
            cues = build_cues(segments)
            files = self.caption_service.write(
                cues,
                project_id=request.project_id,
                formats=request.caption_formats,
            )
        except Exception as exc:  # noqa: BLE001 - normalised into an artifact failure
            code = _code_for(exc)
            typed: CaptionsThumbnailError
            if isinstance(exc, MediaPathError):
                typed = CaptionsThumbnailAssetError(f"Caption path rejected: {exc}")
            elif isinstance(exc, CaptionsThumbnailError):
                typed = exc
            else:
                typed = CaptionsThumbnailCaptionError(f"Caption generation failed: {exc}")
            logger.warning("captions_failed project_id=%s error=%s", request.project_id, exc)
            return _Outcome(
                info=CaptionsInfo(
                    status=ArtifactStatus.FAILED,
                    timing_source=timing_source,
                    error_code=code,
                ),
                error_entry=f"captions: {code}: {exc}",
                error=typed,
            )

        return _Outcome(
            info=CaptionsInfo(
                status=ArtifactStatus.COMPLETED,
                srt_path=files.srt_path,
                vtt_path=files.vtt_path,
                formats=files.formats,
                caption_count=files.caption_count,
                duration_seconds=files.duration_seconds,
                timing_source=timing_source,
                size_bytes=files.size_bytes,
            )
        )

    def _produce_metadata(
        self,
        request: CaptionsThumbnailRequest,
        context: AgentContext,
        grounding: GroundingInfo,
    ) -> _Outcome:
        script = request.script
        warnings: list[str] = []

        narration_parts = [
            part
            for part in (
                script.hook,
                script.introduction,
                script.narration,
                script.conclusion,
                script.cta,
                *(section.narration for section in script.sections),
            )
            if part and part.strip()
        ]
        narration = "\n\n".join(narration_parts)
        if len(narration) > MAX_PROMPT_NARRATION_CHARS:
            narration = narration[:MAX_PROMPT_NARRATION_CHARS] + "\n[truncated]"
            warnings.append(
                "Script narration was truncated for the metadata prompt "
                f"({MAX_PROMPT_NARRATION_CHARS} character cap)."
            )

        claims = (
            "\n".join(
                f"- {claim.claim_id} (status: {claim.status or 'unspecified'}; "
                f"sources: {', '.join(claim.source_ids) or 'none'}"
                + (f"; qualification: {claim.qualification}" if claim.qualification else "")
                + ")"
                for claim in script.claims_used
            )
            or "None"
        )
        source_domains = (
            ", ".join(
                dict.fromkeys(
                    str(source.get("domain", "")).strip()
                    for source in script.sources
                    if str(source.get("domain", "")).strip()
                )
            )
            or "None"
        )

        prompt = PromptBuilder().with_system(SYSTEM_PROMPT.text).with_user(
            USER_PROMPT.render(
                topic=script.title or "YouTube video",
                target_audience=request.target_audience,
                language=request.language,
                narration=narration or "(empty)",
                claims=claims,
                source_domains=source_domains,
            ).text
        )
        options = GenerationOptions(
            think=self._resolve_thinking(context),
            **_METADATA_OPTIONS,
        )

        try:
            draft = self.provider.generate_structured(
                prompt.build(), YouTubeMetadataDraft, options=options
            )
        except Exception as exc:  # noqa: BLE001 - normalised into an artifact failure
            code = _code_for(exc, default="captions_thumbnail_ai_error")
            logger.warning(
                "metadata_ai_failed project_id=%s code=%s", request.project_id, code
            )
            return _Outcome(
                info=MetadataInfo(
                    status=ArtifactStatus.FAILED,
                    language=request.language,
                    audience=request.target_audience,
                    error_code=code,
                ),
                warnings=warnings,
                error_entry=f"metadata: {code}: {exc}",
                error=CaptionsThumbnailAIError(
                    f"Metadata generation failed: {exc}", code=code
                ),
            )

        try:
            info, meta_warnings = compose_metadata(
                draft,
                script=script,
                language=request.language,
                audience=request.target_audience,
            )
        except MetadataValidationError as exc:
            code = _code_for(exc)
            logger.warning(
                "metadata_invalid project_id=%s reason=%s", request.project_id, exc
            )
            return _Outcome(
                info=MetadataInfo(
                    status=ArtifactStatus.FAILED,
                    language=request.language,
                    audience=request.target_audience,
                    error_code=code,
                ),
                warnings=warnings,
                error_entry=f"metadata: {code}: {exc}",
                error=CaptionsThumbnailAIError(f"AI metadata was unusable: {exc}"),
            )

        warnings.extend(meta_warnings)

        # Persist the metadata JSON, then report status from the disk
        # truth: content that could not be written is not a completed
        # artifact (it is still returned in the response, unclaimed).
        try:
            path = self._write_metadata_json(request.project_id, info, grounding)
        except Exception as exc:  # noqa: BLE001 - persistence failure
            code = _code_for(exc)
            if isinstance(exc, MediaPathError):
                code = "captions_thumbnail_asset_error"
            logger.warning(
                "metadata_write_failed project_id=%s error=%s", request.project_id, exc
            )
            return _Outcome(
                info=info.model_copy(
                    update={"status": ArtifactStatus.FAILED, "error_code": code}
                ),
                warnings=warnings
                + [
                    "Metadata was generated but its JSON file could not be "
                    "written; the metadata travels in the response only."
                ],
                error_entry=f"metadata: {code}: {exc}",
                error=CaptionsThumbnailOutputError(
                    f"Metadata file could not be written: {exc}"
                ),
            )

        return _Outcome(
            info=info.model_copy(
                update={"status": ArtifactStatus.COMPLETED, "path": path}
            ),
            warnings=warnings,
        )

    def _produce_thumbnail(
        self,
        request: CaptionsThumbnailRequest,
        metadata: MetadataInfo,
    ) -> _Outcome:
        warnings: list[str] = []
        # Prefer the final, selected metadata title. When metadata failed
        # before proposing one, fall back to the script's own title —
        # loudly, so the caller knows the headline is not the chosen one.
        headline = metadata.title.strip() or request.script.title.strip()
        if not metadata.title.strip() and request.script.title.strip():
            warnings.append(
                "Thumbnail uses the script title because metadata "
                "generation failed."
            )
        if not headline:
            code = "captions_thumbnail_render_error"
            return _Outcome(
                info=ThumbnailInfo(error_code=code),
                error_entry=f"thumbnail: {code}: no title available for the thumbnail",
                error=CaptionsThumbnailRenderError(
                    "No title available for the thumbnail"
                ),
            )

        # Optional local image: containment is enforced, and a missing
        # file degrades honestly (render without it, with a warning).
        image_path: str | None = None
        if request.thumbnail_image:
            try:
                resolved = validate_path(request.thumbnail_image, self.settings.media_dir)
            except MediaPathError as exc:
                # Syntactic traversal is rejected by the request schema;
                # anything that still escapes here is a caller error.
                raise CaptionsThumbnailAssetError(
                    f"thumbnail_image escapes the media directory: {exc}"
                ) from exc
            if resolved.is_file():
                image_path = str(resolved)
            else:
                warnings.append(
                    f"Supplied thumbnail_image {request.thumbnail_image!r} does "
                    "not exist; rendering without it."
                )

        renderer = self.renderer
        if not renderer.health_check():
            reason = getattr(renderer, "unavailable_reason", lambda: "unavailable")()
            code = "captions_thumbnail_render_error"
            logger.warning("thumbnail_unavailable reason=%s", reason)
            return _Outcome(
                info=ThumbnailInfo(error_code=code),
                warnings=warnings,
                error_entry=f"thumbnail: {code}: {reason}",
                error=CaptionsThumbnailRenderError(reason),
            )

        width, height = request.thumbnail_size
        spec = ThumbnailSpec(
            title=headline,
            width=width,
            height=height,
            branding=request.watermark,
            image_path=image_path,
        )
        try:
            output_dir = validate_path(
                Path(request.project_id) / "thumbnails", self.settings.projects_dir
            )
            output_path = output_dir / "thumbnail.png"
            rendered = renderer.render(spec, output_path)
            # Belt and braces: the file we were handed must sit inside
            # the configured projects root.
            validate_path(rendered.path, self.settings.projects_dir)
        except Exception as exc:  # noqa: BLE001 - normalised into an artifact failure
            code = _code_for(exc)
            if isinstance(exc, MediaPathError):
                typed: CaptionsThumbnailError = CaptionsThumbnailAssetError(
                    f"Thumbnail path rejected: {exc}"
                )
            elif isinstance(exc, CaptionsThumbnailError):
                typed = exc
            else:
                typed = CaptionsThumbnailRenderError(f"Thumbnail rendering failed: {exc}")
            logger.warning(
                "thumbnail_failed project_id=%s error=%s", request.project_id, exc
            )
            return _Outcome(
                info=ThumbnailInfo(error_code=code),
                warnings=warnings,
                error_entry=f"thumbnail: {code}: {exc}",
                error=typed,
            )

        if rendered.truncated_title:
            warnings.append("The thumbnail title was shortened to fit the canvas.")

        asset_id = "thumb-" + hashlib.sha256(rendered.path.encode("utf-8")).hexdigest()[:12]
        return _Outcome(
            info=ThumbnailInfo(
                status=ArtifactStatus.COMPLETED,
                asset_id=asset_id,
                path=rendered.path,
                filename=rendered.filename,
                mime_type=rendered.mime_type,
                width=rendered.width,
                height=rendered.height,
                size_bytes=rendered.size_bytes,
                renderer=rendered.renderer,
            ),
            warnings=warnings,
        )

    # ---------- Internals ----------

    def _write_metadata_json(
        self, project_id: str, info: MetadataInfo, grounding: GroundingInfo
    ) -> str:
        """Persist metadata + grounding as JSON and verify the file."""
        output_dir = validate_path(Path(project_id) / "metadata", self.settings.projects_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CaptionsThumbnailOutputError(
                f"cannot create metadata directory {output_dir}: {exc}"
            ) from exc

        payload: dict[str, Any] = {
            "project_id": project_id,
            "metadata": info.model_dump(
                mode="json", exclude={"path", "status", "error_code"}
            ),
            "grounding": grounding.model_dump(mode="json"),
        }
        target = output_dir / "metadata.json"
        try:
            with open(target, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as exc:
            raise CaptionsThumbnailOutputError(
                f"cannot write {target}: {exc}"
            ) from exc

        if not target.is_file() or target.stat().st_size <= 0:
            raise CaptionsThumbnailOutputError(
                f"metadata file was not written correctly: {target}"
            )
        return str(target)

    def _load_request(self, context: AgentContext) -> CaptionsThumbnailRequest:
        raw = context.config.get(REQUEST_KEY)
        if raw is None:
            raise CaptionsThumbnailInputError("No captions/thumbnail request supplied")
        if isinstance(raw, CaptionsThumbnailRequest):
            return raw
        try:
            return CaptionsThumbnailRequest.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - pydantic errors are the signal
            raise CaptionsThumbnailInputError(
                f"Invalid captions/thumbnail request: {exc}"
            ) from exc

    def _resolve_thinking(self, context: AgentContext) -> bool:
        val = context.config.get("thinking")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in {"true", "1", "yes", "on"}
        return self._thinking_default

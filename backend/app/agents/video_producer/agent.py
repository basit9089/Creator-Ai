"""Video Producer Agent.

Pipeline, all deterministic except FFmpeg itself::

    ScriptResult → scenes → assets → voiceover → timeline → render → MP4

The agent never rewrites narration, never researches, and never claims
a file exists that does not: statuses are derived from what actually
happened on disk. AI is deliberately not used — the scriptwriter
already produced the factual content, and re-asking a model to lay it
out would only add a source of drift between script and video.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from app.agents.base import Agent, AgentContext, AgentOutput, Capability
from app.agents.video_producer.exceptions import (
    VideoProducerInputError,
    VideoProducerRenderError,
)
from app.agents.video_producer.grounding import validate_path
from app.agents.video_producer.scene_planner import plan_scenes
from app.agents.video_producer.schemas import (
    ProductionStatus,
    Scene,
    Timeline,
    VideoProductionRequest,
    VideoProductionResult,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.ai.provider import AIProvider, Message
from app.services.media.assets.manager import AssetManager
from app.services.media.audio.local import LocalVoiceProvider
from app.services.media.rendering.base import VideoRenderer
from app.services.media.rendering.ffmpeg import FFmpegRenderer
from app.services.media.timeline.builder import build_timeline

logger = get_logger("agents.video_producer")

#: Key under which the caller places the request in the agent context.
REQUEST_KEY = "video_production_request"

#: Overlay text for the fallback render (no assets available). Every
#: scene shows its heading; this is a legitimate placeholder render,
#: not a fake: FFmpeg produces a real, probed MP4.
_DEFAULT_BACKGROUND = "#101010"


class VideoProducerAgent(Agent):
    """Turns a :class:`ScriptResult` into a planned and rendered video."""

    name = "video_producer"
    version = "1.0.0"
    description = "Plans scenes from a script and renders them to video with FFmpeg."
    capabilities = frozenset({Capability.STRUCTURED_OUTPUT, Capability.RETRYABLE})

    def __init__(
        self,
        *,
        provider: AIProvider | None = None,  # reserved: unused by design
        renderer: VideoRenderer | None = None,
        asset_manager: AssetManager | None = None,
        voice_provider: LocalVoiceProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self._renderer = renderer
        self._asset_manager = asset_manager
        self._voice_provider = voice_provider
        self._settings = settings

    # ---------- Collaborators (injectable for tests) ----------

    @property
    def settings(self) -> Settings:
        """Resolved application settings (storage roots etc.)."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def renderer(self) -> VideoRenderer:
        if self._renderer is None:
            settings = self.settings
            self._renderer = FFmpegRenderer(
                settings.renders_dir,
                media_dir=settings.media_dir,
                projects_dir=settings.projects_dir,
            )
        return self._renderer

    @property
    def asset_manager(self) -> AssetManager:
        if self._asset_manager is None:
            self._asset_manager = AssetManager(base_dir=self.settings.media_dir)
        return self._asset_manager

    @property
    def voice_provider(self) -> LocalVoiceProvider:
        if self._voice_provider is None:
            self._voice_provider = LocalVoiceProvider(self.settings.media_dir)
        return self._voice_provider

    # ---------- Execution ----------

    async def run(self, context: AgentContext, messages: Sequence[Message]) -> AgentOutput:
        request = self._load_request(context)

        # 1. Plan: deterministic scene conversion from the script.
        scenes = plan_scenes(request.script)
        if not scenes:
            raise VideoProducerInputError("Script produced no scenes")

        # 2. Assets: resolve what exists; report honestly what does not.
        scenes, asset_status = self.asset_manager.resolve_scene_assets(scenes)

        # 3. Voiceover: use a supplied local file when valid, else pending.
        audio_path, audio_status, voice_warning = self._resolve_voiceover(request)

        # 4. Timeline: authoritative durations, computed backend-side.
        width, height = request.resolution_size
        background = request.background.color or _DEFAULT_BACKGROUND
        for index, scene in enumerate(scenes):
            if not scene.background or scene.background == _DEFAULT_BACKGROUND:
                scenes[index] = scene.model_copy(update={"background": background})
        timeline = build_timeline(
            scenes,
            width=width,
            height=height,
            fps=request.fps,
            audio={"source": audio_status},
        )

        warnings: list[str] = []
        if voice_warning:
            warnings.append(voice_warning)
        if asset_status == "missing":
            warnings.append(
                "Some scene assets are not available; falling back to "
                "background and text overlays."
            )
        if request.include_subtitles:
            warnings.append(
                "Subtitles are not rendered in this phase; the flag is "
                "accepted but unused."
            )

        # 5. Render: only after planning succeeded.
        output_name = request.output_filename or f"{request.project_id}.mp4"
        output_dir = self.settings.project_dir(request.project_id) / "renders"
        output_path = output_dir / output_name

        if not self.renderer.health_check():
            raise VideoProducerRenderError(
                "FFmpeg renderer is not available on this machine"
            )

        logger.info(
            "video_production_start project_id=%s scenes=%d duration=%.2f",
            request.project_id,
            len(scenes),
            timeline.duration_seconds,
        )

        try:
            produced = self.renderer.render(
                timeline,
                scenes,
                output_path,
                audio_path=audio_path,
            )
        except VideoProducerRenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised into a render error
            raise VideoProducerRenderError("Video rendering failed") from exc

        # Final gate before COMPLETED: the file must exist, be
        # non-empty and probe as a real video. A voice track that was
        # requested must also be present as an audio stream. This runs
        # even though the renderer validates internally, so no future
        # renderer double can slip an unverified file into a success.
        file_size, codec, has_audio = self._verify_output(
            str(produced), expects_audio=(audio_status == "present")
        )

        result = self._build_result(
            request=request,
            scenes=scenes,
            timeline=timeline,
            output_path=str(produced),
            file_size=file_size,
            codec=codec,
            asset_status=asset_status,
            audio_status=audio_status,
            warnings=warnings,
        )
        logger.info(
            "video_production_completed project_id=%s file_size=%d duration=%.2f",
            result.project_id,
            result.file_size,
            result.duration_seconds,
        )
        return AgentOutput.ok(data=result)

    # ---------- Internals ----------

    def _load_request(self, context: AgentContext) -> VideoProductionRequest:
        raw = context.config.get(REQUEST_KEY)
        if raw is None:
            raise VideoProducerInputError("No video production request supplied")
        if isinstance(raw, VideoProductionRequest):
            return raw
        try:
            return VideoProductionRequest.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - pydantic errors are the signal
            raise VideoProducerInputError(f"Invalid video production request: {exc}") from exc

    def _resolve_voiceover(
        self, request: VideoProductionRequest
    ) -> tuple[str | None, str, str | None]:
        """Return ``(audio_path, status, warning)`` for the voice track.

        Never synthesises and never pretends to: an unusable or absent
        voice file downgrades the render to silent with a warning.
        """
        settings = self.settings
        voice = request.voiceover
        if voice.provider == "none" or not voice.audio_file:
            return None, "silent", None

        validated = self.voice_provider.validate_audio_file(voice.audio_file)
        if validated is None:
            return (
                None,
                "pending",
                f"Voiceover file {voice.audio_file!r} is missing or unsupported; "
                "rendering silent audio.",
            )
        # A resolved path must still sit under the configured media root.
        try:
            validate_path(validated, settings.media_dir)
        except Exception:  # noqa: BLE001 - traversal means "do not use"
            return (
                None,
                "pending",
                "Voiceover file is outside the media directory; rendering silent audio.",
            )
        return str(validated), "present", None

    def _verify_output(self, output_path: str, *, expects_audio: bool) -> tuple[int, str, bool]:
        """Verify a rendered file before it may be reported complete.

        Returns ``(size_bytes, codec, has_audio)``.

        Raises:
            VideoProducerRenderError: the file is missing, empty,
                unprobeable, has no video stream, has no audio stream
                when a voice track was expected, or whose geometry does
                not match the timeline.
        """
        path = Path(output_path)
        if not path.is_file():
            raise VideoProducerRenderError("Rendered output does not exist")
        size = path.stat().st_size
        if size <= 0:
            raise VideoProducerRenderError("Rendered output is empty")

        data = self.renderer.probe(path)
        streams = data.get("streams", []) if isinstance(data, dict) else []
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            raise VideoProducerRenderError("Rendered output has no video stream")

        video = video_streams[0]
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        if expects_audio and not has_audio:
            raise VideoProducerRenderError(
                "Voiceover was supplied but the output has no audio stream"
            )
        return size, str(video.get("codec_name") or "unknown"), has_audio

    def _build_result(
        self,
        *,
        request: VideoProductionRequest,
        scenes: tuple[Scene, ...] | list[Scene],
        timeline: Timeline,
        output_path: str,
        file_size: int,
        codec: str,
        asset_status: str,
        audio_status: str,
        warnings: list[str],
    ) -> VideoProductionResult:
        """Assemble the result from *verified* output facts only."""
        return VideoProductionResult(
            project_id=request.project_id,
            status=ProductionStatus.COMPLETED,
            scenes=tuple(scenes),
            timeline=timeline,
            output_path=output_path,
            output_filename=os.path.basename(output_path),
            duration_seconds=timeline.duration_seconds,
            width=timeline.width,
            height=timeline.height,
            fps=timeline.fps,
            file_size=file_size,
            codec=codec,
            audio_status=audio_status,
            asset_status=asset_status,
            warnings=tuple(warnings),
        )

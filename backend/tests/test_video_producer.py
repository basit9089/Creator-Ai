"""Tests for the Video Producer: schemas, planning, timeline, agent, registry.

The through-line of every test here is honesty: statuses must reflect
what actually happened on disk, narration and claim/source references
must survive the conversion untouched, and nothing may report
``COMPLETED`` unless a real, probed video file exists.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agents.base import AgentContext, PlaceholderAgent
from app.agents.registry import (
    IMPLEMENTED_AGENT_NAMES,
    build_default_registry,
)
from app.agents.runner import AgentRunner
from app.agents.scriptwriter.schemas import ScriptClaimReference, ScriptResult, Section
from app.agents.video_producer.agent import REQUEST_KEY, VideoProducerAgent
from app.agents.video_producer.exceptions import (
    VideoProducerInputError,
    VideoProducerRenderError,
)
from app.agents.video_producer.scene_planner import (
    MAX_SCENE_SECONDS,
    MIN_SCENE_SECONDS,
    plan_scenes,
)
from app.agents.video_producer.schemas import (
    AssetStatus,
    ProductionStatus,
    VideoProductionRequest,
    VisualType,
)
from app.core.config import Settings
from app.services.media.rendering.base import VideoRenderer
from app.services.media.rendering.ffmpeg import FONT_CANDIDATES, FFmpegRenderer

# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

#: Tiny geometry so real renders finish in well under a second.
TINY = "320x180"


def make_script(**overrides: object) -> ScriptResult:
    """A small but complete script: hook, two sections, conclusion, CTA."""
    defaults: dict[str, object] = {
        "title": "Test Video",
        "hook": "Here is the hook.",
        "introduction": "Welcome to the introduction.",
        "sections": (
            Section(
                section_id="s1",
                heading="First Section",
                narration="one two three four five six seven eight nine ten",
                claim_references=(
                    {"claim_id": "c1", "source_ids": ("src1", "src2")},
                ),
            ),
            Section(
                section_id="s2",
                heading="Second Section",
                narration="alpha beta gamma",
            ),
        ),
        "conclusion": "In conclusion, it works.",
        "cta": "Subscribe now!",
        "sources": (
            {"id": "src1", "url": "https://a.example/1", "domain": "a.example"},
            {"id": "src2", "url": "https://b.example/2", "domain": "b.example"},
            {"id": "src9", "url": "https://c.example/9", "domain": "c.example"},
        ),
        "words_per_minute": 150,
    }
    defaults.update(overrides)
    return ScriptResult(**defaults)  # type: ignore[arg-type]


def make_request(**overrides: object) -> VideoProductionRequest:
    payload: dict[str, object] = {"script": make_script(), "project_id": "test"}
    payload.update(overrides)
    return VideoProductionRequest(**payload)  # type: ignore[arg-type]


class FakeRenderer(VideoRenderer):
    """Deterministic renderer double for agent tests without FFmpeg."""

    name = "fake"

    def __init__(
        self,
        *,
        healthy: bool = True,
        write_file: bool = True,
        streams: list[dict] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.healthy = healthy
        self.write_file = write_file
        self.fail_with = fail_with
        self.streams = streams if streams is not None else [
            {"codec_type": "video", "codec_name": "h264", "width": 320, "height": 180}
        ]
        self.render_calls = 0

    def health_check(self) -> bool:
        return self.healthy

    def probe(self, path: Path) -> dict:
        return {"streams": self.streams, "format": {"duration": "1.0"}}

    def render(
        self,
        timeline: object,
        scenes: object,
        output_path: Path,
        *,
        audio_path: Path | None = None,
    ) -> Path:
        self.render_calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        if self.write_file:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x00" * 64)
        return output_path


def run_agent(
    agent: VideoProducerAgent,
    request: VideoProductionRequest | None,
    *,
    tmp_path: Path | None = None,
) -> object:
    """Execute the agent through the real AgentRunner."""
    config = {} if request is None else {REQUEST_KEY: request}
    return asyncio.run(AgentRunner().run(agent, AgentContext(config=config), []))


# --------------------------------------------------------------------------
# Request schema validation
# --------------------------------------------------------------------------


class TestVideoProductionRequest:
    def test_defaults_are_youtube_shaped(self) -> None:
        request = make_request()
        assert request.resolution == "1920x1080"
        assert request.resolution_size == (1920, 1080)
        assert request.fps == 30
        assert request.output_format == "mp4"
        assert request.aspect_ratio == "16:9"
        assert request.project_id == "test"

    @pytest.mark.parametrize("bad", ["bad", "1920", "1920X1080", "0x100", "wide", ""])
    def test_invalid_resolution_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_request(resolution=bad)

    @pytest.mark.parametrize("bad", ["99999x99999", "7681x1080", "1920x4321"])
    def test_resolution_beyond_8k_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_request(resolution=bad)

    @pytest.mark.parametrize("bad", [0, 121, -30])
    def test_fps_outside_1_120_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            make_request(fps=bad)

    @pytest.mark.parametrize("bad", ["avi", "gif", "mp4?x", ""])
    def test_output_format_must_be_mp4_mov_mkv(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_request(output_format=bad)

    def test_output_format_is_normalised(self) -> None:
        assert make_request(output_format="  MP4 ").output_format == "mp4"

    @pytest.mark.parametrize("bad", ["21:9", "16_9", ""])
    def test_aspect_ratio_must_be_one_of_the_four(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_request(aspect_ratio=bad)

    @pytest.mark.parametrize(
        "bad",
        ["../evil", "/abs", "a/b", "", ".hidden", "x" * 65, "..", "C:evil"],
    )
    def test_project_id_cannot_escape_storage(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_request(project_id=bad)

    @pytest.mark.parametrize(
        "bad",
        ["../../x.mp4", "/abs/x.mp4", "sub/x.mp4", "x.avi", "x.mp4.exe", "", "..mp4"],
    )
    def test_output_filename_rejects_traversal_and_wrong_types(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_request(output_filename=bad)

    @pytest.mark.parametrize(
        "bad", ["../outside", "/abs", "C:/win", "..\\win", ".."]
    )
    def test_asset_directory_rejects_traversal(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_request(asset_directory=bad)

    @pytest.mark.parametrize("bad", [{}, {"sections": (), "narration": "", "hook": ""}])
    def test_script_without_narratable_content_is_rejected(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            VideoProductionRequest(script=bad)

    def test_missing_script_field_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            VideoProductionRequest()  # type: ignore[call-arg]

    @pytest.mark.parametrize("speed", [0.1, 3.0])
    def test_voiceover_speed_is_bounded(self, speed: float) -> None:
        with pytest.raises(ValidationError):
            make_request(
                voiceover={"speed": speed}  # type: ignore[arg-type]
            )

    def test_resolution_size_parses_after_validation(self) -> None:
        assert make_request(resolution="640x360").resolution_size == (640, 360)


# --------------------------------------------------------------------------
# Scene planning (deterministic conversion, no LLM)
# --------------------------------------------------------------------------


class TestScenePlanning:
    def test_full_script_covers_every_part_in_order(self) -> None:
        scenes = plan_scenes(make_script())
        headings = [s.heading for s in scenes]
        assert headings == [
            "Test Video",  # hook
            "Introduction",
            "First Section",
            "Second Section",
            "Conclusion",
            "Call to action",
        ]
        assert [s.scene_id for s in scenes] == [f"scene_{i:03d}" for i in range(1, 7)]
        assert [s.order for s in scenes] == [1, 2, 3, 4, 5, 6]

    def test_narration_is_preserved_verbatim(self) -> None:
        script = make_script()
        scenes = plan_scenes(script)
        by_heading = {s.heading: s for s in scenes}
        assert by_heading["Test Video"].narration == script.hook
        assert by_heading["Introduction"].narration == script.introduction
        assert by_heading["First Section"].narration == script.sections[0].narration
        assert by_heading["Conclusion"].narration == script.conclusion
        assert by_heading["Call to action"].narration == script.cta

    def test_durations_follow_150_wpm_word_counts(self) -> None:
        scenes = plan_scenes(make_script())
        by_heading = {s.heading: s for s in scenes}
        # 10 words at 150 WPM = 4.0 seconds.
        assert by_heading["First Section"].duration_seconds == 4.0
        # 3 words at 150 WPM = 1.2 seconds.
        assert by_heading["Second Section"].duration_seconds == 1.2

    def test_short_text_gets_the_minimum_duration(self) -> None:
        scenes = plan_scenes(make_script(hook="Go!"))
        assert scenes[0].duration_seconds == MIN_SCENE_SECONDS

    def test_duration_is_bounded_at_the_maximum(self) -> None:
        huge = " ".join(["word"] * 1500)  # 10 minutes at 150 WPM
        scenes = plan_scenes(
            make_script(
                hook="",
                introduction="",
                sections=(),
                conclusion="",
                cta="",
                narration=huge,
            )
        )
        assert len(scenes) == 1
        assert scenes[0].duration_seconds == MAX_SCENE_SECONDS

    def test_wpm_from_the_script_is_used(self) -> None:
        scenes = plan_scenes(make_script(words_per_minute=300))
        by_heading = {s.heading: s for s in scenes}
        assert by_heading["First Section"].duration_seconds == 2.0

    def test_section_claims_are_preserved_verbatim(self) -> None:
        scenes = plan_scenes(make_script())
        section = next(s for s in scenes if s.heading == "First Section")
        assert section.claim_references == (
            {"claim_id": "c1", "source_ids": ("src1", "src2")},
        )
        # The section with no claims gets none invented for it.
        bare = next(s for s in scenes if s.heading == "Second Section")
        assert bare.claim_references == ()

    def test_source_references_are_narrowed_to_cited_sources(self) -> None:
        scenes = plan_scenes(make_script())
        section = next(s for s in scenes if s.heading == "First Section")
        cited = [src["id"] for src in section.source_references]
        assert cited == ["src1", "src2"]  # src9 is never cited here
        # Scenes without claims do not inherit the whole bibliography.
        hook = next(s for s in scenes if s.heading == "Test Video")
        assert hook.source_references == ()

    def test_visual_types_are_production_notes_not_assets(self) -> None:
        scenes = plan_scenes(make_script())
        types = {s.heading: s.visual_type for s in scenes}
        assert types["Test Video"] is VisualType.TEXT
        assert types["Introduction"] is VisualType.BACKGROUND
        assert types["First Section"] is VisualType.IMAGE
        assert types["Call to action"] is VisualType.TEXT
        # Image scenes carry an asset requirement (a prompt, no path).
        image_scene = next(s for s in scenes if s.visual_type is VisualType.IMAGE)
        assert image_scene.asset_requirements
        assert "path" not in image_scene.asset_requirements[0]
        # Text/background scenes require no external asset at all.
        hook = next(s for s in scenes if s.visual_type is VisualType.TEXT)
        assert hook.asset_requirements == ()

    def test_flat_narration_script_becomes_one_body_scene(self) -> None:
        script = make_script(
            hook="",
            introduction="",
            sections=(),
            conclusion="",
            cta="",
            narration="Flat narration words here.",
            claims_used=(
                ScriptClaimReference(claim_id="c9", source_ids=("src1",)),
            ),
        )
        scenes = plan_scenes(script)
        assert len(scenes) == 1
        assert scenes[0].narration == "Flat narration words here."
        # Script-level claims are the only attribution available and are kept.
        assert scenes[0].claim_references == (
            {
                "claim_id": "c9",
                "source_ids": ("src1",),
                "usage": "",
                "status": "",
                "qualification": "",
            },
        )
        assert [src["id"] for src in scenes[0].source_references] == ["src1"]

    def test_transition_is_fade_after_the_first_scene(self) -> None:
        scenes = plan_scenes(make_script())
        assert scenes[0].transition == ""
        assert all(s.transition == "fade" for s in scenes[1:])

    def test_planning_is_deterministic(self) -> None:
        script = make_script()
        first = plan_scenes(script)
        second = plan_scenes(script)
        assert first == second

    def test_empty_script_yields_no_scenes(self) -> None:
        script = ScriptResult(title="")
        assert plan_scenes(script) == []


# --------------------------------------------------------------------------
# Timeline builder
# --------------------------------------------------------------------------


class TestTimeline:
    @pytest.fixture()
    def scenes(self) -> list[object]:
        return plan_scenes(make_script())

    def test_scenes_are_back_to_back_without_gaps_or_overlaps(self, scenes: list) -> None:
        from app.services.media.timeline.builder import build_timeline

        timeline = build_timeline(scenes, width=640, height=360, fps=25)
        previous_end = 0.0
        for tscene in timeline.scenes:
            assert tscene.start_time == previous_end
            assert tscene.end_time > tscene.start_time
            previous_end = tscene.end_time
        assert timeline.duration_seconds == previous_end

    def test_total_duration_is_the_sum_of_scene_durations(self, scenes: list) -> None:
        from app.services.media.timeline.builder import build_timeline

        timeline = build_timeline(scenes, width=640, height=360, fps=25)
        assert timeline.duration_seconds == pytest.approx(
            sum(s.duration_seconds for s in scenes), abs=0.001
        )

    def test_geometry_and_fps_are_carried_through(self, scenes: list) -> None:
        from app.services.media.timeline.builder import build_timeline

        timeline = build_timeline(scenes, width=640, height=360, fps=25)
        assert (timeline.width, timeline.height, timeline.fps) == (640, 360, 25)

    def test_timeline_is_deterministic(self, scenes: list) -> None:
        from app.services.media.timeline.builder import build_timeline

        first = build_timeline(scenes, width=640, height=360, fps=25)
        second = build_timeline(scenes, width=640, height=360, fps=25)
        assert first == second

    def test_overlays_carry_the_heading(self, scenes: list) -> None:
        from app.services.media.timeline.builder import build_timeline

        timeline = build_timeline(scenes)
        assert timeline.scenes[0].overlays[0]["text"] == "Test Video"

    def test_empty_scene_list_yields_zero_duration(self) -> None:
        from app.services.media.timeline.builder import build_timeline

        timeline = build_timeline([])
        assert timeline.duration_seconds == 0.0
        assert timeline.scenes == ()


# --------------------------------------------------------------------------
# Agent pipeline (with injected renderer — no FFmpeg needed)
# --------------------------------------------------------------------------


class TestVideoProducerAgent:
    def test_missing_request_is_an_input_error(self) -> None:
        agent = VideoProducerAgent(renderer=FakeRenderer())
        result = run_agent(agent, None)
        assert not result.success
        assert result.metadata.get("code") == "video_producer_invalid_request"

    def test_unusable_request_dict_is_an_input_error(self) -> None:
        agent = VideoProducerAgent(renderer=FakeRenderer())
        result = run_agent(agent, {"resolution": "banana"})  # type: ignore[arg-type]
        assert not result.success
        assert result.metadata.get("code") == "video_producer_invalid_request"

    def test_happy_path_reports_completed_with_verified_file(
        self, tmp_path: Path
    ) -> None:
        renderer = FakeRenderer()
        agent = VideoProducerAgent(
            renderer=renderer,
            settings=Settings(storage_root=str(tmp_path)),
        )
        result = run_agent(agent, make_request())

        assert result.success
        data = result.data
        assert data.status is ProductionStatus.COMPLETED
        assert data.file_size > 0
        assert data.codec == "h264"
        assert Path(data.output_path).is_file()
        assert data.output_path.startswith(str(tmp_path))
        assert renderer.render_calls == 1

    def test_narration_and_claims_survive_the_pipeline(self, tmp_path: Path) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(),
            settings=Settings(storage_root=str(tmp_path)),
        )
        data = run_agent(agent, make_request()).data
        by_heading = {s.heading: s for s in data.scenes}
        assert by_heading["First Section"].narration == (
            "one two three four five six seven eight nine ten"
        )
        assert by_heading["First Section"].claim_references == (
            {"claim_id": "c1", "source_ids": ("src1", "src2")},
        )
        assert [src["id"] for src in by_heading["First Section"].source_references] == [
            "src1",
            "src2",
        ]

    def test_asset_status_is_honest_when_assets_are_only_planned(
        self, tmp_path: Path
    ) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(),
            settings=Settings(storage_root=str(tmp_path)),
        )
        data = run_agent(agent, make_request()).data
        # Image scenes asked for assets no one has supplied: honest "missing".
        assert data.asset_status == "missing"
        assert any("assets" in w for w in data.warnings)
        image_scene = next(s for s in data.scenes if s.visual_type is VisualType.IMAGE)
        assert image_scene.asset_references[0].status is AssetStatus.PLANNED

    def test_audio_defaults_to_silent_and_says_so(self, tmp_path: Path) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(),
            settings=Settings(storage_root=str(tmp_path)),
        )
        data = run_agent(agent, make_request()).data
        assert data.audio_status == "silent"

    def test_missing_voiceover_file_downgrades_to_pending_with_warning(
        self, tmp_path: Path
    ) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(),
            settings=Settings(storage_root=str(tmp_path)),
        )
        request = make_request(
            voiceover={
                "provider": "local_file",
                "audio_file": "nowhere/voice.wav",
            }  # type: ignore[arg-type]
        )
        data = run_agent(agent, request).data
        assert data.audio_status == "pending"
        assert any("Voiceover" in w for w in data.warnings)

    def test_include_subtitles_is_acknowledged_but_unused(
        self, tmp_path: Path
    ) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(),
            settings=Settings(storage_root=str(tmp_path)),
        )
        data = run_agent(agent, make_request(include_subtitles=True)).data
        assert any("Subtitles" in w for w in data.warnings)

    def test_unavailable_ffmpeg_fails_with_a_retryable_code(self, tmp_path: Path) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(healthy=False),
            settings=Settings(storage_root=str(tmp_path)),
        )
        result = run_agent(agent, make_request())
        assert not result.success
        assert result.metadata.get("code") == "video_producer_render_error"
        assert result.metadata.get("retryable") is True

    def test_renderer_crash_is_normalised_to_a_render_error(
        self, tmp_path: Path
    ) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(fail_with=RuntimeError("disk on fire")),
            settings=Settings(storage_root=str(tmp_path)),
        )
        result = run_agent(agent, make_request())
        assert not result.success
        assert result.metadata.get("code") == "video_producer_render_error"
        # The raw exception text stays server-side.
        assert "disk on fire" not in (result.error or "")

    def test_renderer_returning_without_a_file_is_not_success(
        self, tmp_path: Path
    ) -> None:
        """The core no-false-success rule: an unverifiable output fails."""
        agent = VideoProducerAgent(
            renderer=FakeRenderer(write_file=False),
            settings=Settings(storage_root=str(tmp_path)),
        )
        result = run_agent(agent, make_request())
        assert not result.success
        assert result.metadata.get("code") == "video_producer_render_error"

    def test_output_without_a_video_stream_is_not_success(
        self, tmp_path: Path
    ) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(streams=[{"codec_type": "audio", "codec_name": "aac"}]),
            settings=Settings(storage_root=str(tmp_path)),
        )
        result = run_agent(agent, make_request())
        assert not result.success
        assert result.metadata.get("code") == "video_producer_render_error"

    def test_expected_voiceover_without_audio_stream_is_not_success(
        self, tmp_path: Path
    ) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(streams=[{"codec_type": "video", "codec_name": "h264"}]),
            settings=Settings(storage_root=str(tmp_path)),
        )
        voice_file = tmp_path / "media" / "voice.wav"
        voice_file.parent.mkdir(parents=True, exist_ok=True)
        voice_file.write_bytes(b"RIFF")
        request = make_request(
            voiceover={"provider": "local_file", "audio_file": str(voice_file)}  # type: ignore[arg-type]
        )
        result = run_agent(agent, request)
        # The fake reports "present" intent (valid file) but the output
        # has no audio stream — that mismatch must fail, not succeed.
        assert not result.success
        assert result.metadata.get("code") == "video_producer_render_error"

    def test_background_color_is_applied_to_scenes(self, tmp_path: Path) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(),
            settings=Settings(storage_root=str(tmp_path)),
        )
        request = make_request(background={"color": "#204080"})  # type: ignore[arg-type]
        data = run_agent(agent, request).data
        assert all(s.background == "#204080" for s in data.scenes)

    def test_project_output_is_confined_to_the_project_directory(
        self, tmp_path: Path
    ) -> None:
        agent = VideoProducerAgent(
            renderer=FakeRenderer(),
            settings=Settings(storage_root=str(tmp_path)),
        )
        data = run_agent(agent, make_request(project_id="confined")).data
        expected_root = tmp_path / "projects" / "confined" / "renders"
        assert data.output_path.startswith(str(expected_root))


# --------------------------------------------------------------------------
# Real FFmpeg integration (skipped when FFmpeg is absent)
# --------------------------------------------------------------------------

FFMPEG_AVAILABLE = bool(
    shutil.which("ffmpeg")
    and shutil.which("ffprobe")
    and any(Path(f).is_file() for f in FONT_CANDIDATES)
)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe/font not available")
class TestVideoProducerRendersRealVideo:
    """End-to-end: a real MP4 exists, probes, and matches the timeline."""

    def test_produce_creates_a_probeable_mp4(self, tmp_path: Path) -> None:
        agent = VideoProducerAgent(settings=Settings(storage_root=str(tmp_path)))
        request = VideoProductionRequest(
            script=ScriptResult(title="Render", narration="tiny render test"),
            project_id="renderok",
            resolution=TINY,
            fps=10,
        )
        result = run_agent(agent, request)
        assert result.success, result.error

        data = result.data
        assert data.status is ProductionStatus.COMPLETED
        assert data.file_size > 0
        assert data.codec  # taken from ffprobe, not invented
        output = Path(data.output_path)
        assert output.is_file()
        assert output.stat().st_size == data.file_size
        # A silent AAC track is muxed into every render.
        assert data.audio_status == "silent"

    def test_render_matches_requested_geometry(self, tmp_path: Path) -> None:
        agent = VideoProducerAgent(settings=Settings(storage_root=str(tmp_path)))
        request = VideoProductionRequest(
            script=ScriptResult(title="Geo", narration="geometry check"),
            project_id="geo",
            resolution=TINY,
            fps=10,
        )
        data = run_agent(agent, request).data
        assert (data.width, data.height, data.fps) == (320, 180, 10)
        assert data.timeline is not None
        assert (data.timeline.width, data.timeline.height) == (320, 180)

        renderer = FFmpegRenderer(
            tmp_path / "renders",
            media_dir=tmp_path / "media",
            projects_dir=tmp_path / "projects",
        )
        probe = renderer.probe(Path(data.output_path))
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (320, 180)


# --------------------------------------------------------------------------
# Registry integration
# --------------------------------------------------------------------------


class TestRegistry:
    def test_video_producer_is_registered_as_an_implemented_agent(self) -> None:
        registry = build_default_registry()
        agent = registry.get_agent("video_producer")
        assert isinstance(agent, VideoProducerAgent)
        assert not isinstance(agent, PlaceholderAgent)

    def test_agent_identity(self) -> None:
        agent = VideoProducerAgent()
        assert agent.name == "video_producer"
        assert agent.version == "1.0.0"
        assert agent.description
        assert "video_producer" in IMPLEMENTED_AGENT_NAMES

    def test_registry_describes_it_as_implemented(self) -> None:
        registry = build_default_registry()
        descriptor = registry.get_descriptor("video_producer")
        assert descriptor.implemented is True

    def test_every_other_agent_still_resolves(self) -> None:
        registry = build_default_registry()
        for name in ("researcher", "fact_checker", "scriptwriter",
                     "video_producer", "captions_thumbnail", "publisher"):
            assert registry.has(name)

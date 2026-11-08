"""Tests for media services: grounding, assets, voiceover, renderer.

Covers the safety rails (path traversal, unsupported types), the honest
status reporting (missing vs available vs planned), the voice provider
seam (no TTS, no fake voice), and the FFmpeg renderer's command
construction, failure handling and real-output validation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.agents.video_producer.exceptions import (
    VideoProducerAssetError,
    VideoProducerRenderError,
)
from app.agents.video_producer.grounding import (
    ensure_within,
    safe_filename,
    validate_path,
)
from app.agents.video_producer.schemas import (
    AssetStatus,
    Scene,
)
from app.services.media.assets.local import (
    LocalAssetProvider,
    checksum_of,
    media_type_for,
)
from app.services.media.assets.manager import MEDIA_SUBDIRS, AssetManager
from app.services.media.audio.local import LocalVoiceProvider
from app.services.media.rendering.ffmpeg import (
    FONT_CANDIDATES,
    X264_PRESET,
    X264_THREADS,
    FFmpegRenderer,
    build_concat_command,
    build_mux_command,
    build_segment_command,
    wrap_text,
)
from app.services.media.rendering.base import VideoRenderer

# --------------------------------------------------------------------------
# Path grounding
# --------------------------------------------------------------------------


class TestGrounding:
    def test_relative_path_inside_base_resolves(self, tmp_path: Path) -> None:
        result = validate_path("sub/file.mp4", tmp_path)
        assert result == (tmp_path / "sub" / "file.mp4").resolve()
        assert result.is_relative_to(tmp_path.resolve())

    def test_absolute_path_inside_base_is_accepted(self, tmp_path: Path) -> None:
        target = tmp_path / "out.mp4"
        assert validate_path(target, tmp_path) == target.resolve()

    @pytest.mark.parametrize(
        "candidate",
        ["../outside.mp4", "a/../../outside.mp4", "..\\outside.mp4"],
    )
    def test_traversal_is_rejected(self, tmp_path: Path, candidate: str) -> None:
        with pytest.raises(VideoProducerAssetError):
            validate_path(candidate, tmp_path)

    def test_absolute_path_outside_base_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(VideoProducerAssetError):
            validate_path(tmp_path.parent / "elsewhere.mp4", tmp_path)

    def test_validation_works_for_files_that_do_not_exist_yet(
        self, tmp_path: Path
    ) -> None:
        # FFmpeg output must be validated before it is written.
        result = validate_path("renders/new.mp4", tmp_path)
        assert not result.exists()
        assert result.is_relative_to(tmp_path.resolve())

    def test_ensure_within_tries_every_base(self, tmp_path: Path) -> None:
        other = tmp_path / "projects"
        other.mkdir()
        assert ensure_within(other / "p" / "v.mp4", tmp_path / "renders", other)
        with pytest.raises(VideoProducerAssetError):
            ensure_within(tmp_path.parent / "x.mp4", tmp_path / "renders", other)

    def test_safe_filename(self) -> None:
        assert safe_filename("clip.mp4") == "clip.mp4"
        for bad in ["", ".", "..", "a/b.mp4", "a\\b.mp4", "/abs.mp4"]:
            with pytest.raises(VideoProducerAssetError):
                safe_filename(bad)


# --------------------------------------------------------------------------
# Local asset provider
# --------------------------------------------------------------------------


class TestLocalAssetProvider:
    @pytest.fixture()
    def provider(self, tmp_path: Path) -> LocalAssetProvider:
        (tmp_path / "images").mkdir()
        return LocalAssetProvider(tmp_path)

    def test_existing_file_resolves_available_with_metadata(
        self, tmp_path: Path, provider: LocalAssetProvider
    ) -> None:
        image = tmp_path / "images" / "logo.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        asset = provider.resolve("images/logo.png")

        assert asset.status is AssetStatus.AVAILABLE
        assert asset.path == str(image.resolve())
        assert asset.media_type == "image"
        assert asset.mime_type == "image/png"
        assert asset.size_bytes == image.stat().st_size
        assert len(asset.checksum) == 64

    def test_missing_file_is_not_an_error_but_is_missing(
        self, provider: LocalAssetProvider
    ) -> None:
        asset = provider.resolve("images/nope.png")
        assert asset.status is AssetStatus.MISSING
        assert asset.path == ""
        assert asset.size_bytes == 0

    @pytest.mark.parametrize("bad", ["../secret.png", "/etc/passwd", "..\\x.png"])
    def test_traversal_raises(self, tmp_path: Path, bad: str) -> None:
        provider = LocalAssetProvider(tmp_path)
        with pytest.raises(VideoProducerAssetError):
            provider.resolve(bad)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_reference_raises(self, provider: LocalAssetProvider, bad: str) -> None:
        with pytest.raises(VideoProducerAssetError):
            provider.resolve(bad)

    @pytest.mark.parametrize(
        "name", ["script.txt", "data.json", "archive.zip", "page.html"]
    )
    def test_unsupported_type_raises(self, tmp_path: Path, name: str) -> None:
        (tmp_path / name).write_text("x", encoding="utf-8")
        provider = LocalAssetProvider(tmp_path)
        with pytest.raises(VideoProducerAssetError):
            provider.resolve(name)

    def test_media_type_mapping(self) -> None:
        assert media_type_for(Path("a.PNG")) == "image"
        assert media_type_for(Path("a.mp4")) == "video"
        assert media_type_for(Path("a.mp3")) == "audio"
        assert media_type_for(Path("a.txt")) is None

    def test_checksum_is_content_based(self, tmp_path: Path) -> None:
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"hello")
        second.write_bytes(b"world")
        assert checksum_of(first) == checksum_of(first)
        assert checksum_of(first) != checksum_of(second)


# --------------------------------------------------------------------------
# Asset manager
# --------------------------------------------------------------------------


def scene_with_requirements(*requirements: dict) -> Scene:
    return Scene(
        scene_id="scene_001",
        heading="H",
        narration="n",
        asset_requirements=tuple(requirements),
    )


class TestAssetManager:
    def test_ensure_directories_creates_the_media_layout(self, tmp_path: Path) -> None:
        manager = AssetManager(base_dir=tmp_path)
        manager.ensure_directories()
        for name in MEDIA_SUBDIRS:
            assert (tmp_path / name).is_dir()

    def test_prompt_only_requirement_stays_planned(self, tmp_path: Path) -> None:
        manager = AssetManager(base_dir=tmp_path)
        scene = scene_with_requirements(
            {"media_type": "image", "prompt": "a mountain at dawn"}
        )
        scenes, status = manager.resolve_scene_assets([scene])
        # Honest: no file was ever supplied, so nothing is "available".
        assert status == "missing"
        assert scenes[0].asset_references[0].status is AssetStatus.PLANNED
        assert scenes[0].asset_references[0].path == ""

    def test_existing_path_requirement_resolves_available(self, tmp_path: Path) -> None:
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "real.png").write_bytes(b"\x89PNGfake")
        manager = AssetManager(base_dir=tmp_path)
        scene = scene_with_requirements(
            {"media_type": "image", "path": "images/real.png"}
        )
        scenes, status = manager.resolve_scene_assets([scene])
        assert status == "available"
        assert scenes[0].asset_references[0].status is AssetStatus.AVAILABLE

    def test_missing_path_requirement_reports_missing(self, tmp_path: Path) -> None:
        manager = AssetManager(base_dir=tmp_path)
        scene = scene_with_requirements(
            {"media_type": "image", "path": "images/absent.png"}
        )
        scenes, status = manager.resolve_scene_assets([scene])
        assert status == "missing"
        assert scenes[0].asset_references[0].status is AssetStatus.MISSING

    def test_traversal_requirement_becomes_missing_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        manager = AssetManager(base_dir=tmp_path)
        scene = scene_with_requirements({"media_type": "image", "path": "../evil.png"})
        _, status = manager.resolve_scene_assets([scene])
        assert status == "missing"

    def test_no_requirements_means_no_assets_needed(self, tmp_path: Path) -> None:
        manager = AssetManager(base_dir=tmp_path)
        scene = Scene(scene_id="scene_001", heading="H", narration="n")
        scenes, status = manager.resolve_scene_assets([scene])
        assert status == "none"
        assert scenes[0].asset_references == ()

    def test_one_unavailable_asset_makes_the_whole_run_missing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "ok.png").write_bytes(b"png")
        manager = AssetManager(base_dir=tmp_path)
        scenes_in = [
            scene_with_requirements({"media_type": "image", "path": "images/ok.png"}),
            scene_with_requirements({"media_type": "image", "path": "images/gone.png"}),
        ]
        _, status = manager.resolve_scene_assets(scenes_in)
        assert status == "missing"


# --------------------------------------------------------------------------
# Voiceover (no TTS, no fakes)
# --------------------------------------------------------------------------


class TestLocalVoiceProvider:
    @pytest.fixture()
    def provider(self, tmp_path: Path) -> LocalVoiceProvider:
        return LocalVoiceProvider(tmp_path)

    def test_health_check_admits_no_synthesis(self, provider: LocalVoiceProvider) -> None:
        assert provider.health_check() is False

    def test_synthesis_without_a_file_is_pending(self, provider: LocalVoiceProvider) -> None:
        result = provider.synthesize("hello world")
        assert result.status == "pending"
        assert not result.audio_path

    def test_valid_local_file_is_used(self, tmp_path: Path) -> None:
        voice = tmp_path / "voice.wav"
        voice.write_bytes(b"RIFF....WAVE")
        provider = LocalVoiceProvider(tmp_path)
        result = provider.synthesize("hello", output_path=voice)
        assert result.status == "available"
        assert result.audio_path == str(voice.resolve())

    def test_validate_rejects_missing_files(self, tmp_path: Path) -> None:
        provider = LocalVoiceProvider(tmp_path)
        assert provider.validate_audio_file("nope.wav") is None

    def test_validate_rejects_traversal(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.wav"
        provider = LocalVoiceProvider(tmp_path)
        assert provider.validate_audio_file(outside) is None

    def test_validate_rejects_non_audio_files(self, tmp_path: Path) -> None:
        (tmp_path / "clip.mp4").write_bytes(b"not audio")
        provider = LocalVoiceProvider(tmp_path)
        assert provider.validate_audio_file("clip.mp4") is None


# --------------------------------------------------------------------------
# Renderer: pure command construction
# --------------------------------------------------------------------------


class TestRendererCommands:
    def test_segment_command_is_an_argument_vector(self) -> None:
        command = build_segment_command(
            fontfile="C:/Windows/Fonts/arial.ttf",
            textfile="D:/work/text_001.txt",
            background="#101010",
            width=1920,
            height=1080,
            fps=30,
            duration=2.5,
            output="D:/out/seg.mp4",
        )
        assert isinstance(command, list)
        assert all(isinstance(part, str) for part in command)
        assert command[0] == "ffmpeg"
        assert "libx264" in command
        assert command[command.index("-preset") + 1] == X264_PRESET
        assert command[command.index("-threads") + 1] == X264_THREADS
        # No shell strings: the filter is one argv element.
        vf = command[command.index("-vf") + 1]
        assert vf.startswith("drawtext=")
        # Colon in the font path is escaped for the filtergraph parser.
        assert "C\\:/Windows/Fonts/arial.ttf" in vf
        # The text file path is escaped the same way.
        assert "textfile='D\\:/work/text_001.txt'" in vf

    def test_segment_command_encodes_the_requested_geometry(self) -> None:
        command = build_segment_command(
            fontfile="f.ttf",
            textfile="t.txt",
            background="#204080",
            width=640,
            height=360,
            fps=25,
            duration=1.0,
            output="seg.mp4",
        )
        source = command[command.index("-i") + 1]
        assert "s=640x360" in source
        assert "r=25" in source
        assert "0x204080" in source

    def test_segment_command_tolerates_missing_hash(self) -> None:
        command = build_segment_command(
            fontfile="f.ttf",
            textfile="t.txt",
            background="",
            width=640,
            height=360,
            fps=25,
            duration=1.0,
            output="seg.mp4",
        )
        assert "0x101010" in command[command.index("-i") + 1]

    def test_concat_command_copies_streams(self) -> None:
        command = build_concat_command(list_file="list.txt", output="out.mp4")
        assert command[0] == "ffmpeg"
        assert "-f" in command and "concat" in command
        assert "copy" in command

    def test_mux_command_without_audio_generates_silence(self) -> None:
        command = build_mux_command(
            video="v.mp4", output="o.mp4", audio=None, width=1, height=1
        )
        assert "anullsrc=r=44100:cl=stereo" in command
        assert "aac" in command
        assert "-shortest" in command

    def test_mux_command_with_audio_maps_the_supplied_file(self) -> None:
        command = build_mux_command(
            video="v.mp4", output="o.mp4", audio="voice.wav", width=1, height=1
        )
        assert "voice.wav" in command
        assert "anullsrc" not in command
        assert command[command.index("-map") + 1] == "0:v"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", ""),
            ("   ", ""),
            ("Short", "Short"),
        ],
    )
    def test_wrap_text_passthrough(self, text: str, expected: str) -> None:
        assert wrap_text(text) == expected

    def test_wrap_text_wraps_and_caps_lines(self) -> None:
        long_text = " ".join(f"word{i}" for i in range(100))
        wrapped = wrap_text(long_text)
        lines = wrapped.split("\n")
        assert 1 <= len(lines) <= 3
        assert all(len(line) <= 40 for line in lines)
        assert "word0" in wrapped


# --------------------------------------------------------------------------
# Renderer: behaviour without invoking FFmpeg's encoder
# --------------------------------------------------------------------------


def tiny_timeline() -> tuple[object, list[Scene]]:
    from app.agents.video_producer.schemas import Timeline, TimelineScene

    scenes = [
        Scene(scene_id="scene_001", heading="Hello", narration="hi", duration_seconds=1.0)
    ]
    timeline = Timeline(
        width=320,
        height=180,
        fps=10,
        duration_seconds=1.0,
        scenes=(
            TimelineScene(
                scene_id="scene_001",
                start_time=0.0,
                end_time=1.0,
                duration=1.0,
            ),
        ),
    )
    return timeline, scenes


class TestRendererBehaviour:
    @pytest.fixture()
    def renderer(self, tmp_path: Path) -> FFmpegRenderer:
        return FFmpegRenderer(
            tmp_path / "renders",
            media_dir=tmp_path / "media",
            projects_dir=tmp_path / "projects",
        )

    def test_constructor_confines_itself_to_configured_roots(
        self, tmp_path: Path
    ) -> None:
        r = FFmpegRenderer(
            tmp_path / "renders",
            media_dir=tmp_path / "media",
            projects_dir=tmp_path / "projects",
        )
        assert r.renders_dir == tmp_path / "renders"
        assert r.media_dir == tmp_path / "media"
        assert r.projects_dir == tmp_path / "projects"
        assert r.renders_dir.is_dir()

    def test_render_rejects_an_empty_timeline(self, renderer: FFmpegRenderer) -> None:
        from app.agents.video_producer.schemas import Timeline

        with pytest.raises(VideoProducerRenderError):
            renderer.render(Timeline(), [], renderer.renders_dir / "out.mp4")

    def test_render_rejects_output_outside_allowed_roots(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        timeline, scenes = tiny_timeline()
        with pytest.raises(VideoProducerAssetError):
            renderer.render(timeline, scenes, tmp_path.parent / "escape.mp4")

    def test_render_accepts_output_in_the_projects_root(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        # Accepted at the grounding layer (execution may still fail
        # later, but the path itself is inside an allowed root).
        from app.agents.video_producer.grounding import ensure_within

        target = tmp_path / "projects" / "p1" / "renders" / "v.mp4"
        assert ensure_within(target, renderer.renders_dir, renderer.projects_dir)

    def test_run_reports_failure_with_the_exit_code(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        with pytest.raises(VideoProducerRenderError) as excinfo:
            renderer._run(
                ["ffmpeg", "-nostdin", "-i", str(tmp_path / "missing.mp4"),
                 str(tmp_path / "out.mp4")]
            )
        assert "code" in str(excinfo.value)

    def test_run_reports_missing_binary(self, renderer: FFmpegRenderer) -> None:
        with pytest.raises(VideoProducerRenderError):
            renderer._run(["definitely-not-a-real-binary-xyz"])

    def test_probe_of_garbage_returns_empty_dict(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"not a video")
        assert renderer.probe(junk) == {}

    def test_validation_rejects_a_missing_file(self, renderer: FFmpegRenderer) -> None:
        timeline, _ = tiny_timeline()
        with pytest.raises(VideoProducerRenderError):
            renderer._validate_output(renderer.renders_dir / "absent.mp4", timeline)

    def test_validation_rejects_an_empty_file(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        timeline, _ = tiny_timeline()
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(VideoProducerRenderError):
            renderer._validate_output(empty, timeline)

    def test_validation_rejects_a_file_without_a_video_stream(
        self, renderer: FFmpegRenderer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        timeline, _ = tiny_timeline()
        target = renderer.renders_dir / "audio_only.mp4"
        target.write_bytes(b"x")
        monkeypatch.setattr(
            renderer, "probe", lambda path: {"streams": [{"codec_type": "audio"}]}
        )
        with pytest.raises(VideoProducerRenderError):
            renderer._validate_output(target, timeline)

    def test_validation_rejects_wrong_geometry(
        self, renderer: FFmpegRenderer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        timeline, _ = tiny_timeline()  # 320x180
        target = renderer.renders_dir / "wrong.mp4"
        target.write_bytes(b"x")
        monkeypatch.setattr(
            renderer,
            "probe",
            lambda path: {
                "streams": [
                    {"codec_type": "video", "width": 1920, "height": 1080}
                ],
                "format": {"duration": "1.0"},
            },
        )
        with pytest.raises(VideoProducerRenderError) as excinfo:
            renderer._validate_output(target, timeline)
        assert "geometry" in str(excinfo.value)

    def test_validation_rejects_a_truncated_render(
        self, renderer: FFmpegRenderer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        timeline, _ = tiny_timeline()  # 1 second
        target = renderer.renders_dir / "short.mp4"
        target.write_bytes(b"x")
        monkeypatch.setattr(
            renderer,
            "probe",
            lambda path: {
                "streams": [
                    {"codec_type": "video", "width": 320, "height": 180}
                ],
                "format": {"duration": "5.0"},
            },
        )
        with pytest.raises(VideoProducerRenderError) as excinfo:
            renderer._validate_output(target, timeline)
        assert "duration" in str(excinfo.value)

    def test_health_check_is_false_without_a_font(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "app.services.media.rendering.ffmpeg.FONT_CANDIDATES", ()
        )
        renderer = FFmpegRenderer(tmp_path / "renders")
        assert renderer.health_check() is False
        timeline, scenes = tiny_timeline()
        with pytest.raises(VideoProducerRenderError):
            renderer.render(timeline, scenes, renderer.renders_dir / "x.mp4")


# --------------------------------------------------------------------------
# Real FFmpeg integration
# --------------------------------------------------------------------------

FFMPEG_AVAILABLE = bool(
    shutil.which("ffmpeg")
    and shutil.which("ffprobe")
    and any(Path(f).is_file() for f in FONT_CANDIDATES)
)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe/font not available")
class TestFFmpegIntegration:
    @pytest.fixture()
    def renderer(self, tmp_path: Path) -> FFmpegRenderer:
        return FFmpegRenderer(
            tmp_path / "renders",
            media_dir=tmp_path / "media",
            projects_dir=tmp_path / "projects",
        )

    def test_health_check_passes(self, renderer: FFmpegRenderer) -> None:
        assert renderer.health_check() is True

    def test_render_produces_a_valid_video(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        timeline, scenes = tiny_timeline()
        target = tmp_path / "projects" / "p" / "renders" / "tiny.mp4"

        output = renderer.render(timeline, scenes, target)

        assert output == target
        assert target.is_file()
        assert target.stat().st_size > 0
        probe = renderer.probe(target)
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (320, 180)
        # The silent AAC track is always present.
        assert any(s["codec_type"] == "audio" for s in probe["streams"])

    def test_render_with_a_real_voiceover_file_keeps_the_audio(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        # Generate a real local audio file with FFmpeg itself.
        voice = tmp_path / "media" / "voice.wav"
        voice.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "sine=frequency=440:duration=1.5",
                str(voice),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        assert voice.is_file()

        timeline, scenes = tiny_timeline()  # 1.0 s video
        target = tmp_path / "projects" / "p" / "renders" / "voiced.mp4"
        renderer.render(timeline, scenes, target, audio_path=voice)

        probe = renderer.probe(target)
        audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        assert audio["codec_name"] == "aac"
        # apad + -shortest keeps the container close to the video length.
        duration = float(probe["format"]["duration"])
        assert duration == pytest.approx(1.0, abs=0.5)

    def test_audio_outside_media_root_is_rejected(
        self, renderer: FFmpegRenderer, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.wav"
        import subprocess

        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "sine=frequency=220:duration=0.5",
                str(outside),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        timeline, scenes = tiny_timeline()
        target = tmp_path / "renders" / "untrusted.mp4"
        # An audio path outside media_dir must fail loudly, not be
        # silently accepted or quietly swapped for silence.
        with pytest.raises(VideoProducerAssetError):
            renderer.render(timeline, scenes, target, audio_path=outside)
        assert not target.exists()


def test_renderer_satisfies_the_contract() -> None:
    assert issubclass(FFmpegRenderer, VideoRenderer)

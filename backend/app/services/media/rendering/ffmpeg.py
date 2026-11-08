"""FFmpeg-backed renderer.

Design constraints, all deliberate:

* **No shell strings.** Every invocation is an argument list; there is
  no ``shell=True`` anywhere, so titles, filenames and paths can never
  inject shell syntax.
* **No fabricated success.** ``render()`` validates its own output
  (exists, non-empty, probes as a video stream with the expected
  geometry) before returning; anything less raises.
* **CPU-only settings.** ``ultrafast`` + single thread keeps x264
  stable on constrained hosts where multi-threaded allocation fails.
* **Text via ``textfile``.** Scene text is written to a file and read
  by FFmpeg, so no escaping of quotes, colons or percent signs is ever
  needed — headings containing arbitrary punctuation are safe.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from app.agents.video_producer.exceptions import VideoProducerRenderError
from app.agents.video_producer.grounding import ensure_within, validate_path
from app.agents.video_producer.schemas import Scene, Timeline
from app.core.logging import get_logger
from app.services.media.rendering.base import VideoRenderer

logger = get_logger("services.media.ffmpeg")

#: x264 preset. ``ultrafast`` is deliberate: this project targets
#: CPU-only machines (i5-10210U class) and the default ``medium``
#: preset fails with allocator errors in constrained environments.
X264_PRESET = "ultrafast"

#: Encoder thread count. Single-threaded encoding avoids x264's
#: allocator failures on constrained hosts while staying fast enough
#: for the short renders this phase produces.
X264_THREADS = "1"

#: Fonts tried in order until one exists. Found paths are escaped for
#: the filtergraph (``:`` → ``\:``) rather than quoted by hand.
FONT_CANDIDATES = (
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

#: Characters drawn per overlay line before wrapping, chosen so a
#: heading stays inside a 1080p frame at a readable size.
_WRAP_AT = 36


def _escape_filter_value(value: str) -> str:
    """Escape a value used inside a filtergraph option."""
    return value.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def build_segment_command(
    *,
    fontfile: str,
    textfile: str,
    background: str,
    width: int,
    height: int,
    fps: int,
    duration: float,
    output: str,
) -> list[str]:
    """Command that renders one scene segment (solid bg + heading).

    Pure function so tests can assert on the exact argument vector
    without invoking FFmpeg.
    """
    colour = background.lstrip("#") or "101010"
    vf = (
        f"drawtext=fontfile='{_escape_filter_value(fontfile)}'"
        f":textfile='{_escape_filter_value(textfile)}'"
        f":fontsize={max(width // 24, 18)}"
        f":fontcolor=white"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
    )
    return [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x{colour}:s={width}x{height}:r={fps}:d={duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", X264_PRESET,
        "-threads", X264_THREADS,
        "-pix_fmt", "yuv420p",
        "-t", f"{duration:.3f}",
        output,
    ]


def build_concat_command(*, list_file: str, output: str) -> list[str]:
    """Command that joins pre-rendered segments without re-encoding."""
    return [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output,
    ]


def build_mux_command(
    *,
    video: str,
    output: str,
    audio: str | None = None,
    width: int,
    height: int,
) -> list[str]:
    """Command that adds an audio track (real or silent) to the video.

    With ``audio=None`` a silent AAC track is generated, so every
    output has a predictable stream layout. ``apad`` + ``-shortest``
    make the audio track exactly as long as the video regardless of
    how long a supplied voiceover is.
    """
    command = ["ffmpeg", "-y", "-i", video]
    if audio is not None:
        command += ["-i", audio]
    else:
        command += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    command += [
        "-map", "0:v",
        "-map", "1:a",
        "-af", "apad",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        output,
    ]
    return command


def wrap_text(text: str, *, width: int = _WRAP_AT, max_lines: int = 3) -> str:
    """Greedy word wrap for overlay text, hard-truncated after ``max_lines``."""
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines and current not in lines:
        lines.append(current)
    return "\n".join(lines)


class FFmpegRenderer(VideoRenderer):
    """Renders timelines to MP4 using the local FFmpeg binary."""

    name = "ffmpeg"

    def __init__(
        self,
        renders_dir: Path | None = None,
        *,
        media_dir: Path | None = None,
        projects_dir: Path | None = None,
    ) -> None:
        self.renders_dir = Path(renders_dir) if renders_dir else Path.cwd() / "renders"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        #: Root that supplied audio files must live under.
        self.media_dir = Path(media_dir) if media_dir else self.renders_dir.parent / "media"
        #: Roots final outputs may be written under: the shared renders
        #: directory and the per-project directory (``projects/<id>/``).
        #: Nothing outside these two is ever accepted, so a crafted
        #: ``output_filename`` cannot write anywhere else.
        self.projects_dir = (
            Path(projects_dir) if projects_dir else self.renders_dir.parent / "projects"
        )
        self._font = next(
            (f for f in FONT_CANDIDATES if Path(f).is_file()), ""
        )

    # ---------- Health / probe ----------

    def health_check(self) -> bool:
        """FFmpeg is callable and the font needed for overlays exists."""
        if not self._font:
            return False
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def probe(self, path: Path) -> dict:
        """Stream/format metadata for ``path``; ``{}`` when unreadable."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if result.returncode != 0:
            return {}
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return {}

    # ---------- Render ----------

    def render(
        self,
        timeline: Timeline,
        scenes: list[Scene],
        output_path: Path,
        *,
        audio_path: Path | None = None,
    ) -> Path:
        """Render every scene, concatenate, attach audio, validate.

        Raises:
            VideoProducerRenderError: FFmpeg failed at any step or the
                final file failed validation. A normal return always
                means a probed, non-empty video exists at the output.
        """
        if not timeline.scenes or timeline.duration_seconds <= 0:
            raise VideoProducerRenderError("Timeline has no scenes to render")
        if not self._font:
            raise VideoProducerRenderError("No usable font found for text overlays")

        target = ensure_within(output_path, self.renders_dir, self.projects_dir)
        target.parent.mkdir(parents=True, exist_ok=True)

        scene_by_id = {scene.scene_id: scene for scene in scenes}

        with tempfile.TemporaryDirectory(prefix="vp_", dir=self.renders_dir) as work:
            work_dir = Path(work)
            segments: list[Path] = []

            # 1. One segment per scene: background + heading overlay.
            for index, tscene in enumerate(timeline.scenes, start=1):
                scene = scene_by_id.get(tscene.scene_id)
                heading = wrap_text(scene.heading if scene else tscene.scene_id) or " "
                text_file = work_dir / f"text_{index:03d}.txt"
                # utf-8 without BOM: FFmpeg would draw the BOM bytes.
                text_file.write_text(heading, encoding="utf-8", newline="\n")
                segment = work_dir / f"seg_{index:03d}.mp4"
                background = (
                    (scene.background if scene else "") or "#101010"
                )
                self._run(
                    build_segment_command(
                        fontfile=self._font,
                        textfile=str(text_file).replace("\\", "/"),
                        background=background,
                        width=timeline.width,
                        height=timeline.height,
                        fps=timeline.fps,
                        duration=tscene.duration,
                        output=str(segment),
                    )
                )
                segments.append(segment)

            # 2. Concatenate the segments (stream copy — same codec params).
            # Absolute paths in the list file: resolution of relative
            # entries depends on FFmpeg's working directory, which we
            # do not control.
            concat_list = work_dir / "list.txt"
            concat_list.write_text(
                "".join(
                    f"file '{s.resolve().as_posix()}'\n" for s in segments
                ),
                encoding="utf-8",
            )
            joined = work_dir / "joined.mp4"
            self._run(
                build_concat_command(list_file=str(concat_list), output=str(joined))
            )

            # 3. Attach audio (supplied voiceover or generated silence).
            safe_audio: str | None = None
            if audio_path is not None:
                validated = validate_path(audio_path, self.media_dir)
                if validated.is_file():
                    safe_audio = str(validated)
                else:
                    logger.warning("voiceover_missing path=%s", audio_path)
            self._run(
                build_mux_command(
                    video=str(joined),
                    output=str(target),
                    audio=safe_audio,
                    width=timeline.width,
                    height=timeline.height,
                )
            )

        self._validate_output(target, timeline)
        return target

    # ---------- Internals ----------

    def _run(self, command: list[str]) -> None:
        """Execute an FFmpeg command and fail loudly on any problem."""
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError as exc:
            raise VideoProducerRenderError("ffmpeg is not available on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise VideoProducerRenderError("ffmpeg render timed out") from exc

        if result.returncode != 0:
            tail = (result.stderr or "")[-800:]
            logger.error("ffmpeg_failed code=%s tail=%s", result.returncode, tail)
            raise VideoProducerRenderError(
                f"ffmpeg exited with code {result.returncode}"
            )

    def _validate_output(self, target: Path, timeline: Timeline) -> None:
        """Post-render gate: no output may be reported unverified."""
        if not target.is_file():
            raise VideoProducerRenderError("Rendered output does not exist")
        size = target.stat().st_size
        if size <= 0:
            raise VideoProducerRenderError("Rendered output is empty")

        data = self.probe(target)
        streams = data.get("streams", []) if isinstance(data, dict) else []
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            raise VideoProducerRenderError("Rendered output has no video stream")

        video = video_streams[0]
        width, height = video.get("width"), video.get("height")
        if width != timeline.width or height != timeline.height:
            raise VideoProducerRenderError(
                f"Rendered geometry {width}x{height} does not match "
                f"requested {timeline.width}x{timeline.height}"
            )

        duration_raw = (data.get("format", {}) or {}).get("duration")
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            duration = 0.0
        # Tolerance covers container rounding; anything else is a
        # truncated or runaway render.
        if duration and abs(duration - timeline.duration_seconds) > max(
            1.0, timeline.duration_seconds * 0.25
        ):
            raise VideoProducerRenderError(
                f"Rendered duration {duration:.2f}s does not match "
                f"timeline {timeline.duration_seconds:.2f}s"
            )

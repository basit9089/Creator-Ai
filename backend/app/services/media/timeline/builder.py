"""Deterministic timeline construction.

Times are accumulated in integer milliseconds and only converted to
seconds at the end, so floating-point drift can never make a scene
start before the previous one ends. The same script always yields the
same timeline.
"""

from __future__ import annotations

from app.agents.video_producer.schemas import Scene, Timeline, TimelineScene

#: Everything is scheduled to millisecond precision.
_MS = 1000


def build_timeline(
    scenes: list[Scene],
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    audio: dict | None = None,
) -> Timeline:
    """Place ``scenes`` back to back on a timeline.

    Args:
        scenes: ordered scenes from :func:`plan_scenes`.
        width, height, fps: output format for the render.
        audio: optional audio description copied onto every scene.

    Returns:
        A :class:`Timeline` whose durations sum (within rounding) to
        ``duration_seconds``.
    """
    placed: list[TimelineScene] = []
    cursor_ms = 0

    for scene in scenes:
        duration_ms = max(int(round(scene.duration_seconds * _MS)), 1)
        start = cursor_ms
        end = start + duration_ms
        placed.append(
            TimelineScene(
                scene_id=scene.scene_id,
                start_time=start / _MS,
                end_time=end / _MS,
                duration=duration_ms / _MS,
                visual={
                    "type": scene.visual_type.value,
                    "prompt": scene.visual_prompt,
                    "background": scene.background,
                },
                audio=dict(audio or {}),
                overlays=(
                    {"kind": "heading", "text": scene.heading},
                )
                if scene.heading
                else (),
            )
        )
        cursor_ms = end

    return Timeline(
        width=width,
        height=height,
        fps=fps,
        duration_seconds=cursor_ms / _MS,
        scenes=tuple(placed),
    )

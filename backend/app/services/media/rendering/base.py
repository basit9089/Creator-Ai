"""Renderer contract.

The Video Producer depends on this interface only, so FFmpeg could be
swapped for another engine (or a fake in tests) without touching agent
code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.agents.video_producer.schemas import Scene, Timeline


class VideoRenderer(ABC):
    """Renders a timeline of scenes into a video file."""

    name: str = "renderer"

    @abstractmethod
    def health_check(self) -> bool:
        """Whether the rendering engine is callable right now."""

    @abstractmethod
    def probe(self, path: Path) -> dict:
        """Return stream metadata for ``path`` (empty dict if unreadable)."""

    @abstractmethod
    def render(
        self,
        timeline: Timeline,
        scenes: list[Scene],
        output_path: Path,
        *,
        audio_path: Path | None = None,
    ) -> Path:
        """Render ``timeline`` to ``output_path``.

        Raises:
            VideoProducerRenderError: rendering failed or produced an
                unusable file. Returning normally implies the output
                exists, is non-empty and probes as a real video.
        """

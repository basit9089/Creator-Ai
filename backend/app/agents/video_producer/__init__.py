"""Video Producer Agent package.

Transforms ScriptResult into production plan and renders video via FFmpeg.
"""

from app.agents.video_producer.agent import VideoProducerAgent, REQUEST_KEY
from app.agents.video_producer.schemas import VideoProductionRequest, VideoProductionResult

__all__ = ["VideoProducerAgent", "REQUEST_KEY", "VideoProductionRequest", "VideoProductionResult"]
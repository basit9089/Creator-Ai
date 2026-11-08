"""Exceptions for Video Producer."""

from app.agents.base import AgentError


class VideoProducerError(AgentError):
    code = "video_producer_error"
    retryable = False


class VideoProducerInputError(VideoProducerError):
    code = "video_producer_invalid_request"
    retryable = False


class VideoProducerAssetError(VideoProducerError):
    code = "video_producer_asset_error"
    retryable = False


class VideoProducerPlanningError(VideoProducerError):
    code = "video_producer_planning_error"
    retryable = False


class VideoProducerRenderError(VideoProducerError):
    code = "video_producer_render_error"
    retryable = True
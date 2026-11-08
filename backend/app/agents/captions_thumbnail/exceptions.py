"""Exceptions for the Captions / Thumbnail / SEO agent."""

from app.agents.base import AgentError


class CaptionsThumbnailError(AgentError):
    code = "captions_thumbnail_error"
    retryable = False


class CaptionsThumbnailInputError(CaptionsThumbnailError):
    code = "captions_thumbnail_invalid_request"
    retryable = False


class CaptionsThumbnailAssetError(CaptionsThumbnailError):
    """A supplied path escaped its allowed root."""

    code = "captions_thumbnail_asset_error"
    retryable = False


class CaptionsThumbnailCaptionError(CaptionsThumbnailError):
    code = "captions_thumbnail_caption_error"
    #: Filesystem hiccups (locked volume, transient IO) may clear up.
    retryable = True


class CaptionsThumbnailRenderError(CaptionsThumbnailError):
    code = "captions_thumbnail_render_error"
    #: Matches the Video Producer: render failures are the classic
    #: retry case (missing font on one machine, disk full on another).
    retryable = True


class CaptionsThumbnailAIError(CaptionsThumbnailError):
    code = "captions_thumbnail_ai_error"
    #: The model or its connection may well be back on a retry.
    retryable = True


class CaptionsThumbnailOutputError(CaptionsThumbnailError):
    """Nothing usable was produced at all."""

    code = "captions_thumbnail_output_error"
    retryable = False

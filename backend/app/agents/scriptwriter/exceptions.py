"""Exceptions for Scriptwriter Agent."""

from app.agents.base import AgentError


class ScriptwriterError(AgentError):
    code = "scriptwriter_error"
    retryable = False


class ScriptwriterInputError(ScriptwriterError):
    code = "scriptwriter_invalid_request"
    retryable = False


class ScriptwriterAIError(ScriptwriterError):
    code = "scriptwriter_ai_unavailable"
    retryable = True


class ScriptwriterGroundingError(ScriptwriterError):
    code = "scriptwriter_grounding_failure"
    retryable = False
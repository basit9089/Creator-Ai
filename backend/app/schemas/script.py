"""Public schemas for script API."""

from pydantic import BaseModel

from app.agents.scriptwriter.schemas import ScriptRequest, ScriptResult

ScriptResponse = BaseModel

__all__ = ["ScriptRequest", "ScriptResult"]
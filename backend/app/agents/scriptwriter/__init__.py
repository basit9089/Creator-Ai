"""Scriptwriter Agent package.

Converts fact-checked research into a structured YouTube script.
Only uses claims deemed safe by the Fact Checker.
"""

from app.agents.scriptwriter.agent import ScriptwriterAgent, REQUEST_KEY
from app.agents.scriptwriter.schemas import (
    ScriptRequest,
    ScriptResult,
    ScriptClaimReference,
    Section,
)

__all__ = [
    "ScriptwriterAgent",
    "REQUEST_KEY",
    "ScriptRequest",
    "ScriptResult",
    "ScriptClaimReference",
    "Section",
]
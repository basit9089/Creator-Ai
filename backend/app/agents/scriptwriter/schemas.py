"""Schemas for Scriptwriter Agent."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.agents.fact_checker.schemas import FactCheckResult
from app.agents.researcher.schemas import ResearchResult


class ScriptLength(StrEnum):
    SHORT = "short"  # 1-2 min
    MEDIUM = "medium"  # 3-5 min
    LONG = "long"  # 6-10 min


class Tone(StrEnum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    EDUCATIONAL = "educational"


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_id: str
    heading: str
    purpose: str = ""
    narration: str = Field(default="", min_length=0)
    transition: str = ""
    claim_references: tuple[dict, ...] = ()


class ScriptClaimReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    source_ids: tuple[str, ...] = ()
    usage: str = ""
    status: str = ""
    qualification: str = ""


class ScriptRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    research_result: ResearchResult
    fact_check_result: FactCheckResult
    desired_video_length: ScriptLength = ScriptLength.MEDIUM
    target_audience: str = "general"
    tone: Tone = Tone.EDUCATIONAL
    language: str = Field(default="en", min_length=2, max_length=8)
    topic: str | None = None
    include_cta: bool = True


class ScriptResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    hook: str = ""
    introduction: str = ""
    sections: tuple[Section, ...] = ()
    narration: str = ""
    transitions: tuple[str, ...] = ()
    conclusion: str = ""
    cta: str = ""
    estimated_duration_seconds: int = 0
    word_count: int = 0
    words_per_minute: int = 150
    claims_used: tuple[ScriptClaimReference, ...] = ()
    sources: tuple[dict, ...] = ()
    warnings: tuple[str, ...] = ()
    status: str = "draft"
    generated_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
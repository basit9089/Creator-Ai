"""Public response models for the fact check endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.fact_checker.schemas import FactCheckRequest, FactCheckResult
from app.agents.base import AgentStatus


class FactCheckExecution(BaseModel):
    """Execution metadata."""

    execution_id: str
    agent_name: str
    agent_version: str
    status: AgentStatus
    succeeded: bool
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float | None = None
    code: str | None = None
    retryable: bool | None = None
    error: str | None = None
    metadata: dict = Field(default_factory=dict)


class FactCheckAPIResponse(BaseModel):
    """API response envelope."""

    success: bool
    execution: FactCheckExecution
    result: FactCheckResult | None = None
    error: dict | None = None

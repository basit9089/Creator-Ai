"""Public response models for the agent endpoints.

Kept separate from the domain models in :mod:`app.agents` so the internal
representation can change without silently altering the API contract.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.registry import AgentDescriptor


class AgentInfo(BaseModel):
    """One registered agent, as reported by the API."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Stable identifier, for example 'researcher'.")
    version: str = Field(description="Semantic version of the agent's behaviour.")
    description: str = ""
    status: str = Field(
        description=(
            "Availability: 'available', 'not_implemented' or 'disabled'."
        )
    )
    implemented: bool = Field(
        description="Whether the agent can be executed yet."
    )
    capabilities: list[str] = Field(default_factory=list)

    @classmethod
    def from_descriptor(cls, descriptor: AgentDescriptor) -> AgentInfo:
        return cls(
            name=descriptor.name,
            version=descriptor.version,
            description=descriptor.description,
            status=descriptor.status.value,
            implemented=descriptor.implemented,
            capabilities=list(descriptor.capabilities),
        )


class AgentListResponse(BaseModel):
    """Every registered agent."""

    success: bool = True
    total: int
    agents: list[AgentInfo]

"""Read-only agent discovery routes.

Deliberately minimal. This phase exposes discovery only: there is no
``POST /api/research`` or any other endpoint that executes an agent,
because the agents themselves do not exist yet and there is nothing
useful to trigger.

Listing agents is read-only and cheap, which makes it a reasonable
development aid: it shows which agent names resolve and which are still
placeholders, without running a model.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.registry import AgentRegistry, get_registry
from app.schemas.agents import AgentInfo, AgentListResponse

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List registered agents",
)
def list_agents(registry: AgentRegistry = Depends(get_registry)) -> AgentListResponse:
    """Return every registered agent with its version and availability.

    Agents marked ``not_implemented`` are registered contracts: the name
    resolves, but executing one reports a retryable failure until the
    implementation lands in a later phase.
    """
    descriptors = registry.descriptors()
    agents = [AgentInfo.from_descriptor(descriptor) for descriptor in descriptors]
    return AgentListResponse(total=len(agents), agents=agents)

"""Agent architecture shared by every Creator AI agent.

An agent is a single, versioned, retryable unit of work: the Researcher,
the Fact Checker, the Scriptwriter. Each depends on
:class:`~app.services.ai.provider.AIProvider` and nothing backend
specific.

Layering
--------
``base``
    Contracts only: :class:`Agent`, :class:`AgentContext`,
    :class:`AgentResult`, :class:`AgentStatus`.
``runner``
    :class:`AgentRunner` executes one agent and records timing, status
    and failure. It holds no pipeline logic, so replacing the in-process
    runner with a Redis or Celery queue later changes no agent.
``registry``
    :class:`AgentRegistry` resolves agents by name, so API code never
    hardcodes an agent choice.

Not implemented in this phase: the six concrete agents (researcher,
fact_checker, scriptwriter, video_producer, captions_thumbnail,
publisher). Their names are registered as placeholders so discovery and
lookup work before the logic exists.
"""

from app.agents.base import (
    Agent,
    AgentContext,
    AgentError,
    AgentNotFoundError,
    AgentNotImplementedError,
    AgentOutput,
    AgentResult,
    AgentStatus,
    BaseAgent,
    Capability,
    PlaceholderAgent,
    new_execution_id,
)
from app.agents.registry import (
    AgentAvailability,
    AgentDescriptor,
    AgentRegistry,
    build_default_registry,
    get_registry,
)
from app.agents.runner import AgentRunner, ExecutionRecord

__all__ = [
    "Agent",
    "AgentAvailability",
    "AgentContext",
    "AgentDescriptor",
    "AgentError",
    "AgentNotFoundError",
    "AgentNotImplementedError",
    "AgentOutput",
    "AgentRegistry",
    "AgentResult",
    "AgentRunner",
    "AgentStatus",
    "BaseAgent",
    "Capability",
    "ExecutionRecord",
    "PlaceholderAgent",
    "build_default_registry",
    "get_registry",
    "new_execution_id",
]

"""Scriptwriter Agent."""

from __future__ import annotations

from collections.abc import Sequence

from app.agents.base import Agent, AgentContext, AgentOutput, Capability
from app.agents.scriptwriter.exceptions import (
    ScriptwriterAIError,
    ScriptwriterGroundingError,
    ScriptwriterInputError,
)
from app.agents.scriptwriter.grounding import validate_claim_references, is_safe_for_use
from app.agents.scriptwriter.prompts import SYSTEM_PROMPT, USER_PROMPT
from app.agents.scriptwriter.schemas import (
    ScriptClaimReference,
    ScriptRequest,
    ScriptResult,
    Section,
)
from app.core.logging import get_logger
from app.prompts.builder import PromptBuilder
from app.services.ai.provider import AIProvider, GenerationOptions, Message, StructuredOutputError

logger = get_logger("agents.scriptwriter")

REQUEST_KEY = "script_request"
WORDS_PER_MINUTE_DEFAULT = 150
LEN_MAP = {"short": "1-2 min", "medium": "3-5 min", "long": "6-10 min"}


class ScriptwriterAgent(Agent):
    name = "scriptwriter"
    version = "1.0.0"
    description = "Converts fact-checked research into a structured YouTube script."
    capabilities = frozenset({Capability.STRUCTURED_OUTPUT, Capability.RETRYABLE})

    def __init__(self, *, provider: AIProvider | None = None, thinking: bool = False) -> None:
        super().__init__()
        self.provider = provider
        self._thinking_default = thinking

    async def run(self, context: AgentContext, messages: Sequence[Message]) -> AgentOutput:
        request = self._load_request(context)
        if not request.research_result or not request.fact_check_result:
            raise ScriptwriterInputError("Missing research or fact-check result")

        valid_claim_ids = {c.id for c in request.research_result.claims}
        valid_source_ids = {s.id for s in request.research_result.sources}

        # Build prompt data
        research_claims = "\n".join(f"- {c.id}: {c.statement} (src:{','.join(c.source_ids)})" for c in request.research_result.claims)
        fc_items = []
        for c in request.fact_check_result.checked_claims:
            fc_items.append(f"- {c.claim_id}: status={c.status.value} supp={c.supporting_source_ids} contra={c.contradicting_source_ids}")
        fact_check_results = "\n".join(fc_items) if fc_items else "None"
        sources = "\n".join(f"- {s.id}: {s.url} ({s.domain})" for s in request.research_result.sources)

        prompt = PromptBuilder().with_system(SYSTEM_PROMPT.text).with_user(
            USER_PROMPT.render(
                topic=request.topic or request.research_result.topic,
                length=LEN_MAP.get(request.desired_video_length.value, request.desired_video_length.value),
                tone=request.tone.value,
                target_audience=request.target_audience,
                include_cta="yes" if request.include_cta else "no",
                research_claims=research_claims or "None",
                fact_check_results=fact_check_results,
                sources=sources or "None",
            ).text
        )

        options = GenerationOptions(temperature=0.2, top_p=0.9, max_tokens=2048, think=self._resolve_thinking(context))
        try:
            result = await self.provider.generate_structured(prompt.build(), ScriptResult, options=options)
        except StructuredOutputError as exc:
            raise ScriptwriterAIError("Invalid structured AI output") from exc
        except Exception as exc:
            raise ScriptwriterAIError("AI provider error") from exc

        try:
            script = ScriptResult.model_validate(result.model_dump())
        except Exception as exc:
            raise ScriptwriterGroundingError("Script validation failed") from exc

        # Validate claim references
        all_refs = []
        for s in script.sections:
            for r in s.claim_references:
                if isinstance(r, dict):
                    all_refs.append(r)
                else:
                    all_refs.append(r.model_dump())
        try:
            validate_claim_references(all_refs, valid_claim_ids, valid_source_ids)
        except Exception as exc:
            raise ScriptwriterGroundingError(str(exc)) from exc

        # Compute word count and duration
        full_text = " ".join(
            [
                script.hook,
                script.introduction,
                script.narration,
                script.conclusion,
                script.cta,
                *[s.narration for s in script.sections],
            ]
        )
        word_count = len(full_text.split()) if full_text else 0
        wpm = script.words_per_minute or WORDS_PER_MINUTE_DEFAULT
        duration = int(round(word_count / wpm * 60)) if wpm > 0 else 0

        # Build claim refs
        claims_used = []
        for r in all_refs:
            claims_used.append(ScriptClaimReference(**r))

        # Collect sources
        sources_used = []
        for sid in valid_source_ids:
            for s in request.research_result.sources:
                if s.id == sid:
                    sources_used.append({"id": s.id, "url": s.url, "domain": s.domain})
                    break

        final = script.model_copy(
            update={
                "word_count": word_count,
                "estimated_duration_seconds": duration,
                "claims_used": tuple(claims_used),
                "sources": tuple(sources_used),
            }
        )

        return AgentOutput.ok(data=final)

    def _load_request(self, context: AgentContext) -> ScriptRequest:
        raw = context.config.get(REQUEST_KEY)
        if raw is not None:
            if isinstance(raw, ScriptRequest):
                return raw
            return ScriptRequest.model_validate(raw)
        raise ScriptwriterInputError("No script request provided")

    def _resolve_thinking(self, context: AgentContext) -> bool:
        val = context.config.get("thinking")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in {"true", "1", "yes", "on"}
        return self._thinking_default
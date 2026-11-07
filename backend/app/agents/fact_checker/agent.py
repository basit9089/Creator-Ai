"""Fact Checker Agent.

Consumes Researcher output and independently verifies claims against
supplied source evidence. Does not browse the internet.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from app.agents.base import (
    Agent,
    AgentContext,
    AgentError,
    AgentOutput,
    AgentStatus,
    Capability,
)
from app.agents.fact_checker.exceptions import (
    FactCheckAIError,
    FactCheckGroundingError,
    FactCheckInputError,
    FactCheckNoClaimsError,
)
from app.agents.fact_checker.grounding import ground_fact_check, verify_references
from app.agents.fact_checker.prompts import FACT_CHECK_SCHEMA, SYSTEM_PROMPT, USER_PROMPT
from app.agents.fact_checker.schemas import (
    ClaimVerification,
    EvidenceEntry,
    EvidenceQuality,
    FactCheckAnalysis,
    FactCheckRequest,
    FactCheckResult,
    VerificationStatus,
)
from app.agents.researcher.schemas import ResearchResult
from app.core.logging import get_logger
from app.prompts.builder import PromptBuilder
from app.services.ai.provider import AIProvider, GenerationOptions, Message, StructuredOutputError

logger = get_logger("agents.fact_checker")

REQUEST_CONTEXT_KEY = "fact_check_request"
MAX_EVIDENCE_EXCERPT = 300


class FactCheckerAgent(Agent):
    """Independently verifies claims against Researcher evidence."""

    name = "fact_checker"
    version = "1.0.0"
    description = "Verifies claims against retrieved evidence; does not browse the web."
    capabilities = frozenset({Capability.STRUCTURED_OUTPUT, Capability.RETRYABLE})

    def __init__(
        self,
        *,
        provider: AIProvider | None = None,
        thinking: bool = False,
    ) -> None:
        super().__init__()
        self.provider = provider
        self._thinking_default = thinking

    async def run(
        self,
        context: AgentContext,
        messages: Sequence[Message],
    ) -> AgentOutput:
        request = self._load_request(context, messages)
        logger.info(
            "fact_check_start topic=%s claims=%d sources=%d",
            request.research_result.topic,
            len(request.research_result.claims),
            len(request.research_result.sources),
        )

        claims_to_verify = self._select_claims(request)
        if not claims_to_verify:
            raise FactCheckNoClaimsError("No claims to verify in the provided research result.")

        evidence_block = self._build_evidence_block(request.research_result)
        claims_block = self._build_claims_block(claims_to_verify)

        try:
            analysis = await self._call_provider(
                topic=request.research_result.topic,
                evidence_block=evidence_block,
                claims_block=claims_block,
                thinking=self._resolve_thinking(context),
            )
        except StructuredOutputError as exc:
            raise FactCheckGroundingError("Fact Checker model output could not be parsed as structured JSON.") from exc
        except AgentError:
            raise
        except Exception as exc:
            raise FactCheckAIError("The AI provider could not verify the claims.") from exc

        grounded = ground_fact_check(analysis, request.research_result)
        dangling = verify_references(request.research_result, grounded.checked_claims)
        if dangling:
            logger.warning("fact_check_dangling_references refs=%s", dangling)

        result = self._build_result(request, grounded, claims_to_verify)
        logger.info(
            "fact_check_complete topic=%s checked=%d verified=%d contradicted=%d insufficient=%d",
            result.topic,
            result.claims_checked,
            result.verified_claims,
            result.contradicted_claims,
            result.insufficient_claims,
        )
        success = result.verified_claims > 0 or result.partially_verified_claims > 0
        if not success and result.contradicted_claims == 0:
            success = False
        return AgentOutput(
            success=success,
            data=result,
            metadata={
                "sources_considered": result.sources_considered,
                "claims_checked": result.claims_checked,
                "verified": result.verified_claims,
                "contradicted": result.contradicted_claims,
                "insufficient": result.insufficient_claims,
            },
        )

    def _load_request(
        self,
        context: AgentContext,
        messages: Sequence[Message],
    ) -> FactCheckRequest:
        raw = context.config.get(REQUEST_CONTEXT_KEY)
        if raw is not None:
            if isinstance(raw, FactCheckRequest):
                return raw
            try:
                return FactCheckRequest.model_validate(raw)
            except Exception as exc:
                raise FactCheckInputError("Invalid fact check request in context.") from exc

        if not messages:
            raise FactCheckInputError("No fact check request provided.")

        # Fallback: not expected in normal flow
        raise FactCheckInputError("No fact check request found in context.")

    def _select_claims(self, request: FactCheckRequest):
        research = request.research_result
        if request.claims:
            ids = set(request.claims)
            return [c for c in research.claims if c.id in ids]
        return list(research.claims)

    def _build_evidence_block(self, research: ResearchResult) -> str:
        lines: list[str] = []
        lines.append("RESEARCH SOURCES (supplied evidence, only these):")
        for i, src in enumerate(research.sources, start=1):
            lines.append(f"[{src.id}] {src.title or src.url}")
            lines.append(f"  URL: {src.url}")
            lines.append(f"  Domain: {src.domain}")
            if src.snippet:
                lines.append(f"  Snippet: {src.snippet[:200]}")
            lines.append("")
        return "\n".join(lines)

    def _build_claims_block(self, claims) -> str:
        lines: list[str] = []
        for c in claims:
            conf = c.confidence.value if c.confidence else "unknown"
            lines.append(f"Claim {c.id}: {c.statement}")
            lines.append(f"  Researcher confidence: {conf}")
            if c.source_ids:
                lines.append(f"  Cited sources: {', '.join(c.source_ids)}")
            lines.append("")
        return "\n".join(lines)

    async def _call_provider(
        self,
        *,
        topic: str,
        evidence_block: str,
        claims_block: str,
        thinking: bool,
    ) -> FactCheckResult:
        prompt = PromptBuilder().with_system(SYSTEM_PROMPT.text).with_user(
            USER_PROMPT.render(
                topic=topic,
                research_sources=evidence_block,
                claims=claims_block,
            ).text
        )
        options = GenerationOptions(
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
            think=thinking,
        )
        result = await self.provider.generate_structured(
            prompt.build(),
            FactCheckAnalysis,
            options=options,
        )
        # Ensure required fields for FactCheckResult
        data = result.model_dump()
        if data.get("overall_status") is None:
            data["overall_status"] = VerificationStatus.INSUFFICIENT_EVIDENCE
        return FactCheckResult.model_validate(data)

    def _resolve_thinking(self, context: AgentContext) -> bool:
        val = context.config.get("thinking")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in {"true", "1", "yes", "on"}
        return self._thinking_default

    def _build_result(
        self,
        request: FactCheckRequest,
        analysis: FactCheckResult,
        claims_to_verify,
    ) -> FactCheckResult:
        checked = list(analysis.checked_claims)
        verified = sum(1 for c in checked if c.status == VerificationStatus.VERIFIED)
        partial = sum(1 for c in checked if c.status == VerificationStatus.PARTIALLY_VERIFIED)
        contradicted = sum(1 for c in checked if c.status == VerificationStatus.CONTRADICTED)
        insufficient = sum(
            1 for c in checked if c.status in (VerificationStatus.INSUFFICIENT_EVIDENCE, VerificationStatus.UNSUPPORTED)
        )
        unsupported_ids = [c.claim_id for c in checked if c.status == VerificationStatus.UNSUPPORTED and not c.supporting_source_ids]

        overall = VerificationStatus.INSUFFICIENT_EVIDENCE
        if contradicted > 0:
            overall = VerificationStatus.CONTRADICTED
        elif verified > 0 and partial == 0 and contradicted == 0 and insufficient == 0:
            overall = VerificationStatus.VERIFIED
        elif verified > 0 or partial > 0:
            overall = VerificationStatus.PARTIALLY_VERIFIED

        return analysis.model_copy(
            update={
                "topic": request.research_result.topic,
                "overall_status": overall,
                "checked_claims": tuple(checked),
                "unsupported_claims": tuple(unsupported_ids),
                "sources_considered": len(request.research_result.sources),
                "claims_checked": len(checked),
                "verified_claims": verified,
                "partially_verified_claims": partial,
                "contradicted_claims": contradicted,
                "insufficient_claims": insufficient,
                "unsupported_claims_count": len(unsupported_ids),
                "generated_at": datetime.now().astimezone(),
            }
        )
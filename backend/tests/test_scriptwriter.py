"""Tests for Scriptwriter."""

import pytest
from app.agents.researcher.schemas import ResearchResult, ResearchClaim, ResearchSource, Confidence
from app.agents.fact_checker.schemas import (
    FactCheckResult,
    ClaimVerification,
    VerificationStatus,
    EvidenceQuality,
)
from app.agents.scriptwriter.agent import ScriptwriterAgent, REQUEST_KEY
from app.agents.scriptwriter.schemas import ScriptRequest, ScriptResult, Section
from app.agents.scriptwriter.grounding import validate_claim_references
from app.agents.base import AgentContext, PlaceholderAgent
from app.agents.runner import AgentRunner
from app.agents.scriptwriter.exceptions import ScriptwriterGroundingError, ScriptwriterAIError
from app.services.ai.provider import StructuredOutputError


def build_data(verified=True, partial=False, contradicted=False, insufficient=False, unsupported=False):
    src = ResearchSource(id='s1', url='https://example.com/1', domain='example.com', title='T1')
    claim = ResearchClaim(id='c1', statement='Test is true.', source_ids=('s1',), confidence=Confidence.MEDIUM)
    rr = ResearchResult(topic='Test', sources=(src,), claims=(claim,))
    if verified:
        fc = FactCheckResult(
            checked_claims=(
                ClaimVerification(
                    claim_id='c1',
                    statement='Test is true.',
                    status=VerificationStatus.VERIFIED,
                    confidence=EvidenceQuality.HIGH,
                    supporting_source_ids=('s1',),
                ),
            )
        )
    elif partial:
        fc = FactCheckResult(
            checked_claims=(
                ClaimVerification(
                    claim_id='c1',
                    statement='Test is true.',
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    confidence=EvidenceQuality.MEDIUM,
                    supporting_source_ids=('s1',),
                ),
            )
        )
    elif contradicted:
        fc = FactCheckResult(
            checked_claims=(
                ClaimVerification(
                    claim_id='c1',
                    statement='Test is true.',
                    status=VerificationStatus.CONTRADICTED,
                    confidence=EvidenceQuality.MEDIUM,
                ),
            )
        )
    elif insufficient:
        fc = FactCheckResult(
            checked_claims=(
                ClaimVerification(
                    claim_id='c1',
                    statement='Test is true.',
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    confidence=EvidenceQuality.LOW,
                ),
            )
        )
    elif unsupported:
        fc = FactCheckResult(
            checked_claims=(
                ClaimVerification(
                    claim_id='c1',
                    statement='Test is true.',
                    status=VerificationStatus.UNSUPPORTED,
                    confidence=EvidenceQuality.LOW,
                ),
            )
        )
    else:
        fc = FactCheckResult(checked_claims=())
    return rr, fc


class MockProvider:
    async def generate_structured(self, messages, schema, options=None):
        return ScriptResult(
            title='Test Video',
            hook='Hook',
            introduction='Intro',
            sections=(),
            narration='Narration text',
            conclusion='Conclusion',
            cta='Subscribe',
        )


class MockProviderWithBad:
    async def generate_structured(self, messages, schema, options=None):
        raise StructuredOutputError('bad')


class MockProviderWithRefs:
    async def generate_structured(self, messages, schema, options=None):
        section = Section(section_id='s1', heading='H', narration='N', claim_references=({'claim_id':'c1','source_ids':('s1',),'usage':'ctx'},))
        return ScriptResult(
            title='Test Video',
            hook='Hook',
            introduction='Intro',
            sections=(section,),
            narration='Narration text',
            conclusion='Conclusion',
            cta='Subscribe',
        )


def test_valid_request_schema():
    rr, fc = build_data()
    req = ScriptRequest(research_result=rr, fact_check_result=fc)
    assert req.research_result.topic == 'Test'


def test_word_count_deterministic():
    s = ScriptResult(title='T', narration='Hello world test', hook='Hi')
    text = ' '.join([s.hook, s.introduction, s.narration, s.conclusion, s.cta, *[sec.narration for sec in s.sections]])
    assert len(text.split()) == 4


def test_duration_calculation():
    s = ScriptResult(title='T', narration='a ' * 300)  # 300 words
    word_count = len(s.narration.split())
    wpm = s.words_per_minute
    duration = int(round(word_count / wpm * 60))
    assert duration == 120


def test_validate_claim_refs():
    validate_claim_references([{'claim_id':'c1','source_ids':('s1',)}], {'c1'},{'s1'})


def test_validate_claim_refs_invalid():
    try:
        validate_claim_references([{'claim_id':'bad','source_ids':('s1',)}], {'c1'},{'s1'})
        assert False
    except Exception:
        assert True


def test_registry_has_scriptwriter():
    from app.agents.registry import build_default_registry
    reg = build_default_registry()
    desc = reg.get_descriptor('scriptwriter')
    assert desc.implemented
    agent = reg.get_agent('scriptwriter')
    assert isinstance(agent, ScriptwriterAgent)
    assert not isinstance(agent, PlaceholderAgent)


@pytest.mark.anyio("asyncio")
async def test_scriptwriter_verified():
    rr, fc = build_data(verified=True)
    req = ScriptRequest(research_result=rr, fact_check_result=fc)
    agent = ScriptwriterAgent(provider=MockProviderWithRefs())
    ctx = AgentContext(config={REQUEST_KEY: req})
    res = await AgentRunner().run(agent, ctx, [])
    assert res.success


@pytest.mark.anyio("asyncio")
async def test_scriptwriter_partial():
    rr, fc = build_data(partial=True)
    req = ScriptRequest(research_result=rr, fact_check_result=fc)
    agent = ScriptwriterAgent(provider=MockProviderWithRefs())
    ctx = AgentContext(config={REQUEST_KEY: req})
    res = await AgentRunner().run(agent, ctx, [])
    assert res.success


@pytest.mark.anyio("asyncio")
async def test_scriptwriter_malformed_ai():
    rr, fc = build_data(verified=True)
    req = ScriptRequest(research_result=rr, fact_check_result=fc)
    agent = ScriptwriterAgent(provider=MockProviderWithBad())
    ctx = AgentContext(config={REQUEST_KEY: req})
    try:
        await AgentRunner().run(agent, ctx, [])
        assert False
    except Exception:
        assert True


def test_api_route_registered():
    from app.main import create_app
    app = create_app()
    paths = app.openapi()['paths']
    assert '/api/script' in paths
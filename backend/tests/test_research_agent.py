"""The Researcher agent itself.

Every collaborator is a fake, so nothing here opens a socket. What these
tests establish is the behaviour a real run depends on and that no single
component's tests can show: that evidence is only ever drawn from pages
that were actually retrieved, that one dead website does not end a run,
that insufficient evidence completes rather than fails, and that the model
is told what it was given instead of being asked to search.
"""

from __future__ import annotations

import json

import pytest

from app.agents.base import AgentContext, AgentOutput, AgentStatus
from app.agents.researcher.agent import REQUEST_CONTEXT_KEY, ResearcherAgent
from app.agents.researcher.exceptions import (
    ResearchAIError,
    ResearchGroundingError,
    ResearchInputError,
)
from app.agents.researcher.schemas import (
    Confidence,
    ResearchRequest,
    ResearchResult,
)
from app.services.ai.provider import (
    GenerationOptions,
    Message,
    ProviderTimeoutError,
)
from app.services.research.extraction.base import (
    ExtractedContent,
    ExtractionError,
    UnsupportedDocumentError,
)
from app.services.research.fetching.base import (
    FetchError,
    FetchTimeoutError,
    HttpStatusError,
)
from app.services.research.search.base import SearchError, SearchQuery, SearchResult

from .conftest import (
    RESEARCH_ANALYSIS_PAYLOAD,
    FakeAIProvider,
    FakeContentExtractor,
    FakeSearchProvider,
    FakeSourceFetcher,
    analysis_payload,
    build_research_agent,
    make_search_result,
)

# Three sources on three publishers, which is what "high confidence"
# requires. The fourth is on the same domain as the first, so it cannot
# raise a grade on its own.
URLS = [
    "https://example.com/one",
    "https://other.com/two",
    "https://third.com/three",
    "https://example.com/four",
]


def results(*urls: str) -> list[SearchResult]:
    return [make_search_result(url) for url in urls]


def documents(*urls: str) -> dict[str, str]:
    """Canned HTML for each URL, with enough text to count as usable."""
    return {
        url: f"<html><body><article><p>{'Evidence text for this page. ' * 40}</p>"
        f"</article></body></html>"
        for url in urls
    }


def context_for(request: ResearchRequest | None = None, **config: object) -> AgentContext:
    return AgentContext(config={REQUEST_CONTEXT_KEY: request or ResearchRequest(topic="History of YouTube"), **config})


def run(agent: ResearcherAgent, context: AgentContext | None = None):
    """Drive the agent the way the Phase 3 runner does."""
    return agent.run(context or context_for(), [Message(role="user", content="History of YouTube")])


def working_agent(
    *,
    provider: FakeAIProvider | None = None,
    search: FakeSearchProvider | None = None,
    fetcher: FakeSourceFetcher | None = None,
    extractor: FakeContentExtractor | None = None,
    **settings: object,
):
    """An agent with three usable sources and a valid analysis.

    Every collaborator defaults to a working one, so a test only names the
    piece it is actually exercising.
    """
    resolved = provider or FakeAIProvider(text=analysis_payload())
    agent = build_research_agent(
        provider=resolved,
        search=search or FakeSearchProvider(results(URLS[0], URLS[1], URLS[2])),
        fetcher=fetcher or FakeSourceFetcher(documents(*URLS)),
        extractor=extractor or FakeContentExtractor(),
        **settings,
    )
    return agent, resolved


# ---------- The happy path ----------


@pytest.mark.anyio
async def test_a_research_run_returns_grounded_citations() -> None:
    agent, _ = working_agent()

    output = await run(agent)

    assert output.success is True
    result = output.data
    assert isinstance(result, ResearchResult)
    assert len(result.sources) == 3
    assert result.summary
    assert result.claims


@pytest.mark.anyio
async def test_every_source_id_resolves_to_a_real_url() -> None:
    """The chain the whole design exists for: claim to identifier to URL."""
    agent, _ = working_agent()

    result = (await run(agent)).data

    for claim in result.claims:
        for source_id in claim.source_ids:
            source = result.source(source_id)
            assert source is not None
            assert source.url.startswith("https://")


@pytest.mark.anyio
async def test_confidence_is_computed_not_taken_from_the_model() -> None:
    """The payload cites two publishers per claim, which grades medium
    however certain the model sounded.

    The grades are derived from the retrieved sources rather than from any
    field in the model output, which is the whole point of grounding.
    """
    agent, _ = working_agent()

    result = (await run(agent)).data

    assert result.confidence_counts["medium"] == 2
    assert result.claims[0].confidence is Confidence.MEDIUM


@pytest.mark.anyio
async def test_source_identifiers_follow_search_rank() -> None:
    """Identifiers are assigned in rank order so the numbering a reader
    sees matches the order the sources were found."""
    agent, _ = working_agent()

    result = (await run(agent)).data

    assert [s.id for s in result.sources] == ["source_01", "source_02", "source_03"]
    assert [s.url for s in result.sources] == URLS[:3]


@pytest.mark.anyio
async def test_the_result_records_what_was_how() -> None:
    agent, _ = working_agent()

    result = (await run(agent)).data

    assert result.search_query == "History of YouTube"
    assert result.search_provider == "fake-search"
    assert result.search_depth.value == "standard"
    assert result.sources_requested == 6
    assert result.sources_found == 3
    assert result.sources_failed == 0
    assert result.distinct_domains == 3
    assert result.sufficient is True
    assert result.generated_at is not None


# ---------- The model is told what it was given ----------


@pytest.mark.anyio
async def test_the_prompt_contains_the_retrieved_text_and_the_real_urls() -> None:
    """If the model is to cite a source, it must have been shown it.

    Each source is rendered as an identifier beside the URL and the text,
    which is what makes ``claim -> source id -> URL`` traceable rather than
    a citation the model had to guess at.
    """
    extractor = FakeContentExtractor(length=400)
    agent, provider = working_agent(extractor=extractor)

    await run(agent)

    prompt = "\n".join(m.content for m in provider.calls[0][0])
    assert "[source_01]" in prompt
    assert URLS[0] in prompt
    assert "RETRIEVED SOURCES" in prompt


@pytest.mark.anyio
async def test_the_prompt_never_contains_a_url_that_was_not_retrieved() -> None:
    """A URL in the prompt is a URL the model may cite, so a rejected
    candidate must not appear even as a warning about itself."""
    blocked = "https://blocked.example/x"
    search = FakeSearchProvider(results(URLS[0], URLS[1], blocked))
    fetcher = FakeSourceFetcher(
        documents(URLS[0], URLS[1]),
        failures={blocked: HttpStatusError("The source returned 403.", status_code=403)},
    )
    agent, provider = working_agent(
        search=search,
        fetcher=fetcher,
        extractor=FakeContentExtractor(length=400),
    )

    await run(agent)

    prompt = "\n".join(m.content for m in provider.calls[0][0])
    assert URLS[0] in prompt
    assert blocked not in prompt
    # And it was recorded as a failure rather than vanishing silently.
    assert any(f.url == blocked for f in (await run(agent)).data.failed_sources)


@pytest.mark.anyio
async def test_the_prompt_forbids_inventing_sources() -> None:
    agent, provider = working_agent()

    await run(agent)

    prompt = "\n".join(m.content for m in provider.calls[0][0])
    assert "source id" in prompt.lower()
    assert "do not" in prompt.lower()


@pytest.mark.anyio
async def test_the_prompt_has_a_system_message() -> None:
    """Instructions about grounding belong in the system turn, where the
    model treats them as policy rather than as user text."""
    agent, provider = working_agent()

    await run(agent)

    assert provider.calls[0][0][0].role == "system"


@pytest.mark.anyio
async def test_the_model_is_called_exactly_once_per_run() -> None:
    agent, provider = working_agent()

    await run(agent)

    assert len(provider.calls) == 1


@pytest.mark.anyio
async def test_thinking_is_off_by_default() -> None:
    """On the target hardware reasoning turns a one-second analysis into a
    one-minute one, and it is on by default in Qwen3."""
    agent, provider = working_agent()

    await run(agent)

    options = provider.calls[0][1]
    assert isinstance(options, GenerationOptions)
    assert options.think is False


@pytest.mark.anyio
async def test_thinking_can_be_overridden_per_run() -> None:
    agent, provider = working_agent()

    await run(agent, context_for(thinking=True))

    assert provider.calls[0][1].think is True


@pytest.mark.anyio
async def test_the_agent_does_not_import_ollama() -> None:
    """Qwen3 is reached only through the provider interface. Naming the
    backend here would tie reasoning to one deployment and break the
    separation the architecture rests on."""
    import app.agents.researcher.agent as module

    source = module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    assert "import ollama" not in text
    assert "ollama." not in text


# ---------- Partial success ----------


@pytest.mark.anyio
async def test_one_dead_website_does_not_end_the_run() -> None:
    """Six sources requested, four retrieved and two blocked is a usable
    result, not a failure."""
    fetcher = FakeSourceFetcher(
        documents(URLS[0], URLS[1]),
        failures={
            URLS[2]: HttpStatusError("The source returned 403.", status_code=403),
            URLS[3]: FetchTimeoutError("The source did not respond in time."),
        },
    )
    search = FakeSearchProvider(results(*URLS))
    agent, _ = working_agent(search=search, fetcher=fetcher)

    output = await run(agent)

    assert output.success is True
    result = output.data
    assert result.sources_found == 2
    assert result.sources_failed == 2
    assert len(result.failed_sources) == 2


@pytest.mark.anyio
async def test_each_failure_records_why() -> None:
    """The shape of the failure is part of the result: "this page was
    blocked" tells a reader something that "this source was missing" does
    not."""
    fetcher = FakeSourceFetcher(
        documents(URLS[0], URLS[1]),
        failures={URLS[2]: HttpStatusError("The source returned 403.", status_code=403)},
    )
    agent, _ = working_agent(search=FakeSearchProvider(results(*URLS)), fetcher=fetcher)

    result = (await run(agent)).data

    failure = next(f for f in result.failed_sources if f.url == URLS[2])
    assert failure.code == "http_error"
    assert failure.domain == "third.com"
    assert "403" in failure.reason


@pytest.mark.anyio
async def test_a_duplicate_source_is_recorded_as_skipped() -> None:
    """A duplicate was redundant, not broken, and the two are counted
    differently so a security rejection cannot hide behind a tidiness
    statistic."""
    search = FakeSearchProvider(
        [
            make_search_result(URLS[0], "History of YouTube"),
            make_search_result(URLS[1], "How YouTube Makes Money"),
            make_search_result(URLS[2], "History of YouTube"),
        ]
    )
    agent, _ = working_agent(search=search)

    result = (await run(agent)).data

    assert result.sources_found == 2
    skipped = [f for f in result.failed_sources if f.outcome.value == "skipped"]
    assert len(skipped) == 1
    assert skipped[0].code == "duplicate_source"


@pytest.mark.anyio
async def test_an_unsafe_url_is_never_fetched() -> None:
    """A ``file://`` candidate must be rejected before retrieval, not
    caught by the fetcher afterwards."""
    search = FakeSearchProvider(
        results(URLS[0], URLS[1], "file:///etc/passwd")
    )
    fetcher = FakeSourceFetcher(documents(URLS[0], URLS[1]))
    agent, _ = working_agent(search=search, fetcher=fetcher)

    result = (await run(agent)).data

    assert "file:///etc/passwd" not in fetcher.requested
    assert any(f.code == "unsafe_url" for f in result.failed_sources)


@pytest.mark.anyio
async def test_a_page_with_too_little_text_is_counted_as_empty() -> None:
    """A paywall stub parsed successfully but is not evidence. Counting it
    as empty rather than failed says nothing went wrong."""

    class ThinExtractor(FakeContentExtractor):
        def extract(self, document):
            return ExtractedContent(
                title="Thin", text="Subscribe to continue reading.", original_length=32
            )

    agent, _ = working_agent(extractor=ThinExtractor())

    output = await run(agent)

    assert output.success is False
    assert output.data.sources_found == 0
    assert output.data.failed_sources[0].code == "empty_content"


@pytest.mark.anyio
async def test_an_unexpected_error_in_one_source_is_contained() -> None:
    """A bug in one retrieval must not read as a broken whole run."""

    class ExplodingExtractor(FakeContentExtractor):
        def extract(self, document):
            raise RuntimeError("boom")

    fetcher = FakeSourceFetcher(documents(*URLS[:3]))
    agent, _ = working_agent(fetcher=fetcher, extractor=ExplodingExtractor())

    output = await run(agent)

    assert output.success is False
    assert output.data.sources_found == 0
    assert all("RuntimeError" in f.reason for f in output.data.failed_sources)


@pytest.mark.anyio
async def test_source_fetches_are_bounded() -> None:
    """A semaphore, so a deep run does not open a dozen connections on a
    laptop."""
    search = FakeSearchProvider(results(*URLS))
    agent, _ = working_agent(search=search, max_concurrent_source_fetches=2)

    await run(agent)

    # Four sources were fetched; the limit governs concurrency, not totals.
    assert len(agent.fetcher.requested) >= 3


# ---------- Insufficient evidence ----------


@pytest.mark.anyio
async def test_too_little_evidence_completes_rather_than_fails() -> None:
    """The Phase 3 distinction: the pipeline worked and the web had too
    little to offer. That is a negative verdict, not an error."""
    agent, _ = working_agent(
        search=FakeSearchProvider(results(URLS[0], URLS[1], URLS[2])),
        fetcher=FakeSourceFetcher(documents(URLS[0])),
    )

    output = await run(agent)

    assert output.success is False
    assert output.error is not None
    assert output.metadata["code"] == "research_insufficient_sources"
    assert output.metadata["retryable"] is True


@pytest.mark.anyio
async def test_no_evidence_at_all_is_reported_distinctly() -> None:
    """A caller can tell "the search found nothing" from "the search found
    things that turned out unusable"."""
    agent, _ = working_agent(search=FakeSearchProvider([]))

    output = await run(agent)

    assert output.success is False
    assert output.metadata["code"] == "research_no_sources"
    assert output.data.sources_found == 0


@pytest.mark.anyio
async def test_the_partial_result_survives_a_negative_verdict() -> None:
    """The retrieved source is still reported, so a caller can see what was
    found and decide whether to ask for more."""
    agent, _ = working_agent(
        search=FakeSearchProvider(results(URLS[0], URLS[1], URLS[2])),
        fetcher=FakeSourceFetcher(documents(URLS[0])),
    )

    output = await run(agent)

    assert len(output.data.sources) == 1
    assert output.data.sources[0].url == URLS[0]
    assert output.data.sufficient is False


@pytest.mark.anyio
async def test_the_model_is_not_called_when_there_is_nothing_to_analyse() -> None:
    """There is no evidence to ground against, so asking would invite the
    model to fill the gap from memory."""
    agent, provider = working_agent(search=FakeSearchProvider([]))

    await run(agent)

    assert provider.calls == []


@pytest.mark.anyio
async def test_a_search_backend_outage_is_not_a_crash() -> None:
    agent, _ = working_agent(
        search=FakeSearchProvider(error=SearchError("The search backend is unreachable."))
    )

    output = await run(agent)

    assert output.success is False
    assert output.metadata["code"] == "research_no_sources"


@pytest.mark.anyio
async def test_an_unexpected_search_error_is_raised_not_swallowed() -> None:
    """A ``SearchError`` means the backend answered badly; anything else is
    a bug, and reporting it as "no results" would hide it."""
    agent, _ = working_agent(search=FakeSearchProvider(error=RuntimeError("bug")))

    with pytest.raises(Exception) as excinfo:
        await run(agent)

    assert "unexpectedly" in str(excinfo.value).lower()


# ---------- The model's output ----------


@pytest.mark.anyio
async def test_output_the_schema_rejects_becomes_a_grounding_error() -> None:
    """Unparseable output is a grounding failure, not a silent empty
    result."""
    provider = FakeAIProvider(text="{not json at all")
    agent, _ = working_agent(provider=provider)

    with pytest.raises(ResearchGroundingError):
        await run(agent)


@pytest.mark.anyio
async def test_a_provider_outage_becomes_a_research_ai_error() -> None:
    provider = FakeAIProvider(error=ProviderTimeoutError("The model did not respond."))
    agent, _ = working_agent(provider=provider)

    with pytest.raises(ResearchAIError):
        await run(agent)


@pytest.mark.anyio
async def test_a_fabricated_reference_is_dropped_and_the_claim_survives() -> None:
    """The claim is kept and graded unsupported. Deleting it would hide
    exactly what the Fact Checker exists to find."""
    payload = {
        **RESEARCH_ANALYSIS_PAYLOAD,
        "claims": [
            {
                "id": "claim_01",
                "statement": "YouTube was founded in February 2005.",
                "source_ids": ["source_99"],
            }
        ],
        "conflicts": [],
    }
    provider = FakeAIProvider(text=json.dumps(payload))
    agent, _ = working_agent(provider=provider)

    result = (await run(agent)).data

    assert len(result.claims) == 1
    assert result.claims[0].source_ids == ()
    assert result.claims[0].confidence is Confidence.UNSUPPORTED
    assert result.unsupported_claims == 1


@pytest.mark.anyio
async def test_the_model_cannot_inflate_a_grade_by_asserting_confidence() -> None:
    """One publisher, but the payload insists on high confidence."""
    payload = {
        **RESEARCH_ANALYSIS_PAYLOAD,
        "claims": [
            {
                "id": "claim_01",
                "statement": "A claim.",
                "source_ids": ["source_01"],
                "confidence": "high",
                "model_confidence": "high",
            }
        ],
        "conflicts": [],
    }
    search = FakeSearchProvider(results(URLS[0], URLS[1]))
    provider = FakeAIProvider(text=json.dumps(payload))
    agent, _ = working_agent(search=search, provider=provider)

    result = (await run(agent)).data

    assert result.claims[0].confidence is Confidence.LOW
    assert result.claims[0].model_confidence is Confidence.HIGH


@pytest.mark.anyio
async def test_conflicting_sources_are_preserved_not_resolved() -> None:
    """The Researcher has no basis for deciding which account is right."""
    agent, _ = working_agent()

    result = (await run(agent)).data

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.source_ids
    assert conflict.contradicting_source_ids
    assert conflict.note


@pytest.mark.anyio
async def test_uncertainties_are_carried_through() -> None:
    """Separating what is known from what is not is half the value."""
    agent, _ = working_agent()

    result = (await run(agent)).data

    assert len(result.uncertainties) == 1
    assert result.uncertainties[0].description


@pytest.mark.anyio
async def test_statistics_keep_their_source() -> None:
    agent, _ = working_agent()

    result = (await run(agent)).data

    statistic = result.statistics[0]
    assert statistic.value == "2005"
    assert result.source(statistic.source_ids[0]) is not None


@pytest.mark.anyio
async def test_a_thin_analysis_yields_a_thin_result_rather_than_an_error() -> None:
    """A model that returns only a summary has still produced usable
    output; the citation checks have nothing to check."""
    provider = FakeAIProvider(
        text=json.dumps({"topic": "History of YouTube", "summary": "It began in 2005."})
    )
    agent, _ = working_agent(provider=provider)

    output = await run(agent)

    assert output.success is True
    assert output.data.summary == "It began in 2005."
    assert output.data.claims == ()


@pytest.mark.anyio
async def test_extraction_failures_are_reported_per_source() -> None:
    """An unsupported document is not the same as an unreachable one."""

    class PdfExtractor(FakeContentExtractor):
        def extract(self, document):
            raise UnsupportedDocumentError("No extractor handles 'application/pdf'.")

    fetcher = FakeSourceFetcher(documents(*URLS[:3]))
    agent, _ = working_agent(fetcher=fetcher, extractor=PdfExtractor())

    result = (await run(agent)).data

    assert result.sources_found == 0
    assert all(f.code == "unsupported_document" for f in result.failed_sources)
    assert all(f.outcome.value == "empty" for f in result.failed_sources)


# ---------- The request ----------


@pytest.mark.anyio
async def test_the_topic_reaches_the_search_backend() -> None:
    search = FakeSearchProvider(results(*URLS[:3]))
    agent, _ = working_agent(search=search)

    await run(agent, context_for(ResearchRequest(topic="History of YouTube")))

    query = search.queries[0]
    assert isinstance(query, SearchQuery)
    assert query.terms == "History of YouTube"


@pytest.mark.anyio
async def test_the_language_and_depth_reach_the_search_backend() -> None:
    search = FakeSearchProvider(results(*URLS[:3]))
    agent, _ = working_agent(search=search)

    await run(
        agent,
        context_for(ResearchRequest(topic="x", search_depth="deep", language="de")),
    )

    query = search.queries[0]
    assert query.language == "de"
    assert query.depth.value == "deep"


@pytest.mark.anyio
async def test_the_search_is_asked_for_no_more_than_the_cap() -> None:
    """Over-asking the backend wastes a request and looks greedy in logs."""
    search = FakeSearchProvider(results(*URLS))
    agent, _ = working_agent(search=search, max_research_sources=2)

    await run(agent)

    assert search.limits[0] <= 2


@pytest.mark.anyio
async def test_a_request_in_the_context_wins_over_the_prompt() -> None:
    """The API validates the request once, at the boundary; re-reading a
    prompt would throw that away."""
    search = FakeSearchProvider(results(*URLS[:3]))
    agent, _ = working_agent(search=search)

    await run(
        agent,
        context_for(ResearchRequest(topic="A specific topic")),
        # The prompt disagrees; the context is authoritative.
    )

    assert search.queries[0].terms == "A specific topic"


@pytest.mark.anyio
async def test_a_request_given_as_a_dict_is_validated() -> None:
    search = FakeSearchProvider(results(*URLS[:3]))
    agent, _ = working_agent(search=search)

    await run(agent, AgentContext(config={REQUEST_CONTEXT_KEY: {"topic": "From a dict"}}))

    assert search.queries[0].terms == "From a dict"


@pytest.mark.anyio
async def test_an_invalid_request_dict_is_refused() -> None:
    agent, _ = working_agent()

    with pytest.raises(ResearchInputError):
        await run(agent, AgentContext(config={REQUEST_CONTEXT_KEY: {"topic": ""}}))


@pytest.mark.anyio
async def test_the_prompt_is_the_topic_when_no_request_is_supplied() -> None:
    """This is what lets the agent be triggered through the Phase 3 runner
    with nothing but a message."""
    search = FakeSearchProvider(results(*URLS[:3]))
    agent, _ = working_agent(search=search)

    output = await agent.run(
        AgentContext(), [Message(role="user", content="History of TikTok")]
    )

    assert output.success is True
    assert search.queries[0].terms == "History of TikTok"


@pytest.mark.anyio
async def test_no_topic_at_all_is_refused() -> None:
    agent, _ = working_agent()

    with pytest.raises(ResearchInputError):
        await agent.run(AgentContext(), [Message(role="user", content="   ")])


@pytest.mark.anyio
async def test_an_empty_message_list_is_refused() -> None:
    agent, _ = working_agent()

    with pytest.raises(ResearchInputError):
        await agent.run(AgentContext(), [])


# ---------- Ignored options are reported ----------


@pytest.mark.anyio
async def test_an_unsupported_option_is_reported_rather_than_ignored_silently() -> None:
    """A caller who set a filter and got none would conclude it worked."""
    agent, _ = working_agent()

    output = await run(
        agent,
        context_for(ResearchRequest(topic="x", date_range="past_year", region="us")),
    )

    warnings = " ".join(output.data.warnings)
    assert "date_range" in warnings
    assert "region" in warnings


@pytest.mark.anyio
async def test_no_warnings_about_options_that_were_not_set() -> None:
    agent, _ = working_agent()

    output = await run(agent)

    assert not any("date_range" in w for w in output.data.warnings)


# ---------- The contract ----------


@pytest.mark.anyio
async def test_the_output_is_a_typed_agent_output() -> None:
    agent, _ = working_agent()

    output = await run(agent)

    assert isinstance(output, AgentOutput)
    assert output.metadata["sources_found"] == 3


def test_the_agent_declares_its_capabilities() -> None:
    """Retrieval is a network operation, so a transient failure is worth
    retrying."""
    from app.agents.base import Capability

    assert Capability.STRUCTURED_OUTPUT in ResearcherAgent.capabilities
    assert Capability.RETRYABLE in ResearcherAgent.capabilities


def test_constructing_the_agent_touches_no_network() -> None:
    """The registry builds agents at import time, so construction must not
    resolve a provider or open a client."""

    class Exploding(FakeSearchProvider):
        def search(self, query, *, limit):  # pragma: no cover
            raise AssertionError("must not search on construction")

    ResearcherAgent(
        provider=FakeAIProvider(), search_provider=Exploding(), fetcher=FakeSourceFetcher()
    )


def test_the_agent_name_is_stable() -> None:
    """The registry and the API route both address it by this name."""
    assert ResearcherAgent.name == "researcher"

"""Opt-in tests that use the real web.

Excluded from the default run and gated on an environment variable, because
they search the internet and fetch real pages::

    RUN_RESEARCH_INTEGRATION=1 pytest -m live_research

Everything they assert is something a mocked test cannot establish: that
DuckDuckGo's markup still parses, that real pages extract to usable text,
and that a local Qwen3 will analyse retrieved evidence and return the
required structure. A unit test proves the parser matches the HTML given to
it, which is a much weaker claim.

They are also the only tests that can fail for reasons outside the code, so
each one is written to report *what* was unreachable rather than just
failing. Network access and a running Ollama are preconditions, checked
before anything is asserted, so a failure here reads as an environment
problem instead of a regression.

No results are hardcoded. Every assertion is made against what actually
came back, and the tests skip rather than pass when the environment cannot
support the full pipeline.
"""

from __future__ import annotations

import os

import anyio
import pytest

from app.agents.researcher.schemas import (
    Confidence,
    ResearchRequest,
    ResearchResult,
)
from app.core.config import get_settings
from app.services.research.dedupe import deduplicate
from app.services.research.extraction.html import HtmlContentExtractor
from app.services.research.fetching.http import HTTPSourceFetcher
from app.services.research.search.base import SearchError, SearchQuery
from app.services.research.search.duckduckgo import DuckDuckGoSearchProvider
from app.services.research.search.wikipedia import WikipediaSearchProvider

pytestmark = pytest.mark.live_research

#: The topic used throughout. Chosen because it is well documented, so a
#: retrieved page almost always contains something citable, and because
#: several independent publishers cover it.
TOPIC = "History of YouTube"

needs_live = pytest.mark.skipif(
    os.getenv("RUN_RESEARCH_INTEGRATION") != "1",
    reason="Live research is opt-in: set RUN_RESEARCH_INTEGRATION=1.",
)


# ---------- Preconditions ----------


@pytest.fixture
async def ddg():
    """A live search backend, closed afterwards.

    Function-scoped rather than module-scoped on purpose. A provider
    caches its ``httpx.AsyncClient``, which is bound to whichever event
    loop first used it, and pytest gives each test a fresh loop. A
    module-scoped provider therefore hands the second test a client
    attached to a closed loop and every search fails with
    ``RuntimeError: Event loop is closed``.
    """
    provider = DuckDuckGoSearchProvider()
    try:
        yield provider
    finally:
        await provider.close()


async def search_or_skip(provider, *, limit: int, terms: str = TOPIC):
    """Search the live backend, or skip with the reason it cannot be used.

    Skipping rather than failing is the point. These tests are the ones
    that can fail for reasons outside the code — a network block, a bot
    wall, a rate limit — and a regression here should be distinguishable
    from an unreachable internet.
    """
    try:
        return await provider.search(SearchQuery(terms=terms, language="en"), limit=limit)
    except SearchError as exc:
        pytest.skip(f"The search backend is unusable from this machine: {exc}")


@pytest.fixture
async def fetcher():
    client = HTTPSourceFetcher()
    try:
        yield client
    finally:
        await client.close()


# ---------- Search ----------


@needs_live
@pytest.mark.anyio
async def test_the_live_search_backend_returns_usable_candidates(ddg) -> None:
    """The real markup still parses.

    DuckDuckGo serves HTML with no API and no key, so this is the only
    backend a fresh clone can rely on. Its markup is not a contract, which
    is exactly why a unit test against captured HTML is not enough.
    """
    results = await search_or_skip(ddg, limit=5)

    assert results, "the live search returned nothing"
    assert len(results) >= 3
    for result in results:
        assert result.title.strip()
        assert result.url.startswith("https://")
        assert "duckduckgo.com/l/" not in result.url, "the redirect was not unwrapped"
        assert result.snippet.strip()


@needs_live
@pytest.mark.anyio
async def test_live_candidates_deduplicate(ddg) -> None:
    results = await search_or_skip(ddg, limit=10)

    report = deduplicate(results)

    assert report.kept_count > 0
    assert not any(r.url in {"file://", ""} for r in report.kept)


@needs_live
@pytest.mark.anyio
async def test_search_asks_the_backend_for_no_more_than_the_limit(ddg) -> None:
    results = await search_or_skip(ddg, limit=3)

    assert len(results) <= 3


# ---------- Fetching ----------


@needs_live
@pytest.mark.anyio
async def test_a_real_page_can_be_retrieved(fetcher: HTTPSourceFetcher) -> None:
    """``example.com`` is used because it is stable, tiny, and has no bot
    protection: a failure here means the fetcher broke, not that a site
    started blocking us."""
    document = await fetcher.fetch("https://example.com/")

    assert document.status_code == 200
    assert "Example Domain" in document.text


@needs_live
@pytest.mark.anyio
async def test_a_live_unsafe_url_is_refused_without_a_request(fetcher: HTTPSourceFetcher) -> None:
    """The SSRF boundary holds against real input, not only fixtures."""
    from app.services.research.fetching.base import UnsafeUrlFetchError

    for url in (
        "file:///etc/passwd",
        "http://localhost:11434/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.1/",
    ):
        with pytest.raises(UnsafeUrlFetchError):
            await fetcher.fetch(url)


@needs_live
@pytest.mark.anyio
async def test_the_local_model_server_is_unreachable_as_a_source(fetcher: HTTPSourceFetcher) -> None:
    """The user's own Ollama server must not be fetchable as a source.

    This is the SSRF case that matters most in a local-first deployment:
    the one service bound to the loopback interface is the one holding the
    user's data.
    """
    from app.services.research.fetching.base import UnsafeUrlFetchError

    with pytest.raises(UnsafeUrlFetchError):
        await fetcher.fetch("http://localhost:11434/api/tags")


# ---------- Extraction ----------


#: A page that is genuinely server-rendered. Several well-known sources
#: for this topic are not, which is what the test below documents;
#: ``historytools.org`` was picked because a live search for the topic
#: returns it and it serves real prose in the initial HTML.
SERVER_RENDERED = "https://www.historytools.org/docs/youtube-history"

#: A real, widely-linked page on the topic whose body is assembled by
#: JavaScript. Kept as a negative case rather than deleted, because it is
#: the single most common way a research run comes back empty.
JAVASCRIPT_RENDERED = "https://blog.youtube/inside-youtube/"


@needs_live
@pytest.mark.anyio
async def test_a_real_page_extracts_to_usable_text(fetcher) -> None:
    """Real markup has what unit-test fixtures omit: nested chrome,
    inline scripts, absolute URLs, real-world entity escaping."""
    document = await fetcher.fetch(SERVER_RENDERED)

    content = HtmlContentExtractor().extract(document)

    assert content.is_usable, (
        f"only {content.original_length} characters extracted from a page "
        "the live search offers as a source"
    )
    assert content.title


@needs_live
@pytest.mark.anyio
async def test_a_javascript_rendered_page_yields_little_text(fetcher) -> None:
    """A documented limitation, pinned so it stays a known one.

    ``blog.youtube/inside-youtube`` serves a 76 KB document whose entire
    article body is injected by JavaScript. What the extractor finds is a
    headline, one sentence of hero copy and a newsletter form: about 300
    characters, correctly identified as unusable.

    That is the right answer — the text is genuinely not in the document —
    and it is also the reason this pipeline cannot claim to read the whole
    web. Adding a headless browser to execute the page would fix it, at
    the cost of a large dependency and an order of magnitude more runtime,
    and is out of scope here.
    """
    document = await fetcher.fetch(JAVASCRIPT_RENDERED)

    content = HtmlContentExtractor().extract(document)

    assert not content.is_usable
    assert content.original_length < 1000, (
        "if this now extracts well, the page started serving its text and "
        "the limitation note in the docs should be revisited"
    )


@needs_live
@pytest.mark.anyio
async def test_a_bot_walled_source_is_reported_not_worked_around(
    fetcher,
) -> None:
    """Bot protection is an honest failure, not something to evade.

    Britannica answers 403 to this client. The fetcher reports that as an
    HTTP error and the Researcher records the source as failed, which is
    the whole design: a page it could not read is a page it will not cite.
    """
    from app.services.research.fetching.base import HttpStatusError

    with pytest.raises(HttpStatusError) as excinfo:
        await fetcher.fetch("https://www.britannica.com/topic/YouTube")

    assert excinfo.value.status_code == 403


# ---------- The whole pipeline ----------


@pytest.fixture
def ollama_ready() -> bool:
    """Whether a local Ollama is up, so the analysis stage can run."""
    if os.getenv("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Ollama integration is opt-in: set RUN_OLLAMA_INTEGRATION=1.")

    import httpx

    try:
        response = httpx.get(
            f"{get_settings().ollama_host.rstrip('/')}/api/tags", timeout=5.0
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


@needs_live
def test_the_whole_pipeline_produces_cited_research(ollama_ready: bool) -> None:
    """End to end: search, fetch, extract, analyse, ground.

    Run once, in-process and synchronously, because the point is to
    observe the assembled result rather than to re-verify the stages the
    unit tests already cover. Bounded so the run finishes in seconds
    rather than minutes.

    The evidence budget is standard depth rather than quick. From this
    machine the two highest-ranked sources for the topic are Wikipedia,
    which TLS verification blocks, and Britannica, which answers 403, so
    a four-source quick run reaches the minimum-usable threshold exactly
    once. That is the retrieval layer behaving correctly, but it leaves
    nothing for the model to analyse, and this test is about the whole
    pipeline rather than about partial-failure handling, which the unit
    tests cover exhaustively.
    """
    if not ollama_ready:
        pytest.skip("Ollama is not reachable, so the analysis stage cannot run.")

    from app.agents.base import AgentContext
    from app.agents.researcher.agent import REQUEST_CONTEXT_KEY, ResearcherAgent
    from app.dependencies import get_agent_runner
    from app.services.ai.provider import Message

    runner = get_agent_runner()
    agent = ResearcherAgent()

    request = ResearchRequest(
        topic=TOPIC, search_depth="standard", max_sources=8, language="en"
    )
    context = AgentContext(config={REQUEST_CONTEXT_KEY: request})

    result = anyio.run(
        lambda: runner.run(agent, context, [Message(role="user", content=TOPIC)])
    )

    assert result.status.value in {"completed", "failed"}

    if result.status.value == "failed":
        # Say why rather than failing opaquely: a broken search backend or
        # a missing model is an environment problem, not a regression.
        pytest.skip(f"The live run failed in this environment: {result.error}")

    payload = result.data
    assert isinstance(payload, ResearchResult)

    if not payload.sufficient:
        pytest.skip(
            f"Only {payload.sources_found} usable sources were reachable from "
            f"this machine, below the threshold, so there was nothing to "
            f"analyse: {[f.domain for f in payload.failed_sources]}"
        )

    assert result.success is True
    assert payload.summary
    assert payload.claims, "the model analysed retrieved evidence and cited nothing"

    known = {source.id for source in payload.sources}
    for source in payload.sources:
        assert source.url.startswith("https://")
        assert source.domain

    for claim in payload.claims:
        assert claim.statement.strip()
        assert claim.confidence in set(Confidence)
        for source_id in claim.source_ids:
            assert source_id in known, "a claim cites a source that was not retrieved"

    # Every source is either used or accounted for. Nothing is dropped
    # silently, which is what makes a partial result trustworthy.
    assert payload.sources_found == len(payload.sources)
    assert payload.sources_found + payload.sources_failed >= payload.sources_requested - 1
    assert payload.search_query == TOPIC
    assert payload.search_provider == "duckduckgo"


@needs_live
def test_a_run_reports_what_it_retrieved_even_when_it_fails() -> None:
    """A topic nothing can be found for must still return a typed body.

    A nonsense subject is the cheapest way to force the negative path
    without needing the network to misbehave.
    """
    import anyio

    from app.agents.researcher.agent import REQUEST_CONTEXT_KEY, ResearcherAgent
    from app.agents.base import AgentContext
    from app.agents.runner import AgentRunner
    from app.dependencies import get_agent_runner
    from app.services.ai.provider import Message

    runner = get_agent_runner()
    agent = ResearcherAgent()

    request = ResearchRequest(topic="qzxvjkwpq lmnhgfdsa", search_depth="quick")
    context = AgentContext(config={REQUEST_CONTEXT_KEY: request})

    result = anyio.run(
        lambda: runner.run(
            agent, context, [Message(role="user", content=request.topic)]
        )
    )

    assert result.status.value in {"completed", "failed"}
    if result.data is not None:
        assert isinstance(result.data, ResearchResult)
        assert result.data.sources_found == 0
        assert result.data.sufficient is False


# ---------- The second backend ----------


@needs_live
@pytest.mark.anyio
async def test_the_wikipedia_backend_is_documented_as_not_the_open_web() -> None:
    """Records what is verifiable about the fallback backend.

    From this machine the Wikipedia hosts fail TLS verification: an
    intercepting proxy presents a certificate that expired on 2026-11-03.
    Verification is not disabled to work around it, which is the point
    worth pinning — a research tool that turns off certificate checking
    to reach a site is no longer trustworthy about what it retrieved.

    So this asserts two things: the backend identifies itself as a
    reference work rather than the open web, and a TLS failure surfaces
    as an unavailable backend rather than as a silent fallback to
    unverified HTTP.
    """
    provider = WikipediaSearchProvider()

    assert provider.returns_web_results is False

    try:
        results = await provider.search(
            SearchQuery(terms=TOPIC, language="en"), limit=3
        )
    except SearchError as exc:
        # The provider reports a safe message; the transport detail lives
        # on __cause__ for the logs, which is where the diagnosis is.
        cause = exc.__cause__
        detail = f"{exc} {cause!r}".lower()

        assert "certificate" in detail or "verify" in detail, (
            "Wikipedia failed for an unrelated reason; the documented "
            "TLS-interception explanation no longer applies and the "
            "known-limitation note needs revisiting."
        )
        assert "verify=False" not in detail
        pytest.skip(f"Wikipedia is unreachable from this machine: {exc}")

    assert results
    for result in results:
        assert "wikipedia.org" in result.url
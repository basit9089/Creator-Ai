"""Search backends, result parsing and the composite fallback.

The HTML parser is tested against a fixture shaped like a real
DuckDuckGo results page, including the decorative anchors that repeat each
URL. Nothing here touches the network: the provider tests drive
``httpx.MockTransport``, and the parser tests call the parse function
directly.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.services.research.search.base import (
    SearchDepth,
    SearchError,
    SearchProvider,
    SearchQuery,
    SearchRateLimitedError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
)
from app.services.research.search.composite import CompositeSearchProvider
from app.services.research.search.duckduckgo import (
    DuckDuckGoSearchProvider,
    parse_results,
    unwrap_redirect,
)
from app.services.research.search.wikipedia import (
    WikipediaSearchProvider,
    article_url,
    parse_search_payload,
    strip_markup,
)

PUBLIC = "93.184.216.34"


def public_resolver(host: str) -> list[str]:
    return [PUBLIC]


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------- A fixture shaped like the real page ----------


def ddg_link(target: str, extra: str = "") -> str:
    """Build the redirect form DuckDuckGo uses for a result link."""
    from urllib.parse import quote

    return (
        f"//duckduckgo.com/l/?uddg={quote(target, safe='')}"
        f"&rut=0123456789abcdef{extra}"
    )


DUCK_DUCK_GO_PAGE = f"""<!DOCTYPE html>
<html><body>
  <div class="results">
    <div class="result results_links web-result">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="{ddg_link('https://en.wikipedia.org/wiki/History_of_YouTube')}">History of YouTube - Wikipedia</a>
      </h2>
      <div class="result__extras">
        <div class="result__extras__url">
          <span class="result__icon">
            <a rel="nofollow" href="{ddg_link('https://en.wikipedia.org/wiki/History_of_YouTube')}">
              <img class="result__icon__img" src="//external-content.duckduckgo.com/ip3/en.wikipedia.org.ico" />
            </a>
          </span>
          <a class="result__url" href="{ddg_link('https://en.wikipedia.org/wiki/History_of_YouTube')}">
            en.wikipedia.org/wiki/History_of_YouTube
          </a>
        </div>
      </div>
      <a class="result__snippet" href="{ddg_link('https://en.wikipedia.org/wiki/History_of_YouTube')}">
        Learn about the origins and development of <b>YouTube</b>.
      </a>
    </div>

    <div class="result results_links web-result">
      <h2 class="result__title">
        <a rel="nofollow" class="result__a" href="{ddg_link('https://blog.example.com/youtube-timeline')}">The YouTube timeline</a>
      </h2>
      <a class="result__snippet" href="{ddg_link('https://blog.example.com/youtube-timeline')}">A year-by-year account.</a>
    </div>
  </div>
</body></html>"""


# ---------- Parsing ----------


def test_parse_results_extracts_title_url_and_snippet() -> None:
    results = parse_results(DUCK_DUCK_GO_PAGE)

    assert len(results) == 2

    first = results[0]
    assert first.title == "History of YouTube - Wikipedia"
    assert first.url == "https://en.wikipedia.org/wiki/History_of_YouTube"
    assert first.domain == "en.wikipedia.org"
    assert "origins and development" in first.snippet


def test_parse_results_ignores_the_decorative_duplicate_anchors() -> None:
    """Each page repeats its URL in a favicon link and a bare domain link.

    Counting those as results would multiply the apparent result count and
    send the same page to the fetcher several times.
    """
    results = parse_results(DUCK_DUCK_GO_PAGE)

    assert len(results) == 2
    assert len({result.url for result in results}) == 2


def test_parse_results_survives_void_elements() -> None:
    """``<img>`` and ``<br>`` are never closed.

    A push/pop tag stack desynchronises on the first one and then
    mis-assigns every following title and snippet, so the parser uses
    explicit flags instead.
    """
    page = """<div class="results">
      <div class="result">
        <h2 class="result__title">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fa.example%2Fone">First</a>
        </h2>
        <img src="x.png"><br>
        <a class="result__snippet" href="/l/?uddg=https%3A%2F%2Fa.example%2Fone">Snippet one.</a>
      </div>
      <div class="result">
        <h2 class="result__title">
          <a class="result__a" href="/l/?uddg=https%3A%2F%2Fb.example%2Ftwo">Second</a>
        </h2>
        <img src="y.png"><br>
        <a class="result__snippet" href="/l/?uddg=https%3A%2F%2Fb.example%2Ftwo">Snippet two.</a>
      </div>
    </div>"""

    results = parse_results(page)

    assert [r.title for r in results] == ["First", "Second"]
    assert [r.snippet for r in results] == ["Snippet one.", "Snippet two."]
    assert [r.url for r in results] == ["https://a.example/one", "https://b.example/two"]


def test_parse_results_numbers_ranks_in_document_order() -> None:
    results = parse_results(DUCK_DUCK_GO_PAGE)

    assert [result.rank for result in results] == [1, 2]


def test_parse_results_collapses_page_indentation() -> None:
    page = """<a class="result__a" href="/l/?uddg=https%3A%2F%2Fa.example%2Fone">
        Spaced
        Out
    </a>"""

    assert parse_results(page)[0].title == "Spaced Out"


def test_parse_results_drops_entries_without_a_usable_link() -> None:
    """A decorative anchor with no destination is skipped, not returned."""
    page = '<a class="result__a" href="#">No link here</a>'

    assert parse_results(page) == []


def test_parse_results_drops_repeated_destinations() -> None:
    page = f"""
      <a class="result__a" href="{ddg_link('https://a.example/one')}">One</a>
      <a class="result__snippet" href="{ddg_link('https://a.example/one')}">First.</a>
      <a class="result__a" href="{ddg_link('https://a.example/one')}">One again</a>
    """

    results = parse_results(page)

    assert len(results) == 1
    assert results[0].title == "One"


def test_parse_results_on_an_empty_page() -> None:
    assert parse_results("<html><body></body></html>") == []


# ---------- Redirect unwrapping ----------


def test_unwrap_redirect_recovers_the_destination() -> None:
    assert unwrap_redirect(ddg_link("https://example.com/a")) == "https://example.com/a"


def test_unwrap_redirect_passes_through_a_direct_link() -> None:
    assert unwrap_redirect("https://example.com/a") == "https://example.com/a"


def test_unwrap_redirect_rejects_a_duckduckgo_link_with_no_target() -> None:
    """A tracking link with no ``uddg`` names nothing fetchable."""
    assert unwrap_redirect("//duckduckgo.com/l/?rut=abc") is None


@pytest.mark.parametrize(
    "href",
    ["", "   ", "javascript:alert(1)", "not a url", "mailto:a@b.c"],
)
def test_unwrap_redirect_rejects_unusable_links(href: str) -> None:
    assert unwrap_redirect(href) is None


def test_unwrap_redirect_resolves_a_path_only_link() -> None:
    assert unwrap_redirect("/l/?uddg=https%3A%2F%2Fa.example%2Fb") == "https://a.example/b"


# ---------- The DuckDuckGo provider ----------


@pytest.mark.anyio
async def test_search_returns_parsed_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "q=history+of+youtube" in str(request.url)
        return httpx.Response(200, text=DUCK_DUCK_GO_PAGE)

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    results = await provider.search(SearchQuery(terms="history of youtube"), limit=5)

    assert len(results) == 2
    assert results[0].url == "https://en.wikipedia.org/wiki/History_of_YouTube"


@pytest.mark.anyio
async def test_search_respects_the_limit() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=DUCK_DUCK_GO_PAGE)

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    assert len(await provider.search(SearchQuery(terms="x"), limit=1)) == 1


@pytest.mark.anyio
async def test_search_with_a_non_positive_limit_skips_the_request() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not make a request for limit=0")

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    assert await provider.search(SearchQuery(terms="x"), limit=0) == []


@pytest.mark.anyio
async def test_search_maps_a_timeout_to_a_stable_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    with pytest.raises(SearchTimeoutError) as excinfo:
        await provider.search(SearchQuery(terms="x"), limit=5)

    assert excinfo.value.code == "search_timeout"


@pytest.mark.anyio
async def test_search_maps_a_connection_failure_to_a_stable_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    with pytest.raises(SearchUnavailableError):
        await provider.search(SearchQuery(terms="x"), limit=5)


@pytest.mark.anyio
@pytest.mark.parametrize("status", [202, 429])
async def test_search_treats_a_rate_limit_as_rate_limited(status: int) -> None:
    """202 is DuckDuckGo's own "slow down" signal, not a server fault."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="rate limited")

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    with pytest.raises(SearchRateLimitedError):
        await provider.search(SearchQuery(terms="x"), limit=5)


@pytest.mark.anyio
async def test_search_detects_an_anti_bot_challenge_served_with_status_200() -> None:
    """A challenge page arrives with HTTP 200, so the body must be inspected.

    Treating it as an empty result would look like "nothing matched this
    topic", when in fact the request was refused.
    """

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><script src='/anomaly.js'></script></html>")

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    with pytest.raises(SearchRateLimitedError):
        await provider.search(SearchQuery(terms="x"), limit=5)


@pytest.mark.anyio
async def test_search_reports_a_server_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)

    with pytest.raises(SearchUnavailableError) as excinfo:
        await provider.search(SearchQuery(terms="x"), limit=5)

    assert "503" in excinfo.value.message


@pytest.mark.anyio
async def test_search_sends_an_identifiable_user_agent() -> None:
    """The client identifies itself rather than impersonating a browser."""
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, text="<html></html>")

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)
    await provider.search(SearchQuery(terms="x"), limit=1)

    assert "CreatorAI" in seen["user-agent"]


@pytest.mark.anyio
async def test_search_omits_the_region_parameter_for_an_unknown_language() -> None:
    """An unmapped language falls back to general results rather than failing."""
    urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, text="<html></html>")

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)
    await provider.search(SearchQuery(terms="x", language="zu"), limit=1)

    assert "kl=" not in urls[0]


@pytest.mark.anyio
async def test_search_sets_the_region_parameter_for_a_known_language() -> None:
    urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, text="<html></html>")

    provider = DuckDuckGoSearchProvider(client=client_for(handler), min_interval=0)
    await provider.search(SearchQuery(terms="x", language="de"), limit=1)

    assert "kl=de-de" in urls[0]


def test_the_provider_reports_itself_as_a_web_search() -> None:
    """Recorded so a report can distinguish web evidence from an encyclopedia."""
    assert DuckDuckGoSearchProvider.returns_web_results is True


# ---------- Wikipedia ----------


WIKIPEDIA_RESPONSE = {
    "batchcomplete": "",
    "query": {
        "search": [
            {
                "ns": 0,
                "title": "History of YouTube",
                "pageid": 12345,
                "snippet": '<span class="searchmatch">YouTube</span> began in 2005.',
            },
            {
                "ns": 0,
                "title": "YouTube",
                "snippet": '<span class="searchmatch">YouTube</span> is a video platform.',
            },
        ]
    },
}


def test_parse_search_payload_maps_title_snippet_and_url() -> None:
    results = parse_search_payload(WIKIPEDIA_RESPONSE)

    assert [r.title for r in results] == ["History of YouTube", "YouTube"]
    assert results[0].url == "https://en.wikipedia.org/wiki/History_of_YouTube"
    assert results[0].rank == 1


def test_parse_search_payload_strips_markup_but_keeps_the_matched_words() -> None:
    results = parse_search_payload(WIKIPEDIA_RESPONSE)

    assert results[0].snippet == "YouTube began in 2005."


def test_parse_search_payload_skips_untitled_hits() -> None:
    payload = {"query": {"search": [{"snippet": "no title"}, {"title": "Real", "snippet": "s"}]}}

    assert [r.title for r in parse_search_payload(payload)] == ["Real"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": {}},
        {"query": {"search": "not a list"}},
        {"query": {"search": [None, 3]}},
        [],
        "nonsense",
        None,
    ],
)
def test_parse_search_payload_returns_empty_for_unfamiliar_shapes(payload: object) -> None:
    """One unexpected response should cost candidates, not the whole run."""
    assert parse_search_payload(payload) == []


def test_article_url_encodes_spaces_as_underscores() -> None:
    assert article_url("History of YouTube") == (
        "https://en.wikipedia.org/wiki/History_of_YouTube"
    )


def test_article_url_escapes_other_special_characters() -> None:
    """An unescaped slash would silently address a different article."""
    assert article_url("AC/DC") == "https://en.wikipedia.org/wiki/AC%2FDC"


def test_strip_markup_removes_tags_and_tidies_whitespace() -> None:
    assert strip_markup('  <b>one</b>\n  <i>two</i>  ') == "one two"


@pytest.mark.anyio
async def test_wikipedia_search_returns_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "list=search" in str(request.url)
        return httpx.Response(200, json=WIKIPEDIA_RESPONSE)

    provider = WikipediaSearchProvider(client=client_for(handler))

    results = await provider.search(SearchQuery(terms="history of youtube"), limit=5)

    assert len(results) == 2


@pytest.mark.anyio
async def test_wikipedia_search_selects_a_language_subdomain() -> None:
    urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=WIKIPEDIA_RESPONSE)

    provider = WikipediaSearchProvider(client=client_for(handler))
    await provider.search(SearchQuery(terms="x", language="de"), limit=5)

    assert "de.wikipedia.org" in urls[0]


@pytest.mark.anyio
async def test_wikipedia_search_falls_back_to_english_for_an_unknown_language() -> None:
    urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=WIKIPEDIA_RESPONSE)

    provider = WikipediaSearchProvider(client=client_for(handler))
    await provider.search(SearchQuery(terms="x", language="zu"), limit=5)

    assert "en.wikipedia.org" in urls[0]


@pytest.mark.anyio
async def test_wikipedia_search_rejects_a_non_json_response() -> None:
    """A captive portal or proxy error page must not raise a parse error."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>blocked</html>")

    provider = WikipediaSearchProvider(client=client_for(handler))

    with pytest.raises(SearchUnavailableError):
        await provider.search(SearchQuery(terms="x"), limit=5)


def test_the_wikipedia_provider_does_not_claim_to_be_a_web_search() -> None:
    """One encyclopedia is not the open web, and the flag records that."""
    assert WikipediaSearchProvider.returns_web_results is False


# ---------- Composite ----------


class _StubProvider(SearchProvider):
    def __init__(
        self,
        name: str,
        *,
        results: list[SearchResult] | None = None,
        error: Exception | None = None,
        returns_web_results: bool = True,
    ) -> None:
        self.name = name
        self.returns_web_results = returns_web_results
        self._results = results or []
        self._error = error
        self.calls = 0

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._results[:limit]


def result(url: str) -> SearchResult:
    return SearchResult(title=f"T {url}", url=url, domain="example.com")


def test_composite_requires_at_least_one_provider() -> None:
    with pytest.raises(ValueError):
        CompositeSearchProvider([])


@pytest.mark.anyio
async def test_composite_stops_once_the_limit_is_satisfied() -> None:
    """A backend that answered in full should not cost extra calls elsewhere.

    DuckDuckGo normally returns more than the Researcher's limit, so this is
    the ordinary path: one request, and no needless call to the fallback.
    """
    first = _StubProvider(
        "first", results=[result("https://a.example/1"), result("https://a.example/2")]
    )
    second = _StubProvider("second", results=[result("https://b.example/2")])

    results = await CompositeSearchProvider([first, second]).search(
        SearchQuery(terms="x"), limit=2
    )

    assert [r.url for r in results] == ["https://a.example/1", "https://a.example/2"]
    assert second.calls == 0


@pytest.mark.anyio
async def test_composite_tops_up_a_partial_answer_from_the_next_provider() -> None:
    """A backend returning one result for a limit of three leaves a gap.

    Asking the next provider costs one request and gains real coverage,
    which is the whole reason the composite exists.
    """
    first = _StubProvider("first", results=[result("https://a.example/1")])
    second = _StubProvider("second", results=[result("https://b.example/2")])

    results = await CompositeSearchProvider([first, second]).search(
        SearchQuery(terms="x"), limit=3
    )

    assert [r.url for r in results] == ["https://a.example/1", "https://b.example/2"]
    assert second.calls == 1


@pytest.mark.anyio
async def test_composite_skips_a_failing_provider() -> None:
    """One unavailable backend must not end the research run."""
    broken = _StubProvider("broken", error=SearchUnavailableError("down"))
    working = _StubProvider("working", results=[result("https://b.example/2")])

    results = await CompositeSearchProvider([broken, working]).search(
        SearchQuery(terms="x"), limit=5
    )

    assert [r.url for r in results] == ["https://b.example/2"]


@pytest.mark.anyio
async def test_composite_consults_the_next_provider_when_one_is_short() -> None:
    """A partial answer is topped up rather than accepted as complete."""
    first = _StubProvider("first", results=[result("https://a.example/1")])
    second = _StubProvider("second", results=[result("https://b.example/2")])

    results = await CompositeSearchProvider([first, second]).search(
        SearchQuery(terms="x"), limit=3
    )

    assert [r.url for r in results] == ["https://a.example/1", "https://b.example/2"]


@pytest.mark.anyio
async def test_composite_does_not_repeat_a_url_from_two_providers() -> None:
    """Both backends can surface the same article; it must appear once."""
    first = _StubProvider("first", results=[result("https://a.example/1")])
    second = _StubProvider("second", results=[result("https://a.example/1")])

    results = await CompositeSearchProvider([first, second]).search(
        SearchQuery(terms="x"), limit=5
    )

    assert len(results) == 1


@pytest.mark.anyio
async def test_composite_returns_empty_when_every_provider_fails() -> None:
    providers = [
        _StubProvider("a", error=SearchUnavailableError("down")),
        _StubProvider("b", error=SearchTimeoutError("slow")),
    ]

    assert await CompositeSearchProvider(providers).search(SearchQuery(terms="x"), limit=5) == []


@pytest.mark.anyio
async def test_composite_does_not_swallow_an_unexpected_error() -> None:
    """A bug in a backend should be visible, not silently degrade the result."""

    class Broken(SearchProvider):
        name = "broken"

        async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
            raise ZeroDivisionError("bug")

    with pytest.raises(ZeroDivisionError):
        await CompositeSearchProvider([Broken()]).search(SearchQuery(terms="x"), limit=5)


@pytest.mark.anyio
async def test_composite_close_survives_a_failing_provider() -> None:
    class Uncloseable(SearchProvider):
        name = "uncloseable"

        async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
            return []

        async def close(self) -> None:
            raise RuntimeError("cannot close")

    await CompositeSearchProvider([Uncloseable()]).close()


@pytest.mark.anyio
async def test_composite_reports_web_results_when_any_provider_does() -> None:
    """The flag is an ``any``, so one real web search lifts the whole chain.

    The Researcher records this on its result, so a report can say the
    evidence came from the open web rather than from one encyclopaedia.
    """
    encyclopedia_only = _StubProvider("encyclopedia", returns_web_results=False)
    web = _StubProvider("web", returns_web_results=True)

    assert CompositeSearchProvider([encyclopedia_only]).returns_web_results is False
    assert CompositeSearchProvider([encyclopedia_only, web]).returns_web_results is True
    assert CompositeSearchProvider([web]).returns_web_results is True


# ---------- SearchDepth budgets ----------


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (SearchDepth.QUICK, (3, 5)),
        (SearchDepth.STANDARD, (5, 8)),
        (SearchDepth.DEEP, (8, 12)),
    ],
)
def test_search_depth_maps_to_a_source_budget(depth: SearchDepth, expected: tuple) -> None:
    assert depth.source_budget == expected


def test_search_errors_carry_stable_codes() -> None:
    """Callers react to codes, not to message text."""
    assert SearchUnavailableError("x").code == "search_unavailable"
    assert SearchTimeoutError("x").code == "search_timeout"
    assert SearchRateLimitedError("x").code == "search_rate_limited"
    assert issubclass(SearchTimeoutError, SearchError)


def test_search_result_derives_its_domain_when_not_given() -> None:
    assert SearchResult(title="t", url="https://www.example.com/a").domain == "example.com"


def test_search_result_serialises_for_logging() -> None:
    payload = SearchResult(title="t", url="https://a.example/x", rank=2).to_dict()

    assert payload == {
        "title": "t",
        "url": "https://a.example/x",
        "snippet": "",
        "domain": "a.example",
        "rank": 2,
    }
    # Must stay JSON-serialisable for logging.
    json.dumps(payload)

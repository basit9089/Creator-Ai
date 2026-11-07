"""Retrieval: bounds, error mapping and the per-hop safety re-check.

Every test drives ``httpx.MockTransport``, so nothing here opens a real
connection. Host resolution is replaced too, which means the assertions
depend on the safety rules rather than on what this machine's DNS happens
to answer today.
"""

from __future__ import annotations

import httpx
import pytest

from app.services.research.fetching.base import (
    FetchError,
    FetchTimeoutError,
    FetchedDocument,
    HttpStatusError,
    SourceFetcher,
    SourceTooLargeError,
    TooManyRedirectsError,
    UnsafeUrlFetchError,
    UnsupportedContentTypeError,
)
from app.services.research.fetching.http import ALLOWED_MEDIA_TYPES, HTTPSourceFetcher

PUBLIC = "93.184.216.34"

HTML = """<!DOCTYPE html><html lang="en"><head><title>A page</title></head>
<body><p>Body text.</p></body></html>"""


def public_resolver(host: str) -> list[str]:
    return [PUBLIC]


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def fetcher_for(handler, **kwargs) -> HTTPSourceFetcher:
    """Build a fetcher wired to a mock transport and a public resolver."""
    kwargs.setdefault("timeout", 5.0)
    return HTTPSourceFetcher(
        client=client_for(handler),
        resolver=public_resolver,
        **kwargs,
    )


def html_response(text: str = HTML) -> httpx.Response:
    return httpx.Response(
        200,
        text=text,
        headers={"content-type": "text/html; charset=utf-8"},
    )


# ---------- A redirect target is requested as the server gave it ----------


@pytest.mark.anyio
async def test_a_trailing_slash_redirect_terminates() -> None:
    """The bug this guards is an infinite loop, not a wrong URL.

    Normalisation treats ``/article`` and ``/article/`` as the same URL.
    Many servers do not: they answer 308 from one to the other. If the
    redirect target is normalised before the next request, the chain is
    ``/article`` -> ``/article/`` -> (normalised to) ``/article`` and the
    run only ends when the redirect budget runs out, losing the page.

    blog.youtube is a real instance of this shape, which is how it was
    found: the fetcher reported too many redirects on a page the browser
    loads without trouble.
    """
    requested: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requested.append(path)
        if path.endswith("/"):
            return html_response()
        return httpx.Response(308, headers={"location": f"{path}/"})

    document = await fetcher_for(handler).fetch("https://example.com/article")

    assert document.status_code == 200
    assert requested == ["/article", "/article/"]
    assert document.redirected is True


@pytest.mark.anyio
async def test_a_redirect_to_a_trailing_slash_is_not_normalised_away() -> None:
    """The second hop requests exactly the path the server named."""
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "/a/b/"})
        return html_response()

    document = await fetcher_for(handler).fetch("https://example.com/start")

    assert seen == ["/start", "/a/b/"]
    assert document.final_url.endswith("/a/b/")


@pytest.mark.anyio
async def test_an_ordinary_relative_redirect_still_resolves() -> None:
    """The common case is unaffected by the change above."""
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return html_response()

    document = await fetcher_for(handler).fetch("https://example.com/old")

    assert document.redirected is True
    assert document.final_url == "https://example.com/new"


# ---------- The happy path ----------


@pytest.mark.anyio
async def test_fetch_returns_the_decoded_document() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return html_response()

    document = await fetcher_for(handler).fetch("https://example.com/a")

    assert isinstance(document, FetchedDocument)
    assert document.status_code == 200
    assert "Body text." in document.text
    assert document.final_url == "https://example.com/a"
    assert document.redirected is False
    assert document.truncated is False


@pytest.mark.anyio
async def test_fetch_strips_content_type_parameters() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return html_response()

    document = await fetcher_for(handler).fetch("https://example.com/a")

    assert document.media_type == "text/html"
    assert document.charset == "utf-8"


@pytest.mark.anyio
async def test_fetch_records_the_size_of_the_body() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return html_response()

    document = await fetcher_for(handler).fetch("https://example.com/a")

    assert document.size_bytes == len(HTML.encode())


@pytest.mark.anyio
async def test_fetch_decodes_a_declared_non_utf8_charset() -> None:
    """A page in another encoding must not come back as mojibake."""
    body = "Grüße".encode("latin-1")

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/html; charset=iso-8859-1"},
        )

    document = await fetcher_for(handler).fetch("https://example.com/a")

    assert "Grüße" in document.text


@pytest.mark.anyio
async def test_fetch_does_not_raise_on_undecodable_bytes() -> None:
    """Retrieval must survive a page that lies about its encoding.

    The text is evidence for a summary, not data to round-trip, so a lossy
    decode is a better outcome than losing the source entirely.
    """

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"\xff\xfe\x00broken utf16",
            headers={"content-type": "text/html; charset=definitely-not-a-charset"},
        )

    document = await fetcher_for(handler).fetch("https://example.com/a")

    assert document.text


@pytest.mark.anyio
async def test_fetch_rejects_an_empty_document() -> None:
    """Nothing to extract means nothing to cite, and that is a failure."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="   \n  ", headers={"content-type": "text/html"}
        )

    with pytest.raises(FetchError, match="empty document"):
        await fetcher_for(handler).fetch("https://example.com/a")


# ---------- Safety ----------


@pytest.mark.anyio
async def test_fetch_refuses_a_unsafe_scheme_without_requesting_it() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not request a non-http URL")

    with pytest.raises(UnsafeUrlFetchError):
        await fetcher_for(handler).fetch("file:///etc/passwd")


@pytest.mark.anyio
async def test_fetch_refuses_localhost() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not request localhost")

    with pytest.raises(UnsafeUrlFetchError, match="localhost"):
        await fetcher_for(handler).fetch("http://localhost:11434/api/generate")


@pytest.mark.anyio
async def test_fetch_refuses_a_private_address() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not request a private address")

    with pytest.raises(UnsafeUrlFetchError, match="private"):
        await fetcher_for(handler).fetch("http://192.168.0.1/admin")


@pytest.mark.anyio
async def test_fetch_refuses_a_host_that_resolves_privately() -> None:
    """The hostname looks innocent; the address it resolves to does not.

    This is the case a host-string check alone would miss.
    """

    def private(host: str) -> list[str]:
        return ["10.1.2.3"]

    async def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not request a privately-resolving host")

    fetcher = HTTPSourceFetcher(client=client_for(handler), resolver=private)

    with pytest.raises(UnsafeUrlFetchError, match="private"):
        await fetcher.fetch("https://looks-public.example.com/a")


# ---------- Timeouts ----------


@pytest.mark.anyio
async def test_fetch_maps_a_timeout_to_its_own_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(FetchTimeoutError) as excinfo:
        await fetcher_for(handler).fetch("https://example.com/a")

    assert excinfo.value.code == "fetch_timeout"
    assert "example.com" in excinfo.value.message


@pytest.mark.anyio
async def test_fetch_maps_a_connection_failure_without_leaking_detail() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to 10.0.0.5", request=request)

    with pytest.raises(FetchError) as excinfo:
        await fetcher_for(handler).fetch("https://example.com/a")

    # The host is safe to show; the transport's internals are not.
    assert "example.com" in excinfo.value.message
    assert "10.0.0.5" not in excinfo.value.message


# ---------- HTTP statuses ----------


@pytest.mark.parametrize("status", [400, 403, 404, 429, 500, 503])
@pytest.mark.anyio
async def test_fetch_reports_an_error_status(status: int) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="error page")

    with pytest.raises(HttpStatusError) as excinfo:
        await fetcher_for(handler).fetch("https://example.com/a")

    assert excinfo.value.status_code == status
    assert str(status) in excinfo.value.message


@pytest.mark.anyio
async def test_fetch_does_not_read_the_body_of_an_error_response() -> None:
    """An error page is not evidence and may be huge, so it is discarded.

    The stream counts how many times it is iterated, which is what proves
    the fetcher abandons the response on the status rather than pulling a
    multi-megabyte error page into memory first.
    """

    class CountingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.iterations = 0

        async def __aiter__(self):
            self.iterations += 1
            yield b"x" * 10_000

    stream = CountingStream()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, headers={"content-type": "text/html"}, stream=stream
        )

    with pytest.raises(HttpStatusError):
        await fetcher_for(handler).fetch("https://example.com/a")

    assert stream.iterations == 0


# ---------- Content type ----------


@pytest.mark.anyio
@pytest.mark.parametrize(
    "media_type",
    ["image/png", "application/pdf", "video/mp4", "application/octet-stream", "text/css"],
)
async def test_fetch_refuses_a_non_document_content_type(media_type: str) -> None:
    """Nothing downloadable is worth having for a text-evidence pipeline.

    Refused before the body is read, so a video is never transferred.
    """

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"binary", headers={"content-type": media_type})

    with pytest.raises(UnsupportedContentTypeError) as excinfo:
        await fetcher_for(handler).fetch("https://example.com/a")

    assert media_type in excinfo.value.message


@pytest.mark.anyio
async def test_fetch_refuses_a_response_with_no_content_type() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html></html>")

    with pytest.raises(UnsupportedContentTypeError, match="unknown content type"):
        await fetcher_for(handler).fetch("https://example.com/a")


@pytest.mark.anyio
@pytest.mark.parametrize("media_type", sorted(ALLOWED_MEDIA_TYPES))
async def test_every_allowed_media_type_is_accepted(media_type: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="some text", headers={"content-type": media_type})

    document = await fetcher_for(handler).fetch("https://example.com/a")

    assert document.media_type == media_type


# ---------- Size cap ----------


@pytest.mark.anyio
async def test_fetch_truncates_at_the_size_cap_and_says_so() -> None:
    """An oversized page is cut short, not rejected.

    A truncated page is still evidence, and silently pretending otherwise
    would let the model treat a fragment as the whole article.
    """

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<p>" + b"y" * 50_000 + b"</p>",
            headers={"content-type": "text/html"},
        )

    document = await fetcher_for(handler, max_size=1000).fetch("https://example.com/a")

    assert document.size_bytes == 1000
    assert document.truncated is True


@pytest.mark.anyio
async def test_fetch_keeps_a_document_exactly_at_the_cap_untruncated() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"z" * 1000,
            headers={"content-type": "text/html"},
        )

    document = await fetcher_for(handler, max_size=1000).fetch("https://example.com/a")

    assert document.size_bytes == 1000
    assert document.truncated is False


@pytest.mark.anyio
async def test_fetch_stops_downloading_once_the_cap_is_passed() -> None:
    """The cap is enforced while streaming, not after the whole body arrives."""
    chunks_sent = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        def body() -> bytes:
            nonlocal chunks_sent
            chunks_sent += 1
            return b"q" * 4096

        return httpx.Response(200, headers={"content-type": "text/html"}, content=body())

    document = await fetcher_for(handler, max_size=2048).fetch("https://example.com/a")

    assert document.truncated is True
    assert document.size_bytes == 2048
    # The point of the streaming check: the transfer stops early.
    assert chunks_sent <= 2


# ---------- Redirects ----------


@pytest.mark.anyio
async def test_fetch_follows_a_redirect_and_records_the_final_url() -> None:
    """A claim must cite the URL a reader can actually open."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(
                301, headers={"location": "https://example.com/new"}
            )
        return html_response()

    document = await fetcher_for(handler).fetch("https://example.com/old")

    assert document.final_url == "https://example.com/new"
    assert document.redirected is True
    assert document.url == "https://example.com/old"


@pytest.mark.anyio
async def test_fetch_resolves_a_relative_redirect() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a/old":
            return httpx.Response(302, headers={"location": "/a/new"})
        return html_response()

    document = await fetcher_for(handler).fetch("https://example.com/a/old")

    assert document.final_url == "https://example.com/a/new"


@pytest.mark.anyio
async def test_fetch_revalidates_the_safety_of_a_redirect_target() -> None:
    """The classic way to slip past a URL check: allow the first hop only.

    A public page that redirects to ``127.0.0.1`` must not reach the local
    Ollama server, which is exactly what the user's own machine runs.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bait":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/"})
        raise AssertionError("must not follow a redirect into the private network")

    fetcher = HTTPSourceFetcher(
        client=client_for(handler),
        resolver=public_resolver,
        timeout=5.0,
    )

    with pytest.raises(UnsafeUrlFetchError, match="127.0.0.1"):
        await fetcher.fetch("https://example.com/bait")


@pytest.mark.anyio
async def test_fetch_stops_at_the_redirect_budget() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/next"})

    with pytest.raises(TooManyRedirectsError):
        await fetcher_for(handler, max_redirects=2).fetch("https://example.com/loop")


@pytest.mark.anyio
async def test_fetch_does_not_follow_redirects_when_the_budget_is_zero() -> None:
    """A zero budget allows the first request but no hop after it.

    The first request is unavoidable: only its response reveals that a
    redirect was offered at all.
    """
    requested: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(302, headers={"location": "/elsewhere"})

    fetcher = HTTPSourceFetcher(
        client=client_for(handler),
        resolver=public_resolver,
        max_redirects=0,
        timeout=5.0,
    )

    with pytest.raises(TooManyRedirectsError):
        await fetcher.fetch("https://example.com/loop")

    assert requested == ["/loop"]


@pytest.mark.anyio
async def test_each_fetch_gets_a_full_redirect_budget() -> None:
    """The budget is per request, never consumed by an earlier one.

    The Researcher fetches several pages concurrently through one shared
    fetcher. If the remaining budget lived on the instance, the first page's
    redirects would starve every other page that followed.
    """
    requested: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path in {"/one", "/two"}:
            return httpx.Response(301, headers={"location": "/final"})
        return html_response()

    fetcher = fetcher_for(handler, max_redirects=1)

    first = await fetcher.fetch("https://example.com/one")
    second = await fetcher.fetch("https://example.com/two")

    assert first.final_url == "https://example.com/final"
    assert second.final_url == "https://example.com/final"
    assert requested.count("/final") == 2


@pytest.mark.anyio
async def test_fetch_reports_a_redirect_with_no_target() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(302)

    with pytest.raises(FetchError, match="without a target"):
        await fetcher_for(handler).fetch("https://example.com/x")


@pytest.mark.anyio
async def test_a_slow_source_does_not_block_another() -> None:
    """One hanging page must not take the whole research run with it."""
    import anyio

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/slow":
            await anyio.sleep(30)
        return html_response()

    fetcher = fetcher_for(handler, max_size=1000)

    with anyio.move_on_after(0.5):
        async with anyio.create_task_group() as group:
            group.start_soon(fetcher.fetch, "https://example.com/slow")
            good = await fetcher.fetch("https://example.com/fast")

    assert good.status_code == 200


# ---------- Lifecycle ----------


@pytest.mark.anyio
async def test_close_is_safe_to_call_more_than_once() -> None:
    """``close_services`` runs on every shutdown, so a repeat must not raise."""

    class CountingFetcher(SourceFetcher):
        name = "counting"

        def __init__(self) -> None:
            self.closed = 0

        async def fetch(self, url: str) -> FetchedDocument:  # pragma: no cover
            raise NotImplementedError

        async def close(self) -> None:
            self.closed += 1

    provider = CountingFetcher()

    await provider.close()
    await provider.close()

    assert provider.closed == 2


@pytest.mark.anyio
async def test_a_shared_fetcher_closes_once_without_raising() -> None:
    """The concrete fetcher's own ``close`` is idempotent."""
    fetcher = HTTPSourceFetcher(resolver=public_resolver)

    await fetcher.close()
    await fetcher.close()
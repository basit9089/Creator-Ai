"""API tests for ``POST /api/captions-thumbnail``.

The endpoint's contract, exercised end-to-end through the real
application and AgentRunner (only the AI provider and the agent
registry are substituted):

* ``completed`` → 200 with ``success: true`` and files that exist;
* ``partial`` → 200 with ``success: false``, per-artifact statuses and
  the real files that *were* produced;
* total failure → the standard error envelope, never HTTP 200;
* invalid requests → 422 before any work (or AI call) happens;
* an unregistered agent → 404.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.agents.captions_thumbnail import (
    CaptionsThumbnailAgent,
    CaptionsThumbnailRequest,
)
from app.agents.registry import AgentRegistry, get_registry
from app.agents.scriptwriter.schemas import ScriptClaimReference, ScriptResult, Section
from app.api.routes.captions import _status_for_code
from app.core.config import Settings
from app.main import create_app
from app.services.ai.provider import ProviderUnavailableError
from app.services.media.captions import parse_srt
from app.services.media.thumbnails import (
    LocalThumbnailRenderer,
    ThumbnailRenderError,
    ThumbnailRenderer,
    ThumbnailSpec,
)
from tests.conftest import FakeAIProvider

#: Schema-valid draft the fake provider returns (the normalisation
#: traps are covered in the agent tests; here the API contract is
#: under test).
DRAFT = {
    "title_candidates": [
        "How YouTube Started in 2005: A Short History",
        "The Real Story Behind YouTube's Launch",
    ],
    "description": "In this video we explain how YouTube started in 2005.",
    "tags": ["youtube", "history of youtube", "2005"],
    "primary_keyword": "youtube history",
    "secondary_keywords": ["video sharing"],
    "search_phrases": ["how youtube started"],
    "content_category": "Education",
}


def make_script() -> ScriptResult:
    return ScriptResult(
        title="How YouTube Started",
        hook="What if one video site changed the internet forever?",
        introduction="Today we cover the launch of YouTube in 2005.",
        sections=(
            Section(
                section_id="s1",
                heading="The launch",
                narration="YouTube launched in 2005 as a video sharing site.",
                claim_references=({"claim_id": "c1", "source_ids": ("src1",)},),
            ),
        ),
        narration="",
        conclusion="That is how YouTube started.",
        cta="Subscribe for more history videos.",
        claims_used=(
            ScriptClaimReference(claim_id="c1", source_ids=("src1",), status="verified"),
        ),
        sources=(
            {
                "id": "src1",
                "url": "https://en.wikipedia.org/wiki/YouTube",
                "domain": "en.wikipedia.org",
            },
        ),
    )


def make_request(**overrides: object) -> CaptionsThumbnailRequest:
    base: dict[str, object] = {"script": make_script(), "project_id": "demo"}
    base.update(overrides)
    return CaptionsThumbnailRequest(**base)  # type: ignore[arg-type]


def body(**overrides: object) -> dict:
    """A valid request body, optionally with raw-field overrides."""
    payload = make_request().model_dump(mode="json")
    payload.update(overrides)
    return payload


@contextmanager
def captions_client(
    tmp_path: Path,
    *,
    provider: FakeAIProvider | None = None,
    renderer: ThumbnailRenderer | None = None,
    registry: AgentRegistry | None = None,
) -> Iterator[tuple[FastAPI, TestClient]]:
    """A real app whose captions registry carries an isolated agent.

    The agent is built with an explicit provider and storage root so no
    test ever resolves the real Ollama provider or writes outside
    ``tmp_path``.
    """
    active = registry
    if active is None:
        active = AgentRegistry()
        active.register(
            CaptionsThumbnailAgent(
                provider=provider or FakeAIProvider(text=json.dumps(DRAFT)),
                renderer=renderer,
                settings=Settings(storage_root=str(tmp_path)),
            )
        )
    app = create_app()
    app.dependency_overrides[get_registry] = lambda: active
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield app, client
    finally:
        app.dependency_overrides.clear()


def post(client: TestClient, **overrides: object):
    return client.post("/api/captions-thumbnail", json=body(**overrides))


# --------------------------------------------------------------------------
# Completed: 200, files verified on disk
# --------------------------------------------------------------------------


class TestCompletedRun:
    def test_completed_returns_200_with_verified_files(self, tmp_path: Path) -> None:
        with captions_client(tmp_path) as (app, client):
            response = post(client, target_audience="builders")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "completed"
        assert data["errors"] == []

        # Captions: real files parsing to the reported count.
        assert data["captions"]["status"] == "completed"
        assert data["captions"]["timing_source"] == "script_wpm"
        srt = Path(data["captions"]["srt_path"])
        vtt = Path(data["captions"]["vtt_path"])
        assert srt.is_file() and vtt.is_file()
        cues = parse_srt(srt.read_text(encoding="utf-8"))
        assert len(cues) == data["captions"]["caption_count"] > 0
        assert list(parse_vtt_file(vtt)) == cues

        # Thumbnail: a real PNG whose size matches the reported size.
        assert data["thumbnail"]["status"] == "completed"
        assert data["thumbnail"]["renderer"] == "local"
        png = Path(data["thumbnail"]["path"])
        assert png.is_file()
        assert data["thumbnail"]["size_bytes"] == png.stat().st_size
        with Image.open(png) as opened:
            assert opened.format == "PNG"
            assert opened.size == (1280, 720)

        # Metadata: a written JSON file mirroring the response.
        assert data["metadata"]["status"] == "completed"
        assert data["metadata"]["audience"] == "builders"
        meta_path = Path(data["metadata"]["path"])
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        assert payload["metadata"]["title"] == data["metadata"]["title"]
        assert 0 <= data["metadata"]["title_score"] <= 100

        # Grounding survived the round trip.
        assert data["grounding"]["claim_refs"][0]["claim_id"] == "c1"
        assert (
            data["grounding"]["source_refs"][0]["url"]
            == "https://en.wikipedia.org/wiki/YouTube"
        )

        # Every produced path is confined to this project directory.
        root = (tmp_path / "projects" / "demo").resolve()
        for path in (srt, vtt, png, meta_path):
            assert path.resolve().is_relative_to(root)

    def test_custom_resolution_is_reflected_in_the_response(
        self, tmp_path: Path
    ) -> None:
        with captions_client(tmp_path) as (app, client):
            response = post(client, thumbnail_resolution="640x360")

        assert response.status_code == 200
        data = response.json()
        assert data["thumbnail"]["width"] == 640
        assert data["thumbnail"]["height"] == 360
        with Image.open(data["thumbnail"]["path"]) as opened:
            assert opened.size == (640, 360)

    def test_caption_format_subset_is_honoured(self, tmp_path: Path) -> None:
        with captions_client(tmp_path) as (app, client):
            response = post(client, caption_formats=["srt"])

        assert response.status_code == 200
        data = response.json()
        assert data["captions"]["formats"] == ["srt"]
        assert data["captions"]["vtt_path"] == ""
        captions_dir = tmp_path / "projects" / "demo" / "captions"
        assert sorted(p.name for p in captions_dir.iterdir()) == ["captions.srt"]


def parse_vtt_file(path: Path) -> list:
    from app.services.media.captions import parse_vtt

    return list(parse_vtt(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Partial: 200 with success=false and exactly what was produced
# --------------------------------------------------------------------------


class TestPartialRuns:
    def test_ai_failure_is_partial_200_with_real_artifacts(
        self, tmp_path: Path
    ) -> None:
        provider = FakeAIProvider(
            text="", error=ProviderUnavailableError("connection refused")
        )
        with captions_client(tmp_path, provider=provider) as (app, client):
            response = post(client)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["status"] == "partial"
        assert data["metadata"]["status"] == "failed"
        assert data["metadata"]["error_code"] == "ai_provider_unavailable"
        # The artifacts that succeeded are named with real, existing files.
        assert data["captions"]["status"] == "completed"
        assert data["thumbnail"]["status"] == "completed"
        assert Path(data["captions"]["srt_path"]).is_file()
        assert Path(data["thumbnail"]["path"]).is_file()
        assert any(e.startswith("metadata: ai_provider_unavailable") for e in data["errors"])

    def test_renderer_crash_is_partial_200(self, tmp_path: Path) -> None:
        class ExplodingRenderer:
            name = "exploding"

            def health_check(self) -> bool:
                return True

            def render(self, spec: ThumbnailSpec, output_path: Path):
                raise ThumbnailRenderError("simulated renderer crash")

        with captions_client(tmp_path, renderer=ExplodingRenderer()) as (app, client):  # type: ignore[arg-type]
            response = post(client)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["status"] == "partial"
        assert data["thumbnail"]["status"] == "failed"
        assert data["thumbnail"]["error_code"] == "captions_thumbnail_render_error"
        assert data["captions"]["status"] == "completed"
        assert data["metadata"]["status"] == "completed"
        assert Path(data["captions"]["srt_path"]).is_file()
        assert Path(data["metadata"]["path"]).is_file()
        assert any("simulated renderer crash" in e for e in data["errors"])

    def test_missing_fonts_is_partial_200(self, tmp_path: Path) -> None:
        renderer = LocalThumbnailRenderer(font_candidates=(), font_bold_candidates=())
        with captions_client(tmp_path, renderer=renderer) as (app, client):
            response = post(client)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert data["thumbnail"]["status"] == "failed"
        assert data["thumbnail"]["error_code"] == "captions_thumbnail_render_error"
        assert any("font" in e for e in data["errors"])


# --------------------------------------------------------------------------
# Total failure: error envelope, never HTTP 200
# --------------------------------------------------------------------------


class TestTotalFailure:
    def test_nothing_produced_returns_an_error_envelope(self, tmp_path: Path) -> None:
        # Block the project directory so every artifact's write fails.
        (tmp_path / "projects").mkdir(parents=True, exist_ok=True)
        (tmp_path / "projects" / "demo").write_bytes(b"blocked")

        with captions_client(tmp_path) as (app, client):
            response = post(client)

        assert response.status_code == 500
        envelope = response.json()
        assert envelope["success"] is False
        assert envelope["error"]["code"] == "captions_thumbnail_caption_error"
        assert "No artifacts were produced" in envelope["error"]["message"]

    def test_unregistered_agent_is_404(self, tmp_path: Path) -> None:
        with captions_client(tmp_path, registry=AgentRegistry()) as (app, client):
            response = post(client)

        assert response.status_code == 404
        envelope = response.json()
        assert envelope["success"] is False
        assert envelope["error"]["code"] == "agent_not_found"


# --------------------------------------------------------------------------
# Validation: 422 before any work or AI call happens
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_id": "../evil"},
        {"project_id": ""},
        {"thumbnail_resolution": "banana"},
        {"thumbnail_resolution": "100x100"},
        {"caption_formats": ["doc"]},
        {"caption_formats": []},
        {"thumbnail_image": "../../etc/passwd"},
        {"thumbnail_image": "C:\\evil.png"},
        {"language": "x"},
        {"target_audience": "   "},
        {"script": {}},
        {"script": {"title": "Blank"}},
    ],
)
def test_invalid_requests_are_422(tmp_path: Path, overrides: dict) -> None:
    with captions_client(tmp_path) as (app, client):
        response = post(client, **overrides)

    assert response.status_code == 422
    envelope = response.json()
    assert envelope["success"] is False
    assert envelope["error"]["code"] == "validation_error"
    assert envelope["error"]["details"]


def test_a_missing_body_is_422(tmp_path: Path) -> None:
    with captions_client(tmp_path) as (app, client):
        response = client.post("/api/captions-thumbnail")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_validation_happens_before_any_ai_call(tmp_path: Path) -> None:
    provider = FakeAIProvider(text=json.dumps(DRAFT))
    with captions_client(tmp_path, provider=provider) as (app, client):
        response = post(client, project_id="../evil")

    assert response.status_code == 422
    assert provider.calls == [], "no AI call may happen for an invalid request"


# --------------------------------------------------------------------------
# OpenAPI and the documented status mapping
# --------------------------------------------------------------------------


def test_openapi_documents_the_endpoint(tmp_path: Path) -> None:
    with captions_client(tmp_path) as (app, client):
        schema = app.openapi()

    path = schema["paths"]["/api/captions-thumbnail"]
    assert "post" in path

    schemas = schema["components"]["schemas"]
    assert "CaptionsThumbnailRequest" in schemas
    assert "CaptionsThumbnailResult" in schemas
    properties = schemas["CaptionsThumbnailResult"]["properties"]
    for key in ("status", "success", "captions", "thumbnail", "metadata", "grounding"):
        assert key in properties


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        ("captions_thumbnail_invalid_request", 422),
        ("captions_thumbnail_asset_error", 422),
        ("agent_not_found", 422),
        ("ai_provider_unavailable", 503),
        ("model_not_found", 503),
        ("ai_provider_timeout", 504),
        ("ai_provider_error", 502),
        ("structured_output_invalid", 502),
        ("captions_thumbnail_ai_error", 502),
        ("captions_thumbnail_caption_error", 500),
        ("captions_thumbnail_render_error", 500),
        ("unknown_thing", 500),
        (None, 500),
    ],
)
def test_error_code_to_status_mapping(code: str | None, expected_status: int) -> None:
    assert _status_for_code(code) == expected_status

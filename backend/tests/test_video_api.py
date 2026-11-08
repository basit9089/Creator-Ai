"""Tests for ``POST /api/video/produce``.

The endpoint's contract: a valid request either returns a verified,
rendered video (200) or an error — never a half-truth. Validation
failures are 422, caller-fault agent errors are 422, and renderer
failures are 500.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.base import Agent, AgentContext, AgentError, AgentOutput
from app.agents.registry import AgentRegistry, get_registry
from app.core.config import reset_settings_cache
from app.main import create_app

#: A minimal script body that renders fast at a tiny resolution.
MINIMAL_SCRIPT: dict = {"title": "API Render", "narration": "tiny api render"}


def produce_body(**overrides: object) -> dict:
    body: dict = {
        "script": MINIMAL_SCRIPT,
        "project_id": "apitest",
        "resolution": "320x180",
        "fps": 10,
    }
    body.update(overrides)
    return body


class FailingAgent(Agent):
    """Test double registered as ``video_producer`` that always fails."""

    name = "video_producer"
    version = "1.0.0"
    description = "test double"

    def __init__(self, exc: AgentError) -> None:
        self._exc = exc

    async def run(self, context: AgentContext, messages: object) -> AgentOutput:
        raise self._exc


@pytest.fixture()
def failing_client_factory() -> object:
    """Build clients whose video_producer always raises ``exc``."""

    clients: list[TestClient] = []

    def factory(exc: AgentError) -> TestClient:
        app = create_app()
        registry = AgentRegistry()
        registry.register(FailingAgent(exc))
        app.dependency_overrides[get_registry] = lambda: registry
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield factory

    for client in clients:
        client.__exit__(None, None, None)
        client.app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Route wiring
# --------------------------------------------------------------------------


def test_video_route_is_registered(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/video/produce" in schema["paths"]
    assert "post" in schema["paths"]["/api/video/produce"]


def test_produce_requires_a_body(client: TestClient) -> None:
    response = client.post("/api/video/produce")
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Request validation (422 before any work happens)
# --------------------------------------------------------------------------


def test_missing_script_is_422(client: TestClient) -> None:
    response = client.post("/api/video/produce", json={"project_id": "x"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": "banana"},
        {"resolution": "99999x99999"},
        {"fps": 999},
        {"output_format": "avi"},
        {"project_id": "../escape"},
        {"output_filename": "../../escape.mp4"},
        {"asset_directory": "../outside"},
        {"aspect_ratio": "21:9"},
    ],
)
def test_invalid_fields_are_422(
    client: TestClient, overrides: dict
) -> None:
    response = client.post("/api/video/produce", json=produce_body(**overrides))
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"


def test_script_without_content_is_422(client: TestClient) -> None:
    response = client.post(
        "/api/video/produce",
        json=produce_body(script={"title": "Empty", "narration": "", "hook": "",
                                  "sections": [], "conclusion": "", "cta": ""}),
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Agent failure mapping
# --------------------------------------------------------------------------


def test_render_failure_is_500_with_its_code(
    failing_client_factory: object,
) -> None:
    from app.agents.video_producer.exceptions import VideoProducerRenderError

    client = failing_client_factory(  # type: ignore[operator]
        VideoProducerRenderError("ffmpeg exited with code 1")
    )
    response = client.post("/api/video/produce", json=produce_body())

    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "video_producer_render_error"


def test_agent_input_error_is_422(failing_client_factory: object) -> None:
    from app.agents.video_producer.exceptions import VideoProducerInputError

    client = failing_client_factory(  # type: ignore[operator]
        VideoProducerInputError("No video production request supplied")
    )
    response = client.post("/api/video/produce", json=produce_body())

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "video_producer_invalid_request"


def test_asset_error_is_422(failing_client_factory: object) -> None:
    from app.agents.video_producer.exceptions import VideoProducerAssetError

    client = failing_client_factory(  # type: ignore[operator]
        VideoProducerAssetError("Path escapes its base directory")
    )
    response = client.post("/api/video/produce", json=produce_body())

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "video_producer_asset_error"


def test_unregistered_agent_is_404(client: TestClient) -> None:
    app = create_app()
    app.dependency_overrides[get_registry] = AgentRegistry
    with TestClient(app, raise_server_exceptions=False) as bare_client:
        response = bare_client.post("/api/video/produce", json=produce_body())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "agent_not_found"


# --------------------------------------------------------------------------
# Success path: a real render (skipped without FFmpeg)
# --------------------------------------------------------------------------

FFMPEG_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not available")
def test_produce_renders_a_real_verified_video(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    reset_settings_cache()

    response = client.post("/api/video/produce", json=produce_body())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["file_size"] > 0
    assert body["codec"]
    assert body["width"] == 320 and body["height"] == 180
    assert body["audio_status"] == "silent"
    # Honest reporting: no image assets were supplied.
    assert body["asset_status"] == "missing"
    # The file is real, on disk, inside the configured storage root.
    output = Path(body["output_path"])
    assert output.is_file()
    assert output.stat().st_size == body["file_size"]
    assert str(tmp_path) in body["output_path"]
    # Narration travels through to the response.
    assert body["scenes"][0]["narration"] == "tiny api render"
    assert body["timeline"]["duration_seconds"] == body["duration_seconds"]


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not available")
def test_produce_is_confined_to_the_requested_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    reset_settings_cache()

    response = client.post(
        "/api/video/produce", json=produce_body(project_id="projectX")
    )

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "projectX"
    expected_root = tmp_path / "projects" / "projectX" / "renders"
    assert body["output_path"].startswith(str(expected_root))

"""Tests for basic health and service endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_returns_service_identification(client: TestClient) -> None:
    """GET / identifies the running service."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "Creator AI" in body["service"]


def test_health_returns_healthy(client: TestClient) -> None:
    """GET /health reports the service as alive."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ai_health_reports_provider_state(client: TestClient) -> None:
    """GET /api/health/ai reports provider reachability and model."""
    response = client.get("/api/health/ai")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["provider"] == "fake"
    assert body["model"] == "fake-model"
    assert body["reachable"] is True
    assert body["model_available"] is True


def test_unknown_route_returns_error_envelope(client: TestClient) -> None:
    """An unknown path returns the standard error envelope, not a stack trace."""
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert "error" in body
    assert "message" in body["error"]


def test_ai_health_does_not_expose_internals(client: TestClient) -> None:
    """The health payload must not leak server configuration or tracebacks."""
    response = client.get("/api/health/ai")

    assert response.status_code == 200
    raw = response.text

    # The Ollama host is server-side configuration and must never be sent
    # to a browser.
    assert "11434" not in raw
    assert "Traceback" not in raw
    assert "ollama_host" not in raw
"""Tests for API authentication and application wiring.

Auth is tested against a route that actually depends on it, rather than by
calling the dependency function directly. The distinction matters: a security
control that works in isolation but was never wired into the route is the
usual way authentication silently fails to exist.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.main import create_app

TEST_API_KEY = "test-key-do-not-use-anywhere-real"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_key=TEST_API_KEY,
        app_env="local",
        cors_origins=["http://localhost:5173"],
        gemini_enabled=False,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """An app with one protected route mounted, to exercise the real dependency.

    Entered as a context manager so the lifespan handler actually runs.
    A TestClient used without `with` silently skips startup, which would let
    these tests pass against an app that fails to boot in production.
    """
    app: FastAPI = create_app(settings)

    @app.get("/api/v1/_protected", dependencies=[Depends(require_api_key)])
    async def protected() -> dict[str, str]:
        return {"ok": "true"}

    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_needs_no_credentials(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_reports_model_state_as_a_boolean(self, client: TestClient) -> None:
        assert isinstance(client.get("/health").json()["model_loaded"], bool)

    def test_service_starts_without_artifacts_and_says_so(self, tmp_path) -> None:
        """Missing artifacts must degrade, not crash.

        An API that refuses to boot when a model file is absent is harder to
        diagnose than one that starts, answers its health probe honestly, and
        returns an actionable 503 from the scoring routes. This verifies the
        degraded path rather than assuming it.
        """
        degraded_settings = Settings(
            api_key=TEST_API_KEY,
            app_env="local",
            gemini_enabled=False,
            artifact_dir=tmp_path,   # empty: no model_bundle.joblib
        )
        degraded = create_app(degraded_settings)
        # Without this the auth dependency resolves the process-wide settings
        # from .env and rejects the test key, so the request 401s before it
        # ever reaches the scoring route. Worth noting that ordering is correct:
        # an unauthenticated caller must not learn whether a model is loaded.
        degraded.dependency_overrides[get_settings] = lambda: degraded_settings

        with TestClient(degraded) as degraded_client:
            health = degraded_client.get("/health").json()
            assert health["status"] == "ok"
            assert health["model_loaded"] is False

            scored = degraded_client.post(
                "/api/v1/applications/score",
                headers={"X-API-Key": TEST_API_KEY},
                json={},
            )
            assert scored.status_code == 503
            assert "train.py" in scored.json()["detail"]

    def test_health_leaks_no_configuration(self, client: TestClient) -> None:
        """A liveness probe must not become a reconnaissance endpoint."""
        body = client.get("/health").json()
        assert set(body) == {"status", "model_loaded"}


class TestApiKeyAuth:
    def test_missing_key_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/v1/_protected")
        assert response.status_code == 401
        assert "Missing X-API-Key" in response.json()["detail"]

    def test_wrong_key_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/v1/_protected", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key."

    def test_correct_key_is_accepted(self, client: TestClient) -> None:
        response = client.get("/api/v1/_protected", headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200

    def test_error_does_not_echo_the_expected_key(self, client: TestClient) -> None:
        """Failure messages must never leak the secret they were compared against."""
        response = client.get("/api/v1/_protected", headers={"X-API-Key": "wrong"})
        assert TEST_API_KEY not in response.text

    @pytest.mark.parametrize("near_miss", ["", " ", TEST_API_KEY + "x", TEST_API_KEY[:-1]])
    def test_near_miss_keys_are_rejected(self, client: TestClient, near_miss: str) -> None:
        response = client.get("/api/v1/_protected", headers={"X-API-Key": near_miss})
        assert response.status_code == 401


class TestCors:
    def test_configured_origin_is_allowed(self, client: TestClient) -> None:
        response = client.options(
            "/api/v1/_protected",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_unknown_origin_is_not_allowed(self, client: TestClient) -> None:
        """Guards against a wildcard creeping into the CORS configuration."""
        response = client.options(
            "/api/v1/_protected",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") != "*"
        assert response.headers.get("access-control-allow-origin") != "https://evil.example.com"


class TestProductionHardening:
    def test_docs_are_disabled_in_production(self) -> None:
        """Interactive docs are a contract locally and an attack surface in prod."""
        prod = create_app(
            Settings(api_key=TEST_API_KEY, app_env="production", gemini_enabled=False)
        )
        prod_client = TestClient(prod)
        assert prod_client.get("/docs").status_code == 404
        assert prod_client.get("/openapi.json").status_code == 404

    def test_docs_are_available_locally(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200

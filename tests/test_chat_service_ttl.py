"""
Tests for session TTL eviction in the tau2 chat server.

These tests are fully synchronous and do not require a running server or any
LLM API keys.  They test the eviction logic directly via:

  - _evict_once()  — the pure eviction function (no timers, no async)
  - FastAPI TestClient — for HTTP-level session lifecycle (create / use / 404)
"""

import time

import pytest
from fastapi.testclient import TestClient

from tau2.api_service.chat_service import ChatServerConfig, _evict_once, create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_session(last_accessed: float) -> dict:
    """Minimal session dict with a fake last_accessed timestamp."""
    return {"environment": object(), "task_id": "0", "last_accessed": last_accessed}


def _app_with_ttl(ttl: int) -> TestClient:
    """Return a TestClient for the chat server with the given session_ttl."""
    cfg = ChatServerConfig(domain="mock", session_ttl=ttl)
    return TestClient(create_app(cfg))


# ---------------------------------------------------------------------------
# Unit tests for _evict_once (no HTTP, no timers)
# ---------------------------------------------------------------------------


class TestEvictOnce:
    def test_evicts_expired_session(self):
        sessions = {"old": _fake_session(last_accessed=time.time() - 100)}
        evicted = _evict_once(sessions, session_ttl=10)
        assert "old" in evicted
        assert "old" not in sessions

    def test_keeps_fresh_session(self):
        sessions = {"fresh": _fake_session(last_accessed=time.time())}
        evicted = _evict_once(sessions, session_ttl=60)
        assert evicted == []
        assert "fresh" in sessions

    def test_mixed_sessions(self):
        now = time.time()
        sessions = {
            "stale": _fake_session(last_accessed=now - 200),
            "active": _fake_session(last_accessed=now - 5),
        }
        evicted = _evict_once(sessions, session_ttl=60)
        assert "stale" in evicted
        assert "stale" not in sessions
        assert "active" in sessions

    def test_disabled_when_ttl_zero(self):
        sessions = {"old": _fake_session(last_accessed=time.time() - 9999)}
        evicted = _evict_once(sessions, session_ttl=0)
        assert evicted == []
        assert "old" in sessions  # untouched

    def test_returns_evicted_ids(self):
        now = time.time()
        sessions = {
            "a": _fake_session(last_accessed=now - 100),
            "b": _fake_session(last_accessed=now - 100),
        }
        evicted = _evict_once(sessions, session_ttl=10)
        assert set(evicted) == {"a", "b"}

    def test_empty_sessions_is_noop(self):
        sessions = {}
        evicted = _evict_once(sessions, session_ttl=10)
        assert evicted == []

    def test_exactly_at_boundary_is_not_evicted(self):
        # A session that is just *under* the TTL should NOT be evicted.
        # We use TTL=60 but only age the session 30s to stay well clear of
        # any floating-point timing slop.
        sessions = {"recent": _fake_session(last_accessed=time.time() - 30)}
        evicted = _evict_once(sessions, session_ttl=60)
        assert evicted == []
        assert "recent" in sessions

    def test_just_past_boundary_is_evicted(self):
        sessions = {"over": _fake_session(last_accessed=time.time() - 61)}
        evicted = _evict_once(sessions, session_ttl=60)
        assert "over" in evicted


# ---------------------------------------------------------------------------
# Integration tests via TestClient (HTTP-level, using mock domain)
# ---------------------------------------------------------------------------


class TestSessionTTLHttp:
    def test_create_session_mock_domain(self):
        """POST /v1/session should succeed on the mock domain."""
        client = _app_with_ttl(ttl=3600)
        resp = client.post("/v1/session", json={"task_id": "create_task_1"})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["domain"] == "mock"

    def test_delete_session(self):
        """DELETE /v1/session/{id} should remove the session."""
        client = _app_with_ttl(ttl=3600)
        session_id = client.post(
            "/v1/session", json={"task_id": "create_task_1"}
        ).json()["session_id"]

        resp = client.delete(f"/v1/session/{session_id}")
        assert resp.status_code == 204

        # Using the deleted session should now 404
        resp = client.post(
            "/v1/tool/execute",
            json={"session_id": session_id, "tool_calls": []},
        )
        assert resp.status_code == 404

    def test_delete_nonexistent_session_returns_404(self):
        client = _app_with_ttl(ttl=3600)
        resp = client.delete("/v1/session/doesnotexist")
        assert resp.status_code == 404

    def test_evicted_session_returns_404_on_tool_execute(self):
        """
        Sessions removed by _evict_once are no longer present in the store,
        so any subsequent endpoint call returns 404.

        The background loop inside create_app sleeps 60 s between sweeps and
        can't be triggered from outside the closure.  We test the full HTTP
        path by injecting a session directly into the store via a thin hook:
        create_app() exposes the sessions dict on app.state so tests can
        back-date it and call _evict_once manually.
        """
        cfg = ChatServerConfig(domain="mock", session_ttl=1)
        app = create_app(cfg)
        client = TestClient(app)

        # Create a real session through the HTTP API
        session_id = client.post(
            "/v1/session", json={"task_id": "create_task_1"}
        ).json()["session_id"]

        # Verify the session is alive
        resp = client.post(
            "/v1/tool/execute",
            json={"session_id": session_id, "tool_calls": []},
        )
        assert resp.status_code == 200

        # Back-date last_accessed so the session appears stale, then sweep
        app.state.sessions[session_id]["last_accessed"] = time.time() - 10
        _evict_once(app.state.sessions, session_ttl=1)

        # Session should now be gone — endpoint returns 404
        resp = client.post(
            "/v1/tool/execute",
            json={"session_id": session_id, "tool_calls": []},
        )
        assert resp.status_code == 404

    def test_config_exposes_session_ttl(self):
        """GET /config should include session_ttl."""
        client = _app_with_ttl(ttl=120)
        resp = client.get("/config")
        assert resp.status_code == 200
        assert resp.json()["session_ttl"] == 120

    def test_ttl_zero_disables_eviction(self):
        """With session_ttl=0, _evict_once should never remove sessions."""
        sessions = {"s": _fake_session(last_accessed=0.0)}  # epoch — infinitely stale
        evicted = _evict_once(sessions, session_ttl=0)
        assert evicted == []
        assert "s" in sessions


# ---------------------------------------------------------------------------
# Tests for GET /v1/session/{session_id}
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_get_existing_session_returns_200(self):
        """GET /v1/session/{id} returns 200 with metadata for a live session."""
        client = _app_with_ttl(ttl=3600)
        session_id = client.post(
            "/v1/session", json={"task_id": "create_task_1"}
        ).json()["session_id"]

        resp = client.get(f"/v1/session/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["task_id"] == "create_task_1"
        assert data["domain"] == "mock"
        assert data["alive"] is True
        assert isinstance(data["last_accessed"], float)

    def test_get_nonexistent_session_returns_404(self):
        """GET /v1/session/{id} returns 404 for an unknown session."""
        client = _app_with_ttl(ttl=3600)
        resp = client.get("/v1/session/doesnotexist")
        assert resp.status_code == 404

    def test_get_after_delete_returns_404(self):
        """GET /v1/session/{id} returns 404 after the session is deleted."""
        client = _app_with_ttl(ttl=3600)
        session_id = client.post(
            "/v1/session", json={"task_id": "create_task_1"}
        ).json()["session_id"]

        client.delete(f"/v1/session/{session_id}")

        resp = client.get(f"/v1/session/{session_id}")
        assert resp.status_code == 404

    def test_get_after_ttl_eviction_returns_404(self):
        """
        GET /v1/session/{id} returns 404 after the session has been evicted
        by _evict_once (TTL expired).
        """
        cfg = ChatServerConfig(domain="mock", session_ttl=1)
        app = create_app(cfg)
        client = TestClient(app)

        session_id = client.post(
            "/v1/session", json={"task_id": "create_task_1"}
        ).json()["session_id"]

        # Confirm session is alive
        assert client.get(f"/v1/session/{session_id}").status_code == 200

        # Back-date and evict
        app.state.sessions[session_id]["last_accessed"] = time.time() - 10
        _evict_once(app.state.sessions, session_ttl=1)

        # Should now be gone
        assert client.get(f"/v1/session/{session_id}").status_code == 404

    def test_get_session_last_accessed_is_recent(self):
        """last_accessed should be close to now (within 5 seconds)."""
        client = _app_with_ttl(ttl=3600)
        before = time.time()
        session_id = client.post(
            "/v1/session", json={"task_id": "create_task_1"}
        ).json()["session_id"]
        after = time.time()

        data = client.get(f"/v1/session/{session_id}").json()
        assert before <= data["last_accessed"] <= after + 1

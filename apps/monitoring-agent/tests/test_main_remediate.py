"""main.py 新端点（自愈闭环）单测。

不依赖真实 PG/k8s：store 用 fake，k8s 用 monkeypatch。
Windows 下 psycopg3 需要 WindowsSelectorEventLoopPolicy。
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys

import pytest

# Windows 下 psycopg3 需要 SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_MON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MON not in sys.path:
    sys.path.insert(0, _MON)

# 必须在 import monitoring_agent.main 之前设置 token，否则 _auth_ok 走 default-deny
os.environ.setdefault("MONITORING_WEBHOOK_TOKEN", "test-token")

from monitoring_agent import main, store


# ── Fake store（复用 test_store 的 FakeCursor/FakeConn 逻辑） ──


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn
        self.executed: list[tuple[str, tuple[Any, ...] | None]] = []
        self._rows: list[tuple[Any, ...]] = []
        self.rowcount = 0

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        stripped = sql.strip().upper()
        if stripped.startswith("INSERT") and "RETURNING" in stripped:
            self._rows = [(1,)]
        elif stripped.startswith("UPDATE") and "RETURNING" in stripped:
            if "STATUS = 'PENDING_APPROVAL'" in stripped and isinstance(params, (tuple, list)) and len(params) >= 2:
                plan_id = params[1]
                if plan_id in self._conn._approved_plans:
                    self._rows = []
                else:
                    self._conn._approved_plans.add(plan_id)
                    self._rows = [(1,)]
            else:
                self._rows = []
        elif stripped.startswith("SELECT"):
            self._rows = []

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    async def __aenter__(self) -> "_FakeCursor":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self._approved_plans: set[int] = set()

    def cursor(self) -> _FakeCursor:
        if self._cursor is None or self._cursor.executed:
            self._cursor = _FakeCursor(self)
        return self._cursor

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "_FakeConn":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


_fake_conn_singleton: _FakeConn | None = None


async def _fake_connect(dsn: str) -> _FakeConn:
    global _fake_conn_singleton
    if _fake_conn_singleton is None:
        _fake_conn_singleton = _FakeConn(None)
    return _fake_conn_singleton


# ── 工具 ──


def _reload_main():
    importlib.reload(store)
    importlib.reload(main)
    return main.app


@pytest.fixture()
def app():
    return _reload_main()


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_store():
    store._conn_factory = None
    yield
    store._conn_factory = None


@pytest.fixture()
def fake_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(store, "_conn_factory", lambda: _fake_connect("x"))


@pytest.fixture()
def auth_headers():
    return {"Authorization": "Bearer test-token"}


# ── 端点注册 ──


def test_new_endpoints_registered(app):
    paths = {getattr(r, "path", "") for r in app.routes}
    for need in (
        "/remediate/preview",
        "/remediate/approve",
        "/remediate/apply",
        "/rca/history",
        "/remediate/plans",
    ):
        assert need in paths, f"缺少端点 {need}"


# ── 鉴权 ──


def test_preview_requires_auth(client, auth_headers):
    r = client.post("/remediate/preview", json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}})
    assert r.status_code == 401


def test_approve_requires_auth(client, auth_headers):
    r = client.post("/remediate/approve", json={"plan_id": 1, "decision": "approved"})
    assert r.status_code == 401


def test_apply_requires_auth(client, auth_headers):
    r = client.post("/remediate/apply", json={"plan_id": 1, "script": {"action": "scale_up", "target": "redis", "params": {"replicas": 1}}, "approved": True})
    assert r.status_code == 401


def test_history_requires_auth(client, auth_headers):
    r = client.get("/rca/history")
    assert r.status_code == 401


def test_plans_requires_auth(client, auth_headers):
    r = client.get("/remediate/plans")
    assert r.status_code == 401


# ── preview → approve → apply 流 ──


def test_preview_returns_plan_id(client, auth_headers, fake_store):
    r = client.post(
        "/remediate/preview",
        json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
        headers=auth_headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert "plan_id" in data
    assert data["status"] == "pending_approval"
    assert "evidence" in data


def test_preview_invalid_action(client, auth_headers):
    r = client.post(
        "/remediate/preview",
        json={"action": "delete_all", "target": "redis"},
        headers=auth_headers,
    )
    assert r.status_code == 400


def test_approve_plan(client, auth_headers, fake_store):
    r = client.post(
        "/remediate/preview",
        json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
        headers=auth_headers,
    )
    plan_id = r.json()["plan_id"]

    r = client.post(
        "/remediate/approve",
        json={"plan_id": plan_id, "decision": "approved", "approver": "admin", "reason": "ok"},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "approved"


def test_duplicate_approve_returns_409(client, auth_headers, fake_store):
    r = client.post(
        "/remediate/preview",
        json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
        headers=auth_headers,
    )
    plan_id = r.json()["plan_id"]

    client.post(
        "/remediate/approve",
        json={"plan_id": plan_id, "decision": "approved"},
        headers=auth_headers,
    )
    r = client.post(
        "/remediate/approve",
        json={"plan_id": plan_id, "decision": "rejected"},
        headers=auth_headers,
    )
    assert r.status_code == 409


def test_apply_without_approve(client, auth_headers, fake_store):
    r = client.post(
        "/remediate/preview",
        json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
        headers=auth_headers,
    )
    plan_id = r.json()["plan_id"]

    r = client.post(
        "/remediate/apply",
        json={"plan_id": plan_id, "script": {"action": "scale_up", "target": "redis", "params": {"replicas": 1}}, "approved": True},
        headers=auth_headers,
    )
    assert r.status_code == 200


def test_apply_returns_execution_id(client, auth_headers, fake_store, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("monitoring_agent.k8s_client.set_replicas", lambda name, replicas: {"name": name, "replicas": replicas})
    r = client.post(
        "/remediate/preview",
        json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
        headers=auth_headers,
    )
    plan_id = r.json()["plan_id"]

    r = client.post(
        "/remediate/apply",
        json={"plan_id": plan_id, "script": {"action": "scale_up", "target": "redis", "params": {"replicas": 1}}, "approved": True},
        headers=auth_headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["applied"] is True
    assert "execution_id" in data


# ── 查询端点 ──


def test_rca_history_empty(client, auth_headers, fake_store):
    r = client.get("/rca/history", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == []


def test_remediate_plans_empty(client, auth_headers, fake_store):
    r = client.get("/remediate/plans", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == []


def test_remediate_plans_with_status(client, auth_headers, fake_store):
    r = client.get("/remediate/plans?status=pending_approval", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == []


# ── MONITORING_PERSIST=0 ──


def test_persist_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MONITORING_PERSIST", "0")
    app = _reload_main()
    from fastapi.testclient import TestClient
    c = TestClient(app)

    r = c.post(
        "/remediate/preview",
        json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["plan_id"] is None
    assert data["status"] == "pending_approval"


# ── DB 降级（store 抛 StoreUnavailable 时端点仍 200） ──


def test_preview_db_unavailable(monkeypatch: pytest.MonkeyPatch, client, auth_headers):
    async def boom(dsn: str):
        raise store.StoreUnavailable("simulated")

    monkeypatch.setattr(store, "_conn_factory", lambda: boom)
    r = client.post(
        "/remediate/preview",
        json={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
        headers=auth_headers,
    )
    assert r.status_code == 200

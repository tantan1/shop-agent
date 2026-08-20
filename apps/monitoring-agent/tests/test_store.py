"""store.py 单测（不依赖真实 PG）。

通过 monkeypatch 把 psycopg.AsyncConnection.connect 替换为 FakeConn，
在内存中模拟 cursor/execute/fetchone/fetchall/commit/rollback。
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import patch

import psycopg
import pytest

_MON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MON not in sys.path:
    sys.path.insert(0, _MON)

from monitoring_agent import store


# ── Fake 基础设施 ──


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...] | None]] = []
        self._rows: list[tuple[Any, ...]] = []
        self.rowcount = 0

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        stripped = sql.strip().upper()
        if stripped.startswith("INSERT") and "RETURNING" in stripped:
            self._rows = [(1,)]
        elif stripped.startswith("UPDATE") and "RETURNING" in stripped:
            self._rows = []
        elif stripped.startswith("SELECT"):
            self._rows = []

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    async def __aenter__(self) -> FakeCursor:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class FakeConn:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> FakeConn:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


async def _fake_connect(dsn: str) -> FakeConn:
    return FakeConn(FakeCursor())


@pytest.fixture(autouse=True)
def _reset_factory():
    store._conn_factory = None
    yield
    store._conn_factory = None


@pytest.fixture()
def fake_pg(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)


# ── 用例 ──


def test_ensure_schema(fake_pg: None):
    store._conn_factory = None

    async def run():
        await store.ensure_schema()

    import asyncio

    asyncio.run(run())
    # 无异常即视为建表成功；真实 PG 上验证见 Phase 1 验收


def test_save_rca_run(fake_pg: None):
    store._conn_factory = None

    async def run():
        run_id = await store.save_rca_run(
            source="alertmanager",
            severity="P1",
            root_cause="redis down",
            affected=["redis"],
            recommendations=["scale up"],
            evidence={"topology_down": ["redis"]},
            remediation={"action": "scale_up", "target": "redis", "replicas": 1},
            used_llm=False,
        )
        assert run_id == 1

    import asyncio

    asyncio.run(run())


def test_create_plan(fake_pg: None):
    store._conn_factory = None

    async def run():
        plan_id = await store.create_plan(
            run_id=1,
            script={"action": "scale_up", "target": "redis", "params": {"replicas": 1}},
            evidence={"plan": {}, "call_sequence": []},
            risk="",
        )
        assert plan_id == 1

    import asyncio

    asyncio.run(run())


def test_approve_plan_guard(fake_pg: None, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    store._conn_factory = None

    async def run():
        updated = await store.approve_plan(plan_id=1, decision="approved", approver="admin")
        assert updated is False

    import asyncio

    asyncio.run(run())
    assert "审批被拒" in caplog.text


def test_save_execution_guard(fake_pg: None, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    store._conn_factory = None

    async def run():
        exec_id = await store.save_execution(
            plan_id=1, applied=True, already_at_target=False, result={}, error=""
        )
        assert exec_id is not None

    import asyncio

    asyncio.run(run())
    assert "执行被拒" in caplog.text


def test_store_unavailable_propagates(monkeypatch: pytest.MonkeyPatch):
    async def boom(dsn: str):
        raise store.StoreUnavailable("simulated")

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", staticmethod(boom))
    store._conn_factory = None

    async def run():
        with pytest.raises(store.StoreUnavailable):
            await store.save_rca_run(
                source="x", severity="P3", root_cause="", affected=[], recommendations=[], evidence={}, remediation=None
            )

    import asyncio

    asyncio.run(run())


def test_list_rca_runs(fake_pg: None):
    store._conn_factory = None

    async def run():
        rows = await store.list_rca_runs(limit=10)
        assert rows == []

    import asyncio

    asyncio.run(run())


def test_list_plans(fake_pg: None):
    store._conn_factory = None

    async def run():
        rows = await store.list_plans(status="pending_approval", limit=5)
        assert rows == []

    import asyncio

    asyncio.run(run())

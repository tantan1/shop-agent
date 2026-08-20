"""Postgres 异步访问层（psycopg3 原生异步）。

职责：
- 幂等建表 / 索引（ensure_schema）
- 4 类写操作：save_rca_run / create_plan / approve_plan / save_execution
- 读操作：get_plan / list_plans / list_rca_runs
- 状态机守卫（条件 UPDATE，防并发双批）
- 失败降级：所有 DB 异常包装为 StoreUnavailable，由调用方捕获后仅打日志 + 计数
"""

from __future__ import annotations

import logging
import os
from typing import Any

import psycopg
from psycopg.types import json as psycopg_json

logger = logging.getLogger("monitoring_agent.store")


class StoreUnavailable(Exception):
    """DB 不可达或写库失败（主链路应捕获并降级）。"""


# ── DSN 构造 ──


def _build_dsn() -> str:
    dsn = os.getenv("MONITORING_DB_DSN", "").strip()
    if dsn:
        return dsn
    host = os.getenv("POSTGRES_HOST", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "local-postgres-password")
    db = os.getenv("MONITORING_DB_NAME", "postgres")
    return f"postgresql://{user}:{password}@{host}:5432/{db}"


# ── 惰性连接（便于单测 monkeypatch） ──

_conn_factory = None


def _get_conn_factory():
    global _conn_factory
    if _conn_factory is None:
        dsn = _build_dsn()

        async def factory():
            return await psycopg.AsyncConnection.connect(dsn)

        _conn_factory = factory
    return _conn_factory


async def _get_conn():
    factory = _get_conn_factory()
    try:
        return await factory()
    except Exception as exc:
        raise StoreUnavailable(f"无法连接数据库: {exc}") from exc


# ── 建表（幂等） ──

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rca_runs (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    source          TEXT NOT NULL,
    severity        TEXT NOT NULL,
    root_cause      TEXT NOT NULL,
    affected        JSONB NOT NULL DEFAULT '[]',
    recommendations JSONB NOT NULL DEFAULT '[]',
    evidence        JSONB NOT NULL DEFAULT '{}',
    remediation     JSONB,
    used_llm        BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS remediation_plans (
    id         BIGSERIAL PRIMARY KEY,
    run_id     BIGINT REFERENCES rca_runs(id) ON DELETE SET NULL,
    created_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    script     JSONB NOT NULL,
    evidence   JSONB NOT NULL DEFAULT '{}',
    risk       TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'plan_ready',
    error      TEXT
);

CREATE TABLE IF NOT EXISTS approvals (
    id       BIGSERIAL PRIMARY KEY,
    plan_id  BIGINT NOT NULL REFERENCES remediation_plans(id) ON DELETE CASCADE,
    approver TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL,
    reason   TEXT NOT NULL DEFAULT '',
    meta     JSONB NOT NULL DEFAULT '{}',
    ts       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS executions (
    id                BIGSERIAL PRIMARY KEY,
    plan_id           BIGINT REFERENCES remediation_plans(id) ON DELETE SET NULL,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied           BOOLEAN NOT NULL DEFAULT FALSE,
    already_at_target BOOLEAN NOT NULL DEFAULT FALSE,
    result            JSONB,
    error             TEXT
);

CREATE INDEX IF NOT EXISTS idx_plans_status ON remediation_plans(status);
CREATE INDEX IF NOT EXISTS idx_plans_run ON remediation_plans(run_id);
CREATE INDEX IF NOT EXISTS idx_plans_script_action ON remediation_plans ((script->>'action'));
CREATE INDEX IF NOT EXISTS idx_approvals_plan ON approvals(plan_id);
CREATE INDEX IF NOT EXISTS idx_exec_plan ON executions(plan_id);
"""


async def ensure_schema() -> None:
    conn = await _get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_SCHEMA_SQL)
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    finally:
        await conn.close()


# ── 写操作 ──


async def save_rca_run(
    source: str,
    severity: str,
    root_cause: str,
    affected: list[str],
    recommendations: list[str],
    evidence: dict[str, Any],
    remediation: dict[str, Any] | None,
    used_llm: bool = False,
) -> int | None:
    conn = await _get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO rca_runs
                    (source, severity, root_cause, affected, recommendations, evidence, remediation, used_llm)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    source,
                    severity,
                    root_cause,
                    psycopg_json.Json(affected),
                    psycopg_json.Json(recommendations),
                    psycopg_json.Json(evidence),
                    psycopg_json.Json(remediation),
                    used_llm,
                ),
            )
            row = await cur.fetchone()
            run_id = row[0] if row else None
        await conn.commit()
        logger.info("RCA 已落库 run_id=%s", run_id)
        return run_id
    except StoreUnavailable:
        raise
    except Exception as exc:
        await conn.rollback()
        raise StoreUnavailable(f"save_rca_run 失败: {exc}") from exc
    finally:
        await conn.close()


async def create_plan(
    run_id: int | None,
    script: dict[str, Any],
    evidence: dict[str, Any],
    risk: str = "",
) -> int | None:
    conn = await _get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO remediation_plans (run_id, script, evidence, risk, status)
                VALUES (%s, %s, %s, %s, 'pending_approval')
                RETURNING id
                """,
                (
                    run_id,
                    psycopg_json.Json(script),
                    psycopg_json.Json(evidence),
                    risk,
                ),
            )
            row = await cur.fetchone()
            plan_id = row[0] if row else None
        await conn.commit()
        logger.info("Plan 已创建 plan_id=%s run_id=%s", plan_id, run_id)
        return plan_id
    except StoreUnavailable:
        raise
    except Exception as exc:
        await conn.rollback()
        raise StoreUnavailable(f"create_plan 失败: {exc}") from exc
    finally:
        await conn.close()


async def approve_plan(plan_id: int, decision: str, approver: str = "", reason: str = "") -> bool:
    """条件 UPDATE：仅 pending_approval 可转 approved/rejected。返回是否生效。"""
    if decision not in ("approved", "rejected"):
        raise ValueError(f"非法 decision: {decision}")

    conn = await _get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE remediation_plans
                SET status = %s
                WHERE id = %s AND status = 'pending_approval'
                RETURNING id
                """,
                (decision, plan_id),
            )
            row = await cur.fetchone()
            updated = row is not None
            if updated:
                await cur.execute(
                    """
                    INSERT INTO approvals (plan_id, decision, approver, reason)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (plan_id, decision, approver, reason),
                )
                logger.info("Plan %s 审批 %s（审批人=%s）", plan_id, decision, approver)
            else:
                logger.warning("Plan %s 审批被拒：当前状态非 pending_approval", plan_id)
        await conn.commit()
        return updated
    except StoreUnavailable:
        raise
    except Exception as exc:
        await conn.rollback()
        raise StoreUnavailable(f"approve_plan 失败: {exc}") from exc
    finally:
        await conn.close()


async def save_execution(
    plan_id: int,
    applied: bool,
    already_at_target: bool,
    result: dict[str, Any] | None = None,
    error: str = "",
) -> int | None:
    """条件 UPDATE + 插入 execution：仅 approved 可转 executed/failed。"""
    conn = await _get_conn()
    try:
        async with conn.cursor() as cur:
            status = "executed" if applied else "failed"
            await cur.execute(
                """
                UPDATE remediation_plans
                SET status = %s, error = %s
                WHERE id = %s AND status = 'approved'
                RETURNING id
                """,
                (status, error, plan_id),
            )
            row = await cur.fetchone()
            updated = row is not None
            if not updated:
                logger.warning("Plan %s 执行被拒：当前状态非 approved", plan_id)

            await cur.execute(
                """
                INSERT INTO executions (plan_id, applied, already_at_target, result, error)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    plan_id,
                    applied,
                    already_at_target,
                    psycopg_json.Json(result),
                    error,
                ),
            )
            row = await cur.fetchone()
            exec_id = row[0] if row else None
        await conn.commit()
        if updated:
            logger.info("Plan %s 执行已落库 exec_id=%s applied=%s", plan_id, exec_id, applied)
        return exec_id
    except StoreUnavailable:
        raise
    except Exception as exc:
        await conn.rollback()
        raise StoreUnavailable(f"save_execution 失败: {exc}") from exc
    finally:
        await conn.close()


# ── 读操作 ──


async def get_plan(plan_id: int) -> dict[str, Any] | None:
    conn = await _get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, run_id, created_ts, script, evidence, risk, status, error
                FROM remediation_plans
                WHERE id = %s
                """,
                (plan_id,),
            )
            row = await cur.fetchone()
        if not row:
            return None
        cols = ["id", "run_id", "created_ts", "script", "evidence", "risk", "status", "error"]
        return dict(zip(cols, row))
    except Exception as exc:
        raise StoreUnavailable(f"get_plan 失败: {exc}") from exc
    finally:
        await conn.close()


async def list_plans(status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    conn = await _get_conn()
    try:
        sql = """
            SELECT id, run_id, created_ts, script, evidence, risk, status, error
            FROM remediation_plans
        """
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = %s"
            params = (status,)
        sql += " ORDER BY id DESC LIMIT %s"
        params = params + (limit,)
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        cols = ["id", "run_id", "created_ts", "script", "evidence", "risk", "status", "error"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        raise StoreUnavailable(f"list_plans 失败: {exc}") from exc
    finally:
        await conn.close()


async def list_rca_runs(limit: int = 20) -> list[dict[str, Any]]:
    conn = await _get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, ts, source, severity, root_cause, affected, recommendations,
                       evidence, remediation, used_llm
                FROM rca_runs
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
        cols = [
            "id",
            "ts",
            "source",
            "severity",
            "root_cause",
            "affected",
            "recommendations",
            "evidence",
            "remediation",
            "used_llm",
        ]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as exc:
        raise StoreUnavailable(f"list_rca_runs 失败: {exc}") from exc
    finally:
        await conn.close()


# ── 保留策略（TTL 清理）──


async def cleanup_old(days: int = 30) -> int:
    """删除超过 days 天的历史记录（保留策略）。

    - rca_runs 按 ts 过期；
    - remediation_plans 按 created_ts 过期（approvals 设 ON DELETE CASCADE、
      executions 设 ON DELETE SET NULL，随之清理）。
    返回删除的总行数（估算）。DB 不可用时抛 StoreUnavailable，由调用方降级。
    """
    if days <= 0:
        raise ValueError("days 必须为正")
    conn = await _get_conn()
    try:
        deleted = 0
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM rca_runs WHERE ts < now() - make_interval(days => %s)",
                (days,),
            )
            deleted += cur.rowcount or 0
            await cur.execute(
                "DELETE FROM remediation_plans WHERE created_ts < now() - make_interval(days => %s)",
                (days,),
            )
            deleted += cur.rowcount or 0
        await conn.commit()
        logger.info("TTL 清理完成：删除约 %s 行（>%s 天）", deleted, days)
        return deleted
    except StoreUnavailable:
        raise
    except Exception as exc:
        await conn.rollback()
        raise StoreUnavailable(f"cleanup_old 失败: {exc}") from exc
    finally:
        await conn.close()

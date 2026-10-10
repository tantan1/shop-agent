"""SQLite 真值库:去重 + 标注 + 用户信号 + 增量水位线 + 自动银标闸门(§2.3/§5.1/§5.3)。

设计要点(Option A):
- Langfuse 仅作 trace 原料;标注真值(seen_queries.gold_tool)只落此处,不回写 Langfuse score。
- canonical_tid 为去重主键,首次见到即锁定,后续同 query 只累加 freq(幂等)。
- 增量同步水位线存 sync_meta(overlap 抗晚到)。
- gate_auto_label 在同步阶段对高 margin 样本自动银标,不进人工队列;抽部分做 spotcheck。
"""
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _qhash(query_text: str) -> str:
    return hashlib.sha256(query_text.strip().encode("utf-8")).hexdigest()


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.SQLITE_DB) or ".", exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    conn = get_conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen_queries (
                canonical_tid      TEXT PRIMARY KEY,
                trace_id           TEXT,
                conversation_id    TEXT,
                query_hash         TEXT,
                query_text         TEXT,
                sample_content     TEXT,
                category           TEXT,
                is_multi_intent    INTEGER DEFAULT 0,
                candidate_tools    TEXT,            -- JSON list
                available_tools    TEXT,            -- JSON list
                llm_suggested_tool TEXT,            -- 单独的 Phase-2 预标草稿(§4),与 top1_tool 区分
                rejected_tools     TEXT,            -- JSON list
                gold_tool          TEXT,            -- 真值(人工或自动银标)
                review_status      TEXT,            -- pending/correct/rejected/auto/spotcheck
                freq               INTEGER DEFAULT 1,
                last_freq_at       TEXT,
                first_seen         TEXT,
                last_seen          TEXT,
                top1_conf          REAL,
                top2_conf          REAL,
                margin             REAL,            -- top1_conf - top2_conf
                top1_tool          TEXT,            -- 模型运行时选定工具名(闸门提升目标)
                label_source       TEXT,            -- NULL/auto/confirm/full/spotcheck
                auto_passed        INTEGER DEFAULT 0,
                exported           INTEGER DEFAULT 0,
                updated_at         TEXT
            );
            CREATE TABLE IF NOT EXISTS sync_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback_signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id        TEXT,
                conversation_id TEXT,
                signal          TEXT,
                reason          TEXT,
                context         TEXT,             -- JSON
                created_at      TEXT
            );
            CREATE TABLE IF NOT EXISTS gold_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_tid TEXT,
                query_hash   TEXT,
                old_gold     TEXT,
                new_gold     TEXT,
                changed_by   TEXT,
                reason       TEXT,
                created_at   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_seen_query_hash ON seen_queries(query_hash);
            CREATE INDEX IF NOT EXISTS idx_seen_queue ON seen_queries(gold_tool, auto_passed);
            CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback_signals(trace_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


# ── 增量水位线(§5.1)──
def get_watermark() -> Optional[str]:
    conn = get_conn()
    try:
        row = conn.execute("SELECT value FROM sync_meta WHERE key='watermark'").fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def set_watermark(ts: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO sync_meta(key, value) VALUES('watermark', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (ts,),
        )
        conn.commit()
    finally:
        conn.close()


# ── 幂等 upsert(§5.1 canonical 锁死)──
def upsert_trace(rec: dict) -> str:
    """rec: trace_id, conversation_id, query_text, sample_content, category, is_multi_intent,
    candidate_tools(list), available_tools(list), llm_suggested_tool, top1_tool, top1_conf, top2_conf, margin, ts。
    返回 canonical_tid。"""
    query_text = (rec.get("query_text") or "").strip()
    canonical_tid = rec.get("trace_id") or rec.get("conversation_id") or _qhash(query_text)
    qh = _qhash(query_text)

    def _j(v):
        return json.dumps(v, ensure_ascii=False) if v is not None else None

    conn = get_conn()
    try:
        existing = conn.execute(
            "SELECT 1 FROM seen_queries WHERE canonical_tid=?", (canonical_tid,)
        ).fetchone()
        now = _now()
        if existing is None:
            conn.execute(
                """INSERT INTO seen_queries (
                    canonical_tid, trace_id, conversation_id, query_hash, query_text, sample_content,
                    category, is_multi_intent, candidate_tools, available_tools, llm_suggested_tool,
                    top1_tool, top1_conf, top2_conf, margin, first_seen, last_seen, last_freq_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    canonical_tid, rec.get("trace_id"), rec.get("conversation_id"), qh, query_text,
                    rec.get("sample_content"), rec.get("category"),
                    1 if rec.get("is_multi_intent") else 0, _j(rec.get("candidate_tools")),
                    _j(rec.get("available_tools")), rec.get("llm_suggested_tool"),
                    rec.get("top1_tool"), rec.get("top1_conf"), rec.get("top2_conf"), rec.get("margin"),
                    now, now, now, now,
                ),
            )
        else:
            # canonical 锁死:仅累加 freq / 刷新 last_seen;不覆盖已锁定的 top/margin/真值
            conn.execute(
                "UPDATE seen_queries SET freq=freq+1, last_seen=?, last_freq_at=?, updated_at=? "
                "WHERE canonical_tid=?",
                (now, now, now, canonical_tid),
            )
        conn.commit()
        return canonical_tid
    finally:
        conn.close()


# ── 自动银标闸门(§5.3)──
def gate_auto_label(canonical_tid: str) -> str:
    """对高 margin 样本自动银标;抽部分进 spotcheck。返回 'auto' / 'spotcheck' / 'manual'。"""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT gold_tool, margin, top1_tool FROM seen_queries WHERE canonical_tid=?",
            (canonical_tid,),
        ).fetchone()
        if row is None or row["gold_tool"] is not None:
            return "manual"
        margin = row["margin"]
        top1_tool = row["top1_tool"]
        if top1_tool is None or margin is None or margin < config.AUTO_PASS_MARGIN:
            return "manual"
        # 稳定抽样:决定该样本是否进入 spotcheck(仍进人工队列)
        h = int(hashlib.sha256(canonical_tid.encode()).hexdigest(), 16) % 100
        is_spotcheck = h < int(config.AUTO_SPOTCHECK_RATE * 100)
        now = _now()
        if is_spotcheck:
            conn.execute(
                "UPDATE seen_queries SET label_source='spotcheck', review_status='spotcheck', "
                "auto_passed=0, updated_at=? WHERE canonical_tid=?",
                (now, canonical_tid),
            )
            conn.commit()
            return "spotcheck"
        conn.execute(
            "UPDATE seen_queries SET gold_tool=?, label_source='auto', review_status='auto', "
            "auto_passed=1, exported=0, updated_at=? WHERE canonical_tid=?",
            (json.dumps([top1_tool], ensure_ascii=False), now, canonical_tid),
        )
        conn.commit()
        return "auto"
    finally:
        conn.close()


# ── 人工队列(§2.1 / §5.3)──
def list_candidates(category: Optional[str], page: int, page_size: int) -> list:
    conn = get_conn()
    try:
        sql = (
            "SELECT * FROM seen_queries "
            "WHERE gold_tool IS NULL AND auto_passed=0"
        )
        params = []
        if category:
            sql += " AND category=?"
            params.append(category)
        sql += " ORDER BY margin ASC, freq DESC LIMIT ? OFFSET ?"
        params.extend([page_size, max(0, page) * page_size])
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_candidate(r) for r in rows]
    finally:
        conn.close()


def _row_to_candidate(r) -> dict:
    def _l(v):
        try:
            return json.loads(v) if v else []
        except Exception:
            return []

    return {
        "canonical_tid": r["canonical_tid"],
        "trace_id": r["trace_id"],
        "conversation_id": r["conversation_id"],
        "query_text": r["query_text"],
        "category": r["category"],
        "is_multi_intent": bool(r["is_multi_intent"]),
        "candidate_tools": _l(r["candidate_tools"]),
        "available_tools": _l(r["available_tools"]),
        "llm_suggested_tool": r["llm_suggested_tool"],
        "top1_tool": r["top1_tool"],
        "margin": r["margin"],
        "freq": r["freq"],
        "label_source": r["label_source"],
    }


def get_candidate(canonical_tid: str) -> Optional[dict]:
    conn = get_conn()
    try:
        r = conn.execute(
            "SELECT * FROM seen_queries WHERE canonical_tid=?", (canonical_tid,)
        ).fetchone()
        return _row_to_candidate(r) if r else None
    finally:
        conn.close()


# ── 标注提交(§2.1)──
def _gold_list_from_req(req) -> list:
    """合并单意图 / 多意图正例为去重列表(顺序保持)。"""
    out = []
    if req.correct_tools:
        out.extend([t for t in req.correct_tools if t])
    if req.correct_tool:
        out.append(req.correct_tool)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def submit_gold(req) -> dict:
    conn = get_conn()
    try:
        r = conn.execute(
            "SELECT gold_tool, candidate_tools FROM seen_queries WHERE canonical_tid=?",
            (req.canonical_tid,),
        ).fetchone()
        if r is None:
            return {"ok": False, "error": "canonical_tid not found"}
        old = r["gold_tool"]
        cands = json.loads(r["candidate_tools"]) if r["candidate_tools"] else []
        gold_list = _gold_list_from_req(req)
        # 候选非空时,正例必须落在候选集(允许自由文本则 cands 为空时放行)
        if cands and gold_list and any(t not in cands for t in gold_list):
            bad = [t for t in gold_list if t not in cands]
            return {"ok": False, "error": "correct_tool not in candidate_tools",
                    "candidates": cands, "invalid": bad}
        gold_json = json.dumps(gold_list, ensure_ascii=False) if gold_list else None
        now = _now()
        # 修订(Relabel):值变化 → 置 exported=0 + 写 gold_history(§5.2)
        changed = (old or "") != (gold_json or "")
        exported_val = 0 if changed else (conn.execute(
            "SELECT exported FROM seen_queries WHERE canonical_tid=?",
            (req.canonical_tid,),
        ).fetchone()["exported"])
        conn.execute(
            """UPDATE seen_queries
               SET gold_tool=?, rejected_tools=?,
                   review_status=?, label_source=?,
                   exported=?, updated_at=?
               WHERE canonical_tid=?""",
            (
                gold_json,
                json.dumps(req.rejected_tools or [], ensure_ascii=False),
                "correct" if gold_list else "rejected",
                req.label_source or "full",
                exported_val,
                now,
                req.canonical_tid,
            ),
        )
        if changed:
            conn.execute(
                """INSERT INTO gold_history(canonical_tid, query_hash, old_gold, new_gold, changed_by, reason, created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    req.canonical_tid,
                    (conn.execute("SELECT query_hash FROM seen_queries WHERE canonical_tid=?",
                                  (req.canonical_tid,)).fetchone()["query_hash"]),
                    old, gold_json, req.changed_by or "annotator", req.reason, now,
                ),
            )
        conn.commit()
        return {"ok": True, "relabeled": changed, "auto_passed": False, "gold_tools": gold_list}
    finally:
        conn.close()


def submit_correction(req) -> dict:
    """来自 chat 纠正按钮(§2.5)。按 conversation_id / trace_id 定位;无 canonical 行则按 query 建。"""
    conn = get_conn()
    try:
        canonical_tid = req.trace_id or req.conversation_id
        if not canonical_tid:
            return {"ok": False, "error": "need trace_id or conversation_id"}
        r = conn.execute(
            "SELECT 1 FROM seen_queries WHERE canonical_tid=?", (canonical_tid,)
        ).fetchone()
        now = _now()
        gold_json = json.dumps([req.correct_tool], ensure_ascii=False) if req.correct_tool else None
        if r is None:
            conn.execute(
                """INSERT INTO seen_queries(canonical_tid, trace_id, conversation_id, category,
                   llm_suggested_tool, gold_tool, rejected_tools, review_status, label_source,
                   first_seen, last_seen, last_freq_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    canonical_tid, req.trace_id, req.conversation_id, "user_correct",
                    req.original_tool, gold_json,
                    json.dumps(req.rejected_tools or [], ensure_ascii=False),
                    "correct" if req.correct_tool else "rejected",
                    "full", now, now, now, now,
                ),
            )
        else:
            old = conn.execute(
                "SELECT gold_tool FROM seen_queries WHERE canonical_tid=?", (canonical_tid,)
            ).fetchone()["gold_tool"]
            conn.execute(
                """UPDATE seen_queries SET gold_tool=?, rejected_tools=?, review_status=?,
                   label_source='full', exported=?, updated_at=? WHERE canonical_tid=?""",
                (
                    gold_json,
                    json.dumps(req.rejected_tools or [], ensure_ascii=False),
                    "correct" if req.correct_tool else "rejected",
                    0 if (old or "") != (gold_json or "") else 1,
                    now, canonical_tid,
                ),
            )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def submit_feedback(req) -> dict:
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO feedback_signals(trace_id, conversation_id, signal, reason, context, created_at)
               VALUES(?,?,?,?,?,?)""",
            (
                req.trace_id, req.conversation_id, req.signal, req.reason,
                json.dumps(req.context or {}, ensure_ascii=False), _now(),
            ),
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


# ── 统计(§5.3 校准 / §4)──
def get_stats() -> dict:
    conn = get_conn()
    try:
        def _c(sql, p=()):
            return conn.execute(sql, p).fetchone()[0]

        total = _c("SELECT COUNT(*) FROM seen_queries")
        unlabeled = _c(
            "SELECT COUNT(*) FROM seen_queries WHERE gold_tool IS NULL AND auto_passed=0"
        )
        auto_labeled = _c("SELECT COUNT(*) FROM seen_queries WHERE label_source='auto'")
        manual_labeled = _c(
            "SELECT COUNT(*) FROM seen_queries WHERE label_source IN ('full','confirm')"
        )
        spotcheck = _c("SELECT COUNT(*) FROM seen_queries WHERE label_source='spotcheck'")
        feedback = _c("SELECT COUNT(*) FROM feedback_signals")
        relabel = _c("SELECT COUNT(*) FROM gold_history")
        return {
            "total": total,
            "unlabeled_in_queue": unlabeled,
            "auto_labeled": auto_labeled,
            "manual_labeled": manual_labeled,
            "spotcheck": spotcheck,
            "feedback_signals": feedback,
            "relabel_count": relabel,
        }
    finally:
        conn.close()

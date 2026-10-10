"""标注 Web 薄层 FastAPI 服务(§2.1)。

端点:
- POST /sync                增量同步 Langfuse trace → SQLite + 自动银标闸门(§5.1/§5.3)
- GET  /candidates          人工队列(WHERE gold_tool IS NULL AND auto_passed=0)
- GET  /candidate/{tid}     单条详情(+可选预填 llm_suggested_tool)
- GET  /tools               工具清单(供前端下拉)
- POST /prelabel            手动触发预标注(可选,默认前端自动)
- POST /submit/gold         提交标注(含 Relabel 修订,§5.2)
- POST /submit/correction  纠正回灌(§2.5)
- POST /submit/feedback     点赞/点踩(§2.5)
- GET  /stats               统计(§4/§5.3 校准)
"""
import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from . import config, index_store, llm_prelabel, langfuse_client
from .schemas import (
    CandidateRow,
    StatsResponse,
    SubmitCorrectionRequest,
    SubmitFeedbackRequest,
    SubmitGoldRequest,
    SyncResponse,
)

app = FastAPI(title="tool-select-annotator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_OVERLAP = timedelta(minutes=5)  # 增量重叠窗口抗晚到(§5.1)


@app.on_event("startup")
def _startup():
    index_store.init_db()


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


@app.post("/sync", response_model=SyncResponse)
def sync():
    """增量拉取新 trace → upsert → 自动银标闸门;更新水位线。"""
    try:
        wm = index_store.get_watermark()
        from_ts = _parse_ts(wm)
        if from_ts is not None:
            from_ts = from_ts - _OVERLAP  # 重叠窗口
        pulled = 0
        auto_labeled = 0
        spotcheck = 0
        latest_ts: Optional[str] = wm
        while True:
            traces = langfuse_client.pull_traces(from_ts, limit=100)
            if not traces:
                break
            for rec in traces:
                canonical_tid = index_store.upsert_trace(rec)
                res = index_store.gate_auto_label(canonical_tid)
                if res == "auto":
                    auto_labeled += 1
                elif res == "spotcheck":
                    spotcheck += 1
                pulled += 1
                if rec.get("ts") and (latest_ts is None or rec["ts"] > latest_ts):
                    latest_ts = rec["ts"]
            if len(traces) < 100:
                break
            # 翻页:以最新 ts 继续(升序)
            from_ts = _parse_ts(latest_ts)
        if latest_ts is not None:
            index_store.set_watermark(latest_ts)
        return SyncResponse(
            pulled=pulled, auto_labeled=auto_labeled, spotcheck=spotcheck, watermark=latest_ts
        )
    except Exception as e:  # noqa: BLE001
        return SyncResponse(pulled=0, auto_labeled=0, spotcheck=0, error=str(e)[:300])


@app.get("/candidates", response_model=List[CandidateRow])
def candidates(
    category: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(config.PAGE_SIZE, ge=1, le=200),
):
    rows = index_store.list_candidates(category, page, page_size)
    # 可选预填(§2.4):轻量,仅在 PRELABEL_URL 配置时
    for r in rows:
        if config.PRELABEL_URL and r.get("llm_suggested_tool") is None:
            sug = llm_prelabel.prelabel(r["query_text"], r["candidate_tools"], r["available_tools"])
            r["llm_suggested_tool"] = sug
    return [CandidateRow(**r) for r in rows]


@app.get("/candidate/{canonical_tid}", response_model=CandidateRow)
def candidate_detail(canonical_tid: str):
    r = index_store.get_candidate(canonical_tid)
    if r is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="not found")
    if config.PRELABEL_URL and r.get("llm_suggested_tool") is None:
        r["llm_suggested_tool"] = llm_prelabel.prelabel(
            r["query_text"], r["candidate_tools"], r["available_tools"]
        )
    return CandidateRow(**r)


_TOOLS_CACHE = {"data": None, "ts": 0.0}


def _bundled_catalog() -> list:
    """本地内置工具清单兜底(来自 shop-agent tool_registry docstring,§2.1)。
    当 SHOP_AGENT_TOOLS_URL 未配置或不可达时使用,保证标注台始终有候选与说明。
    shop-agent 重建并暴露 /agent/tools 后,实时清单优先。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_catalog.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/tools")
def tools():
    """工具清单(名称+说明):优先从 shop-agent /agent/tools 拉取(§2.1),缓存 5 分钟;
    失败/未配置则回退内置 tools_catalog.json。供标注台展示候选工具描述,
    trace 无 available_tools 时作为兜底候选集。
    """
    import time as _t

    if config.TOOLS_URL:
        now = _t.time()
        if _TOOLS_CACHE["data"] is not None and now - _TOOLS_CACHE["ts"] < 300:
            return {"tools": _TOOLS_CACHE["data"]}
        try:
            resp = requests.get(config.TOOLS_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("tools", [])
            if data:
                _TOOLS_CACHE.update(data=data, ts=now)
                return {"tools": data}
        except Exception:
            pass
    # 回退:缓存或内置
    if _TOOLS_CACHE["data"] is not None:
        return {"tools": _TOOLS_CACHE["data"]}
    return {"tools": _bundled_catalog()}


@app.post("/submit/gold")
def submit_gold(req: SubmitGoldRequest):
    return index_store.submit_gold(req)


@app.post("/submit/correction")
def submit_correction(req: SubmitCorrectionRequest):
    return index_store.submit_correction(req)


@app.post("/submit/feedback")
def submit_feedback(req: SubmitFeedbackRequest):
    return index_store.submit_feedback(req)


@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse(**index_store.get_stats())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT, reload=False)

"""Langfuse 只读客户端:拉取 tool_select_review trace 作为标注原料(§2/§3)。

Web 薄层只从 Langfuse 读,不回写 score(Option A)。
"""
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from langfuse import Langfuse

from . import config

logger = logging.getLogger(__name__)


def get_client() -> Langfuse:
    return Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )


def _parse_query_text(sample_content: Optional[str]) -> str:
    if not sample_content:
        return ""
    for line in str(sample_content).splitlines():
        if line.startswith("【用户】"):
            return line[len("【用户】"):].strip()
    return sample_content.strip()


def _margin_from_top_tools(top_tools) -> Optional[float]:
    if not top_tools or len(top_tools) < 2:
        return None
    try:
        c1 = float(top_tools[0].get("confidence") or 0)
        c2 = float(top_tools[1].get("confidence") or 0)
        return round(c1 - c2, 6)
    except Exception:
        return None


# 服务端 metadata 过滤(字符串化 JSON):仅拉取含真实候选排序的 trace,
# 跳过 error / 单动作退化的捕获,海量数据下大幅缩减 ClickHouse 扫描量。
# 用 numberObject(整数 1)匹配 capture 端写入的 has_top_tools,因 metadata 嵌套键
# 仅支持 stringObject/numberObject/categoryOptions 三种带 key 的过滤类型。
_TRACE_FILTER = json.dumps(
    [{"type": "numberObject", "column": "metadata", "key": "has_top_tools", "operator": "=", "value": 1}]
)


def pull_traces(from_timestamp: Optional[datetime], limit: int = 100) -> List[dict]:
    """拉取 from_timestamp(含重叠窗口,见 §5.1)之后的 trace,按时间升序。

    仅取 ``tool_select_review`` 命名 + metadata ``has_top_tools=true`` 的 trace,
    直接跳过无标注价值的退化捕获。``filter`` 不被自托管版本识别时会 400,
    故失败时退回纯 name 过滤(当前行为),保证同步流程不中断。
    """
    client = get_client()
    params = {
        "name": config.TRACE_NAME,
        "limit": min(limit, 100),
    }
    if from_timestamp is not None:
        params["from_timestamp"] = from_timestamp
    try:
        params["filter"] = _TRACE_FILTER
        page = client.api.trace.list(order_by="timestamp.asc", **params)
    except Exception as e:
        logger.warning("langfuse trace.list(filter) 失败，退回 name 过滤", error=str(e))
        params.pop("filter", None)
        page = client.api.trace.list(order_by="timestamp.asc", **params)
    out: List[dict] = []
    for t in getattr(page, "data", []) or []:
        meta = getattr(t, "metadata", {}) or {}
        top_tools = meta.get("top_tools") or []
        margin = _margin_from_top_tools(top_tools)
        top1_conf = top_tools[0].get("confidence") if top_tools else meta.get("confidence")
        top2_conf = top_tools[1].get("confidence") if len(top_tools) > 1 else None
        rec = {
            "trace_id": getattr(t, "id", None),
            "conversation_id": getattr(t, "session_id", None),
            "ts": (getattr(t, "timestamp", None).isoformat()
                   if getattr(t, "timestamp", None) else None),
            "sample_content": meta.get("sample_content"),
            "query_text": _parse_query_text(meta.get("sample_content")),
            "category": meta.get("category"),
            "is_multi_intent": bool(meta.get("is_multi_intent")),
            "candidate_tools": meta.get("candidate_tools") or [],
            "available_tools": meta.get("available_tools") or [],
            "llm_suggested_tool": meta.get("selected"),  # 模型 selected(顶层);Phase-2 预标另算
            "top1_tool": meta.get("selected"),
            "top1_conf": top1_conf,
            "top2_conf": top2_conf,
            "margin": margin,
        }
        out.append(rec)
    return out


def now_utc() -> datetime:
    return datetime.now(timezone.utc)

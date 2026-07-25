"""网关 JSON 日志（与 shop-agent structlog JSON 对齐字段契约）。

字段契约（阶段二 2.4）：service / level / ts / msg，JSON body 供 OTel json_parser
提取为日志 attribute；LOG_FORMAT != json 时回退人类可读格式。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextvars import ContextVar

_SERVICE = os.getenv("SERVICE_NAME", "gateway")


def _get_otel_trace_ids() -> tuple:
    """尽力从 OTel context 取 (trace_id, span_id)；不可用返回 (None, None)。"""
    try:
        from gateway.otel_tracing import get_current_trace_ids
        return get_current_trace_ids()
    except Exception:
        return None, None

# 当前请求的 trace_id / span_id（由 bind_trace_middleware 从 traceparent 头解析注入）
_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id_var: ContextVar[str | None] = ContextVar("span_id", default=None)

# LogRecord 内建字段（extra 之外不重复输出）
_BUILTIN_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName", "processName",
    "process", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "service": _SERVICE,
            "level": record.levelname,
            "ts": _iso_ts(record.created),
            "msg": record.getMessage(),
        }
        if record.name:
            payload["logger"] = record.name
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # trace 关联字段（阶段二 2.7 / 4.3 方案 A）：
        # 优先取当前 OTel context（由 FastAPI/httpx instrumentation 维护，统一 W3C traceparent）
        otel_tid, otel_sid = _get_otel_trace_ids()
        tid = otel_tid or _trace_id_var.get() or getattr(record, "trace_id", None)
        sid = otel_sid or _span_id_var.get() or getattr(record, "span_id", None)
        payload["trace_id"] = tid
        payload["span_id"] = sid
        # extra={...} 注入的业务字段（如 rule_id/text_hash/stage），逐个转成 JSON 字段
        for k, v in record.__dict__.items():
            if k in _BUILTIN_FIELDS or k in payload:
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def _iso_ts(created: float) -> str:
    t = time.gmtime(created)
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S', t)}.{int(record_msecs(created)):03d}Z"


def record_msecs(created: float) -> int:
    return int((created - int(created)) * 1000)


def setup_logging(level: str = "INFO") -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if os.getenv("LOG_FORMAT", "json") == "json":
        handlers[0].setFormatter(JsonFormatter())
    else:
        handlers[0].setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=handlers)


async def trace_binding_middleware(request, call_next):
    """FastAPI 中间件：从 W3C traceparent / SW8 header 提取 trace_id 注入 contextvar。

    使本请求内全部日志携带同一 trace_id，跨服务用 W3C traceparent 传播时，
    gateway ↔ shop-agent 的日志可按 trace_id 全链关联（阶段二 2.7 / 4.3.3）。
    """
    tid, sid = _parse_trace_headers(request.headers)
    _tok_t = _trace_id_var.set(tid)
    _tok_s = _span_id_var.set(sid)
    try:
        return await call_next(request)
    finally:
        _trace_id_var.reset(_tok_t)
        _span_id_var.reset(_tok_s)


def _parse_trace_headers(headers) -> tuple[str | None, str | None]:
    """优先读 W3C traceparent（00-<tid>-<sid>-01），兜底 SkyWalking sw8/sw6。"""
    tp = headers.get("traceparent")
    if tp and "-" in tp:
        parts = tp.split("-")
        if len(parts) >= 4:
            return parts[1], parts[2]
    # SkyWalking sw8: 1-<traceid>-<segmentid>-...（traceid 前 8 位+时间戳可作稳定 id）
    for key in ("sw8", "sw6", "sw8-correlation"):
        val = headers.get(key)
        if val:
            seg = val.split("-")
            if len(seg) >= 2:
                return seg[1], seg[2] if len(seg) >= 3 else None
    return None, None

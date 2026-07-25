"""monitoring-agent JSON 日志（与 gateway/shop-agent 对齐字段契约）。

字段契约（阶段二 2.4）：service / level / ts / msg；extra 注入的业务字段
（如 sev/cause/used_llm）转为 JSON 字段。LOG_FORMAT != json 时回退人类可读。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextvars import ContextVar

_SERVICE = os.getenv("SERVICE_NAME", "monitoring-agent")

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id_var: ContextVar[str | None] = ContextVar("span_id", default=None)

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
        tid = _trace_id_var.get() or getattr(record, "trace_id", None)
        sid = _span_id_var.get() or getattr(record, "span_id", None)
        payload["trace_id"] = tid
        payload["span_id"] = sid
        for k, v in record.__dict__.items():
            if k in _BUILTIN_FIELDS or k in payload:
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def _iso_ts(created: float) -> str:
    t = time.gmtime(created)
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S', t)}.{int((created - int(created)) * 1000):03d}Z"


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
    """FastAPI 中间件：从 W3C traceparent / SW8 header 提取 trace_id 注入 contextvar。"""
    tid, sid = _parse_trace_headers(request.headers)
    _tok_t = _trace_id_var.set(tid)
    _tok_s = _span_id_var.set(sid)
    try:
        return await call_next(request)
    finally:
        _trace_id_var.reset(_tok_t)
        _span_id_var.reset(_tok_s)


def _parse_trace_headers(headers) -> tuple[str | None, str | None]:
    tp = headers.get("traceparent")
    if tp and "-" in tp:
        parts = tp.split("-")
        if len(parts) >= 4:
            return parts[1], parts[2]
    for key in ("sw8", "sw6", "sw6-a", "sw6-c"):
        val = headers.get(key)
        if val:
            seg = val.split("-")
            if len(seg) >= 2:
                return seg[1], seg[2] if len(seg) >= 3 else None
    return None, None

"""统一 OTel trace（4.3 方案 A）。

- 用 OTel SDK + OTLP gRPC exporter 直接上报到 otel-gateway（共享 Collector）。
- W3C traceparent 传播：FastAPI 入站解析 + httpx 出站注入。
- 从 OTel context 提取当前 trace_id / span_id，供 structlog 注入日志 → Loki 可按 trace_id 过滤。

与 SkyWalking（SW6，skywalking_client.py）并存策略：
- 传播统一走 W3C traceparent（本模块），SkyWalking 仅作旁路（SW_AGENT_ENABLED 可整体关闭）。
- 日志 trace_id 优先取 OTel context（见 src/shared/logger.py），保证 shop-agent → gateway 同一 trace。

注意：必须在 import 任何 opentelemetry 库之前设置 OTEL_* 环境变量（本模块顶层只做 import）。
"""

from __future__ import annotations  # noqa: E402

import os  # noqa: E402
from typing import Any, Optional  # noqa: E402

from opentelemetry import trace  # noqa: E402
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    BatchSpanProcessor,
    SpanProcessor,
)

_DEFAULT_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-gateway:4317")
_SERVICE_NAME = os.getenv("SERVICE_NAME", "shop-agent")
# 传播协议：OTel SDK 默认全局 propagator 已是 W3C traceparent（CompositePropagator 含
# traceparent/tracestate/baggage），无需显式设置；如被其他库覆盖，这里兜底恢复。
from opentelemetry import propagate  # noqa: E402

try:
    from opentelemetry.propagators.tracecontext import TraceContextTextMapPropagator  # noqa: E402

    propagate.set_global_textmap(TraceContextTextMapPropagator())
except Exception:  # noqa: B902
    # propagator-tracecontext 不可用时，依赖 SDK 默认 W3C propagator
    pass

from src.shared.redact import redact as _pii_redact  # noqa: E402

_provider: Optional[TracerProvider] = None
_initialized = False


class PiiRedactionSpanProcessor(SpanProcessor):
    """出站 PII 脱敏：在 span 被导出前，对 attributes 中的字符串值做脱敏。"""

    def __init__(self) -> None:
        self._redact = _pii_redact

    def on_start(self, span, parent_context=None) -> None:
        pass

    def on_end(self, span) -> None:
        attrs = getattr(span, "_attributes", None)
        if not attrs:
            return
        modified = False
        new_attrs: dict[str, Any] = {}
        for key, value in attrs.items():
            if isinstance(value, str):
                redacted = self._redact(value)
                if redacted != value:
                    new_attrs[key] = redacted
                    modified = True
                else:
                    new_attrs[key] = value
            elif (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, str) for v in value)
            ):
                redacted = [self._redact(v) for v in value]
                if redacted != value:
                    new_attrs[key] = redacted
                    modified = True
                else:
                    new_attrs[key] = value
            else:
                new_attrs[key] = value
        if modified:
            span._attributes = new_attrs  # type: ignore[attr-defined]


def init_otel_tracing() -> None:
    """幂等初始化全局 TracerProvider。失败静默降级（不阻断主链路）。"""
    global _provider, _initialized
    if _initialized:
        return
    _initialized = True
    try:
        resource = Resource.create(
            attributes={
                "service.name": _SERVICE_NAME,
                "service.version": os.getenv("APP_VERSION", "local"),
                "deployment.environment": os.getenv("ENVIRONMENT", "local"),
            }
        )
        provider = TracerProvider(resource=resource)
        # 出站 PII 脱敏（在 OTel 层兜底，确保所有导出数据均无明文手机号/身份证/密钥）
        provider.add_span_processor(PiiRedactionSpanProcessor())
        # 全量采样；如需按比例可设 OTEL_TRACES_SAMPLER / OTEL_TRACES_SAMPLER_ARG
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_DEFAULT_ENDPOINT))
        )
        trace.set_tracer_provider(provider)
        _provider = provider
    except Exception:  # pragma: no cover - 观测层不得影响业务
        _provider = None


def shutdown_otel_tracing() -> None:
    """关闭 provider，flush 缓冲 span。"""
    global _initialized
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:  # pragma: no cover
            pass
    _initialized = False


def get_current_trace_ids() -> tuple[Optional[str], Optional[str]]:
    """从 OTel context 提取 (trace_id, span_id)。无活动 span 返回 (None, None)。

    trace_id 以 32 位十六进制输出（OTel 标准），与日志 trace_id 字段契约一致。
    """
    try:
        span = trace.get_current_span()
        if span is None:
            return None, None
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return None, None
        tid = format(ctx.trace_id, "032x")
        sid = format(ctx.span_id, "016x")
        return tid, sid
    except Exception:
        return None, None

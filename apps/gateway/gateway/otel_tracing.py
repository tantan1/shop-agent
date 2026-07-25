"""统一 OTel trace（4.3 方案 A）。

- 用 OTel SDK + OTLP gRPC exporter 直接上报到 otel-gateway（共享 Collector）。
- W3C traceparent 传播：FastAPI 入站解析 + httpx 出站注入。
- 从 OTel context 提取当前 trace_id / span_id，供 logging_json 写入日志 → Loki 可按 trace_id 过滤。

注意：
- 必须在 import 任何 opentelemetry 库之前设置 OTEL_* 环境变量（本模块顶层只做 import）。
- 与 SkyWalking（SW6）并存：网关侧日志 trace 关联以 W3C traceparent 为准。
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_DEFAULT_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-gateway:4317")
_SERVICE_NAME = os.getenv("SERVICE_NAME", "gateway")
# 传播协议：OTel SDK 默认全局 propagator 已是 W3C traceparent（CompositePropagator 含
# traceparent/tracestate/baggage），无需显式设置；如被其他库覆盖，这里兜底恢复。
from opentelemetry import propagate  # noqa: E402
try:
    from opentelemetry.propagators.tracecontext import TraceContextTextMapPropagator
    propagate.set_global_textmap(TraceContextTextMapPropagator())
except Exception:  # noqa: B902
    # propagator-tracecontext 不可用时，依赖 SDK 默认 W3C propagator
    pass

_provider: Optional[TracerProvider] = None
_initialized = False


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
        # 全量采样；如需按比例可设 OTEL_TRACES_SAMPLER=parentbased_traceidratio / OTEL_TRACES_SAMPLER_ARG
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

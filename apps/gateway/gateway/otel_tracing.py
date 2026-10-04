"""统一 OTel trace（4.3 方案 A）。

- 用 OTel SDK + OTLP gRPC exporter 直接上报到 otel-gateway（共享 Collector）。
- W3C traceparent 传播：FastAPI 入站解析 + httpx 出站注入。
- 从 OTel context 提取当前 trace_id / span_id，供 logging_json 写入日志 → Loki 可按 trace_id 过滤。

注意：
- 必须在 import 任何 opentelemetry 库之前设置 OTEL_* 环境变量（本模块顶层只做 import）。
- 与 SkyWalking（SW6）并存：网关侧日志 trace 关联以 W3C traceparent 为准。
"""
from __future__ import annotations

import base64
import os
import uuid
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)

# 探活与监控端点不产生 trace（k8s 探活、Prometheus 抓取、上游探活都很高频，
# 落到 Langfuse 全是噪声）。均为叶子节点，丢弃不影响任何业务 span。
# 不用 OTEL_PYTHON_EXCLUDED_URLS：本服务在 lifespan 里才调 instrument_app，
# 而 uvicorn 的 lifespan 启动会先触发 Starlette 构建中间件栈，插桩晚于栈构建，
# 带 excluded_urls 的 OTel 中间件进不了服务栈，环境变量实测不生效。
_DROP_SPAN_NAMES = frozenset(
    {
        "fastapi.dependencies",
        "fastapi.serialization",
        "GET /metrics",
        "GET /health",
        "GET /v1/health",
        "GET /v1/health/liveliness",
    }
)

# ASGI 插桩为每个请求额外产出 "<METHOD> <path> http send" / "... http receive" 两个叶子
# span，实测占单次请求 span 数的 ~40%，不含任何业务信息，属纯噪声。
_DROP_SPAN_SUFFIXES = (" http send", " http receive")


class DropNoisySpanSampler(Sampler):
    """按 span 名丢弃噪声 span，其余回退 ParentBased(ALWAYS_ON)。"""

    def __init__(self, delegate: Optional[Sampler] = None) -> None:
        self._delegate = delegate if delegate is not None else ParentBased(ALWAYS_ON)

    def should_sample(
        self,
        parent_context: Optional[object],
        trace_id: int,
        name: str,
        kind: Optional[object] = None,
        attributes: Optional[object] = None,
        links: Optional[object] = None,
        trace_state: Optional[object] = None,
    ) -> SamplingResult:
        if name in _DROP_SPAN_NAMES:
            return SamplingResult(Decision.DROP)
        if name.endswith(_DROP_SPAN_SUFFIXES):
            return SamplingResult(Decision.DROP)
        return self._delegate.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return "DropNoisySpanSampler(drop=%s)" % ",".join(sorted(_DROP_SPAN_NAMES))

# Langfuse v3.22+ 仅支持 HTTP OTLP（无 gRPC/4317）。端点须含完整信号路径 /v1/traces；
# 若设置 OTEL_EXPORTER_OTLP_ENDPOINT 环境变量，OTel HTTP exporter 会把它当作 base 并再追加
# 一次 /v1/traces，故本服务不设置该 env（见 docker-compose .env 已注释），由构造参数直给完整路径。
_DEFAULT_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://langfuse-web:3000/api/public/otel/v1/traces")


def _otel_auth_headers() -> dict:
    """Langfuse OTLP 需要 Basic 认证（pk:sk）。"""
    pk = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip().strip('"').strip("'")
    sk = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip().strip('"').strip("'")
    if pk and sk:
        token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        return {"Authorization": f"Basic {token}"}
    return {}
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
        # Langfuse OTLP Basic 认证：OTel HTTP exporter 只认 OTEL_EXPORTER_OTLP_HEADERS
        # 环境变量（构造参数的 headers 在 HTTP 模式下不生效），故在此注入。
        _auth = _otel_auth_headers().get("Authorization")
        if _auth:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization={_auth}"

        resource = Resource.create(
            attributes={
                "service.name": _SERVICE_NAME,
                "service.version": os.getenv("APP_VERSION", "local"),
                "deployment.environment": os.getenv("ENVIRONMENT", "local"),
            }
        )
        provider = TracerProvider(resource=resource, sampler=DropNoisySpanSampler())
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

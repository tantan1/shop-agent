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

import base64  # noqa: E402
import os  # noqa: E402
from typing import Any, Optional  # noqa: E402

from opentelemetry import trace  # noqa: E402
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    BatchSpanProcessor,
    SpanProcessor,
)
from opentelemetry.sdk.trace.sampling import (  # noqa: E402
    ALWAYS_ON,
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)

_DEFAULT_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://langfuse-web:3000/api/public/otel/v1/traces")


def _otel_auth_headers() -> dict:
    """构造 Langfuse OTLP 的 Basic 认证头（pk:sk）。

    Langfuse v3.22+ 仅支持 HTTP OTLP，且要求 Authorization 头。
    兼容 .env 中键值的引号包裹。
    """
    pk = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip().strip('"').strip("'")
    sk = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip().strip('"').strip("'")
    if pk and sk:
        token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        return {"Authorization": f"Basic {token}"}
    return {}
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


# 框架层噪声 span：FastAPI 依赖注入求解与响应序列化。二者均为叶子节点、无子节点，
# 对 LLM 排障无价值，故按名丢弃。
# 注意：绝不可加入 "fastapi.endpoint" —— 它是所有业务 span 的父节点，丢弃会被
# ParentBased 采样级联传播，导致整棵业务子树（LLM 调用/检索/工具执行）全部丢失。
_DROP_SPAN_NAMES = frozenset(
    {
        "fastapi.dependencies",
        "fastapi.serialization",
        # 探活与监控端点（k8s 探活、monitoring-agent 轮询、Prometheus 抓取）。
        # 注意：不要用 OTEL_PYTHON_EXCLUDED_URLS 环境变量做这件事——本服务在 lifespan 里
        # 才调 FastAPIInstrumentor.instrument_app(app)，而 uvicorn 的 lifespan 启动会先触发
        # Starlette 构建中间件栈，插桩晚于栈构建，带 excluded_urls 的 OTel 中间件进不了
        # 服务栈，环境变量因此完全不生效（实测 /metrics 仍每 10s 产生一条 trace）。
        # 采样器在 span 创建处判断，与中间件无关，可靠。
        "GET /metrics",
        "GET /health",
        "GET /v1/health",
        "GET /v1/health/liveliness",
    }
)

# ASGI 插桩为每个请求额外产出 "<METHOD> <path> http send" / "... http receive" 两个 span，
# 实测占单次请求 span 数的 ~40%，且不含任何业务信息，属纯噪声。二者均为叶子节点。
_DROP_SPAN_SUFFIXES = (" http send", " http receive")


# 全局抑制开关：置 True 时丢弃所有 span（用于启动预热期）。
# 预热（embedding / FAISS 索引 / reranker）在任何请求之前执行、无请求上下文，
# 其 HTTP 客户端 span 会成为孤儿根 trace；@observe 侧由 langfuse_callback 的同名开关抑制，
# 这里覆盖 OTel span。
_SUPPRESS_ALL_SPANS = False


def set_span_suppression(flag: bool) -> None:
    """开启/关闭 span 抑制（启动预热期用）。"""
    global _SUPPRESS_ALL_SPANS
    _SUPPRESS_ALL_SPANS = bool(flag)


class DropNoisySpanSampler(Sampler):
    """按 span 名丢弃指定噪声 span，其余回退委托采样器。

    委托 ``ParentBased(ALWAYS_ON)`` 保证子 span 正常继承父决策；名称判断放在委托之前，
    使被点名的叶子 span 直接 DROP，而不会影响其兄弟或父节点。
    """

    def __init__(self, delegate: Optional[Sampler] = None) -> None:
        self._delegate = delegate if delegate is not None else ParentBased(ALWAYS_ON)

    def should_sample(
        self,
        parent_context: Optional[Any],
        trace_id: int,
        name: str,
        kind: Optional[Any] = None,
        attributes: Optional[Any] = None,
        links: Optional[Any] = None,
        trace_state: Optional[Any] = None,
    ) -> SamplingResult:
        if _SUPPRESS_ALL_SPANS:
            return SamplingResult(Decision.DROP)
        if name in _DROP_SPAN_NAMES:
            return SamplingResult(Decision.DROP)
        if name.endswith(_DROP_SPAN_SUFFIXES):
            return SamplingResult(Decision.DROP)
        return self._delegate.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return "DropNoisySpanSampler(drop=%s)" % ",".join(sorted(_DROP_SPAN_NAMES))


def init_otel_tracing() -> None:
    """幂等初始化全局 TracerProvider。失败静默降级（不阻断主链路）。"""
    global _provider, _initialized
    if _initialized:
        return
    _initialized = True
    try:
        # Langfuse OTLP 需要 Basic 认证。OTel HTTP exporter 只认
        # OTEL_EXPORTER_OTLP_HEADERS 环境变量（构造参数的 headers 在 HTTP 模式下不生效），
        # 故在此注入标准 OTEL 头变量，exporter 构造时会读取。
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
        # 出站 PII 脱敏（在 OTel 层兜底，确保所有导出数据均无明文手机号/身份证/密钥）
        provider.add_span_processor(PiiRedactionSpanProcessor())
        # 全量采样；如需按比例可设 OTEL_TRACES_SAMPLER / OTEL_TRACES_SAMPLER_ARG
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_DEFAULT_ENDPOINT, headers=_otel_auth_headers()))
        )
        trace.set_tracer_provider(provider)
        _provider = provider
    except Exception:  # pragma: no cover - 观测层不得影响业务
        _provider = None


def instrument_httpx_clients() -> None:
    """注册 httpx / httpx2 出站 HTTP 插桩（幂等，失败静默降级）。

    为什么必须显式注册 **httpx2**（实测结论）：
      openai SDK 自 3.x 起默认使用 httpx2（``SyncHttpxClientWrapper`` 继承自
      ``httpx2.Client``），而 ``HTTPXClientInstrumentor`` 只插桩 ``httpx``。
      本项目 LLM 出站（shop-agent → gateway）全部走 openai SDK，若只插桩 httpx，
      这些请求既无 client span、也不携带 traceparent —— gateway 侧 server span
      收不到上游上下文只能成为新根 span，链路被切成两条独立 trace。

    两个插桩器各自覆盖一个库（httpx / httpx2），互不重复：官方插桩包装的是
    ``send()``，若对同一库再叠加自定义补丁会产生嵌套双 span，故此处不自行打补丁。
    """
    try:
        from opentelemetry.instrumentation.httpx import (  # noqa: PLC0415
            HTTPX2ClientInstrumentor,
            HTTPXClientInstrumentor,
        )
    except Exception:  # pragma: no cover - 观测层不得影响业务
        return
    for _instrumentor in (HTTPXClientInstrumentor, HTTPX2ClientInstrumentor):
        try:
            _instrumentor().instrument()
        except Exception:  # pragma: no cover - 缺包或结构变化不影响业务
            continue


def bind_context(func, *args, **kwargs):  # type: ignore[no-untyped-def]
    """把 func 与**当前**观测上下文绑定，返回一个可直接提交给线程池的无参 callable。

    与 ``run_with_context`` 的区别（关键）：本函数在**调用线程**立即 copy 上下文并闭包捕获，
    返回的可调用对象到任何线程执行都携带该上下文。若改成在工作线程里才 copy（例如写成
    ``pool.submit(lambda: run_with_context(...))``），拷到的是工作线程的空上下文，等于没做。
    """
    import contextvars  # noqa: PLC0415

    ctx = contextvars.copy_context()
    return lambda: ctx.run(func, *args, **kwargs)


def run_with_context(func, *args, **kwargs):  # type: ignore[no-untyped-def]
    """在线程池中执行 func，但继承调用方的观测上下文。

    问题：``contextvars`` **不会跨线程自动传播**。而 OTel 与 Langfuse 的当前 span /
    trace 都存放在 contextvars 里。因此凡是把同步阻塞调用丢进 executor 的地方
    （``loop.run_in_executor``、``ThreadPoolExecutor.submit``），被调函数内的
    ``@observe`` 与 HTTP 客户端 span 都会失去父上下文，变成一条**独立的根 trace**——
    一次请求于是被拆成多条互不相干的 trace，无法串联。

    解决：在提交前 copy 当前上下文，在线程内用 ``ctx.run(...)`` 执行。

    用途示例::

        ranked = await loop.run_in_executor(
            None, lambda: run_with_context(reranker.rerank, query=q, documents=docs)
        )
    """
    import contextvars  # noqa: PLC0415

    ctx = contextvars.copy_context()
    return ctx.run(func, *args, **kwargs)


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

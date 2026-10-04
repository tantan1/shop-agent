"""LLM 流量网关（FastAPI，批次0 骨架）。

职责：作为所有 LLM 流量的唯一出口（见 docs/platform-engineering/01）。
- 本文件仅做 app 装配 + 挂载各 controller，业务逻辑在 controllers/ 与各模块。
- 三道检测点位（①入向注入 → ②治理面 → 路由 → 出向 fail 开关 → 转发）在 controllers/proxy.py 串联。
  注：fail 开关属 01 §5 降级机制，非注入防线，不占三道闸编号（①②③ 见 scope-10 §1）。
- 流式（text/event-stream）保持 StreamingResponse，逐块过治理钩子。
- 成本/计量为租户维度占位（批次3 接 Prometheus+Langfuse）。
- 限流/Guardrails/脱敏/语义缓存为钩子（批次1~2 填充）。

生产级路由/故障转移由 LiteLLM Router 接管（gateway/litellm_router.py）：model_list 路由、
负载均衡、重试、fallback 全部由 Router 完成，本实现仅落地 01 拓扑与三道治理防线骨架。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .controllers import health_router, metrics_router, proxy_router
from .litellm_router import init_llm_router
from .logging_json import setup_logging, trace_binding_middleware
from .otel_tracing import init_otel_tracing, shutdown_otel_tracing

setup_logging(level=settings.log_level)

# 生产级路由基座：启动时构建 LiteLLM Router（model_list 路由/负载均衡/重试/fallback）。
# 失败即启动失败（不静默），确保 fail-closed 不变量在进程入口即成立。
init_llm_router()

app = FastAPI(title="LLM Traffic Gateway", version="1.0.0")

# 统一 OTel trace（4.3 方案 A）：在挂载路由前初始化 provider + FastAPI instrumentation。
# 入站请求自动创建 span 并解析 W3C traceparent → 日志 trace_id 取自 OTel context。
init_otel_tracing()
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
    # httpx 出站（LLM 转发）：自动注入 W3C traceparent 并创建 client span
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    HTTPXClientInstrumentor().instrument()
except Exception:
    # 观测层失败不阻断网关
    pass

app.middleware("http")(trace_binding_middleware)

# health_router 必须在 proxy_router（catch-all /v1/{path}）之前挂载，
# 否则 /v1/health 会被代理当成 LLM 请求转发而返 503。
app.include_router(health_router)
app.include_router(proxy_router)
app.include_router(metrics_router)

# 演示页面静态托管（兼容 /demo 和 /demo/，避免 307 重定向）
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

_demo_dir = Path("gateway/static/demo")
_demo_index = (_demo_dir / "index.html").read_text(encoding="utf-8")


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
@app.get("/demo/", response_class=HTMLResponse, include_in_schema=False)
async def demo_root():
    return _demo_index


# 静态资源（js/css 等）挂载在 /demo/static/ 下
app.mount("/demo/static", StaticFiles(directory=_demo_dir), name="demo-static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

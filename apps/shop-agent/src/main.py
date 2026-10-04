import sys  # noqa: E402
import warnings  # noqa: E402

# 抑制 langgraph 内部 JsonPlusSerializer 的 allowed_objects 弃用警告
# filterwarnings 对该警告无效（langchain_core 使用自定义 deprecation 机制）
# 直接 patch warnings.warn 按消息内容拦截
_original_warn = warnings.warn


def _patched_warn(*args, **kwargs):
    # args[0] 可能直接是 Warning 实例（langchain 传入 warning_cls(message)），而非字符串
    msg = args[0]
    if isinstance(msg, Warning):
        msg = str(msg)
    if isinstance(msg, str) and "allowed_objects" in msg:
        return
    return _original_warn(*args, **kwargs)


warnings.warn = _patched_warn

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi import __version__ as fastapi_version  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402

from src.core.config import config  # noqa: E402
from src.core.probe import start_probe_server, stop_probe_server  # noqa: E402
from src.core.rate_limiter import get_rate_limiter  # noqa: E402
from src.modules.auth.routers import router as auth_router  # noqa: E402
from src.modules.chat.a2a_routers import router as a2a_router  # noqa: E402
from src.modules.chat.digital_human.digital_human_router import (  # noqa: E402
    router as digital_human_router,  # noqa: E402
)
from src.modules.chat.routers import router as chat_router  # noqa: E402
from src.modules.chat.routers_mockapi import router as mockapi_router  # noqa: E402
from src.modules.items.routers import router as reports_router  # noqa: E402
from src.modules.monitoring.metrics import app_info  # noqa: E402
from src.modules.monitoring.langfuse_mlops_router import router as langfuse_mlops_router  # noqa: E402
from src.modules.monitoring.router import router as monitoring_router  # noqa: E402
from src.modules.monitoring.skywalking_client import (  # noqa: E402
    init_skywalking,
    shutdown_skywalking,
    skywalking_middleware,
)
from src.shared.exceptions import (  # noqa: E402
    BusinessException,
    business_exception_handler,
    general_exception_handler,
    http_exception_handler,
)
from src.shared.logger import configure_logging, logging_middleware  # noqa: E402
from src.shared.otel_tracing import (  # noqa: E402
    init_otel_tracing,
    instrument_httpx_clients,
    shutdown_otel_tracing,
)
from src.shared.responses import success_response  # noqa: E402


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """应用生命周期管理"""
    # ── -1. 独立探针服务优先启动（liveness/readiness，独立进程线程，不受业务事件循环/线程池影响）──
    probe_started = False
    try:
        if getattr(config, "PROBE_ENABLED", True):
            start_probe_server(
                host=getattr(config, "PROBE_HOST", "0.0.0.0"),
                port=getattr(config, "PROBE_PORT", 8001),
            )
            probe_started = True
    except Exception as e:
        print(f"[startup] 独立探针服务启动跳过: {e}")

    # ── 0. Agent Card 预热（必须最先执行，依赖最少，确保 /agent/card 端点立即可用）──
    try:
        from src.modules.chat.core.agent_card import warmup_agent_card  # noqa: E402

        warmup_agent_card()
    except Exception as e:
        print(f"[startup] Agent Card 预热跳过: {e}")

    # ── 0.1 预热 Redis 语义缓存：启动即创建版本化向量索引（hospital_questions_idx_{version}）──
    try:
        from src.modules.chat.core.redis_cache_service import get_redis_cache_service  # noqa: E402

        svc = get_redis_cache_service()
        if svc.is_available:
            print("[startup] Redis 语义缓存就绪，版本化向量索引已创建")
        else:
            print("[startup] Redis 不可达，语义缓存已禁用（索引将在首次使用时创建）")
    except Exception as e:
        print(f"[startup] Redis 语义缓存预热跳过: {e}")

    # 启动时执行
    configure_logging()  # 配置日志

    # 统一 OTel trace（4.3 方案 A）：先于 SkyWalking 初始化，日志 trace_id 以 OTel context 为准。
    # 必须在应用启动早期调用（FastAPI instrumentation 会解析 W3C traceparent → 日志可关联）。
    init_otel_tracing()
    # 统一脱敏：以 SDK 自带 mask 钩子初始化 Langfuse 客户端（幂等，未配置静默跳过）
    try:
        from src.modules.monitoring.langfuse_callback import init_langfuse_masking  # noqa: E402

        init_langfuse_masking()
    except Exception as e:
        print(f"[startup] Langfuse 脱敏注入跳过: {e}")
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402

        FastAPIInstrumentor.instrument_app(app)
        # 关键：uvicorn 的 lifespan 启动会先触发 Starlette 构建中间件栈，而本插桩发生在
        # lifespan 内部（晚于栈构建），若不重置，带 excluded_urls 的 OTel 中间件永远进不了
        # 服务栈 → 探活/监控端点照样产生 trace。置 None 让下次请求重建栈以纳入插桩。
        app.middleware_stack = None
        # 出站 HTTP：同时覆盖 httpx 与 httpx2。openai SDK 3.x 走 httpx2，
        # 只插桩 httpx 会导致 LLM 出站请求无 span 且不注入 traceparent（断链）。
        instrument_httpx_clients()
    except Exception as e:
        print(f"[startup] OTel instrumentation 跳过: {e}")

    # 设置应用信息指标
    app_info.info(
        {
            "version": "1.0.0",
            "app_name": "shop-agent",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
    )

    # 启动 Prometheus instrumentation（instrument 已在模块级完成）
    instrumentator.expose(app_instance, endpoint="/metrics", include_in_schema=False)
    # 初始化 SkyWalking 链路追踪
    init_skywalking()

    # 预热模型（避免首次请求等待模型加载）
    # 注意: lifespan 内事件循环已在运行，不能使用 loop.run_until_complete()
    # 预热期抑制 @observe：warmup 调用发生在任何请求之前、无请求上下文，
    # 其 span 会成为孤儿根 trace 污染 Langfuse（每次启动固定 3 条）。
    try:
        from src.modules.monitoring.langfuse_callback import (  # noqa: E402
            set_observe_suppressed,
        )
        from src.shared.otel_tracing import set_span_suppression  # noqa: E402

        set_observe_suppressed(True)  # @observe 侧
        set_span_suppression(True)  # OTel span 侧（预热期的 HTTP 客户端 span）
    except Exception as e:
        print(f"[startup] 预热期 trace 抑制开启失败（忽略）: {e}")

    try:
        from src.modules.chat.core.embedding_service import EmbeddingService  # noqa: E402

        # 直接同步加载 embedding 模型
        emb_svc = EmbeddingService.get_instance()
        embeddings = emb_svc.get_embeddings()  # 触发模型加载
        embeddings.embed_query("warmup")  # 首次推理预热
        print("[startup] Embedding 模型预热完成")

        # 预热 FAISS 意图索引（同步构建，避免首次意图识别 ~12s）
        try:
            from src.modules.chat.core.intent_recognizer import IntentRecognizer  # noqa: E402

            IntentRecognizer.warmup_sync(embedding_service=emb_svc)
            print("[startup] FAISS 意图索引预热完成")
        except Exception as e:
            print(f"[startup] FAISS 意图索引预热跳过: {e}")
    except Exception as e:
        print(f"[startup] Embedding 模型预热跳过: {e}")

    try:
        from src.modules.chat.core.reranker_service import RerankerService  # noqa: E402

        reranker = RerankerService.get_instance()
        # Reranker 推理是同步的，需放入线程池避免阻塞启动
        import concurrent.futures  # noqa: E402

        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(reranker.rerank, "warmup", ["预热文档"]).result()
        print("[startup] BGE-Reranker 模型预热完成")
    except Exception as e:
        print(f"[startup] BGE-Reranker 模型预热跳过: {e}")

    # 预热本地小模型（避免首次参数抽取等待 ~13s 模型加载）
    # 仅 transformers 后端需要；vllm/ollama 是进程外服务，无需也不应进程内加载
    try:
        from src.modules.chat.core.local_model_service import LocalModelService  # noqa: E402

        local = LocalModelService.get_instance()
        if LocalModelService._using_remote():
            print(f"[startup] 小模型走远程后端 ({LocalModelService._backend()})，跳过进程内预热")
        else:
            # 模型加载是同步阻塞的（transformers from_pretrained），放入线程池
            import concurrent.futures  # noqa: E402

            with concurrent.futures.ThreadPoolExecutor() as pool:
                loaded = pool.submit(local._ensure_loaded).result()
            if loaded:
                print("[startup] 本地小模型 (参数抽取) 预热完成")
            else:
                print("[startup] 本地小模型预热跳过（加载失败）")
    except Exception as e:
        print(f"[startup] 本地小模型预热跳过: {e}")

    # 预热 P2 线性头分类器（避免首次工具选择 ~27s 模型加载）
    try:
        from src.modules.chat.core.tool_head_classifier import ToolHeadClassifier  # noqa: E402

        head_clf = ToolHeadClassifier.get_instance()
        # 模型加载是同步阻塞的，放入线程池
        import concurrent.futures  # noqa: E402

        with concurrent.futures.ThreadPoolExecutor() as pool:
            loaded = pool.submit(head_clf.warmup).result()
        if loaded:
            print("[startup] P2 线性头分类器预热完成")
        else:
            print("[startup] P2 线性头分类器预热跳过（加载失败或 torch 不可用）")
    except Exception as e:
        print(f"[startup] P2 线性头分类器预热跳过: {e}")

    # 预热结束，恢复正常埋点
    try:
        set_observe_suppressed(False)
        set_span_suppression(False)
    except Exception:
        pass

    # ── MCP Server 挂载（如果 MCP_ENABLED=true） ──
    try:
        from src.core.config import config as _cfg  # noqa: E402

        mcp_enabled = getattr(_cfg, "MCP_ENABLED", False)
        if mcp_enabled:
            from src.modules.chat.core.mcp_server import create_mcp_server  # noqa: E402

            _mcp_port = getattr(_cfg, "FASTMCP_PORT", 3001) or 3001
            _mcp_host = getattr(_cfg, "FASTMCP_HOST", "127.0.0.1") or "127.0.0.1"
            _mcp_fastmcp = create_mcp_server(
                streamable_http_path="/",
                host=_mcp_host,
                port=_mcp_port,
            )
            _mcp_app = _mcp_fastmcp.streamable_http_app()
            app_instance.mount("/mcp", _mcp_app)
            # 手动启动 SessionManager
            _mcp_session_ctx = _mcp_fastmcp._session_manager.run()
            await _mcp_session_ctx.__aenter__()
            print(f"[startup] MCP Server 已挂载 http://{_mcp_host}:{_mcp_port}/mcp")
        else:
            print("[startup] MCP Server 已禁用（MCP_ENABLED=false）")
    except Exception as e:
        print(f"[startup] MCP Server 挂载跳过: {e}")

    yield

    # 关闭时清理
    # 先停探针服务
    if probe_started:
        try:
            stop_probe_server()
            print("[shutdown] 独立探针服务已停止")
        except Exception as e:
            print(f"[shutdown] 独立探针服务停止跳过: {e}")

    # Flush Langfuse 追踪缓冲区 — 确保所有数据在退出前发送
    try:
        from src.modules.monitoring.langfuse_callback import flush_langfuse  # noqa: E402

        flush_langfuse()
        print("[shutdown] Langfuse 追踪数据已刷新")
    except Exception as e:
        print(f"[shutdown] Langfuse flush 跳过: {e}")

    # 关闭 SkyWalking Agent
    shutdown_skywalking()

    # 关闭 OTel provider（flush 缓冲 span）
    shutdown_otel_tracing()

    # 关闭 MCP Server SessionManager
    try:
        if mcp_enabled:
            await _mcp_session_ctx.__aexit__(None, None, None)
            print("[shutdown] MCP Server SessionManager 已关闭")
    except Exception:
        pass

    # 卸载 instrumentation 以避免重复注册
    try:
        instrumentator.uninstrument(app_instance)
    except AttributeError:
        pass  # 部分版本的 prometheus_fastapi_instrumentator 不支持 uninstrument


app = FastAPI(
    title="大数据服务API",
    description="为业务方提供数据查询服务的API接口",
    version="1.0.0",
    debug=config.DEBUG_MODE,
    docs_url="/docs" if config.DEBUG_MODE else None,  # 开发环境启用文档
    redoc_url="/redoc" if config.DEBUG_MODE else None,
    lifespan=lifespan,
)

# SkyWalking 链路追踪中间件（最外层，覆盖全链路）
app.middleware("http")(skywalking_middleware)

# 添加日志中间件
app.middleware("http")(logging_middleware)

# 速率限制中间件（Redis + 内存降级，全局 30req/60s）
_rate_limiter = get_rate_limiter()
app.middleware("http")(_rate_limiter.middleware)

# 注册异常处理器
app.add_exception_handler(BusinessException, business_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# ============ Prometheus 监控集成 ============
# 创建 Instrumentator 实例并配置
instrumentator = Instrumentator(
    should_group_status_codes=True,  # 分组状态码 (2xx, 3xx, 4xx, 5xx)
    should_ignore_untemplated=True,  # 忽略未模板化的端点
    should_respect_env_var=True,  # 支持环境变量禁用 (ENV VAR: ENABLE_METRICS)
    excluded_handlers=[  # 排除的端点
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/monitoring/metrics",
        "/.well-known/agent-card.json",
        "/a2a/health",
    ],
)

# 添加默认指标（模块级 instrument，不能在 lifespan 中调用否则 middleware 报错）
instrumentator.instrument(app)

# ============ 注册路由 ============
print(f"[DEBUG] API_V1_PREFIX = {config.API_V1_PREFIX!r}")
app.include_router(auth_router, prefix=config.API_V1_PREFIX)
app.include_router(reports_router, prefix=config.API_V1_PREFIX)
app.include_router(chat_router, prefix=config.API_V1_PREFIX)
app.include_router(mockapi_router, prefix=config.API_V1_PREFIX)
app.include_router(monitoring_router, prefix=config.API_V1_PREFIX)
app.include_router(a2a_router)  # A2A 端点不带 API 前缀，直接 /a2a/*
app.include_router(digital_human_router, prefix=config.API_V1_PREFIX)
app.include_router(langfuse_mlops_router, prefix=config.API_V1_PREFIX)

# 演示页面静态托管（同源，无 CORS 问题）
app.mount("/demo", StaticFiles(directory="static/demo", html=True), name="demo")
# 标注控制台已迁移到 Langfuse UI（自研 static/mlops 已删除）


@app.get("/health", include_in_schema=False)
async def health_check():
    """健康检查接口"""
    # MCP Client 状态（无依赖注入，避免循环引用；未启用或异常时安静降级）
    mcp_health = {"enabled": bool(config.MCP_CLIENT_ENABLED), "connected": False, "sessions": 0, "tools_count": 0}
    if config.MCP_CLIENT_ENABLED:
        try:
            from src.modules.chat.core.mcp_client import mcp_manager

            if mcp_manager is not None:
                sessions = [
                    c for c in mcp_manager._servers.values() if getattr(c, "connected", False)
                ]
                mcp_health = {
                    "enabled": True,
                    "connected": bool(sessions),
                    "sessions": len(sessions),
                    "tools_count": sum(len(c.tools) for c in mcp_manager._servers.values()),
                }
                # 档 B：暴露 schema 失配总数，便于运维巡检
                try:
                    from src.modules.monitoring.metrics import mcp_schema_mismatch_total

                    total = 0
                    for sample in mcp_schema_mismatch_total.collect()[0].samples:
                        total += sample.value
                    mcp_health["schema_mismatch_total"] = int(total)
                except Exception:
                    mcp_health["schema_mismatch_total"] = 0
        except Exception:
            pass

    return success_response(
        data={
            "server_status": "running",
            "fastapi_version": fastapi_version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "debug_mode": config.DEBUG_MODE,
            "mcp": mcp_health,
        },
        message="服务运行正常",
    )


# Agent Card 已在 lifespan 中预热，直接导入使用
from src.modules.chat.core.agent_card import build_agent_card as _build_card  # noqa: E402


@app.get("/.well-known/agent-card.json", include_in_schema=False)
async def well_known_agent_card():
    """A2A 标准 Agent Card 端点（无需认证，<1ms 缓存命中）。

    符合 A2A 协议规范：外部系统通过 GET /.well-known/agent-card.json
    自动发现 Agent 的能力声明。
    """
    card = _build_card()
    return card.model_dump(by_alias=True)


# 手动暴露 /metrics（instrumentator.expose 失效时的兜底）
from fastapi.responses import Response  # noqa: E402
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: E402


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn  # noqa: E402

    uvicorn.run(app, host="127.0.0.1", port=8000)

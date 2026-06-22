import sys
import warnings

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

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi import __version__ as fastapi_version
from prometheus_fastapi_instrumentator import Instrumentator

from src.core.config import config
from src.shared.logger import configure_logging, logging_middleware
from src.shared.exceptions import (
    BusinessException,
    business_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from src.shared.responses import success_response
from src.modules.auth.routers import router as auth_router
from src.modules.items.routers import router as reports_router
from src.modules.chat.routers import router as chat_router
from src.modules.chat.routers_mockapi import router as mockapi_router
from src.modules.chat.a2a_routers import router as a2a_router
from src.modules.chat.digital_human.digital_human_router import router as digital_human_router
from src.modules.monitoring.router import router as monitoring_router
from src.modules.monitoring.metrics import app_info
from src.core.rate_limiter import get_rate_limiter
from src.modules.monitoring.skywalking_client import (
    init_skywalking, skywalking_middleware, shutdown_skywalking
)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """应用生命周期管理"""
    # ── 0. Agent Card 预热（必须最先执行，依赖最少，确保 /agent/card 端点立即可用）──
    try:
        from src.modules.chat.core.agent_card import warmup_agent_card
        warmup_agent_card()
    except Exception as e:
        print(f"[startup] Agent Card 预热跳过: {e}")

    # 启动时执行
    configure_logging()  # 配置日志
    
    # 设置应用信息指标
    app_info.info({
        'version': '1.0.0',
        'app_name': 'shop-agent',
        'python_version': f'{sys.version_info.major}.{sys.version_info.minor}'
    })
    
    # 启动 Prometheus instrumentation
    instrumentator.instrument(app_instance)

    # 初始化 SkyWalking 链路追踪
    init_skywalking()
    
    # 预热模型（避免首次请求等待模型加载）
    # 注意: lifespan 内事件循环已在运行，不能使用 loop.run_until_complete()
    try:
        from src.modules.chat.core.embedding_service import EmbeddingService
        # 直接同步加载 embedding 模型
        emb_svc = EmbeddingService.get_instance()
        embeddings = emb_svc.get_embeddings()       # 触发模型加载
        embeddings.embed_query("warmup")             # 首次推理预热
        print("[startup] Embedding 模型预热完成")

        # 预热 FAISS 意图索引（同步构建，避免首次意图识别 ~12s）
        try:
            from src.modules.chat.core.intent_recognizer import IntentRecognizer
            IntentRecognizer.warmup_sync(embedding_service=emb_svc)
            print("[startup] FAISS 意图索引预热完成")
        except Exception as e:
            print(f"[startup] FAISS 意图索引预热跳过: {e}")
    except Exception as e:
        print(f"[startup] Embedding 模型预热跳过: {e}")
    
    try:
        from src.modules.chat.core.reranker_service import RerankerService
        reranker = RerankerService.get_instance()
        # Reranker 推理是同步的，需放入线程池避免阻塞启动
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(reranker.rerank, "warmup", ["预热文档"]).result()
        print("[startup] BGE-Reranker 模型预热完成")
    except Exception as e:
        print(f"[startup] BGE-Reranker 模型预热跳过: {e}")
    
    # 预热本地小模型（避免首次参数抽取等待 ~13s 模型加载）
    try:
        from src.modules.chat.core.local_model_service import LocalModelService
        local = LocalModelService.get_instance()
        # 模型加载是同步阻塞的（transformers from_pretrained），放入线程池
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            loaded = pool.submit(local._ensure_loaded).result()
        if loaded:
            print("[startup] 本地小模型 (参数抽取) 预热完成")
        else:
            print("[startup] 本地小模型预热跳过（加载失败）")
    except Exception as e:
        print(f"[startup] 本地小模型预热跳过: {e}")

    # ── MCP Server 挂载（如果 MCP_ENABLED=true） ──
    try:
        from src.core.config import config as _cfg
        mcp_enabled = getattr(_cfg, "MCP_ENABLED", False)
        if mcp_enabled:
            from src.modules.chat.core.mcp_server import create_mcp_server
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
    # Flush Langfuse 追踪缓冲区 — 确保所有数据在退出前发送
    try:
        from src.modules.monitoring.langfuse_callback import flush_langfuse
        flush_langfuse()
        print("[shutdown] Langfuse 追踪数据已刷新")
    except Exception as e:
        print(f"[shutdown] Langfuse flush 跳过: {e}")

    # 关闭 SkyWalking Agent
    shutdown_skywalking()
    
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
    lifespan=lifespan
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
    should_respect_env_var=True,     # 支持环境变量禁用 (ENV VAR: ENABLE_METRICS)
    excluded_handlers=[             # 排除的端点
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/monitoring/metrics",
        "/.well-known/agent-card.json",
        "/a2a/health",
    ]
)

# 添加默认指标 (instrument app 已移到 lifespan 中)
# 注意：instrumentator.instrument(app) 已在 lifespan 事件中调用

# ============ 注册路由 ============
print(f"[DEBUG] API_V1_PREFIX = {config.API_V1_PREFIX!r}")
app.include_router(auth_router, prefix=config.API_V1_PREFIX)
app.include_router(reports_router, prefix=config.API_V1_PREFIX)
app.include_router(chat_router, prefix=config.API_V1_PREFIX)
app.include_router(mockapi_router, prefix=config.API_V1_PREFIX)
app.include_router(monitoring_router, prefix=config.API_V1_PREFIX)
app.include_router(a2a_router)  # A2A 端点不带 API 前缀，直接 /a2a/*
app.include_router(digital_human_router, prefix=config.API_V1_PREFIX)


@app.get("/health", include_in_schema=False)
async def health_check():
    """健康检查接口"""
    return success_response(
        data={
            "server_status": "running",
            "fastapi_version": fastapi_version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "debug_mode": config.DEBUG_MODE
        },
        message="服务运行正常"
    )


# Agent Card 已在 lifespan 中预热，直接导入使用
from src.modules.chat.core.agent_card import build_agent_card as _build_card


@app.get("/.well-known/agent-card.json", include_in_schema=False)
async def well_known_agent_card():
    """A2A 标准 Agent Card 端点（无需认证，<1ms 缓存命中）。

    符合 A2A 协议规范：外部系统通过 GET /.well-known/agent-card.json
    自动发现 Agent 的能力声明。
    """
    card = _build_card()
    return card.model_dump(by_alias=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
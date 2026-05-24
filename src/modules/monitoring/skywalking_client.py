"""
SkyWalking 分布式链路追踪客户端
==============================

为 FastAPI 应用集成 Apache SkyWalking Python Agent，
实现分布式链路追踪（APM），与现有 Prometheus + Grafana 指标监控互补。

功能介绍:
- 初始化 SkyWalking Python Agent（gRPC 上报到 OAP Server）
- FastAPI HTTP 中间件：为每个请求创建 EntrySpan，自动记录响应状态/耗时/异常
- 支持环境变量配置（SW_AGENT_NAME, SW_AGENT_COLLECTOR_BACKEND_SERVICES 等）
- 优雅降级：SkyWalking 不可用时不影响主业务
- 提供 @traced 装饰器，方便对关键业务函数自定义 Span

使用方式:
    from src.modules.monitoring.skywalking_client import (
        init_skywalking, skywalking_middleware, traced, shutdown_skywalking
    )
    init_skywalking()          # 在 lifespan 启动时调用
    app.middleware("http")(skywalking_middleware)  # 注册中间件
    shutdown_skywalking()      # 在 lifespan 关闭时调用
"""

import time
import logging
import os
from typing import Callable
from functools import wraps

from fastapi import Request, Response

logger = logging.getLogger("shop-agent.skywalking")

# ---------------------------------------------------------------------------
# 配置默认值
# ---------------------------------------------------------------------------
DEFAULT_OAP_ADDRESS = os.getenv("SW_AGENT_COLLECTOR_BACKEND_SERVICES", "skywalking-oap:11800")
DEFAULT_SERVICE_NAME = os.getenv("SW_AGENT_NAME", "shop-agent")
DEFAULT_SAMPLE_RATE = int(os.getenv("SW_AGENT_SAMPLE_RATE", "1"))  # 1 = 全量，-1 = 强制全量
DEFAULT_AGENT_ENABLED = os.getenv("SW_AGENT_ENABLED", "true").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Agent 初始化（幂等）
# ---------------------------------------------------------------------------
_agent_initialized = False


def init_skywalking(
    service_name: str = DEFAULT_SERVICE_NAME,
    oap_address: str = DEFAULT_OAP_ADDRESS,
    enabled: bool = DEFAULT_AGENT_ENABLED,
) -> bool:
    """
    初始化 SkyWalking Agent（幂等调用）。

    应在 FastAPI lifespan 启动阶段调用。

    Args:
        service_name: SkyWalking 中展示的服务名
        oap_address: OAP Server gRPC 地址（如 skywalking-oap:11800）
        enabled: 是否启用（默认从 SW_AGENT_ENABLED 环境变量读取）

    Returns:
        True 如果成功初始化（或已初始化），False 如果禁用或异常
    """
    global _agent_initialized

    if _agent_initialized:
        return True

    if not enabled:
        logger.info("SkyWalking Agent 已禁用（SW_AGENT_ENABLED=false）")
        return False

    try:
        from skywalking import agent, config
        # 注：skywalking-python 从环境变量自动读取绝大部分配置，
        # 此处仅补充程序化配置作为 fallback
        os.environ.setdefault("SW_AGENT_NAME", service_name)
        os.environ.setdefault("SW_AGENT_COLLECTOR_BACKEND_SERVICES", oap_address)
        # 日志级别：与 App 日志保持同步
        os.environ.setdefault("SW_LOGGING_LEVEL", "WARNING")

        config.init(
            agent_name=service_name,
            agent_collector_backend_services=oap_address,
        )
        agent.start()

        _agent_initialized = True
        logger.info(
            "SkyWalking Agent 已启动 | service=%s | oap=%s",
            service_name, oap_address,
        )
        return True

    except ImportError:
        logger.warning("apache-skywalking 未安装，链路追踪已跳过。如需启用请: pip install apache-skywalking")
        return False
    except Exception as exc:
        logger.error("SkyWalking Agent 初始化失败: %s，链路追踪功能不可用", exc)
        return False


def shutdown_skywalking():
    """关闭 SkyWalking Agent，确保缓冲区数据上报。"""
    global _agent_initialized
    if not _agent_initialized:
        return
    try:
        from skywalking import agent as agent_module

        # apache-skywalking 0.7.x 模块层不暴露 stop()，
        # 需要访问内部 __agent 实例
        try:
            _agent = getattr(agent_module, '__agent', None)
            if _agent is not None and hasattr(_agent, 'stop'):
                _agent.stop()
                logger.info("SkyWalking Agent 已停止")
            else:
                logger.info("SkyWalking Agent 内部实例不可用，跳过主动停止")
        except Exception:
            pass

        _agent_initialized = False
        logger.info("SkyWalking Agent 已关闭（后台守护线程将随进程退出）")
    except Exception as exc:
        logger.warning("SkyWalking Agent 关闭时出错: %s", exc)
        _agent_initialized = False


# ---------------------------------------------------------------------------
# FastAPI HTTP 中间件
# ---------------------------------------------------------------------------
async def skywalking_middleware(request: Request, call_next: Callable) -> Response:
    """
    SkyWalking 链路追踪中间件。

    为每个 HTTP 请求创建 EntrySpan，记录：
    - 请求方法 & 路径
    - 响应状态码
    - 请求耗时（ms）
    - 异常（如有）

    挂载到所有路由之前，确保全链路覆盖。
    """
    if not _agent_initialized:
        # 降级：直接放行，不影响业务
        return await call_next(request)

    try:
        from skywalking.trace.context import SpanContext, get_context
        from skywalking.trace.tags import Tag
    except ImportError:
        return await call_next(request)

    start_time = time.perf_counter()

    # ---- Entry Span ----
    try:
        from skywalking.trace.carrier import CarrierItem, Setter
        from skywalking.trace.context import get_context
        from skywalking.trace.span import EntrySpan
        from skywalking import Component

        method = request.method
        url_path = request.url.path
        peer = request.client.host if request.client else "unknown"

        # 读取上游 trace 上下文（如果有 SW6 header）
        carrier = dict(request.headers)
        span_ctx = get_context().new_entry_span(
            op=f"{method} {url_path}",
            carrier=carrier,
        )

        with span_ctx as span:
            span.layer = "Http"
            component_id = await _detect_component()
            if component_id:
                span.component = component_id
            span.tag(Tag(key="http.method", val=method))
            span.tag(Tag(key="http.url", val=str(request.url)))

            # 执行业务
            try:
                response = await call_next(request)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # 记录成功
                if 200 <= response.status_code < 400:
                    span.tag(Tag(key="status_code", val="true"))
                else:
                    span.error_occurred = True
                    span.tag(Tag(key="status_code", val="false"))
                span.tag(Tag(key="http.status_code", val=str(response.status_code)))
                span.tag(Tag(key="elapsed_ms", val=f"{elapsed_ms:.1f}"))

                return response

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.error_occurred = True
                span.tag(Tag(key="error", val="true"))
                span.tag(Tag(key="error.message", val=str(exc)))
                span.tag(Tag(key="error.type", val=type(exc).__name__))
                span.tag(Tag(key="http.status_code", val="500"))
                span.tag(Tag(key="elapsed_ms", val=f"{elapsed_ms:.1f}"))
                raise

    except Exception:
        # 任何 SkyWalking 内部异常都不应影响业务
        return await call_next(request)


async def _detect_component():
    """检测 HTTP 组件 ID（FastAPI 基于 ASGI/HTTP）。"""
    try:
        return 49  # HTTP component
    except Exception:
        return None


# ---------------------------------------------------------------------------
# @traced 装饰器：自定义函数级 Span
# ---------------------------------------------------------------------------
def traced(op: str = None, layer: str = None):
    """
    装饰器：为关键业务函数创建自定义 Span。

    用法:
        @traced(op="rag:embedding")
        async def embed_query(text: str):
            ...

        @traced(op="tool:refund_order", layer="RPCFramework")
        async def execute_refund(order_id: str):
            ...

    Args:
        op: Span 的操作名（建议用 "模块:功能" 命名风格）
        layer: Span 层级（如 Http, RPCFramework, Database 等）
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not _agent_initialized:
                return await func(*args, **kwargs)

            span_op = op or f"{func.__module__}.{func.__name__}"
            try:
                from skywalking.trace.context import get_context

                context = get_context()
                with context.new_exit_span(op=span_op, peer="local") as span:
                    if layer:
                        span.layer = layer
                    return await func(*args, **kwargs)

            except Exception:
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not _agent_initialized:
                return func(*args, **kwargs)

            span_op = op or f"{func.__module__}.{func.__name__}"
            try:
                from skywalking.trace.context import get_context

                context = get_context()
                with context.new_exit_span(op=span_op, peer="local") as span:
                    if layer:
                        span.layer = layer
                    return func(*args, **kwargs)

            except Exception:
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

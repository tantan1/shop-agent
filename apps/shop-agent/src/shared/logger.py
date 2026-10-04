import logging
import os
import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response

from src.core.config import config

_SERVICE = os.getenv("SERVICE_NAME", "shop-agent")


def _service_processor(logger, method_name, event_dict):
    """注入 service 字段（与 gateway / monitoring-agent 字段契约一致）。"""
    event_dict.setdefault("service", _SERVICE)
    return event_dict


def _trace_processor(logger, method_name, event_dict):
    """注入 trace_id / span_id（阶段二 2.7，日志↔trace 关联）。

    从 structlog contextvars 读取（由 logging_middleware 绑定）；未绑定则记入
    trace_id=None，便于 Loki 过滤异常请求。
    """
    ctx = structlog.contextvars.get_contextvars()
    event_dict.setdefault("trace_id", ctx.get("trace_id"))
    event_dict.setdefault("span_id", ctx.get("span_id"))
    return event_dict


def configure_logging():
    """配置结构化日志"""
    # 设置标准库 logging 级别（structlog 依赖它做 level 过滤）
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.DEBUG),
        format="%(message)s",
    )
    # 统一脱敏：日志任何字段出站前先经 redact（键名 + 值内 PII）。
    # 放于 JSONRenderer/ConsoleRenderer 之前，保证 Loki 收到的也是脱敏后内容。
    from src.shared.redact import redact_processor

    # 配置structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", key="ts"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _service_processor,
            _trace_processor,
            redact_processor,
            structlog.processors.JSONRenderer(ensure_ascii=False)
            if config.LOG_FORMAT == "json"
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None):
    """获取日志记录器"""
    return structlog.get_logger(name)


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """请求日志中间件"""
    start_time = time.time()

    # 绑定 trace_id（阶段二 2.7）：从 SkyWalking context 提取当前 trace id，
    # 使本请求的全部日志带上同一 trace_id，Loki 可 {trace_id="..."} 全链关联。
    trace_id, span_id = _extract_current_trace()
    ctx = structlog.contextvars.get_contextvars()
    if trace_id:
        ctx.setdefault("trace_id", trace_id)
    if span_id:
        ctx.setdefault("span_id", span_id)

    # 获取请求信息
    logger = get_logger("api")
    request_id = uuid.uuid4().hex  # 跨进程可关联的请求ID
    # 将 request_id 注入 structlog context，使本请求全链日志均携带（C 维度修复）
    structlog.contextvars.bind_contextvars(request_id=request_id)

    try:
        # 处理请求
        response = await call_next(request)

        # 计算处理时间
        process_time = time.time() - start_time

        # 记录请求完成
        logger.info(
            "请求完成",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=f"{process_time:.4f}s",
        )

        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-Id"] = request_id  # 回传请求ID，便于客户端关联

        return response

    except Exception as e:
        # 计算处理时间
        process_time = time.time() - start_time

        # 记录请求异常
        logger.error(
            "请求异常",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            exception=e.__class__.__name__,
            error_detail=str(e),
            process_time=f"{process_time:.4f}s",
        )

        # 重新抛出异常，让异常处理器处理
        raise
    finally:
        # 请求结束清理 contextvars，避免污染下一个请求
        structlog.contextvars.clear_contextvars()


def _extract_current_trace():
    """从当前 trace 上下文提取 (trace_id, span_id)（尽力而为）。

    4.3 方案 A：优先取 OTel context（W3C traceparent 统一传播）；无则回退 SkyWalking
    context（SW6）。返回 (trace_id, span_id)；均不可用时返回 (None, None)。
    """
    # OTel（优先）：FastAPI/httpx instrumentation 维护，跨服务一致
    try:
        from src.shared.otel_tracing import get_current_trace_ids

        tid, sid = get_current_trace_ids()
        if tid:
            return tid, sid
    except Exception:
        pass
    # SkyWalking（旁路回退）
    try:
        from skywalking.trace.context import get_context

        ctx = get_context()
        segment = getattr(ctx, "segment", None)
        if segment is None:
            return None, None
        seg_id = getattr(segment, "segment_id", None)
        trace_id = None
        related = getattr(segment, "related_traces", None) or []
        if related:
            first = related[0]
            trace_id = getattr(first, "value", None) or str(first)
        return (trace_id or seg_id, seg_id)
    except Exception:
        return None, None


class APILogger:
    """API业务日志记录器"""

    def __init__(self, name: str = "business"):
        self.logger = get_logger(name)

    @staticmethod
    def _fmt(msg, args):
        """兼容 printf 风格位置参数：logger.info("x=%s", v) -> "x=v" """
        if args:
            try:
                return msg % args
            except Exception:
                return msg
        return msg

    def info(self, msg: str, *args, **kwargs):
        """记录信息日志（支持 printf 风格位置参数）"""
        self.logger.info(self._fmt(msg, args), **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """记录错误日志（支持 printf 风格位置参数）"""
        self.logger.error(self._fmt(msg, args), **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """记录警告日志（支持 printf 风格位置参数）"""
        self.logger.warning(self._fmt(msg, args), **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """记录调试日志（支持 printf 风格位置参数）"""
        self.logger.debug(self._fmt(msg, args), **kwargs)

    def log_api_call(self, api_key: str, endpoint: str, success: bool, **kwargs):
        """记录API调用"""
        self.logger.info(
            "API调用",
            api_key=api_key[:8] + "****" if api_key else None,  # 脱敏处理
            endpoint=endpoint,
            success=success,
            **kwargs,
        )

    def log_database_operation(self, operation: str, table: str, success: bool, **kwargs):
        """记录数据库操作"""
        self.logger.info("数据库操作", operation=operation, table=table, success=success, **kwargs)

    def log_business_event(self, event: str, **kwargs):
        """记录业务事件"""
        self.logger.info("业务事件", business_event=event, **kwargs)

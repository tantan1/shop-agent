"""
Langfuse 追踪服务 — 遵循 Langfuse v4.x 官方最佳实践

- 使用 langfuse.langchain.CallbackHandler（框架集成，自动捕获 model/tokens/span 层级）
- 使用 propagate_attributes() 上下文管理器设置 trace-level 属性（session_id/user_id/tags/trace_name）
  （v4.x 中这些属性不再通过 CallbackHandler 构造参数传入，而是通过 OTel 上下文传播）
- 返回 (handler, ctx_manager) 元组，调用方负责管理 ctx 生命周期
- 在应用 shutdown 时调用 flush() 确保数据不丢失
"""

import os  # noqa: E402
from typing import List, Optional, Tuple  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# 🔴 关键：在 import Langfuse 之前确保 .env 已加载
load_dotenv()

from langfuse import Langfuse, propagate_attributes  # noqa: E402
from langfuse.langchain import CallbackHandler  # noqa: E402

from src.shared.logger import APILogger  # noqa: E402
from src.shared.redact import mask_for_langfuse  # noqa: E402

_logger = APILogger("langfuse_callback")

_NAME = "_initialized_for_mask"  # 标记：本回调是否已注入统一脱敏（只注入一次）

_LANGFUSE_OBSERVE_AVAILABLE = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)


def init_langfuse_masking() -> None:
    """以统一脱敏钩子初始化 Langfuse 客户端（幂等）。

    Langfuse v4 SDK 自带 ``mask`` 过滤钩子（MaskFunction），在 SDK 属性创建时
    就地脱敏 input / output / metadata，敏感数据不出应用边界。本函数在应用
    启动早期调用一次：先以 mask 构造默认客户端，后续 get_client() / CallbackHandler
    会复用该单例（见 langfuse._client.get_client）。

    未配置 LANGFUSE_PUBLIC_KEY / SECRET_KEY 时静默跳过。
    """
    if getattr(_logger, _NAME, False):
        return
    setattr(_logger, _NAME, True)
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return
    try:
        Langfuse(mask=mask_for_langfuse)
        _logger.info("Langfuse 统一脱敏钩子已注入")
    except Exception as e:
        _logger.warning(f"Langfuse mask 注入失败（非致命）: {str(e)}")


def create_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    trace_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[Tuple[CallbackHandler, object]]:
    """
    为单个请求创建 Langfuse CallbackHandler + propagate_attributes 上下文管理器。

    v4.x 变更说明：
      - CallbackHandler.__init__() 只接受 public_key 和 trace_context
      - session_id / user_id / tags / trace_name / metadata 需通过 propagate_attributes() 设置
      - 返回 (handler, ctx) 元组，调用方必须在使用前 ctx.__enter__()，结束后 ctx.__exit__()

    用法:
      result = create_langfuse_handler(session_id="...", tags=["ecommerce"])
      if result:
          handler, ctx = result
          ctx.__enter__()
          try:
              # ... agent 执行期间 handler 产生的 span 自动继承这些属性
          finally:
              ctx.__exit__(None, None, None)

    如果未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，返回 None 静默禁用。
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    try:
        # v4.x: 显式传入 public_key，避免 get_client() 在多客户端场景下返回 disabled client
        handler = CallbackHandler(public_key=public_key)

        # 创建 propagate_attributes 上下文管理器（设置 user_id/session_id/tags/trace_name/metadata）
        ctx = propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            trace_name=trace_name,
            metadata=metadata,
        )

        _logger.info(
            "Langfuse trace handler 已创建",
            trace_name=trace_name or "(auto)",
            session_id=session_id,
            tags=tags,
        )
        return handler, ctx
    except Exception as e:
        _logger.error(f"Langfuse CallbackHandler 创建失败: {str(e)}")
        return None


def flush_langfuse() -> None:
    """
    刷新 Langfuse 客户端缓冲区，确保所有追踪数据已发送。

    在以下场景必须调用：
    - FastAPI lifespan shutdown（应用退出）
    - 短生命周期脚本（CLI、notebook）
    - Serverless 函数退出前

    在长生命周期服务（如 FastAPI）中，Langfuse 后台线程会定期自动发送，
    但 shutdown 时 flush 能防止数据截断丢失。
    """
    try:
        # CallbackHandler 底层使用全局 Langfuse 客户端
        # flush() 是类方法，不依赖具体实例；get_client() 复用已注入脱敏的单例
        from langfuse import get_client  # noqa: E402

        client = get_client()  # 从环境变量自动获取凭证（复用 mask 单例）
        client.flush()
        _logger.info("Langfuse 数据已刷新")
    except Exception as e:
        _logger.warning(f"Langfuse flush 失败（非致命）: {str(e)}")


if _LANGFUSE_OBSERVE_AVAILABLE:
    from langfuse import observe as _langfuse_observe  # noqa: E402
else:
    _langfuse_observe = None


def observe(name: Optional[str] = None):
    """条件性 Langfuse @observe 装饰器。

    未配置 LANGFUSE_PUBLIC_KEY / SECRET_KEY 时降级为 no-op，避免初始化报错。
    """
    if not _LANGFUSE_OBSERVE_AVAILABLE or _langfuse_observe is None:

        def noop_decorator(func):
            return func

        return noop_decorator
    return _langfuse_observe(name=name)

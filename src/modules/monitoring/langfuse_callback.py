"""
Langfuse 追踪服务 — 遵循 Langfuse v4.x 官方最佳实践

- 使用 langfuse.langchain.CallbackHandler（框架集成，自动捕获 model/tokens/span 层级）
- 使用 propagate_attributes() 上下文管理器设置 trace-level 属性（session_id/user_id/tags/trace_name）
  （v4.x 中这些属性不再通过 CallbackHandler 构造参数传入，而是通过 OTel 上下文传播）
- 返回 (handler, ctx_manager) 元组，调用方负责管理 ctx 生命周期
- 在应用 shutdown 时调用 flush() 确保数据不丢失
"""
import os
from typing import Optional, List, Tuple

from dotenv import load_dotenv

# 🔴 关键：在 import Langfuse 之前确保 .env 已加载
load_dotenv()

from langfuse import propagate_attributes
from langfuse.langchain import CallbackHandler

from src.shared.logger import APILogger

_logger = APILogger("langfuse_callback")


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
        # v4.x: 构造参数只需 public_key，trace-level 属性通过 propagate_attributes 传播
        handler = CallbackHandler()

        # 创建 propagate_attributes 上下文管理器（设置 user_id/session_id/tags/trace_name/metadata）
        ctx = propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            trace_name=trace_name,
            metadata=metadata,
        )

        _logger.info("Langfuse trace handler 已创建",
                     trace_name=trace_name or "(auto)",
                     session_id=session_id,
                     tags=tags)
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
        # flush() 是类方法，不依赖具体实例
        from langfuse import Langfuse
        client = Langfuse()  # 从环境变量自动获取凭证
        client.flush()
        _logger.info("Langfuse 数据已刷新")
    except Exception as e:
        _logger.warning(f"Langfuse flush 失败（非致命）: {str(e)}")

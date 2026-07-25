"""LLM 调用 trace 钩子（scope-04 Langfuse 预留）。

设计纪律（architecture.md「防 SPOF」）：
- **默认关闭**：仅当 env LANGFUSE_ENABLED=true 时启用推送；否则全程 no-op。
- **失败降级**：推送异常静默吞掉，绝不因 trace 失败阻断网关主链路。
- 不引入运行时强依赖：langfuse 包为可选 import，缺失即 no-op。

本批仅留钩子与接线点，proxy 在「模型调用成功/失败」处调用 trace_llm，
实际 trace 数据接入留后续批次（依赖真实 Langfuse 部署）。
"""
from __future__ import annotations

import os

_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() in ("1", "true", "yes")

_client = None


def enabled() -> bool:
    return _ENABLED


def init_tracing() -> None:
    """可选初始化 Langfuse client；失败则降级为 no-op。"""
    global _client
    if not _ENABLED:
        return
    try:
        from langfuse import Langfuse  # 可选依赖

        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        # 任何初始化失败都不应阻断网关
        _client = None


def trace_llm(tenant: str, model: str, est_tokens: int, ok: bool) -> None:
    """记录一次 LLM 调用（成功/失败）。异常静默降级。"""
    if not _ENABLED or _client is None:
        return
    try:
        _client.trace(
            name="gateway_llm_call",
            metadata={
                "tenant": tenant,
                "model": model,
                "est_tokens": est_tokens,
                "ok": ok,
            },
        )
    except Exception:
        pass

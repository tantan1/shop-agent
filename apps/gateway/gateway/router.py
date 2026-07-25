"""路由决策（01 §2 拓扑 / §4 引擎可替换）。

路由键 = model 字段（含前缀匹配）。实际的上游选择、负载均衡、重试、fallback 已交由
LiteLLM Router 接管（gateway/litellm_router.py）；本模块只做「网关侧决策 + 模型名改写」：
  - 本地小模型任务键（tool_select/param/local/*/models/*/qwen3-unified）→ 改写 upstream_model 为 qwen3-unified
  - 云端键（gpt-*/claude*/qwen*/mock*） → 原样透传 model 名给 Router
  - fallback_chain 不再由网关手写，改为占位 ["__litellm_router__"]，声明「由 Router 内部接管」

映射表与 litellm_router 的 model_list（YAML 或 legacy）保持一致，二者同源。
"""
from __future__ import annotations

from .config import settings
from .types import RouteDecision

# 本地统一模型 served 名（vLLM 提供），所有本地小模型任务以该名发往 vLLM
_VLLM_MODEL = "qwen3-unified"

# 需改写 upstream_model 的本地任务键
_LOCAL_ALIASES = {"tool_select", "param", "qwen3-unified"}


def _is_local_alias(model: str) -> bool:
    m = model.lower()
    return (
        m in _LOCAL_ALIASES
        or m.startswith("local/")
        or m.startswith("models/")
    )


def route(model: str | None, tenant: str = "default") -> RouteDecision:
    """网关侧路由决策：仅做模型名改写 + 声明 fallback 归属，实际调度交 LiteLLM Router。

    tenant 预留给批次1 做租户级路由/限流。
    返回：
      - upstream_model：本地任务改写为 qwen3-unified，其余留空（保持请求原样）
      - backend：仅用于可观测标注（vllm / mock / cloud / default）
      - fallback_chain：占位 ["__litellm_router__"]，声明故障转移由 Router 内部接管（02 §4）
    """
    key = (model or "").lower()
    if _is_local_alias(key):
        backend = "vllm"
        upstream_model = _VLLM_MODEL
    elif key.startswith("mock"):
        backend = "mock"
        upstream_model = ""
    elif key.startswith(("gpt-", "claude", "qwen")):
        backend = "cloud"
        upstream_model = ""
    else:
        backend = "default"
        upstream_model = ""
    return RouteDecision(
        upstream_base_url="",  # 不再由网关持有 base_url；Router 从 model_list 解析
        model=model or "",
        backend=backend,
        fallback_allowed=True,  # 02 §4 语义：允许 Router 内部故障转移
        # 占位声明：实际 fallback 链在 LiteLLM Router 的 model_list 中定义
        fallback_chain=["__litellm_router__"],
        upstream_model=upstream_model,
    )


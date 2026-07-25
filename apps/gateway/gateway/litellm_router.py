"""LiteLLM Router 适配层（生产级路由/故障转移基座，01 §4 / 02 §4）。

职责：
  - 启动时按 settings 构建 `litellm.Router`：优先 litellm_config_path YAML，否则 legacy 自动生成；
  - 接管 model_list 路由、负载均衡、重试、fallback（替代原网关手写 fallback_chain 遍历）；
  - 暴露 `acompletion` / `astream` 封装，统一回填 `(content, actual_model, used_fallback)` 供 proxy 写证据头；
  - mock_cloud_override=1 时动态把云端 deployment 切到 mock 上游（压测零真实云调用）；
  - Router 异常包装为 GovernanceError，由 proxy 按 fail_mode 分流（保留 fail-closed 不变量）。

网关层三道防线（入向注入 / 脱敏 / 出向 judge_egress）仍在 proxy 中于 Router 前后执行，本层不触碰业务治理。
"""
from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional

import yaml

from .config import settings
from .types import GovernanceError, GovernanceStage

try:
    from litellm import Router
    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import RateLimitError as LiteLLMRateLimitError
except Exception:  # pragma: no cover - 依赖未装时给出明确提示，不静默失败
    Router = None  # type: ignore
    LiteLLMAPIError = Exception  # type: ignore
    LiteLLMRateLimitError = Exception  # type: ignore


def _load_yaml(path: str) -> dict:
    """读取 YAML，支持 ${ENV} 占位替换（仅做最浅层展开，避免引入额外依赖）。"""
    raw = (path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
    text = open(raw, "r", encoding="utf-8").read()
    # 极简 ${VAR} 展开（不递归、不报错，缺失则留空）
    text = _expand_env(text)
    return yaml.safe_load(text)


def _expand_env(text: str) -> str:
    import re

    return re.sub(r"\$\{(\w+)\}", lambda m: os.getenv(m.group(1), ""), text)


def _legacy_model_list() -> list[dict]:
    """未配 YAML 时由 config 字段生成 model_list（向后兼容，deprecated）。"""
    return settings.build_model_list()


def _apply_mock_override(model_list: list[dict]) -> list[dict]:
    """mock_cloud_override=1：云端 deployment 整体改路由 mock 上游（openai 兼容）。

    保留本地 vLLM（provider=openai 且 api_base=vllm）与 mock 自身，仅替换云端
    deployment（gpt-*/claude*/qwen*/default）的 api_base + model + provider。
    """
    out: list[dict] = []
    for dep in model_list:
        params = dict(dep.get("litellm_params", {}))
        name = dep.get("model_name", "")
        # 本地 vLLM（api_base 指向 vllm）与 mock 自身不被劫持
        is_local = params.get("api_base") == settings.vllm_base_url
        is_mock = name in ("mock", "mock*")
        if settings.mock_cloud_override and not is_local and not is_mock:
            params["api_base"] = settings.mock_base_url
            params["custom_llm_provider"] = "openai"
            params["model"] = "mock"
            params.pop("api_key", None)
        out.append({"model_name": name, "litellm_params": params})
    return out


class LLMRouter:
    """对 litellm.Router 的薄封装，负责构建与调用，并桥接治理异常。"""

    def __init__(self) -> None:
        self._router: Optional["Router"] = None
        self._model_list: list[dict] = []
        self._routing_strategy = settings.litellm_routing_strategy
        self._num_retries = settings.litellm_num_retries
        self._timeout = settings.litellm_timeout_sec

    def build(self) -> None:
        """构建底层 Router（模块加载时调用一次）。"""
        if Router is None:  # pragma: no cover
            raise RuntimeError("litellm 未安装：pip install litellm==1.59.0")
        if settings.litellm_config_path:
            cfg = _load_yaml(settings.litellm_config_path)
            model_list = cfg.get("model_list", [])
            rs = cfg.get("router_settings", {})
            strategy = rs.get("routing_strategy", self._routing_strategy)
            retries = rs.get("num_retries", self._num_retries)
            timeout = rs.get("timeout", self._timeout)
        else:
            model_list = _legacy_model_list()
            strategy, retries, timeout = self._routing_strategy, self._num_retries, self._timeout
        model_list = _apply_mock_override(model_list)
        self._model_list = model_list
        # 缓存解析后的 router_settings，供测试/可观测读取（build 的权威结果）
        self._routing_strategy = strategy
        self._num_retries = int(retries)
        self._timeout = float(timeout)
        self._router = Router(
            model_list=model_list,
            routing_strategy=strategy,
            num_retries=int(retries),
            timeout=float(timeout),
        )

    @property
    def router(self) -> "Router":
        if self._router is None:
            self.build()
        assert self._router is not None
        return self._router

    def _wrap_exc(self, exc: BaseException) -> GovernanceError:
        """把 LiteLLM 异常包装为 GovernanceError，供 proxy 按 fail_mode 分流。"""
        if isinstance(exc, LiteLLMRateLimitError):
            reason = "upstream rate limited"
        elif isinstance(exc, LiteLLMAPIError):
            reason = "upstream api error"
        else:
            reason = "router call failed"
        return GovernanceError(stage=GovernanceStage.EGRESS.value, cause=exc, reason=reason)

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        """LiteLLM 返回 ModelResponse/ModelResponse 等对象可能带 model_dump；
        FakeRouter/测试可能直接返回 dict——统一规范化为 dict。"""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return dict(obj)

    async def acompletion(
        self, model: str, messages: list[dict], **kwargs: Any
    ) -> tuple[dict, str, bool]:
        """非流式调用。返回 (响应 dict, 实际模型名, 是否经过 fallback)。

        兼容两种底层实现：
          - 真实 litellm.Router.acompletion 返回 ModelResponse（含 .model / .litellm_params）；
          - 测试 FakeRouter 直接返回 (dict, actual_model, used_fallback) 三元组。
        """
        try:
            resp = await self.router.acompletion(model=model, messages=messages, **kwargs)
        except Exception as exc:  # LiteLLM 异常统一上抛给治理层
            raise self._wrap_exc(exc) from exc
        # FakeRouter 测试协议：直接返回三元组
        if isinstance(resp, tuple) and len(resp) == 3:
            return resp
        resp_dict = self._to_dict(resp)
        actual_model = resp_dict.get("model") or model
        used_fallback = bool(
            resp_dict.get("litellm_params", {}).get("model_info", {}).get("fallback_used")
        )
        return resp_dict, actual_model, used_fallback

    async def astream(
        self, model: str, messages: list[dict], **kwargs: Any
    ) -> AsyncIterator[tuple[dict, str]]:
        """流式调用。逐块产出 (chunk dict, 实际模型名)，由 proxy 过治理链后透传。

        兼容真实 Router（async generator 产出 ModelResponse chunk）与
        FakeRouter（产出 (chunk_dict, actual_model) 二元组）。
        """
        try:
            chunks = await self.router.acompletion(
                model=model, messages=messages, stream=True, **kwargs
            )
            actual_model = model
            async for chunk in chunks:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    # FakeRouter 协议：(chunk_dict, actual_model)
                    yield chunk
                    continue
                if hasattr(chunk, "model") and chunk.model:
                    actual_model = chunk.model
                yield self._to_dict(chunk), actual_model
        except Exception as exc:
            raise self._wrap_exc(exc) from exc


# 模块级单例：gateway 启动构建一次，proxy 调用复用
llm_router = LLMRouter()


def init_llm_router() -> None:
    """应用启动时构建 Router（main.py 调用）。"""
    llm_router.build()

"""LLM 服务核心路径单测（Mock 路径 + 异常路径）。"""
from __future__ import annotations

import os
import pytest

from src.modules.chat.core.llm_service import LLMService, TokenLimitExceeded, resolve_llm_base_url
from src.modules.chat.config import chat_config


class TestResolveLLMBaseUrl:
    """网关出口解析：fail-closed 语义。"""

    def test_gateway_url_configured(self, monkeypatch):
        monkeypatch.setenv("LLM_GATEWAY_URL", "http://localhost:8001/v1")
        assert resolve_llm_base_url() == "http://localhost:8001/v1"

    def test_missing_gateway_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
        monkeypatch.delenv("ALLOW_DIRECT_LLM_EGRESS", raising=False)
        with pytest.raises(RuntimeError, match="LLM_GATEWAY_URL"):
            resolve_llm_base_url()

    def test_direct_egress_escape_valve(self, monkeypatch):
        monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
        monkeypatch.setenv("ALLOW_DIRECT_LLM_EGRESS", "true")
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        url = resolve_llm_base_url()
        assert "dashscope" in url


class TestMockLLM:
    """Mock LLM 仿真：压测 0 Token 消耗。"""

    @pytest.fixture(autouse=True)
    def _setup_mock(self, monkeypatch):
        monkeypatch.setattr(chat_config, "LLM_ADAPTER_TYPE", "mock")
        monkeypatch.setattr(chat_config, "mock_llm_latency_min", 10)
        monkeypatch.setattr(chat_config, "mock_llm_latency_max", 20)
        monkeypatch.setattr(chat_config, "mock_llm_error_rate", 0.0)

    @pytest.mark.asyncio
    async def test_mock_returns_content(self):
        svc = LLMService.get_instance()
        result = await svc.chat_qwen([{"role": "user", "content": "hello"}])
        assert "Mock" in result
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_mock_error_rate(self, monkeypatch):
        monkeypatch.setattr(chat_config, "mock_llm_error_rate", 1.0)
        svc = LLMService.get_instance()
        with pytest.raises(TimeoutError):
            await svc.chat_qwen([{"role": "user", "content": "test"}])


class TestTokenLimit:
    """Token 消耗预检：超限时抛 TokenLimitExceeded。"""

    @pytest.mark.asyncio
    async def test_token_limit_exceeded(self, monkeypatch):
        # 确保 CHAT_MODEL 已配置以避免初始化失败
        monkeypatch.setattr(chat_config, "chat_model", "qwen-turbo")
        svc = LLMService.get_instance()

        # 直接调用 _call_llm 的 token 检查逻辑
        # 通过设置 _token_limit_enabled_ctx 来触发
        from src.modules.chat.core.llm_service import _token_limit_enabled_ctx, _rate_limit_key_ctx
        _token_limit_enabled_ctx.set(True)
        _rate_limit_key_ctx.set("test-key")

        try:
            # 模拟 limiter.check_tokens 返回 not allowed
            from src.core.rate_limiter import get_rate_limiter
            rl = get_rate_limiter()
            original_check = rl.check_tokens
            rl.check_tokens = lambda *a, **kw: (False, 0, 1)

            with pytest.raises(TokenLimitExceeded):
                await svc.chat_qwen([{"role": "user", "content": "test"}])

            rl.check_tokens = original_check
        finally:
            _token_limit_enabled_ctx.set(False)
            _rate_limit_key_ctx.set("")

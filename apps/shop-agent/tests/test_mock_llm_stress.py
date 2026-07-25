"""Mock LLM 压测仿真单元测试。

被测：LLM_ADAPTER_TYPE=mock 时 LLMService 走 _mock_simulate：
- 不发起任何外部调用（不依赖 ChatOpenAI / TONGYI_API_KEY / LLM_GATEWAY_URL）；
- 按 MOCK_LLM_LATENCY_MIN/MAX 产生延迟；
- 按 MOCK_LLM_ERROR_RATE 抛 TimeoutError。

同时验证速率限制配置项（GLOBAL_RATE_LIMIT / CHAT_RATE_LIMIT）可从 env 覆盖，
且默认值与历史一致（30 / 15），保证默认行为不回退。
"""

import asyncio

import pytest
from pydantic import BaseModel

from src.core.config import config
from src.modules.chat.config import chat_config
from src.modules.chat.core.llm_service import LLMService


class _DummySchema(BaseModel):
    """结构化输出测试 schema（任意字段）"""
    ok: bool = True


@pytest.mark.asyncio
async def test_mock_chat_returns_simulated_reply():
    """mock 模式下 chat_qwen 返回模拟文本，不报错。"""
    svc = LLMService()
    svc._qwen_llm = None  # 强制走 mock 分支，不触碰真实 LLM

    # 用 monkeypatch 把 LLM_ADAPTER_TYPE 固定为 mock（chat_config 是实例属性）
    chat_config.LLM_ADAPTER_TYPE = "mock"
    try:
        out = await svc.chat_qwen([{"role": "user", "content": "查一下我的订单"}])
        assert "[Mock]" in out
    finally:
        del chat_config.LLM_ADAPTER_TYPE


@pytest.mark.asyncio
async def test_mock_chat_error_rate_zero():
    """error_rate=0 时永不抛错。"""
    chat_config.LLM_ADAPTER_TYPE = "mock"
    chat_config.mock_llm_error_rate = 0.0
    chat_config.mock_llm_latency_min = 0
    chat_config.mock_llm_latency_max = 0
    try:
        for _ in range(20):
            out = await LLMService().chat_qwen([{"role": "user", "content": "x"}])
            assert "[Mock]" in out
    finally:
        del chat_config.LLM_ADAPTER_TYPE
        chat_config.mock_llm_error_rate = 0.01
        chat_config.mock_llm_latency_min = 100
        chat_config.mock_llm_latency_max = 500


@pytest.mark.asyncio
async def test_mock_chat_error_rate_high():
    """error_rate=1.0 时必然抛 TimeoutError。"""
    chat_config.LLM_ADAPTER_TYPE = "mock"
    chat_config.mock_llm_error_rate = 1.0
    chat_config.mock_llm_latency_min = 0
    chat_config.mock_llm_latency_max = 0
    try:
        with pytest.raises(TimeoutError):
            await LLMService().chat_qwen([{"role": "user", "content": "x"}])
    finally:
        del chat_config.LLM_ADAPTER_TYPE
        chat_config.mock_llm_error_rate = 0.01
        chat_config.mock_llm_latency_min = 100
        chat_config.mock_llm_latency_max = 500


@pytest.mark.asyncio
async def test_mock_structured_returns_schema():
    """mock 模式下 chat_qwen_structured 返回空 schema 实例。"""
    chat_config.LLM_ADAPTER_TYPE = "mock"
    chat_config.mock_llm_latency_min = 0
    chat_config.mock_llm_latency_max = 0
    try:
        out = await LLMService().chat_qwen_structured(
            [{"role": "user", "content": "x"}], _DummySchema
        )
        assert out.ok is True
    finally:
        del chat_config.LLM_ADAPTER_TYPE
        chat_config.mock_llm_latency_min = 100
        chat_config.mock_llm_latency_max = 500


def test_rate_limit_defaults_preserved():
    """默认速率限制与历史一致：全局 30、chat 15。"""
    assert config.GLOBAL_RATE_LIMIT == 30
    assert config.CHAT_RATE_LIMIT == 15
    assert chat_config.global_rate_limit == 30
    assert chat_config.chat_rate_limit == 15


def test_rate_limit_env_override(monkeypatch):
    """GLOBAL_RATE_LIMIT/CHAT_RATE_LIMIT env 可覆盖，压测期可放开。"""
    monkeypatch.setenv("GLOBAL_RATE_LIMIT", "5000")
    monkeypatch.setenv("CHAT_RATE_LIMIT", "5000")
    from pydantic_settings import BaseSettings
    # pydantic-settings 缓存类级字段，这里直接验证 Settings 类能读新值
    monkeypatch.setattr("src.core.config.Settings.Config.env_file", ())
    reloaded = config.__class__(**{})
    assert reloaded.GLOBAL_RATE_LIMIT == 5000
    assert reloaded.CHAT_RATE_LIMIT == 5000

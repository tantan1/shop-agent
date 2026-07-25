"""LLM 出口「无旁路不变量」单元测试（平台工程 01 §L3）。

被测：src.modules.chat.core.llm_service 的
resolve_llm_base_url / _direct_egress_allowed / GatewayNotConfigured。

不变量：出网 LLM 流量必须收敛到唯一网关 LLM_GATEWAY_URL；缺失即 fail-closed
拒启；仅非生产下允许显式逃生阀直连，生产额外要求 OVERRIDE_PROD=true。

注意：resolve_llm_base_url 内部对模块级 prometheus Counter 计数。该 Counter
在模块导入时只构造一次，用例中绝不重建（同名重复构造会 ValueError），
只调用函数本身。
"""

import pytest

from src.modules.chat.core.llm_service import (
    GatewayNotConfigured,
    _direct_egress_allowed,
    resolve_llm_base_url,
)

# 出口相关全部环境变量：每个用例前统一清空，保证用例间零串味。
_EGRESS_ENVS = (
    "LLM_GATEWAY_URL",
    "ALLOW_DIRECT_LLM_EGRESS",
    "ALLOW_DIRECT_LLM_EGRESS_OVERRIDE_PROD",
    "ENVIRONMENT",
    "APP_ENV",
)

# 逃生阀放行时的公网兜底地址（与 llm_service._DIRECT_EGRESS_BASE_URL 保持一致）
DIRECT_EGRESS_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@pytest.fixture()
def egress_env(monkeypatch):
    """清空所有出口相关 env，返回 monkeypatch 供用例按需设置。"""
    for name in _EGRESS_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_resolve_returns_gateway_url_when_set(egress_env):
    """配置了网关地址 -> 直接返回该地址（唯一出口）。"""
    egress_env.setenv("LLM_GATEWAY_URL", "http://gw")

    assert resolve_llm_base_url() == "http://gw"


def test_resolve_raises_when_gateway_missing_and_no_escape_hatch(egress_env):
    """未配置网关且未开逃生阀 -> fail-closed 抛 GatewayNotConfigured。"""
    with pytest.raises(GatewayNotConfigured):
        resolve_llm_base_url()


def test_direct_egress_allowed_in_non_prod(egress_env):
    """非生产 + 逃生阀开启 -> 允许直连，降级为公网兜底地址。"""
    egress_env.setenv("ALLOW_DIRECT_LLM_EGRESS", "true")
    egress_env.setenv("ENVIRONMENT", "development")

    assert _direct_egress_allowed() is True
    assert resolve_llm_base_url() == DIRECT_EGRESS_BASE_URL


def test_prod_escape_hatch_blocked_without_override(egress_env):
    """生产 + 逃生阀开启但无 OVERRIDE_PROD -> 护栏拦截，拒绝直连。"""
    egress_env.setenv("ENVIRONMENT", "production")
    egress_env.setenv("ALLOW_DIRECT_LLM_EGRESS", "true")

    assert _direct_egress_allowed() is False
    with pytest.raises(GatewayNotConfigured):
        resolve_llm_base_url()


def test_prod_escape_hatch_allowed_with_override(egress_env):
    """生产 + 逃生阀开启 + OVERRIDE_PROD -> 显式应急直连放行。"""
    egress_env.setenv("ENVIRONMENT", "production")
    egress_env.setenv("ALLOW_DIRECT_LLM_EGRESS", "true")
    egress_env.setenv("ALLOW_DIRECT_LLM_EGRESS_OVERRIDE_PROD", "true")

    assert _direct_egress_allowed() is True
    assert resolve_llm_base_url() == DIRECT_EGRESS_BASE_URL


def test_app_env_recognized_as_prod(egress_env):
    """APP_ENV=prod 也应被识别为生产环境（多 env 名兼容）。"""
    egress_env.setenv("APP_ENV", "prod")
    egress_env.setenv("ALLOW_DIRECT_LLM_EGRESS", "true")

    assert _direct_egress_allowed() is False
    with pytest.raises(GatewayNotConfigured):
        resolve_llm_base_url()

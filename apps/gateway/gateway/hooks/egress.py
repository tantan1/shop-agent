"""出向 fail-closed/open 开关（01 §5 网关宕机处置边界）。

注意：本开关属上游可达性/降级机制，**不是注入防线**，不占用三道闸编号
（①入向 / ②出向 judge_egress / ③工具参数，编号唯一权威见 scope-10 §1）。


在「建连前」决策，流式/非流式统一。closed 模式：上游不可达/未配置则拒绝建立连接，
绝不退回直连供应商（无旁路不变量）。open 模式：允许回退（裸跑环境）。
"""
from __future__ import annotations

from ..config import settings
from ..types import GatewayMode


def decide(model: str) -> bool:
    """返回 True 表示允许转发；False 表示拒绝（fail-closed）。

    LiteLLM Router 接管后，网关不再持有 base_url；「上游可达性」等价于
    「目标 model 是否在 Router 的 model_list 中注册」。未注册（如 Bedrock 未配、
    或请求了未知模型）时：closed 模式拒绝（绝不退回直连供应商，无旁路不变量），
    open 模式放行（裸跑环境，由 Router 自行失败）。
    """
    # 注意：用模块属性（gateway.litellm_router.llm_router）而非 `from .. import`，
    # 避免 reload 产生新单例后本模块仍持有旧（空）单例引用。
    import gateway.litellm_router as _lr_mod

    llm_router = _lr_mod.llm_router

    # Router 的 model_name 支持 glob（gpt-* / local/*），用 fnmatch 通配匹配；
    # "*" 通配部署存在时任意 model 均视为已注册。
    import fnmatch

    registered = any(
        fnmatch.fnmatch(model, dep.get("model_name", "")) for dep in llm_router._model_list
    )
    if registered:
        return True
    # 未注册：closed 拒绝，open 放行回退
    return settings.fail_mode == GatewayMode.FAIL_OPEN

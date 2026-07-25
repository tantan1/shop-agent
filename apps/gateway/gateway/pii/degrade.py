"""脱敏引擎降级阶梯（07，D5 关联）。

本批只实现 ``healthy`` 结构占位，降级决策由调用方（governance hook）结合
``GATEWAY_FAIL_MODE`` 全局硬下限判定。引擎自身健康度检测留此处扩展。

降级等级：
- ``healthy``  ：引擎正常，全量脱敏。
- ``degraded`` ：部分规则不可用（如中央源超时回退快照），仍按快照脱敏。
- ``down``     ：引擎彻底不可用（基线缺失/规则全坏）。此时能否放行取决于
                  ``GATEWAY_FAIL_MODE``：closed 下被全局硬下限压制（一律拒发），
                  open 下按「路由敏感度分级」放行（低敏 fail-open，高敏仍拦）。

注意：``down`` 的低敏 fail-open 在 ``GATEWAY_FAIL_MODE=closed`` 下**不生效**
（D5 全局硬下限），决策入口须先读 ``settings.fail_mode``。
"""
from __future__ import annotations

from enum import Enum

# 降级阶梯阈值（07 §4 示意；不依赖语义层联网，确定性层即可用）。
# 后台刷新超时上限：超过此值视为源不可达，保留 last-known-good（degraded）。
RULE_REFRESH_TIMEOUT_SEC: float = 2.0
# 熔断窗口：连续刷新失败达到此次数，引擎健康度由 degraded 降为 down（交由
# fail_mode 全局硬下限裁决）。仅作阈值约定，熔断计数由调用方按此配置实现。
RULE_REFRESH_FAILURE_BUDGET: int = 5


class EngineHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


def decide(health: EngineHealth, sensitivity: str, fail_mode: str) -> bool:
    """引擎降级放行决策。

    - fail_mode=closed：任何降级等级都拒发（全局硬下限，D5）。
    - fail_mode=open ：healthy/degraded 放行；down 按敏感度分级（低敏放行、高敏拒发）。
    """
    if fail_mode == "closed":
        return False  # 硬下限压制一切子模块分级放行
    if health in (EngineHealth.HEALTHY, EngineHealth.DEGRADED):
        return True
    # down + open：按敏感度分级
    return sensitivity == "low"

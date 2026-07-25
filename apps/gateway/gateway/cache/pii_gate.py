"""写入前 PII 门禁（08 §3 / 06 §3 跨平面硬门禁）。

写入缓存前调 07 ``PiiEngine`` 检测：命中 PII 则整条不写缓存（绕过，不脱敏后存）。
理由：缓存是长生命周期存储，脱敏后再存会丢失「原始 PII 已被掩码」的信号，且
业务方可能从缓存读到他人 PII——故整条拒绝写入，宁可不命中也绝不存 PII。

硬约束层：代码强制，不参与 cache-policy.yml 配置优先级，任何分片策略不可关闭。
"""
from __future__ import annotations

from typing import Any

from gateway.hooks.governance import hooks
from gateway.types import GovernanceError, GovernanceStage


class PiiGate:
    """缓存写入 PII 门禁（整条不写）。"""

    def check(self, text: str) -> bool:
        """返回 True 表示允许写入；False 表示命中 PII 整条拦截。

        C1：检测异常按 D3 上抛 GovernanceError（不静默放行/拒绝），由 cache_store
        调用方按 fail_mode 分流——但写入门禁异常时保守拒绝写入（不存）。
        """
        if not text:
            return True
        try:
            # 复用 07 引擎（hooks 已持有实例）；非流式 run 的脱敏即等价于扫描。
            # 此处直接用底层 redact 判定（命中即含 PII），避免改动 payload 语义。
            engine = hooks._engine
            if engine is None:
                # pii_enabled=false：引擎未加载，按「无 PII」放行（运维显式关闭脱敏）。
                return True
            _, hit_types = engine.redact(text)
            return len(hit_types) == 0
        except Exception as e:  # noqa: BLE001
            # C1：异常上抛，由 store 调用方决定写入分流（保守：拒绝写入）。
            raise GovernanceError(
                GovernanceStage.INGRESS.value, cause=e, reason="pii gate scan failed"
            ) from e


gate = PiiGate()

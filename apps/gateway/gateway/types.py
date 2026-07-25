"""跨批次共享数据结构（批次0 定义，批次1~2 复用，避免漂移）。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class GatewayMode(str, Enum):
    """网关整体不可用时的处置边界（01 §5）。"""

    FAIL_CLOSED = "closed"  # 拒绝全部 LLM 流量
    FAIL_OPEN = "open"      # 放行绕过（回退直连/上游）

    @classmethod
    def from_env(cls, raw: Optional[str]) -> "GatewayMode":
        if raw is None:
            return cls.FAIL_CLOSED  # 默认 closed：治理完整性优先
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.FAIL_CLOSED


@dataclass
class RouteDecision:
    """路由决策结果（01 §4 引擎可替换 / 02 §4 故障转移）。

    注：自嵌入 LiteLLM Router 后，网关不再持有后端 base_url，实际调度（路由/负载均衡/
    重试/fallback）由 Router 接管。本结构仅表达网关侧决策：
      - upstream_base_url 留空（Router 从 model_list 解析）；
      - upstream_model 为本地任务改写键（qwen3-unified），其余留空；
      - fallback_chain 仅占位 ["__litellm_router__"]，声明故障转移归属 Router 内部。
    """

    upstream_base_url: str
    model: str
    backend: str  # "vllm" | "mock" | "cloud" | "default"
    fallback_allowed: bool
    # 发往 Router 时实际使用的模型名：本地（vLLM）任务统一映射 qwen3-unified；
    # 空字符串 = 保持请求原样（云端键）转发给 Router。
    upstream_model: str = ""
    # 占位声明：实际故障转移链在 LiteLLM Router 的 model_list 中定义（02 §4）。
    fallback_chain: List[str] = field(default_factory=list)


@dataclass
class TenantUsage:
    """租户维度计量占位（03/批次3 接真实计数）。"""

    tenant: str
    requests: int = 0
    est_tokens: int = 0


@dataclass
class Verdict:
    """注入/合规判定结果（批次2/06/10 细化字段）。

    review: 06 合规模糊边界标记（E3，三通道可观测，不阻断）。
      - 仅当 allowed=True 且命中模糊边界时由 ``guardrails_check`` 填；
      - 字段不含命中原文：``rule_id`` 为规则标识，``text_hash`` 为命中文本
        sha256 前缀（供离线人工复核队列对账，不回原文）；
      - 调用方据此注入响应头 ``X-Guardrails-Review`` + metrics + 结构化日志。

    flagged: 出向（非 user 角色）注入命中标记（信任模型分级处置）。
      - 仅当 allowed=True 且命中来自 assistant/tool/system 角色时由注入闸填；
      - 含义：该消息被标记为不可信（可能含间接注入），**不阻断业务**，
        由上游做告警打点 + 隔离/人在回路确认；
      - 元素为 dict（rule_id / role / text_hash），不含命中原文（E3 同纪律）。
      - user 角色命中走 deny 路径（allowed=False + reason），不进 flagged。
    """

    allowed: bool
    reason: str = ""
    review: "VerdictReview | None" = None
    flagged: "list[dict]" = None  # 出向不可信标记（非 user 角色命中）

    @classmethod
    def allow(
        cls,
        reason: str = "",
        review: "VerdictReview | None" = None,
        flagged: "list[dict] | None" = None,
    ) -> "Verdict":
        return cls(allowed=True, reason=reason, review=review, flagged=flagged)

    @classmethod
    def deny(cls, reason: str = "") -> "Verdict":
        return cls(allowed=False, reason=reason)


@dataclass
class VerdictReview:
    """06 human_review 标记（E3）：只带 rule_id + text_hash，绝不带原文。"""

    rule_id: str
    text_hash: str

    def header_value(self) -> str:
        # HTTP 响应头须 latin-1 安全：rule_id 可能含中文，做 percent-encode（全 ASCII）。
        # 下游/日志/metrics 仍用原始 rule_id（内部通道可含中文）。
        from urllib.parse import quote

        return f"{quote(self.rule_id, safe='')}:{self.text_hash}"


class GovernanceStage(str, Enum):
    """治理链阶段标识（C1/D3：供 proxy 按阶段打点分流）。"""

    INGRESS = "ingress"
    EGRESS = "egress"
    STREAM = "stream"


class GovernanceError(Exception):
    """治理链异常（C1/D3）。

    纪律：治理钩子（`hooks/governance.py` 的 run / run_stream / guardrails_check /
    judge_egress / cache_lookup / cache_store）内部**不得自吞异常**——自吞等于把
    fail-open 藏进治理层，`controllers/proxy.py` 的 `GATEWAY_FAIL_MODE` 就管不到。
    钩子须把原始异常包装为本类型向上抛，由 proxy 按 fail_mode 统一分流。

    携带 `stage` 供 proxy 打点（`gateway_governance_error_total{stage, mode}`）与
    日志（`governance_error{stage, exc_type, ts}`）。

    安全约束：**不得携带原文**。`reason` 只允许放非敏感的定位信息（规则 id、阶段名），
    禁止把命中的 prompt/响应内容塞进异常消息——异常会进日志，等于新开泄露面（与 E3 同纪律）。
    """

    def __init__(
        self,
        stage: "GovernanceStage | str",
        cause: Optional[BaseException] = None,
        reason: str = "",
    ) -> None:
        self.stage: str = stage.value if isinstance(stage, GovernanceStage) else str(stage)
        self.cause: Optional[BaseException] = cause
        self.reason: str = reason
        # exc_type 取根因类型名，供打点区分（如 RulesLoadError / re.error / KeyError）
        self.exc_type: str = type(cause).__name__ if cause is not None else "GovernanceError"
        super().__init__(f"governance failed at {self.stage}: {self.exc_type}")

    def log_fields(self) -> dict[str, str]:
        """结构化日志字段（不含原文，供 proxy 直接展开）。"""
        return {
            "stage": self.stage,
            "exc_type": self.exc_type,
            "reason": self.reason,
        }

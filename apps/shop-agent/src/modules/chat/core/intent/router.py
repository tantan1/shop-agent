"""路由层 —— 只回答「怎么执行」，不关心意图是怎么判出来的。

产物 ``ExecutionPlan.mode`` 取值对齐 yaml_flow 的 ``node.type``：
``rag_pipeline`` / ``direct_tool`` / ``react``。

策略来源（均为外部注入的数据，不含硬编码业务常量）：
- 阈值与触发词：``RoutingPolicy``（后续接 config）
- 是否敏感（需多步 / 需更高置信）：``SkillDef.risk`` / ``hitl``，
  取代原先硬编码的 ``ALWAYS_AGENT_ACTIONS`` / ``WRITE_ACTIONS``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .candidate import ExecutionPlan, IntentCandidate


@dataclass
class RoutingPolicy:
    """路由策略（纯数据）。"""

    min_confidence: float = 0.65  # 低于此分不信任 → 回落 RAG
    write_min_confidence: float = 0.78  # 敏感技能（risk=high / hitl）的更高门槛
    ambiguity_zone: float = 0.72  # 歧义带：低于此分倾向交 Agent
    react_triggers: List[str] = field(default_factory=list)


def is_sensitive_skill(skill) -> bool:
    """敏感（资金/隐私/多步）技能：``risk=high`` 或 ``hitl=true``。

    与 ``yaml_flow/validator.py::_is_skill_sensitive`` 同语义 ——
    全系统统一这一个判定口径，避免各处各写一份集合。
    """
    if skill is None:
        return False
    risk = str(getattr(skill, "risk", "low") or "low").strip().lower()
    hitl = bool(getattr(skill, "hitl", False))
    return risk == "high" or hitl


class ExecutionRouter:
    """把分类结果翻译成执行计划。

    不感知 FAISS / LLM / 否定词，也不感知 SkillRegistry 的加载方式 ——
    技能元数据通过 ``skill_lookup`` 注入，便于单测。
    """

    def __init__(
        self,
        policy: Optional[RoutingPolicy] = None,
        skill_lookup: Optional[Callable[[str], object]] = None,
    ):
        self._policy = policy or RoutingPolicy()
        self._skill_lookup = skill_lookup or (lambda _name: None)

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    def _skill_of(self, action: Optional[str]):
        if not action:
            return None
        try:
            return self._skill_lookup(action)
        except Exception:
            return None

    def route(
        self,
        candidate: Optional[IntentCandidate],
        message: str = "",
        complexity_override: Optional[str] = None,
        reason_override: Optional[str] = None,
    ) -> ExecutionPlan:
        """把候选意图映射为执行计划。

        Args:
            candidate: 分类结果；``None`` 或 ``action`` 为空表示无工具意图。
            message: 原始消息（用于推理类关键词匹配）。
            complexity_override: LLM 二次分类给出的复杂度，覆盖本地判定。
            reason_override: LLM 二次分类给出的理由。
        """
        p = self._policy

        # 1) 无工具意图（否定词 / 未命中）→ RAG
        if candidate is None or not candidate.action:
            return ExecutionPlan(
                mode="rag_pipeline",
                skill=None,
                confidence=0.0,
                reason="无工具意图命中（否定词或未命中）",
            )

        action = candidate.action
        score = float(candidate.score)

        # 2) 置信度不足 → 回落 RAG
        if score <= p.min_confidence:
            return ExecutionPlan(
                mode="rag_pipeline",
                skill=action,
                confidence=score,
                reason=f"置信度 {score:.3f} <= {p.min_confidence}，回落 RAG",
            )

        skill = self._skill_of(action)

        # 3) 敏感技能（risk=high / hitl=true）低置信降级 → RAG，防误触副作用
        if is_sensitive_skill(skill) and score <= p.write_min_confidence:
            return ExecutionPlan(
                mode="rag_pipeline",
                skill=action,
                confidence=score,
                reason=f"敏感技能低置信降级: {score:.3f} <= {p.write_min_confidence}",
            )

        # 4) 复杂度判定 → react（多步） / direct_tool（单步）
        if complexity_override is not None:
            complexity = complexity_override
            reason = reason_override or "LLM 二次分类"
        else:
            complexity, reason = self._assess(message, action, score, skill)

        mode = "react" if complexity == "multi_step" else "direct_tool"
        return ExecutionPlan(mode=mode, skill=action, confidence=score, reason=reason)

    def _assess(
        self, message: str, action: str, score: float, skill
    ) -> Tuple[str, str]:
        """本地复杂度判定（替代原 ``assess_complexity``）。"""
        p = self._policy

        # 信号1：敏感技能本质多步（如退货：查政策→验条件→执行）
        if is_sensitive_skill(skill):
            return (
                "multi_step",
                f"{action} 为敏感/多步骤技能（SKILL.md 的 risk/hitl 声明）",
            )

        # 信号2：含推理/多步关键词
        matched = [t for t in p.react_triggers if t and t in message]
        in_zone = score < p.ambiguity_zone

        if len(matched) >= 2:
            return "multi_step", f"含多个推理/多步关键词: {matched}"
        if len(matched) == 1 and in_zone:
            return (
                "multi_step",
                f"含推理关键词且相似分在歧义带: {matched}, score={score:.3f}",
            )
        if in_zone:
            return (
                "multi_step",
                f"相似分 {score:.3f} < {p.ambiguity_zone}，可能存在歧义",
            )
        return "simple", "表达清晰，直接调用工具即可"

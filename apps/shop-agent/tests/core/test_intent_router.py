"""路由层测试（关注点分离重构 · 第 2 步）。

核心断言：
1. ``ExecutionPlan.mode`` 取值对齐 yaml_flow 的 ``node.type``；
2. 「多步骤 / 写类」判定来自 SKILL.md 的 ``risk`` / ``hitl``，
   **不再依赖硬编码的 ALWAYS_AGENT_ACTIONS / WRITE_ACTIONS**；
3. ``IntentResult`` 由 ``plan`` 唯一派生，deprecated 字段已移除。
"""

from src.modules.chat.core.intent.candidate import ExecutionPlan, IntentCandidate
from src.modules.chat.core.intent.router import (
    ExecutionRouter,
    RoutingPolicy,
    is_sensitive_skill,
)


class _Skill:
    """最小 SkillDef 替身（只需 name / risk / hitl）。"""

    def __init__(self, name: str, risk: str = "low", hitl: bool = False):
        self.name = name
        self.risk = risk
        self.hitl = hitl


def _router(skills=None):
    skills = skills or []
    lookup = lambda name: next((s for s in skills if s.name == name), None)  # noqa: E731
    return ExecutionRouter(
        RoutingPolicy(
            min_confidence=0.65,
            write_min_confidence=0.78,
            ambiguity_zone=0.72,
            react_triggers=["为什么", "怎么办", "帮我处理"],
        ),
        lookup,
    )


def _hit(action: str, score: float, source: str = "faiss"):
    return IntentCandidate(action=action, score=score, source=source)


# ── 敏感性判定（唯一口径）──


def test_is_sensitive_skill_reads_risk_and_hitl():
    assert is_sensitive_skill(_Skill("a", risk="high"))
    assert is_sensitive_skill(_Skill("a", hitl=True))
    assert is_sensitive_skill(_Skill("a", risk="HIGH"))  # 大小写不敏感
    assert not is_sensitive_skill(_Skill("a"))
    assert not is_sensitive_skill(None)


# ── 无意图 / 低置信 → RAG ──


def test_no_candidate_routes_to_rag():
    plan = _router().route(None, "随便问问")
    assert plan.mode == "rag_pipeline"
    assert plan.skill is None


def test_negation_candidate_routes_to_rag():
    cand = IntentCandidate(action=None, score=1.0, source="negation", matched="退货政策")
    plan = _router().route(cand, "退货政策是什么")
    assert plan.mode == "rag_pipeline"


def test_low_confidence_routes_to_rag():
    plan = _router([_Skill("query-order")]).route(_hit("query-order", 0.50), "查订单")
    assert plan.mode == "rag_pipeline"
    assert "置信度" in plan.reason


# ── 敏感技能（risk=high / hitl）──


def test_sensitive_skill_low_confidence_downgrades_to_rag():
    """0.65 < 0.70 <= 0.78：写类操作必须更高置信，否则降级 RAG 防误触。"""
    router = _router([_Skill("request-return", risk="high", hitl=True)])
    plan = router.route(_hit("request-return", 0.70), "我要退货")
    assert plan.mode == "rag_pipeline"
    assert "降级" in plan.reason


def test_sensitive_skill_high_confidence_routes_to_react():
    router = _router([_Skill("request-return", risk="high", hitl=True)])
    plan = router.route(_hit("request-return", 0.90), "我要退货")
    assert plan.mode == "react"
    assert plan.skill == "request-return"


def test_new_sensitive_skill_needs_no_code_change():
    """关键：新增一个 risk=high 的 skill，无需改任何代码即被判为多步。

    这是「用 skill 元数据取代硬编码集合」的直接证明。
    """
    router = _router([_Skill("brand-new-write-action", risk="high")])
    plan = router.route(_hit("brand-new-write-action", 0.90), "帮我执行")
    assert plan.mode == "react"


# ── 普通技能 ──


def test_normal_skill_high_confidence_routes_to_direct_tool():
    router = _router([_Skill("query-order")])
    plan = router.route(_hit("query-order", 0.90), "查一下订单")
    assert plan.mode == "direct_tool"
    assert plan.skill == "query-order"


def test_multiple_react_trigger_words_routes_to_react():
    router = _router([_Skill("query-order")])
    plan = router.route(_hit("query-order", 0.90), "为什么订单没到，帮我处理一下")
    assert plan.mode == "react"


def test_single_trigger_with_high_confidence_stays_direct_tool():
    """单个触发词 + 高相似度 → 仍按简单调用处理（沿用原决策表：

    「用户口吻不一定真的多步」，避免把「为什么说还没发货」误判成多步流程）。
    """
    router = _router([_Skill("query-order")])
    plan = router.route(_hit("query-order", 0.90), "为什么我的订单还没到")
    assert plan.mode == "direct_tool"


def test_single_trigger_in_ambiguity_zone_routes_to_react():
    """单个触发词 + 相似分落在歧义带 → 交 Agent。"""
    router = _router([_Skill("query-order")])
    plan = router.route(_hit("query-order", 0.70), "为什么我的订单还没到")  # 0.70 < 0.72
    assert plan.mode == "react"


def test_ambiguity_zone_routes_to_react():
    router = _router([_Skill("query-order")])
    plan = router.route(_hit("query-order", 0.70), "查订单")  # 0.70 < 0.72
    assert plan.mode == "react"


# ── LLM 覆盖 ──


def test_complexity_override_wins_over_local_assessment():
    router = _router([_Skill("query-order")])
    plan = router.route(
        _hit("query-order", 0.90),
        "查订单",
        complexity_override="multi_step",
        reason_override="LLM 判定",
    )
    assert plan.mode == "react"
    assert plan.reason == "LLM 判定"


# ── IntentResult 由 plan 唯一派生（无 deprecated 双写）──


def test_to_result_maps_plan_to_intent_result():
    from src.modules.chat.core.intent_recognizer import IntentRecognizer

    rag = IntentRecognizer._to_result(
        ExecutionPlan(mode="rag_pipeline", reason="无工具意图"), 0.9
    )
    assert rag.plan is not None and rag.plan.mode == "rag_pipeline"
    assert rag.action is None and rag.similarity_score == 0.9

    tool = IntentRecognizer._to_result(
        ExecutionPlan(mode="direct_tool", skill="query-order", confidence=0.9, reason="r"),
        0.9,
    )
    assert tool.plan.mode == "direct_tool"
    assert tool.action == "query-order"
    assert tool.similarity_score == 0.9

    agent = IntentRecognizer._to_result(
        ExecutionPlan(mode="react", skill="request-return", confidence=0.9, reason="r2"),
        0.9,
    )
    assert agent.plan.mode == "react"
    assert agent.action == "request-return"
    assert agent.similarity_score == 0.9


# ── 兼容期 mode 读取口 ──


def test_mode_derives_from_plan():
    """mode 统一由 plan.mode 派生；无 plan 兜底为 rag_pipeline。"""
    from src.modules.chat.schemas import IntentResult

    assert IntentResult(plan=ExecutionPlan(mode="rag_pipeline")).mode == "rag_pipeline"
    assert (
        IntentResult(plan=ExecutionPlan(mode="direct_tool", skill="query-order")).mode
        == "direct_tool"
    )
    assert (
        IntentResult(plan=ExecutionPlan(mode="react", skill="request-return", reason="r")).mode
        == "react"
    )
    # 所有生产路径都会设置 plan；无 plan 兜底为 rag_pipeline（防御性）
    assert IntentResult().mode == "rag_pipeline"


def test_mode_prefers_plan_when_present():
    from src.modules.chat.schemas import IntentResult

    res = IntentResult(
        plan=ExecutionPlan(mode="react", skill="request-return", reason="r"),
        action="request-return",
    )
    assert res.mode == "react"
    assert res.plan.reason == "r"


# ── 回归护栏：禁止把集合硬编码回识别器 ──


def test_recognizer_no_longer_hardcodes_action_sets():
    import src.modules.chat.core.intent_recognizer as ir

    assert not hasattr(ir, "ALWAYS_AGENT_ACTIONS"), (
        "多步骤意图应由 SKILL.md 的 risk/hitl 推导，不得硬编码集合"
    )
    assert not hasattr(ir, "WRITE_ACTIONS"), (
        "写类意图应由 SKILL.md 的 risk/hitl 推导，不得硬编码集合"
    )

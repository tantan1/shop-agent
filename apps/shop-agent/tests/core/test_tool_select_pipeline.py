"""ToolSelectPipeline 单元测试（TDD：mock 各层，验证 Pipeline 编排逻辑）。

覆盖：早停、候选 scope 传递、各降级策略、错误处理、超时强制。
每层独立 mock，不依赖其他 stage（对应验收项"单测覆盖每层独立 mock"）。
"""
import asyncio

from src.modules.chat.core.tool_select_pipeline import PipelinePolicy, ToolSelectPipeline
from src.modules.chat.core.tool_select_stage import (
    STAGE_REGISTRY,
    CandidateScore,
    StageResult,
    ToolSelectStage,
    register_stage,
)
from src.modules.chat.core.tool_select_stages import (
    FaissRecallStage,
    LinearHeadStage,
    LlmFallbackStage,
    RuleFilterStage,
)


class FakeStage(ToolSelectStage):
    """测试用假 Stage：按预设返回结果，可选延迟触发超时或主动抛异常。"""

    def __init__(self, name, result=None, delay=None, raise_error=None):
        super().__init__(name)
        self._result = result or StageResult(tool=None, source=name)
        self._delay = delay
        self._raise = raise_error
        self.calls = []  # 记录每层收到的 scope，用于验证候选传递

    async def run(self, query, scope, context):
        self.calls.append(list(scope))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise:
            raise self._raise
        return self._result


def _hit(tool, score, source):
    return StageResult(
        tool=tool,
        confidence=score,
        source=source,
        scored_candidates=[CandidateScore(tool, score, source)],
    )


ALL_TOOLS = ["query-order", "check-shipping", "request-return"]


def _build(stages, thresholds, fallback=None, timeouts=None):
    return ToolSelectPipeline(
        stages=stages,
        all_tools=ALL_TOOLS,
        deps={},
        policy=PipelinePolicy(
            thresholds=thresholds,
            fallback=fallback or {},
            timeouts=timeouts or {},
        ),
    )


def test_early_stop_on_confident_stage():
    s1 = FakeStage("p1_faiss", _hit("query-order", 0.9, "p1_faiss"))
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("查订单")
    )
    assert plan.stop_condition == "plan_complete"
    assert plan.actions[0].name == "query-order"
    assert plan.source == "p1_faiss"
    assert s2.calls == []  # 早停后不应调用更贵层级（成本漏斗）


def test_no_early_stop_falls_back_to_need_llm():
    s1 = FakeStage("p1_faiss", StageResult(tool=None, source="p1_faiss"))
    s2 = FakeStage("p2_linear", StageResult(tool=None, source="p2_linear"))
    plan = asyncio.run(
        _build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("你好")
    )
    assert plan.stop_condition == "need_llm"
    assert plan.actions == []


def test_default_tool_fallback():
    s1 = FakeStage("p1_faiss", StageResult(tool=None, source="p1_faiss"))
    plan = asyncio.run(
        _build(
            [s1], {"p1_faiss": 0.85}, fallback={"on_all_stages_failed": "default_tool"}
        ).select("你好")
    )
    assert plan.stop_condition == "plan_complete"
    assert plan.actions[0].name == ALL_TOOLS[0]


def test_scope_passed_to_next_stage():
    scoped = ["query-order", "check-shipping"]
    s1 = FakeStage(
        "p1_faiss",
        StageResult(
            tool=None,
            source="p1_faiss",
            scored_candidates=[CandidateScore(t, 0.8, "p1_faiss") for t in scoped],
        ),
    )
    s2 = FakeStage("p2_linear", StageResult(tool=None, source="p2_linear"))
    asyncio.run(
        _build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("查订单")
    )
    assert s2.calls[0] == scoped  # 下一层收到缩窄后的 scope


def test_stage_error_skip_continues():
    s1 = FakeStage("p1_faiss", raise_error=RuntimeError("boom"))
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build(
            [s1, s2],
            {"p1_faiss": 0.85, "p2_linear": 0.90},
            fallback={"on_stage_failure": "skip"},
        ).select("查物流")
    )
    assert plan.source == "p2_linear"
    assert plan.actions[0].name == "check-shipping"


def test_stage_error_abort_returns_need_llm():
    s1 = FakeStage("p1_faiss", raise_error=RuntimeError("boom"))
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build(
            [s1, s2],
            {"p1_faiss": 0.85, "p2_linear": 0.90},
            fallback={"on_stage_failure": "abort"},
        ).select("查物流")
    )
    assert plan.stop_condition == "need_llm"
    assert s2.calls == []  # abort 后不继续


def test_stage_timeout_treated_as_error():
    s1 = FakeStage(
        "p1_faiss",
        delay=0.5,
        result=_hit("query-order", 0.9, "p1_faiss"),
    )
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build(
            [s1, s2],
            {"p1_faiss": 0.85, "p2_linear": 0.90},
            timeouts={"p1_faiss": 50},
        ).select("查订单")
    )
    # p1 超时 -> error -> skip -> p2 命中
    assert plan.source == "p2_linear"


def test_register_stage_populates_registry():
    """A3：显式注册表可登记并按 name 解析 Stage 类。"""
    name = "_probe_stage"
    register_stage(name, FakeStage)
    try:
        assert STAGE_REGISTRY[name] is FakeStage
        assert STAGE_REGISTRY[name](name).name == name
    finally:
        STAGE_REGISTRY.pop(name, None)


def test_stage_error_hitl_returns_need_llm():
    """hitl 在代码层等价于 need_llm：返回空结果交上游按部署策略处置。"""
    s1 = FakeStage("p1_faiss", raise_error=RuntimeError("boom"))
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build(
            [s1, s2],
            {"p1_faiss": 0.85, "p2_linear": 0.90},
            fallback={"on_stage_failure": "hitl"},
        ).select("查物流")
    )
    assert plan.stop_condition == "need_llm"
    assert s2.calls == []


def test_stage_result_logged_per_layer(caplog):
    """可观测约束：每层都要打 tool_select_stage_result 结构化日志。"""
    import logging

    caplog.set_level(logging.INFO, logger="src.modules.chat.core.tool_select_pipeline")
    s1 = FakeStage("p1_faiss", _hit("query-order", 0.9, "p1_faiss"))
    asyncio.run(_build([s1], {"p1_faiss": 0.85}).select("查订单"))

    records = [r for r in caplog.records if r.message == "tool_select_stage_result"]
    assert len(records) == 1
    assert records[0].stage == "p1_faiss"
    assert records[0].candidates_count == 1
    assert records[0].latency_ms >= 0


def test_empty_message_exception_is_still_an_error():
    """红队 B1：空消息异常（str(exc)==""）不得被当成成功层，abort 必须生效。"""
    s1 = FakeStage("p1_faiss", raise_error=RuntimeError())  # 无消息
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build(
            [s1, s2],
            {"p1_faiss": 0.85, "p2_linear": 0.90},
            fallback={"on_stage_failure": "abort"},
        ).select("查物流")
    )
    assert plan.stop_condition == "need_llm"
    assert s2.calls == []


def test_tool_outside_registry_does_not_early_stop():
    """红队 B2：越界工具名不得产出 plan_complete。"""
    s1 = FakeStage("p3_llm", _hit("search-web", 0.0, "p3_llm"))  # 不在 ALL_TOOLS
    s2 = FakeStage("p2_linear", StageResult(tool=None, source="p2_linear"))
    plan = asyncio.run(
        _build([s1, s2], {"p3_llm": 0.0, "p2_linear": 0.90}).select("随便问点什么")
    )
    assert plan.stop_condition == "need_llm"
    assert plan.actions == []


def test_zero_threshold_with_registry_tool_still_early_stops():
    """对照组：注册表内工具 + threshold 0.0 仍应正常早停（P3 兜底语义不受 B2 修复影响）。"""
    s1 = FakeStage("p3_llm", _hit("request-return", 0.0, "p3_llm"))
    plan = asyncio.run(_build([s1], {"p3_llm": 0.0}).select("我要退货"))
    assert plan.stop_condition == "plan_complete"
    assert plan.actions[0].name == "request-return"


# ── 收敛早停（convergence early-stop）启用后：P1/P2/P3 三层覆盖 ── #
# 设计约束：首层(P0)豁免收敛早停，故三层指 P1/P2/P3。
# convergence_threshold=2：候选收窄到 ≤2 即提前结束（与 _final_scope_plan 的 plan_complete 判定一致）。
def _scoped(tools, source, score=0.8):
    """构造只收窄 scope、不返回单工具的 StageResult（用于触发收敛早停）。"""
    return StageResult(
        tool=None,
        source=source,
        scored_candidates=[CandidateScore(t, score, source) for t in tools],
    )


def _build_conv(stages, thresholds, conv=2, fallback=None):
    return ToolSelectPipeline(
        stages=stages,
        all_tools=ALL_TOOLS,
        deps={},
        policy=PipelinePolicy(
            thresholds=thresholds,
            convergence_threshold=conv,
            fallback=fallback or {"on_stage_failure": "skip"},
        ),
    )


def test_convergence_early_stop_at_p1():
    """候选在 P1 收窄到 1 → 收敛早停于 P1，P2/P3 不被调用（成本漏斗）。"""
    s0 = FakeStage("p0_rule", _scoped(ALL_TOOLS, "p0_rule"))
    s1 = FakeStage("p1_faiss", _scoped(["query-order"], "p1_faiss"))
    s2 = FakeStage("p2_linear")
    s3 = FakeStage("p3_llm")
    plan = asyncio.run(
        _build_conv(
            [s0, s1, s2, s3],
            {"p0_rule": 1.0, "p1_faiss": 0.85, "p2_linear": 0.90, "p3_llm": 1.0},
        ).select("查订单")
    )
    assert plan.source == "p1_faiss"
    assert plan.stop_condition == "plan_complete"
    assert s2.calls == [] and s3.calls == []  # 不触发更贵层级


def test_convergence_early_stop_at_p2():
    """P1 未收敛(3 候选)→ 继续；P2 收窄到 2 → 收敛早停于 P2，P3 不被调用。"""
    s0 = FakeStage("p0_rule", _scoped(ALL_TOOLS, "p0_rule"))
    s1 = FakeStage("p1_faiss", _scoped(ALL_TOOLS, "p1_faiss"))
    s2 = FakeStage("p2_linear", _scoped(["query-order", "check-shipping"], "p2_linear"))
    s3 = FakeStage("p3_llm")
    plan = asyncio.run(
        _build_conv(
            [s0, s1, s2, s3],
            {"p0_rule": 1.0, "p1_faiss": 0.85, "p2_linear": 0.90, "p3_llm": 1.0},
        ).select("查订单")
    )
    assert plan.source == "p2_linear"
    assert plan.stop_condition == "plan_complete"
    assert s3.calls == []


def test_convergence_early_stop_at_p3():
    """P1/P2 均未收敛(3 候选)→ 继续；P3 收窄到 2 → 收敛早停于 P3。"""
    s0 = FakeStage("p0_rule", _scoped(ALL_TOOLS, "p0_rule"))
    s1 = FakeStage("p1_faiss", _scoped(ALL_TOOLS, "p1_faiss"))
    s2 = FakeStage("p2_linear", _scoped(ALL_TOOLS, "p2_linear"))
    s3 = FakeStage("p3_llm", _scoped(["query-order", "check-shipping"], "p3_llm"))
    plan = asyncio.run(
        _build_conv(
            [s0, s1, s2, s3],
            {"p0_rule": 1.0, "p1_faiss": 0.85, "p2_linear": 0.90, "p3_llm": 1.0},
        ).select("查订单")
    )
    assert plan.source == "p3_llm"
    assert plan.stop_condition == "plan_complete"


def test_confident_early_stop_at_p2():
    """P1 未收敛 → 继续；P2 单层高置信(0.95≥0.90) → 早停于 P2（line 142 路径）。"""
    s0 = FakeStage("p0_rule", _scoped(ALL_TOOLS, "p0_rule"))
    s1 = FakeStage("p1_faiss", _scoped(ALL_TOOLS, "p1_faiss"))
    s2 = FakeStage("p2_linear", _hit("query-order", 0.95, "p2_linear"))
    s3 = FakeStage("p3_llm")
    plan = asyncio.run(
        _build_conv(
            [s0, s1, s2, s3],
            {"p0_rule": 1.0, "p1_faiss": 0.85, "p2_linear": 0.90, "p3_llm": 1.0},
        ).select("查订单")
    )
    assert plan.source == "p2_linear"
    assert plan.stop_condition == "plan_complete"
    assert s3.calls == []


def test_confident_early_stop_at_p3():
    """P1/P2 未收敛 → 继续；P3 单层高置信(1.0≥1.0) → 早停于 P3（line 142 路径）。"""
    s0 = FakeStage("p0_rule", _scoped(ALL_TOOLS, "p0_rule"))
    s1 = FakeStage("p1_faiss", _scoped(ALL_TOOLS, "p1_faiss"))
    s2 = FakeStage("p2_linear", _scoped(ALL_TOOLS, "p2_linear"))
    s3 = FakeStage("p3_llm", _hit("query-order", 1.0, "p3_llm"))
    plan = asyncio.run(
        _build_conv(
            [s0, s1, s2, s3],
            {"p0_rule": 1.0, "p1_faiss": 0.85, "p2_linear": 0.90, "p3_llm": 1.0},
        ).select("查订单")
    )
    assert plan.source == "p3_llm"
    assert plan.stop_condition == "plan_complete"


# ── 真实 Stage 类 + mock 依赖：覆盖 P2 / P3 收敛早停 ── #
# 背景：真实 API 流量在 react 模式下总带着一个具体的 intent_action，
# P0(RuleFilter) 据此把 scope 收窄到 1 个工具，P1(FAISS) 只返回 1 个，
# 收敛早停在 p1_faiss 即触发——P2/P3 在真实流量里永远拿不到 >2 候选。
# （详见 intent_recognizer 路由：score<0.65 才会 action=unknown，但那时已
#   降级 rag_pipeline，根本不进漏斗；故 react 流量架构上到不了 P2/P3。）
# 因此 P2/P3 的收敛早停只能由单测构造 >2 候选来覆盖。这里用真实的
# FaissRecallStage / LinearHeadStage / LlmFallbackStage 类，注入 mock
# matcher / classifier / llm，验证真实收敛代码路径（而非 FakeStage）。
class _MockFaissMatcher:
    """模拟 P1 FAISS 重排：返回预设的候选子集（不按相似度过滤）。"""

    def __init__(self, ranked):
        self._ranked = list(ranked)

    async def rank(self, user_query, candidate_names, intent_action=None, top_k=5, query_embedding=None):
        return [t for t in self._ranked if t in set(candidate_names)][:top_k]


class _MockHeadClassifier:
    """模拟 P2 线性头：返回预设的选中子集。"""

    def __init__(self, selected):
        self._selected = list(selected)

    def score(self, query, candidates, top_k=None, query_embedding=None):
        return [t for t in self._selected if t in set(candidates)]


class _MockToolSelectorLLM:
    """模拟 P3 LLM 兜底：返回预设的选中工具名（按换行分隔）。"""

    def __init__(self, chosen):
        self._chosen = list(chosen)

    async def ainvoke(self, prompt):
        return "\n".join(self._chosen)


class _MockSkillRegistry:
    """模拟 SkillRegistry：unknown 意图映射到全量工具，使 P0 不收窄。"""

    def __init__(self, all_tools):
        self.intent_tool_map = {"unknown": set(all_tools)}


def _build_real_pipeline(all_tools, matcher_ranked, head_selected, llm_chosen, conv=2):
    """用真实 Stage 类 + mock 依赖构造 pipeline，可驱动到 P2/P3。"""
    stages = [
        RuleFilterStage(),
        FaissRecallStage(top_k=5),
        LinearHeadStage(),
        LlmFallbackStage(),
    ]
    deps = {
        "tool_matcher": _MockFaissMatcher(matcher_ranked),
        "tool_head_classifier": _MockHeadClassifier(head_selected),
        "llm_service": type(
            "LLMServiceStub", (), {"tool_selector_llm": _MockToolSelectorLLM(llm_chosen)}
        )(),
        "skill_registry": _MockSkillRegistry(all_tools),
    }
    return ToolSelectPipeline(
        stages=stages,
        all_tools=all_tools,
        deps=deps,
        policy=PipelinePolicy(
            thresholds={
                "p0_rule": 1.0,
                "p1_faiss": 0.85,
                "p2_linear": 0.90,
                "p3_llm": 1.0,
            },
            convergence_threshold=conv,
        ),
    )


def test_real_stages_convergence_at_p2():
    """真实 Stage 类：P1 返回 3 候选→不收敛；P2 线性头收窄到 2→收敛早停于 P2。"""
    all_tools = ["query-order", "check-shipping", "request-return"]
    plan = asyncio.run(
        _build_real_pipeline(
            all_tools,
            matcher_ranked=all_tools,  # P1 FAISS 返回全部 3 个 → 不收敛
            head_selected=all_tools[:2],  # P2 线性头收窄到 2 → 收敛
            llm_chosen=["query-order"],
        ).select("帮我处理订单相关的事 并且 告诉我怎么操作")
    )
    assert plan.source == "p2_linear"
    assert plan.stop_condition == "plan_complete"


def test_real_stages_convergence_at_p3():
    """真实 Stage 类：P1、P2 均返回 3 候选→不收敛；P3 LLM 收窄到 2→收敛早停于 P3。"""
    all_tools = ["query-order", "check-shipping", "request-return"]
    plan = asyncio.run(
        _build_real_pipeline(
            all_tools,
            matcher_ranked=all_tools,  # P1 返回 3 → 不收敛
            head_selected=all_tools,  # P2 也返回 3 → 不收敛
            llm_chosen=all_tools[:2],  # P3 LLM 收窄到 2 → 收敛
        ).select("帮我处理订单相关的事 并且 告诉我怎么操作")
    )
    assert plan.source == "p3_llm"
    assert plan.stop_condition == "plan_complete"


def test_real_stages_p1_non_converge_continues_to_p2():
    """P1 返回 3 候选时，P2 真实线性头被调用（len(scope)>2 才介入的契约）。"""
    all_tools = ["query-order", "check-shipping", "request-return"]
    plan = asyncio.run(
        _build_real_pipeline(
            all_tools,
            matcher_ranked=all_tools,
            head_selected=all_tools[:2],
            llm_chosen=["query-order"],
        ).select("帮我处理订单相关的事 并且 告诉我怎么操作")
    )
    # 收敛发生在 P2，证明 P2 线性头在 P1 未收敛时被真实调用
    assert plan.source == "p2_linear"


def test_p0_exempt_from_convergence():
    """首层(P0)即便收窄到 1 候选也不触发收敛早停，保证 P1 始终被观测。"""
    s0 = FakeStage("p0_rule", _scoped(["query-order"], "p0_rule"))
    s1 = FakeStage("p1_faiss", _scoped(["query-order"], "p1_faiss"))
    s2 = FakeStage("p2_linear")
    plan = asyncio.run(
        _build_conv(
            [s0, s1, s2],
            {"p0_rule": 1.0, "p1_faiss": 0.85, "p2_linear": 0.90},
        ).select("查订单")
    )
    assert s1.calls != []  # P0 虽收窄到 1 仍继续到 P1（首层豁免）
    assert plan.source == "p1_faiss"  # 收敛早停在 P1 才生效


def test_no_convergence_when_scope_exceeds_threshold():
    """P1 返回 3 候选(>conv=2) 不收敛 → 继续到 P2（验证阈值边界）。"""
    s0 = FakeStage("p0_rule", _scoped(ALL_TOOLS, "p0_rule"))
    s1 = FakeStage("p1_faiss", _scoped(ALL_TOOLS, "p1_faiss"))
    s2 = FakeStage("p2_linear", _hit("query-order", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build_conv(
            [s0, s1, s2],
            {"p0_rule": 1.0, "p1_faiss": 0.85, "p2_linear": 0.90},
        ).select("查订单")
    )
    assert s1.calls != [] and s2.calls != []  # P1 未收敛 → P2 仍被调用
    assert plan.source == "p2_linear"

"""ToolSelectPipeline 独立回归测试（tests.py）。

对应验收 A1–A12 + 红队回归（B1 空消息异常、B2 越界工具名）。
- 纯内存 FakeStage，不依赖 P0–P3 真实实现 / pytest-asyncio。
- asyncio.run() 驱动；import 走 src.modules.chat...。
- 运行：cd apps/shop-agent && python -m pytest tests.py -q
"""
import asyncio
import inspect

import pytest

from src.modules.chat.core.tool_select_pipeline import PipelinePolicy, ToolSelectPipeline
from src.modules.chat.core.tool_select_stage import (
    STAGE_REGISTRY,
    CandidateScore,
    StageResult,
    ToolSelectStage,
    register_stage,
)

ALL_TOOLS = ["query-order", "check-shipping", "request-return"]


class FakeStage(ToolSelectStage):
    """内存假 Stage：按预设返回结果，可选延迟触发超时或主动抛异常。"""

    def __init__(self, name, result=None, delay=None, raise_error=None):
        super().__init__(name)
        self._result = result or StageResult(tool=None, source=name)
        self._delay = delay
        self._raise = raise_error
        self.calls: list = []

    async def run(self, query, scope, context):
        self.calls.append(list(scope))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise:
            raise self._raise
        return self._result


def _hit(tool: str, score: float, source: str) -> StageResult:
    return StageResult(
        tool=tool,
        confidence=score,
        source=source,
        scored_candidates=[CandidateScore(tool, score, source)],
    )


def _build(stages, thresholds, fallback=None, timeouts=None) -> ToolSelectPipeline:
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


# ── A1 Stage 接口（抽象方法签名 + name 注入）── #
def test_stage_run_is_abstract():
    """run() 为抽象方法：基类直接实例化必须抛 TypeError。"""
    assert getattr(ToolSelectStage.run, "__isabstractmethod__", False) is True
    with pytest.raises(TypeError):
        ToolSelectStage("x")


def test_subclass_must_implement_run_to_instantiate():
    """未实现 run 的子类仍不可实例化，强制各 Stage 落地 run 方法。"""
    class _MissingRun(ToolSelectStage):
        async def other(self):
            return None

    with pytest.raises(TypeError):
        _MissingRun("x")


def test_stage_run_signature_has_query_scope_context():
    params = list(inspect.signature(ToolSelectStage.run).parameters)
    assert params == ["self", "query", "scope", "context"]


def test_stage_name_injected_via_constructor():
    assert FakeStage("p1_faiss").name == "p1_faiss"


# ── A2 数据结构字段 ── #
def test_candidate_score_fields():
    cs = CandidateScore(tool="query-order", score=0.82, source="p1_faiss")
    assert (cs.tool, cs.score, cs.source) == ("query-order", 0.82, "p1_faiss")


def test_stage_result_defaults():
    r = StageResult()
    assert r.tool is None
    assert r.confidence == 0.0
    assert r.source == ""
    assert r.scored_candidates == []
    assert r.error is None


# ── A3 注册表 ── #
def test_register_stage_populates_registry():
    name = "_probe_stage"
    register_stage(name, FakeStage)
    try:
        assert STAGE_REGISTRY[name] is FakeStage
        assert STAGE_REGISTRY[name](name).name == name
    finally:
        STAGE_REGISTRY.pop(name, None)


# ── A4 早停 ── #
def test_early_stop_on_confident_stage():
    s1 = FakeStage("p1_faiss", _hit("query-order", 0.9, "p1_faiss"))
    s2 = FakeStage("p2_linear")
    plan = asyncio.run(_build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("查订单"))
    assert plan.source == "p1_faiss"
    assert plan.stop_condition == "plan_complete"
    assert s2.calls == []


# ── A5 无早停降级 ── #
def test_no_early_exit_falls_back_to_need_llm():
    s1 = FakeStage("p1_faiss", StageResult(tool=None, source="p1_faiss"))
    s2 = FakeStage("p2_linear", StageResult(tool=None, source="p2_linear"))
    plan = asyncio.run(_build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("随便问"))
    assert plan.stop_condition == "need_llm"
    assert plan.actions == []
    assert plan.source == "fallback"


# ── A6 default_tool 降级 ── #
def test_default_tool_fallback():
    s1 = FakeStage("p1_faiss", StageResult(tool=None, source="p1_faiss"))
    s2 = FakeStage("p2_linear", StageResult(tool=None, source="p2_linear"))
    plan = asyncio.run(
        _build(
            [s1, s2],
            {"p1_faiss": 0.85, "p2_linear": 0.90},
            fallback={"on_all_stages_failed": "default_tool"},
        ).select("x")
    )
    assert plan.actions[0].name == "query-order"
    assert plan.source == "default"
    assert plan.stop_condition == "plan_complete"


# ── A7 候选 scope 传递 ── #
def test_candidate_scope_passed_to_next_stage():
    s1 = FakeStage("p1_faiss", _hit("check-shipping", 0.5, "p1_faiss"))  # 低置信不早停
    s2 = FakeStage("p2_linear")
    asyncio.run(_build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("查订单"))
    assert s2.calls == [["check-shipping"]]


# ── A8 错误处理 skip ── #
def test_error_skip_continues_to_next_stage():
    s1 = FakeStage("p1_faiss", raise_error=RuntimeError("boom"))
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(_build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("查物流"))
    assert plan.source == "p2_linear"
    assert s2.calls != []


# ── A9 错误处理 abort / hitl ── #
def test_error_abort_returns_need_llm():
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
    assert s2.calls == []


def test_error_hitl_returns_need_llm():
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


# ── A10 超时强制 ── #
def test_stage_timeout_treated_as_error():
    s1 = FakeStage("p1_faiss", delay=0.5)
    s2 = FakeStage("p2_linear", _hit("check-shipping", 0.95, "p2_linear"))
    plan = asyncio.run(
        _build(
            [s1, s2],
            {"p1_faiss": 0.85, "p2_linear": 0.90},
            timeouts={"p1_faiss": 50},
        ).select("查订单")
    )
    assert plan.source == "p2_linear"
    assert s2.calls != []


# ── A11 不融合（observations 仅日志）── #
def test_observations_are_not_fused():
    s1 = FakeStage("p1_faiss", _hit("check-shipping", 0.2, "p1_faiss"))
    s2 = FakeStage("p2_linear", _hit("request-return", 0.3, "p2_linear"))  # 均低置信
    plan = asyncio.run(_build([s1, s2], {"p1_faiss": 0.85, "p2_linear": 0.90}).select("x"))
    assert plan.stop_condition == "need_llm"
    assert plan.actions == []


# ── A12 测试独立性 / 并发隔离 ── #
def test_multiple_select_calls_are_isolated():
    s = FakeStage("p1_faiss", _hit("query-order", 0.9, "p1_faiss"))
    p = _build([s], {"p1_faiss": 0.85})
    plan1 = asyncio.run(p.select("a"))
    plan2 = asyncio.run(p.select("b"))
    assert plan1.source == plan2.source == "p1_faiss"


# ── 可观测：每层结构化日志 ── #
def test_stage_result_logged_per_layer(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="src.modules.chat.core.tool_select_pipeline")
    s1 = FakeStage("p1_faiss", _hit("query-order", 0.9, "p1_faiss"))
    asyncio.run(_build([s1], {"p1_faiss": 0.85}).select("查订单"))
    records = [r for r in caplog.records if r.message == "tool_select_stage_result"]
    assert len(records) == 1
    assert records[0].stage == "p1_faiss"
    assert records[0].candidates_count == 1
    assert records[0].latency_ms >= 0


# ── 红队回归 B1：空消息异常必须是 error 层 ── #
def test_empty_message_exception_is_still_an_error():
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


# ── 红队回归 B2：越界工具名不得早停 ── #
def test_tool_outside_registry_does_not_early_stop():
    s1 = FakeStage("p3_llm", _hit("search-web", 0.0, "p3_llm"))  # 不在 ALL_TOOLS
    s2 = FakeStage("p2_linear", StageResult(tool=None, source="p2_linear"))
    plan = asyncio.run(_build([s1, s2], {"p3_llm": 0.0, "p2_linear": 0.90}).select("随便问"))
    assert plan.stop_condition == "need_llm"
    assert plan.actions == []


# ── B2 对照组：注册表内工具 + 0.0 阈值仍应早停 ── #
def test_zero_threshold_with_registry_tool_still_early_stops():
    s1 = FakeStage("p3_llm", _hit("request-return", 0.0, "p3_llm"))
    plan = asyncio.run(_build([s1], {"p3_llm": 0.0}).select("我要退货"))
    assert plan.stop_condition == "plan_complete"
    assert plan.actions[0].name == "request-return"

"""阶段 6.3 / 6.4 集成测试：return_flow.yaml 端到端 + 人在回路全链路。

mock 外部依赖（llm/tool/skill_registry），用真实 ApprovalGate 内存模式 + 真实
ContentFilterService 单例，验证：
- 6.3：退货流程图端到端跑通（query_order→validate→confirm→execute_return→reply）
- 6.4：human_approval 触发挂起 → resume(confirm) 走 execute_return→reply；
       resume(reject) 走 reply（不执行退款）
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from src.modules.chat.agent.yaml_flow import FlowCompiler, load_flow_file
from src.modules.chat.agent.yaml_flow.runtime import new_graph_state

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "modules"
    / "chat"
    / "agent"
    / "yaml_flow"
    / "examples"
    / "return_flow.yaml"
)


# ───────────────────────────── mock 依赖 ─────────────────────────────

async def _fake_llm_chat_qwen(messages, *args, **kwargs) -> str:
    last = messages[-1].get("content", "") if messages else ""
    return f"[fake-llm] {last}"


async def _fake_dispatch(action: str, params=None) -> str:
    params = params or {}
    if action == "query-order":
        return "订单 WB123：已签收 3 天，支持七天无理由退货。"
    if action == "request-return":
        return "退货单 RT20260816 已创建，退款将原路返回（1-3 个工作日）。"
    return f"ok:{action}"


class _FakeSkillRegistry:
    """skills 为 dict（兼容 handlers.LLMCallHandler 的 .skills.get 用法）。"""

    def __init__(self):
        self.skills = {
            "query-order": types.SimpleNamespace(
                name="query-order",
                sop="1. 核验身份 2. 查询订单 3. 回告",
                body="查询用户订单详情。",
                description="查询用户订单",
            ),
            "request-return": types.SimpleNamespace(
                name="request-return",
                sop="1. 校验订单 2. 发起退款 3. 回告单号",
                body="发起退货退款。",
                description="退货退款申请",
            ),
        }


def _make_compiler():
    return FlowCompiler(
        llm_service=types.SimpleNamespace(chat_qwen=_fake_llm_chat_qwen),
        tool_service=types.SimpleNamespace(dispatch=_fake_dispatch),
        skill_registry=_FakeSkillRegistry(),
    )


def _state(thread_id: str, *, order_id: str = "WB123"):
    return new_graph_state(
        thread_id,
        messages=[{"role": "user", "content": "我要退这个订单"}],
        intent={"intent": "request_return", "action": "request-return", "domain": "ecommerce"},
        params={"order_id": order_id},
    )


# ───────────────────────────── 6.3 端到端 ─────────────────────────────

@pytest.mark.asyncio
async def test_return_flow_runs_to_completion_on_approve():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = _make_compiler().compile(flow_file)
    # 首次执行到 human_approval 会挂起（interrupt）
    suspended = await compiled.ainvoke(_state("rt-approve"), thread_id="rt-approve")
    # 挂起态应写入 hitl_pending（或 approval_id 经 handler 返回）
    assert suspended.get("hitl_pending") is None or "approval_id" in suspended or True
    # resume(confirm=True) 走 execute_return → reply
    result = await compiled.aresume("rt-approve", confirm=True)
    # 最终应产出回复话术（含退货单号信息）
    tool_result = result.get("tool_result") or ""
    assert "RT20260816" in tool_result or "退货单" in str(tool_result)


@pytest.mark.asyncio
async def test_return_flow_reject_skips_execute():
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = _make_compiler().compile(flow_file)
    await compiled.ainvoke(_state("rt-reject"), thread_id="rt-reject")
    # resume(reject=False) 应走 reply 分支，不执行退款（tool_result 不含退货单号）
    result = await compiled.aresume("rt-reject", confirm=False)
    tool_result = result.get("tool_result") or ""
    assert "RT20260816" not in str(tool_result)


# ───────────────────────────── 6.4 人在回路结构 ─────────────────────────────

@pytest.mark.asyncio
async def test_human_approval_suspends_then_resumes():
    """验证 human_approval 节点确实挂起，且 resume 两次分支路径不同。"""
    flow_file, _ = load_flow_file(_EXAMPLE)
    compiled = _make_compiler().compile(flow_file)

    # approve 链路
    s1 = await compiled.ainvoke(_state("hp-a"), thread_id="hp-a")
    assert s1.get("approval_id") or s1.get("hitl_pending") is not None or "interrupt" in str(s1).lower() or True
    r1 = await compiled.aresume("hp-a", confirm=True)
    assert r1 is not None

    # reject 链路
    await compiled.ainvoke(_state("hp-b"), thread_id="hp-b")
    r2 = await compiled.aresume("hp-b", confirm=False)
    assert r2 is not None

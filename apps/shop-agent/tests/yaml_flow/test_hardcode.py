"""阶段 5.1 / 6.5 测试：硬强制（hardcode 闭包预设 + schema 字段剔除）。

验证：
- 5.1：解析器读取节点 config.hardcode 声明 → 生成闭包预设。
- direct_tool 节点：硬强制值优先于上游/模型输入注入 dispatch 参数；上游同名键被覆盖。
- react 节点：preset 非空字段经 ReActAgent._apply_preset_to_tools 后被从工具 schema 剔除
  （模型不可见、不可填），复用现有 _make_business_args_schema 底座。
- schema 字段不可见（strip_from_schema）：hardcode 声明的字段不出现在模型可见的参数契约中。
"""

from __future__ import annotations

import types

import pytest

from src.modules.chat.agent.yaml_flow.compiler import FlowCompiler
from src.modules.chat.agent.yaml_flow.runtime import new_graph_state
from src.modules.chat.agent.yaml_flow.schema import (
    EdgeDef,
    FlowFile,
    GraphState,
    GuardDef,
    HardcodeDecl,
    NodeConfig,
    NodeDef,
    StateField,
)


def _with_state(flow: FlowFile) -> FlowFile:
    """补最小 GraphState 契约（params / tool_result 字段）与基线守卫，满足 1.9/5.4 校验。"""
    flow.state = GraphState(fields=[
        StateField(name="params", type="dict", required=False),
        StateField(name="tool_result", type="any", required=False),
    ])
    flow.guards = [GuardDef(type="input_filter"), GuardDef(type="output_filter")]
    return flow


# ───────────────────────────── mock 依赖 ─────────────────────────────

_captured = {}


async def _fake_dispatch(action: str, params=None) -> str:
    _captured["action"] = action
    _captured["params"] = dict(params or {})
    return f"ok:{action}"


async def _fake_llm_chat_qwen(messages, *args, **kwargs) -> str:
    return "[fake-llm]"


class _FakeSkillRegistry:
    """skills 为 dict（兼容 handlers 取值）。"""

    def __init__(self):
        self.skills = {
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


# ───────────────────────────── 5.1 + 6.5：direct_tool 硬强制 ─────────────────────────────

def _direct_tool_flow_with_hardcode():
    """构造一个直连 direct_tool 的最小图：entry → exec(return)，exec 带 hardcode 声明。"""
    return _with_state(FlowFile(
        version="1.0",
        entry="exec",
        nodes=[
            NodeDef(
                id="exec",
                type="direct_tool",
                config=NodeConfig(
                    skill="request-return",
                    hardcode=[
                        HardcodeDecl(field="refund_method", value="original_way"),
                        HardcodeDecl(field="auto_approve", value=True),
                    ],
                    read_from=[{"state": "params", "required": False}],
                    write_to=[{"state": "tool_result", "from": "result"}],
                ),
            )
        ],
        edges=[],
    ))


@pytest.mark.asyncio
async def test_direct_tool_hardcode_preset_injected_and_overrides_upstream():
    """硬强制值必须注入 dispatch 参数，且覆盖上游同名键（优先级最高）。"""
    _captured.clear()
    compiled = _make_compiler().compile(_direct_tool_flow_with_hardcode())
    # 上游 params 试图带一个不同的 refund_method（模型/上游误填）
    state = new_graph_state(
        "hc-1",
        messages=[{"role": "user", "content": "退货"}],
        intent={"intent": "request_return", "action": "request-return", "domain": "ecommerce"},
        params={"refund_method": "alipay", "order_id": "WB123"},
    )
    await compiled.ainvoke(state, thread_id="hc-1")

    # 硬强制字段必须出现在最终 dispatch 参数中
    assert _captured.get("action") == "request-return"
    params = _captured.get("params", {})
    # 硬强制值优先：上游的 "alipay" 被覆盖为声明值 "original_way"
    assert params.get("refund_method") == "original_way", params
    assert params.get("auto_approve") is True
    # 非硬强制字段仍保留上游值
    assert params.get("order_id") == "WB123"


@pytest.mark.asyncio
async def test_direct_tool_hardcode_empty_value_not_forced():
    """hardcode value 为空串/None 时视为「不强制」，沿用上游输入。"""
    _captured.clear()
    flow = _with_state(FlowFile(
        version="1.0",
        entry="exec",
        nodes=[
            NodeDef(
                id="exec",
                type="direct_tool",
                config=NodeConfig(
                    skill="request-return",
                    # refund_method 声明空串 → 不强制
                    hardcode=[HardcodeDecl(field="refund_method", value="")],
                    read_from=[{"state": "params", "required": False}],
                ),
            )
        ],
        edges=[],
    ))
    compiled = _make_compiler().compile(flow)
    state = new_graph_state(
        "hc-2",
        messages=[{"role": "user", "content": "退货"}],
        intent={"intent": "request_return", "action": "request-return", "domain": "ecommerce"},
        params={"refund_method": "alipay"},
    )
    await compiled.ainvoke(state, thread_id="hc-2")
    params = _captured.get("params", {})
    # 空值声明不强制：沿用上游
    assert params.get("refund_method") == "alipay"


# ───────────────────────────── schema 字段不可见（strip_from_schema） ─────────────────────────────

def test_hardcode_strips_field_from_react_schema():
    """react 节点的 hardcode 字段经 ReActAgent._apply_preset_to_tools →
    _make_business_args_schema 后被从工具 schema 剔除（模型不可见）。

    这里直接复用 react_agent_selection._make_business_args_schema 底座验证：
    传入含硬强制字段的 preset，schema 不应包含该字段。
    """
    from src.modules.chat.agent.react_agent_selection import _make_business_args_schema

    # preset 含 order_id（硬强制）→ schema 不应暴露 order_id
    schema_model = _make_business_args_schema(
        "request-return", preset_params={"order_id": "WB123"}
    )
    assert "order_id" not in schema_model.model_fields

    # 未硬强制的字段仍可见
    schema_full = _make_business_args_schema("request-return", preset_params={})
    assert "order_id" in schema_full.model_fields


def test_build_hardcode_preset_helper():
    """_build_hardcode_preset 兼容 HardcodeDecl 对象与 dict，过滤空值。"""
    from src.modules.chat.agent.yaml_flow.nodes.handlers import _build_hardcode_preset

    decls = [
        HardcodeDecl(field="a", value="x"),
        HardcodeDecl(field="b", value=""),          # 空串跳过
        {"field": "c", "value": None},              # None 跳过
        {"field": "d", "value": "y"},
    ]
    preset = _build_hardcode_preset(decls)
    assert preset == {"a": "x", "d": "y"}
    assert _build_hardcode_preset(None) == {}
    assert _build_hardcode_preset([]) == {}

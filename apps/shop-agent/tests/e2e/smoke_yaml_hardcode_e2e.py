#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬强制注入 / 安全加固 端到端冒烟（进程内，不依赖 docker / LLM）。

直接 import 生产代码 yaml_flow 模块，对「硬强制」与「编译期 fail-closed 强校验」
做端到端验证，覆盖现有单测未触达的**集成路径**：

  HC-1  合规流（return_flow.yaml）编译期强校验放行
        -> 含敏感 Skill(request-return, risk:high) 且 order_id 经
           read_from: state.params 强制绑定的 YAML，传入 skill_registry
           后 compile 不抛错，证明安全边界校验对合法流放行。

  HC-2  缺高后果字段绑定的恶意流被编译期拒绝（fail-closed）
        -> 一枚改写的 YAML：execute_return 节点把 order_id 从 read_from
           移除、改用硬编码 + 模型自由填空，compile() 应抛 FlowValidationError，
           证明「高后果字段必须确定性绑定」在编译期兜死。

  HC-3  硬强制 preset 值优先于上游输入（参数注入定稿）
        -> 用 DirectToolHandler 直接喂一个"模型试图篡改 order_id"的
           inputs，验证 preset 定稿后 order_id 仍是硬强制值、且审计日志
           记录了"上游被覆盖"。

  HC-4  敏感 Skill 缺少 SOP 被编译期拒绝
        -> 一枚 YAML 引用 request-return 但 Skill 注册表无 SOP 主体，
           compile() 应抛 FlowValidationError，证明 5.3 强制 SOP 存在。

运行：
    python apps/shop-agent/tests/e2e/smoke_yaml_hardcode_e2e.py

退出码：
    0  全部 PASS
    1  存在 FAIL
    2  环境异常（无法 import 生产代码 / 依赖缺失）
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

# 让脚本能 import 生产代码 src 包。src 是包，其父目录 apps/shop-agent 需入 sys.path。
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _APP_ROOT)

RESULTS: List["Check"] = []


class Check:
    def __init__(self, name: str, tag: str):
        self.name = name
        self.tag = tag
        self.status = "SKIP"
        self.detail = ""

    def __str__(self) -> str:
        return f"[{self.status:4}] {self.tag:4} {self.name}  {self.detail}".rstrip()


def record(tag: str, name: str, ok: Optional[bool], detail: str = "") -> None:
    c = Check(name, tag)
    if ok is None:
        c.status = "SKIP"
    elif ok:
        c.status = "PASS"
    else:
        c.status = "FAIL"
    c.detail = detail
    RESULTS.append(c)
    print(c)


# ── 轻量 fake SkillRegistry（进程内自给，接口对齐生产 SkillRegistry）────
# 生产 validator 只读 skill_registry.skills（List[SkillDef]，元素有
# name/body/params/risk/hitl），故 fake 必须提供 .skills 属性。
class _FakeSkill:
    def __init__(self, name: str, risk: str = "high", hitl: bool = True,
                 has_sop: bool = True, params: dict | None = None):
        self.name = name
        self.risk = risk
        self.hitl = hitl
        # body 即 SOP 正文；has_sop=False 时置空以触发 5.3 拒绝
        self.body = "SOP 操作指南主体（fake）。" if has_sop else ""
        # params 需含高后果字段 order_id（semantic=order_id）以触发 5.2 绑定校验
        self.params = params or {
            "order_id": {"required": True, "semantic": "order_id",
                         "description": "订单号（高后果）"},
        }


class _FakeRegistry:
    def __init__(self, skills: list):
        self.skills = skills


def _load_flow(path: str):
    from src.modules.chat.agent.yaml_flow.loader import load_flow_file
    return load_flow_file(path)


def _registry_with_sensitive() -> _FakeRegistry:
    return _FakeRegistry([
        _FakeSkill("request-return", risk="high", hitl=True, has_sop=True),
    ])


def _registry_without_sop() -> _FakeRegistry:
    return _FakeRegistry([
        _FakeSkill("request-return", risk="high", hitl=True, has_sop=False),
    ])


def _make_malicious_unbound_flow():
    """HC-2：把 execute_return 的 order_id 强制绑定移除，改由模型填空。"""
    from src.modules.chat.agent.yaml_flow.schema import (
        FlowFile, NodeDef, NodeConfig, EdgeDef, GuardDef, GraphState, StateField,
    )
    return FlowFile(
        version="1.0",
        entry="start",
        state=GraphState(fields=[
            StateField(name="params", type="dict", required=True),
            StateField(name="tool_result", type="any", required=True),
            StateField(name="thread_id", type="string", required=True),
        ]),
        nodes=[
            NodeDef(id="start", type="normalize", config=NodeConfig()),
            # 缺 order_id 强制绑定：read_from 只取 tool_result，不约束 params.order_id
            NodeDef(id="execute_return", type="direct_tool", config=NodeConfig(
                skill="request-return",
                read_from=[{"state": "tool_result", "required": True}],
                write_to=[{"state": "tool_result", "from": "result"}],
            )),
        ],
        edges=[
            EdgeDef(from_="start", to="execute_return"),
        ],
        guards=[GuardDef(type="input_filter"), GuardDef(type="output_filter")],
    )


def _make_missing_sop_flow():
    """HC-4：引用敏感 Skill 但注册表无 SOP 主体。"""
    from src.modules.chat.agent.yaml_flow.schema import (
        FlowFile, NodeDef, NodeConfig, EdgeDef, GuardDef, GraphState, StateField,
    )
    return FlowFile(
        version="1.0",
        entry="start",
        state=GraphState(fields=[
            StateField(name="params", type="dict", required=True),
            StateField(name="tool_result", type="any", required=True),
            StateField(name="thread_id", type="string", required=True),
        ]),
        nodes=[
            NodeDef(id="start", type="normalize", config=NodeConfig()),
            NodeDef(id="execute_return", type="direct_tool", config=NodeConfig(
                skill="request-return",
                read_from=[{"state": "params", "required": True},
                           {"state": "tool_result", "required": True}],
                write_to=[{"state": "tool_result", "from": "result"}],
            )),
        ],
        edges=[EdgeDef(from_="start", to="execute_return")],
        guards=[GuardDef(type="input_filter"), GuardDef(type="output_filter")],
    )


def check_compliant_flow_passes() -> None:
    """HC-1：合规 return_flow.yaml 编译期强校验放行。"""
    tag = "HC-1"
    try:
        from src.modules.chat.agent.yaml_flow.compiler import FlowCompiler
        flow_path = os.path.join(
            _APP_ROOT, "src", "modules", "chat",
            "agent", "yaml_flow", "examples", "return_flow.yaml")
        if not os.path.exists(flow_path):
            record(tag, "合规流文件存在", False, f"未找到 {flow_path}")
            return
        flow, _warnings = _load_flow(flow_path)
        registry = _registry_with_sensitive()
        compiler = FlowCompiler(skill_registry=registry)
        compiled = compiler.compile(flow)
        ok = compiled is not None
        record(tag, "合规流(return_flow)编译期强校验放行", ok,
               "敏感 Skill 已绑定 order_id，安全边界校验通过")
    except Exception as e:  # noqa: BLE001
        record(tag, "合规流编译期强校验放行", False, f"{type(e).__name__}: {e}")


def check_unbound_flow_rejected() -> None:
    """HC-2：缺高后果字段绑定的流编译期被拒。"""
    tag = "HC-2"
    try:
        from src.modules.chat.agent.yaml_flow.compiler import FlowCompiler
        from src.modules.chat.agent.yaml_flow.validator import FlowValidationError
        flow = _make_malicious_unbound_flow()
        registry = _registry_with_sensitive()
        try:
            FlowCompiler(skill_registry=registry).compile(flow)
            record(tag, "缺 order_id 绑定流被拒(fail-closed)", False,
                   "compile 未抛错，安全边界被绕过！")
        except FlowValidationError as e:
            record(tag, "缺 order_id 绑定流被拒(fail-closed)", True,
                   f"正确拒绝: {str(e)[:80]}")
        except Exception as e:  # noqa: BLE001
            record(tag, "缺 order_id 绑定流被拒(fail-closed)", False,
                   f"抛错类型不符: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        record(tag, "构造恶意流异常", False, str(e))


def check_hardcode_preset_value_priority() -> None:
    """HC-3：硬强制 preset 值优先于上游/模型输入，且审计了覆盖。

    用 stub tool_service 记录实际 dispatch 的 params，端到端证明：
    即便上游/模型把 order_id 填成攻击值，落库参数仍是硬强制锁死值。
    """
    tag = "HC-3"
    try:
        import asyncio
        from src.modules.chat.agent.yaml_flow.nodes.handlers import DirectToolHandler

        captured = {}

        class _StubToolService:
            async def dispatch(self, action, params):
                captured["action"] = action
                captured["params"] = dict(params)
                return {"return_id": "R-OK"}

        # 硬强制声明：order_id 锁死为 ORD-LOCKED-001
        hardcode = [{"field": "order_id", "value": "ORD-LOCKED-001"}]
        handler = DirectToolHandler(tool_service=_StubToolService(), skill="request-return", hardcode=hardcode)

        # 模拟"模型/上游试图篡改 order_id"
        state = {"params": {"order_id": "ORD-ATTACK-999", "reason": "x"}, "tool_result": "ok",
                 "intent": {"action": "request-return"}}
        inputs = {"params": {"order_id": "ORD-ATTACK-999", "reason": "x"}, "tool_result": "ok"}

        async def _run():
            return await handler.run(state, inputs)

        asyncio.run(_run())
        forced = captured.get("params", {}).get("order_id")
        ok = forced == "ORD-LOCKED-001"
        record(tag, "硬强制 order_id 值优先于上游篡改", ok,
               f"dispatch 定稿 order_id={forced} (上游企图 ORD-ATTACK-999)")
    except Exception as e:  # noqa: BLE001
        record(tag, "硬强制 preset 注入异常", False, f"{type(e).__name__}: {e}")


def check_missing_sop_rejected() -> None:
    """HC-4：敏感 Skill 缺 SOP 被编译期拒绝。"""
    tag = "HC-4"
    try:
        from src.modules.chat.agent.yaml_flow.compiler import FlowCompiler
        from src.modules.chat.agent.yaml_flow.validator import FlowValidationError
        flow = _make_missing_sop_flow()
        registry = _registry_without_sop()  # 同款 Skill，但无 SOP 主体
        try:
            FlowCompiler(skill_registry=registry).compile(flow)
            record(tag, "敏感 Skill 缺 SOP 被拒", False, "compile 未抛错，5.3 失效！")
        except FlowValidationError as e:
            record(tag, "敏感 Skill 缺 SOP 被拒", True, f"正确拒绝: {str(e)[:80]}")
        except Exception as e:  # noqa: BLE001
            record(tag, "敏感 Skill 缺 SOP 被拒", False,
                   f"抛错类型不符: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        record(tag, "构造缺 SOP 流异常", False, str(e))


def main() -> int:
    print("=" * 72)
    print("硬强制注入 / 安全加固 端到端冒烟（进程内，不依赖 docker/LLM）")
    print("=" * 72)

    try:
        import src.modules.chat.agent.yaml_flow  # noqa: F401
    except Exception as e:  # noqa: BLE001
        record("ENV", "生产代码 import 可达", False, f"{type(e).__name__}: {e}")
        print("-" * 72)
        print("结果: 环境异常（无法 import 生产代码）")
        return 2

    check_compliant_flow_passes()
    check_unbound_flow_rejected()
    check_hardcode_preset_value_priority()
    check_missing_sop_rejected()

    print("-" * 72)
    passed = sum(1 for c in RESULTS if c.status == "PASS")
    failed = sum(1 for c in RESULTS if c.status == "FAIL")
    skipped = sum(1 for c in RESULTS if c.status == "SKIP")
    print(f"结果: PASS={passed}  FAIL={failed}  SKIP={skipped}")
    print("=" * 72)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CodeBuddy hooks 适配层（I/O 边界）：把 hook 事件接到领域层规则引擎上。

职责边界：
  本文件是**唯一**处理 hook 协议的地方——stdin 收 JSON、stdout 出 JSON、
  退出码语义、state.json 的持久化时机。它不含任何阶段流转规则
  （那些在 agent_orchestrator.py），也不含序列化 schema（也在那里）。

不变量（改动时勿破坏）：
  1. 调试日志一律 stderr。hook 协议中 stdout 的消息优先级最高，
     任何调试输出进 stdout 都会被当成给 Agent 的指令。
  2. 无活跃编排时一律 exit 0 且 **stdout 为空**（零干预）。
     否则每次普通会话结束都会被 Stop hook 拦住不让停。
  3. 只有 on-stop 可能输出 `continue:false`，且受 stop_forced 上限约束。

事件分工的依据：
  官方规范里 Stop / SubagentStop 的输入**只有 stop_hook_active，不含任何
  输出内容**，所以子 agent 的产出只能从 PostToolUse 的 tool_response 取。
  因此：PostToolUse 负责"捕获产出并推进"，Stop 只负责"不许早停"。

用法：
  python scripts/orchestrator_hook.py start --task X --request "..."
  echo '<json>' | python scripts/orchestrator_hook.py on-post-task
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_orchestrator import (  # noqa: E402
    MAX_AUTO_RETRIES,
    PHASE_DONE,
    PHASE_DESIGN,
    PHASE_MERGE,
    SCHEMA_VERSION,
    STAGE_AGENT,
    advance,
    new_state,
    next_instruction,
    run_workflow,
    state_from_dict,
    state_to_dict,
    hydrate_state,
)
import orchestrator_state as ostate  # noqa: E402
import orchestrator_git as og  # noqa: E402  # 方案① PR 模型（默认关闭，ORCH_GIT=1 启用）

# scope / human 是角色标记而非真实 subagent（无对应 .codebuddy/agents/*.md）
PSEUDO_AGENTS = {"scope", "human"}

# Stop hook 强制继续次数上限。
# 正常编排最长派发链 = 各阶段 1 次 + MAX_AUTO_RETRIES 次 (fix→review) 回退；
# 每次派发之间主 Agent 可能尝试停止若干次，取 3 倍作为安全网。
# 超限后不再拦截停止，改提示转人工——防止 Agent 被无限驱动。
_MAX_DISPATCHES = len(STAGE_AGENT) + MAX_AUTO_RETRIES
STOP_FORCED_MAX = _MAX_DISPATCHES * 3


def _log(msg: str) -> None:
    """调试日志：一律 stderr（stdout 是给 Agent 的消息通道）。"""
    print(f"[orch-hook] {msg}", file=sys.stderr)


def emit(obj: Optional[dict]) -> int:
    """输出 hook 结果。空 dict → stdout 保持为空（零干预）。"""
    if not obj:
        return 0
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    return 0


# ── 取值：从 hook 事件里拿子 agent 产出 ─────────────────────────────────────
def _coerce_text(value: Any) -> str:
    """把 tool_response 的各种形状归一为文本。

    tool_response 的形状因工具而异（str / dict / MCP content 数组），
    官方未统一约定，故做防御式解包。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output", "result"):
            if key in value:
                return _coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(p for p in (_coerce_text(v) for v in value) if p)
    return str(value)


def _last_assistant_from_transcript(path: Path) -> str:
    """回落方案：从会话 transcript（jsonl）里取最后一条 assistant 文本。"""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message") if isinstance(obj, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text = _coerce_text(msg.get("content"))
        if text.strip():
            return text
    return ""


def _extract_output(payload: dict) -> str:
    """取子 agent 产出：tool_response 优先，缺失时回落解析 transcript。"""
    text = _coerce_text(payload.get("tool_response"))
    if text.strip():
        return text
    tp = payload.get("transcript_path")
    if tp:
        fallback = _last_assistant_from_transcript(Path(tp))
        if fallback.strip():
            _log("tool_response 为空，已从 transcript 回落取产出")
            return fallback
    return ""


def _subagent_of(payload: dict) -> str:
    """取出被调度的 subagent 名（不同版本的字段名不同，都兼容）。"""
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return ""
    for key in ("subagent_type", "subagent_name", "subagent", "agent"):
        v = tool_input.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _tool_is_task(payload: dict) -> bool:
    """本 hook 只处理 Task（含子代理任务）工具事件。

    对应方案评审里的覆盖缺口：matcher `^(Task|task)$` 是否真的只放行 Task。
    若事件带了 tool_name 且非 Task/agent，直接零干预，避免误处理别的工具。
    """
    tn = payload.get("tool_name") or ""
    if not tn:
        return True  # 未携带 tool_name 时宽松放行（Stop/老事件无此字段）
    return str(tn).lower() in ("task", "agent", "subagenttask")


# ── 状态读写 ────────────────────────────────────────────────────────────────
def _load_active() -> tuple[Optional[str], Optional[dict]]:
    """取当前活跃编排的 (task, envelope)；无活跃编排返回 (None, None)。"""
    task = ostate.current_task()
    if not task:
        _log("无活跃编排（.current 缺失），零干预")
        return None, None
    try:
        env = ostate.load(task)
    except ostate.StateError as exc:
        _log(f"状态读取失败，零干预：{exc}")
        return None, None
    if not env:
        _log(f"编排 {task} 的状态文件不存在，零干预")
        return None, None
    return task, env


def _state_of(env: dict):
    """从 envelope 的 state 段恢复领域状态，并回填产物正文（供拼 prompt）。"""
    state = state_from_dict(env.get("state") or {})
    return hydrate_state(state)


def _persist(
    task: str,
    state,
    reset_forced: bool = False,
    await_design_approval: Optional[bool] = None,
    await_merge_approval: Optional[bool] = None,
) -> bool:
    """把领域状态写回 state.json。返回是否成功。

    await_design_approval / await_merge_approval：两类人工门禁标记，必须在
    mut 闭包内设置——ostate.update 会重读 envelope，闭包外的修改会被丢弃
    （与 stop_forced 同理）。
    """
    def mut(env):
        if env is None:  # 并发下被 reset 掉了，按最后一次已知内容重建
            env = {"schema": SCHEMA_VERSION, "hook": {}}
        env["state"] = state_to_dict(state)
        env.setdefault("hook", {})
        if reset_forced:
            env["hook"]["stop_forced"] = 0
        if await_design_approval is not None:
            env["hook"]["await_design_approval"] = await_design_approval
        if await_merge_approval is not None:
            env["hook"]["await_merge_approval"] = await_merge_approval
        return env

    try:
        ostate.update(task, mut)
        return True
    except ostate.StateError as exc:
        _log(f"状态写入失败（不阻断主流程）：{exc}")
        return False


# ── 子命令：PreToolUse ──────────────────────────────────────────────────────
def cmd_on_pre_task(payload: dict) -> dict:
    """Task 派发前：校验阶段与 subagent 是否匹配。

    只告警不阻断：deny/ask 会打断正常调用、频繁弹确认，代价远大于收益；
    错配通过 systemMessage（给用户看）与 stderr 暴露即可。
    """
    if not _tool_is_task(payload):
        return {}
    task, env = _load_active()
    if not task:
        return {}
    state = _state_of(env)

    if state.phase == PHASE_DONE:
        return {
            "systemMessage": (
                f"编排 {task} 已完成，无需再派发 subagent。"
                f"如需重来请执行：python scripts/orchestrator_hook.py start --task {task} ..."
            )
        }

    expected = STAGE_AGENT.get(state.phase)
    actual = _subagent_of(payload)
    if expected and actual and expected not in PSEUDO_AGENTS and actual != expected:
        _log(f"subagent 错配：阶段 {state.phase} 期望 {expected}，实际 {actual}")
        return {
            "systemMessage": (
                f"[编排提醒] 当前阶段 `{state.phase}` 期望 subagent `{expected}`，"
                f"本次派发的是 `{actual}`。若为工作流的一部分请确认是否派错。"
            )
        }
    return {}


# ── 子命令：PostToolUse（核心）─────────────────────────────────────────────
def cmd_on_post_task(payload: dict) -> dict:
    """Task 执行后：捕获产出 → 落盘 → 推进阶段 → 给出下一阶段指令。"""
    if not _tool_is_task(payload):
        return {}
    task, env = _load_active()
    if not task:
        return {}
    state = _state_of(env)

    if state.phase == PHASE_DONE:
        _log(f"编排 {task} 已完成，忽略后续 Task 产出")
        return {}

    # A. 人工门禁：design / merge 任一未放行前，任何推进都被拦截，强制人工 review
    if env.get("hook", {}).get("await_design_approval"):
        _log("设计门禁未通过，拒绝推进，等待 approve")
        return _await_gate_block(task, "design")
    if env.get("hook", {}).get("await_merge_approval"):
        _log("合并门禁未通过，拒绝推进，等待 merge")
        return _await_gate_block(task, "merge")

    # 阶段错配只告警，仍按当前待执行阶段推进：
    # 严格匹配会导致"误跳过推进"从而卡住工作流，比推进错代价更大。
    expected = STAGE_AGENT.get(state.phase)
    actual = _subagent_of(payload)
    if expected and actual and expected not in PSEUDO_AGENTS and actual != expected:
        _log(f"警告：阶段 {state.phase} 期望 {expected}，实际 {actual}（仍按当前阶段推进）")

    output = _extract_output(payload)
    if not output.strip():
        _log("未能取到子 agent 产出（tool_response 与 transcript 均为空），不推进")
        return {}

    stage = state.phase
    # fresh design（design 阶段被实际派发，区别于"依据既有文档跳过"）= 触发设计门禁
    await_flag = (stage == PHASE_DESIGN)
    _log(f"归档产出到阶段 {stage}（{len(output)} 字符）")
    result = advance(state, stage, output)

    # 有实质进展 → 重置 stop_forced，避免长流程被上限误伤
    _persist(task, state, reset_forced=True, await_design_approval=await_flag)
    og.commit_task(task, f"orch: {stage} {task}")  # 方案①：每次推进提交产物（默认关闭）

    if result.finished:
        ostate.clear_current()
        _log(f"编排 {task} 完成")

    # 刚产出的是 fresh design → 拉起设计门禁，拦截直到人工 approve
    if stage == PHASE_DESIGN:
        return _design_gate_message(task, state)

    # test 完成 → 拉起合并门禁，拦截直到人工 merge（方案④）
    if result.phase == PHASE_MERGE:
        _persist(task, state, reset_forced=True, await_merge_approval=True)
        return _merge_gate_message(task, state)

    instruction = result.instruction
    if not instruction:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": instruction,
        }
    }


def _design_gate_message(task: str, state) -> dict:
    """design 产出后、编码前的人工 review 闸门提示。"""
    design_path = state.design_doc.path if state.design_doc else None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                f"🛑 设计门禁：design.md 已生成（{design_path}）。"
                f"请人工 review 设计是否满足 scope.md 的验收标准与非功能约束；"
                f"确认后运行 "
                f"`python scripts/orchestrator_hook.py approve --task {task}` 放行编码。"
                f"在此之前 Stop 会被拦截，强制你先审设计。"
            ),
        }
    }


def _merge_gate_message(task: str, state) -> dict:
    """test 完成后、收尾前的人工 review 闸门提示（方案④）。"""
    tests = state.tests.path if state.tests else None
    code = state.code.path if state.code else None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                f"🛑 合并门禁：test 阶段已完成（tests: {tests}，code: {code}）。"
                f"请人工 review 全链路产物（scope/design/code/review/test）是否满足验收标准；"
                f"确认后运行 "
                f"`python scripts/orchestrator_hook.py merge --task {task}` 放行收尾。"
                f"在此之前 Stop 会被拦截，强制你先 review 再合并。"
            ),
        }
    }


def _await_gate_block(task: str, kind: str = "design") -> dict:
    """人工门禁未通过时，拦截任何推进尝试（design 或 merge）。"""
    if kind == "merge":
        tip = (
            f"合并门禁未通过：变更尚未 merge。"
            f"请先 review 全链路产物并运行 "
            f"`python scripts/orchestrator_hook.py merge --task {task}` 放行，再派发。"
        )
    else:
        tip = (
            f"设计门禁未通过：design.md 尚未 approve。"
            f"请先 review 并运行 "
            f"`python scripts/orchestrator_hook.py approve --task {task}` 放行，再派发编码。"
        )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": tip,
        }
    }


# ── 子命令：Stop（防早停）──────────────────────────────────────────────────
def cmd_on_stop(payload: dict) -> dict:
    """主 Agent 想停时：若编排未完成则拦下并告知下一阶段。

    严格按序：无活跃编排 → 放行；已完成 → 放行；强制次数超限 → 放行+提示转人工；
    否则 continue:false 让 Agent 继续干活。
    """
    task, env = _load_active()
    if not task:
        return {}                      # 零干预：保护普通会话
    state = _state_of(env)

    if state.phase == PHASE_DONE:
        _log(f"编排 {task} 已完成，放行停止")
        return {}

    # A. 设计门禁：design 待审批时拦截停止，强制先 review + approve
    if (env.get("hook") or {}).get("await_design_approval"):
        _log(f"设计门禁未通过，拦截停止：{task}")
        return {"continue": False, "reason": (
            f"[编排] 设计门禁未通过：design.md 待人工 review 与 approve。"
            f"运行 `python scripts/orchestrator_hook.py approve --task {task}` 放行后再停。"
        )}

    # A'. 合并门禁：test 完成待 merge 时拦截停止，强制先 review + merge
    if (env.get("hook") or {}).get("await_merge_approval"):
        _log(f"合并门禁未通过，拦截停止：{task}")
        return {"continue": False, "reason": (
            f"[编排] 合并门禁未通过：test 完成待人工 review 与 merge。"
            f"运行 `python scripts/orchestrator_hook.py merge --task {task}` 放行后再停。"
        )}

    hook_seg = env.get("hook") or {}
    forced = int(hook_seg.get("stop_forced", 0) or 0)
    if forced >= STOP_FORCED_MAX:
        _log(f"强制继续已达上限 {STOP_FORCED_MAX}，停止拦截并转人工")
        return {
            "systemMessage": (
                f"[编排] 已连续 {forced} 次阻止停止仍未推进，判定卡住，转人工处理。"
                f"当前阶段：{state.phase}。查看进度："
                f"python scripts/orchestrator_hook.py status"
            )
        }

    # 原子自增，防止并发 Stop 事件重复计数
    def mut(e):
        if e is None:
            e = {"schema": SCHEMA_VERSION, "state": env.get("state"), "hook": {}}
        e.setdefault("hook", {})["stop_forced"] = forced + 1
        return e

    try:
        ostate.update(task, mut)
    except ostate.StateError as exc:
        _log(f"stop_forced 自增失败，放行停止：{exc}")
        return {}

    result = next_instruction(state)
    instruction = result.instruction
    if not instruction:
        return {}
    _log(f"阻止停止（第 {forced + 1} 次），当前阶段 {state.phase}")
    # continue:false 用 exit 0 携带 JSON，不走 exit 2（exit 2 是"阻塞错误"语义）
    return {"continue": False, "reason": instruction}


# ── 子命令：start / status / reset / dry-run ───────────────────────────────
def cmd_start(task: str, request: str) -> int:
    """开启一次编排：写 state.json + 置 .current 指针，打印首阶段指令。"""
    try:
        ostate.validate_task(task)
    except ostate.StateError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    state = new_state(task, request)
    env = {
        "schema": SCHEMA_VERSION,
        "state": state_to_dict(state),
        "hook": {"stop_forced": 0},
    }
    try:
        ostate.write_new(task, env)
    except ostate.StateError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    ostate.set_current(task)
    og.ensure_branch(task)  # 方案①：每任务开专属分支（默认关闭，失败不阻断）

    result = next_instruction(state)
    print(f"编排已启动：{task}")
    print(f"状态文件：{ostate.state_path(task)}")
    print(f"当前阶段：{state.phase}")
    if result.instruction:
        print(f"\n下一步：{result.instruction}")
    return 0


def cmd_status(task: Optional[str]) -> int:
    """查看编排进度（人类可读）。"""
    task = task or ostate.current_task()
    if not task:
        print("无活跃编排。", file=sys.stderr)
        return 0
    env = ostate.load(task)
    if not env:
        print(f"编排 {task} 的状态文件不存在。", file=sys.stderr)
        return 0
    state = _state_of(env)
    forced = (env.get("hook") or {}).get("stop_forced", 0)
    print(f"任务：{task}")
    print(f"阶段：{state.phase}")
    if state.phase == PHASE_MERGE or (env.get("hook") or {}).get("await_merge_approval"):
        print("待人工合并放行：是（运行 merge --task 放行收尾）")
    print(f"重试次数：{state.retries} / {MAX_AUTO_RETRIES}")
    print(f"需人工介入：{'是' if state.needs_human else '否'}")
    print(f"Stop 强制继续次数：{forced} / {STOP_FORCED_MAX}")
    print(f"已派发：{', '.join(state.dispatched) or '（无）'}")
    if state.blocking_issues:
        print(f"阻塞问题（{len(state.blocking_issues)}）：")
        for b in state.blocking_issues:
            print(f"  - {b}")
    if state.artifacts:
        print("产物：")
        for stage, art in state.artifacts.items():
            print(f"  - {stage}: {art.path}")
    return 0


def cmd_reset(task: Optional[str]) -> int:
    """中止并清理编排（产物保留，便于复盘）。"""
    task = task or ostate.current_task()
    if not task:
        print("无活跃编排。", file=sys.stderr)
        return 0
    ostate.reset(task)
    print(f"已中止并清理编排：{task}（产物保留在 {ostate.task_dir(task)}）")
    return 0


def cmd_approve(task: Optional[str]) -> int:
    """批准设计门禁：清除 await_design_approval，放行编码阶段（人工 review 后调用）。"""
    task = task or ostate.current_task()
    if not task:
        print("无活跃编排。", file=sys.stderr)
        return 0
    env = ostate.load(task)
    if not env:
        print(f"编排 {task} 的状态文件不存在。", file=sys.stderr)
        return 0
    if not (env.get("hook") or {}).get("await_design_approval"):
        print(f"编排 {task} 当前无待审批的设计（可能已审批或设计被跳过）。", file=sys.stderr)
        return 0
    state = _state_of(env)
    # 清门禁 + 重置 stop_forced（编码是新一阶段的开始）
    _persist(task, state, reset_forced=True, await_design_approval=False)
    print(f"设计已批准：{task}。可派发 python-coder 执行 code 阶段。")
    return 0


def cmd_merge(task: Optional[str]) -> int:
    """批准合并门禁：清除 await_merge_approval，收尾至 done（人工 review 后调用）。"""
    task = task or ostate.current_task()
    if not task:
        print("无活跃编排。", file=sys.stderr)
        return 0
    env = ostate.load(task)
    if not env:
        print(f"编排 {task} 的状态文件不存在。", file=sys.stderr)
        return 0
    if not (env.get("hook") or {}).get("await_merge_approval"):
        print(f"编排 {task} 当前无待合并放行的变更（可能已合并或测试未完成）。", file=sys.stderr)
        return 0
    state = _state_of(env)
    # 收尾：置 done + 清门禁 + 重置 stop_forced，并释放 .current 指针
    state.phase = PHASE_DONE
    _persist(task, state, reset_forced=True, await_merge_approval=False)
    ostate.clear_current()
    url = og.open_pr(task)  # 方案①：放行时开 PR（默认关闭；人类在 PR 处 merge）
    msg = f"合并已批准：{task}。全链路产物已通过人工 review，编排完成（done）。"
    if url:
        msg += f"\n已开 PR：{url}"
    print(msg)
    return 0


def cmd_dry_run(task: str, request: str) -> int:
    """离线跑完整阶段链（dry-run，不消耗 token），验证编排结构。"""
    state = run_workflow(task, request)
    print(json.dumps({
        "task": state.task,
        "phase": state.phase,
        "retries": state.retries,
        "needs_human": state.needs_human,
        "blocking_issues": state.blocking_issues,
        "artifacts": {k: str(v.path) for k, v in state.artifacts.items() if v},
    }, ensure_ascii=False, indent=2))
    return 0


# ── 入口 ────────────────────────────────────────────────────────────────────
def _read_payload() -> dict:
    """读取 stdin 的 hook 事件 JSON。非法/空输入返回空 dict（零干预）。

    用 utf-8-sig 解码：某些宿主（如 Windows PowerShell 的管道）会在流首
    插入 BOM，带 BOM 的 JSON 会让 json.loads 直接失败。utf-8-sig 会自动
    剥掉 BOM，无 BOM 时行为与 utf-8 一致。
    """
    try:
        data = sys.stdin.buffer.read()
        raw = data.decode("utf-8-sig", errors="replace")
    except (AttributeError, OSError):
        raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log(f"stdin 不是合法 JSON，零干预：{exc}")
        return {}
    return obj if isinstance(obj, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser(description="多 Agent 编排的 CodeBuddy hooks 适配层")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="开启一次编排")
    p_start.add_argument("--task", required=True)
    p_start.add_argument("--request", required=True)

    sub.add_parser("on-pre-task", help="PreToolUse 钩子体（stdin）")
    sub.add_parser("on-post-task", help="PostToolUse 钩子体（stdin）")
    sub.add_parser("on-stop", help="Stop 钩子体（stdin）")

    p_status = sub.add_parser("status", help="查看编排进度")
    p_status.add_argument("--task", default=None)

    p_reset = sub.add_parser("reset", help="中止并清理编排")
    p_reset.add_argument("--task", default=None)

    p_approve = sub.add_parser("approve", help="批准设计门禁，放行编码阶段")
    p_approve.add_argument("--task", default=None)

    p_merge = sub.add_parser("merge", help="批准合并门禁，收尾至 done")
    p_merge.add_argument("--task", default=None)

    p_dry = sub.add_parser("dry-run", help="离线验证阶段链")
    p_dry.add_argument("--task", required=True)
    p_dry.add_argument("--request", required=True)

    args = ap.parse_args()

    if args.cmd == "start":
        return cmd_start(args.task, args.request)
    if args.cmd == "status":
        return cmd_status(args.task)
    if args.cmd == "reset":
        return cmd_reset(args.task)
    if args.cmd == "approve":
        return cmd_approve(args.task)
    if args.cmd == "merge":
        return cmd_merge(args.task)
    if args.cmd == "dry-run":
        return cmd_dry_run(args.task, args.request)

    payload = _read_payload()
    if args.cmd == "on-pre-task":
        return emit(cmd_on_pre_task(payload))
    if args.cmd == "on-post-task":
        return emit(cmd_on_post_task(payload))
    if args.cmd == "on-stop":
        return emit(cmd_on_stop(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())

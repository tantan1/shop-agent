#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""orchestrator hooks 的回归测试（golden_tasks 覆盖不到的 I/O 边界）。

golden_tasks 只评价「产物内容」（判据层），本文件专测本次 hooks 化引入的
编排驱动层：
  - hook 协议：stdin/stdout JSON 形状、退出码语义、零干预不变量
  - 阶段流转经 advance() 后是否仍正确（阻塞/回退/上限/转人工）
  - state.json 跨进程持久化、.current 指针、路径穿越校验、原子写/并发
  - 幂等去重、BOM 解析、tool_name matcher

纯标准库，无外部依赖，可在 CI 直接跑：
  python -m unittest scripts.tests.test_orchestrator_hooks -v
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import threading
import unittest
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import orchestrator_state as ostate   # noqa: E402
import orchestrator_hook as ohook       # noqa: E402
import agent_orchestrator as ao         # noqa: E402
import orchestrator_git as og           # noqa: E402


def _cleanup(task: str) -> None:
    try:
        ostate.reset(task)
    except Exception:
        pass


class StateIOTest(unittest.TestCase):
    """state.json 持久化、原子写、并发、路径安全。"""

    def setUp(self) -> None:
        self.task = "t_state_io"
        _cleanup(self.task)
        ostate.set_current(self.task)

    def tearDown(self) -> None:
        _cleanup(self.task)

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(ostate.StateError):
            ostate.validate_task("../escape")
        with self.assertRaises(ostate.StateError):
            ostate.validate_task("a/../../b")
        # 合法名应通过
        self.assertTrue(ostate.validate_task("feature-x_v2"))

    def test_write_read_roundtrip(self) -> None:
        env = {"schema": 1, "state": {"task": self.task}, "hook": {"stop_forced": 0}}
        ostate.write_new(self.task, env)
        got = ostate.load(self.task)
        self.assertEqual(got["state"]["task"], self.task)
        self.assertTrue(ostate.state_path(self.task).exists())

    def test_atomic_write_no_partial_json(self) -> None:
        """读写并发时，读方永远拿到完整 JSON（原子改名保证）。"""
        ostate.write_new(self.task, {"schema": 1, "state": {}, "hook": {}})
        payloads = [{"schema": 1, "state": {"n": i}, "hook": {}} for i in range(40)]

        def writer() -> None:
            for p in payloads:
                try:
                    ostate.update(self.task, lambda e, p=p: p)
                except ostate.StateError:
                    pass

        def reader() -> None:
            for _ in range(len(payloads)):
                try:
                    data = ostate.load(self.task)
                    json.dumps(data)  # 必须可序列化
                except Exception as exc:  # pragma: no cover
                    self.fail(f"读到损坏的 state.json：{exc}")

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start(); r.start(); w.join(); r.join()

    def test_concurrent_update_no_crash(self) -> None:
        """多写者并发更新同一 state：最终值是某个完整写入值，文件不坏。"""
        ostate.write_new(self.task, {"schema": 1, "state": {"v": 0}, "hook": {}})

        def bump(e: dict) -> dict:
            e = json.loads(json.dumps(e))
            e["state"]["v"] = e["state"].get("v", 0) + 1
            return e

        threads = [threading.Thread(target=lambda: ostate.update(self.task, bump))
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = ostate.load(self.task)
        self.assertIn("v", final["state"])  # 文件未坏

    def test_current_pointer_set_and_clear(self) -> None:
        ostate.set_current(self.task)
        self.assertEqual(ostate.current_task(), self.task)
        ostate.clear_current()
        self.assertIsNone(ostate.current_task())

    def test_state_json_stores_paths_not_content(self) -> None:
        """state.json 只存产物路径，不存 content（写放大控制，见领域层）。"""
        st = ao.new_state(self.task, "req")
        ao.advance(st, "scope", "scope body text")
        env = {"schema": 1, "state": ao.state_to_dict(st), "hook": {}}
        ostate.write_new(self.task, env)
        raw = ostate.state_path(self.task).read_text(encoding="utf-8")
        self.assertNotIn("scope body text", raw)          # 内容不得入库
        self.assertIn("scope.md", raw)                     # 路径应入库
        # 设计与 review 同在此（注意：在 state 段内，不在顶层）
        self.assertIn("artifacts", json.loads(raw)["state"])


class DomainFlowTest(unittest.TestCase):
    """推进规则：happy path / 回退 / 上限 / 跳过 / 伪绿 / 幂等 / 序列化。"""

    def setUp(self) -> None:
        self.task = "t_domain"
        self.st = ao.new_state(self.task, "req")

    def _drive(self, st: "ao.OrchestrationState", max_steps: int = 20) -> list:
        """用与 run_workflow/hooks 完全相同的驱动循环跑完，返回见过的阶段。"""
        seen: list = []
        res = ao.next_instruction(st)
        while res.next_stage and len(seen) < max_steps:
            stage = res.next_stage
            out = ao._call(ao.STAGE_AGENT[stage], res.next_prompt)
            seen.append(stage)
            res = ao.advance(st, stage, out)
        # 离线模式自动批准合并门禁（与 run_workflow 一致）
        if st.phase == ao.PHASE_MERGE:
            st.phase = ao.PHASE_DONE
        return seen

    def test_happy_path_full_chain(self) -> None:
        st = self.st
        ao.set_dispatcher(lambda s, p: f"out-{s}")
        seen = self._drive(st)
        self.assertEqual(seen, ["scope", "design", "code", "review", "test"])
        self.assertEqual(st.phase, ao.PHASE_DONE)
        self.assertFalse(st.needs_human)

    def test_review_blocking_triggers_fix_loop_then_human(self) -> None:
        st = self.st
        ao.set_dispatcher(
            lambda s, p: "[BLOCKING] 必须修复" if s == "code-reviewer" else f"out-{s}"
        )
        seen = self._drive(st)
        # scope design code review fix review fix review test
        self.assertIn("fix", seen)
        self.assertEqual(seen.count("fix"), ao.MAX_AUTO_RETRIES)
        self.assertTrue(st.needs_human)
        self.assertEqual(st.phase, ao.PHASE_DONE)

    def test_design_skip_when_doc_exists(self) -> None:
        # 既有设计文档名必须包含任务关键词（find_design_doc 按关键词子串匹配）
        doc = ao.ARCH_DIR / "ratelimitx_design.md"
        doc.write_text("# 既有设计\n按租户限流\n", encoding="utf-8")
        try:
            st = ao.new_state("ratelimitx", "r")
            ao.set_dispatcher(lambda s, p: f"out-{s}")
            seen = self._drive(st)
            self.assertNotIn("design", seen)            # 跳过设计阶段
            self.assertIsNotNone(st.design_doc)          # 既有设计仍入 state
        finally:
            doc.unlink(missing_ok=True)

    def test_fakegreen_detection(self) -> None:
        st = self.st
        # 走完 scope/design/code/review，最后 test 产出伪绿
        fake_tests = "def test_x():\n    assert True\n    pass\n"
        st.phase = ao.PHASE_TEST
        st.scope_ref = str(ao.RUN_DIR / self.task / "scope.md")
        (ao.RUN_DIR / self.task).mkdir(parents=True, exist_ok=True)
        (ao.RUN_DIR / self.task / "scope.md").write_text("scope", encoding="utf-8")
        p = ao._save(self.task, "test", "tests.py", fake_tests)
        problems = ao.check_test_not_fakegreen(p)
        self.assertTrue(any("恒真断言" in x or "占位" in x for x in problems))

    # ── 硬性质量门（方案②）──
    def _quality_state(self, task: str) -> "ao.OrchestrationState":
        st = ao.new_state(task, "r")
        st.phase = ao.PHASE_TEST
        (ao.RUN_DIR / task).mkdir(parents=True, exist_ok=True)
        return st

    def test_quality_gate_no_artifacts_is_noop(self) -> None:
        task = "t_qg_noop"
        st = self._quality_state(task)
        try:
            self.assertEqual(ao.check_quality_gate(st), [])
        finally:
            shutil.rmtree(ao.RUN_DIR / task, ignore_errors=True)

    def test_quality_gate_low_coverage_blocks(self) -> None:
        task = "t_qg_cov"
        st = self._quality_state(task)
        try:
            (ao.RUN_DIR / task / "coverage.json").write_text(
                json.dumps({"totals": {"percent_covered": 42.0}}), encoding="utf-8"
            )
            probs = ao.check_quality_gate(st)
            self.assertTrue(any("覆盖率" in p for p in probs))
        finally:
            shutil.rmtree(ao.RUN_DIR / task, ignore_errors=True)

    def test_quality_gate_sast_high_blocks(self) -> None:
        task = "t_qg_sast"
        st = self._quality_state(task)
        try:
            (ao.RUN_DIR / task / "sast.json").write_text(json.dumps({
                "results": [{"issue_severity": "HIGH", "filename": "code.py",
                             "line_number": 10, "issue_text": "可能命令注入"}]
            }), encoding="utf-8")
            probs = ao.check_quality_gate(st)
            self.assertTrue(any("SAST" in p for p in probs))
        finally:
            shutil.rmtree(ao.RUN_DIR / task, ignore_errors=True)

    def test_quality_gate_blocks_via_advance(self) -> None:
        task = "t_qg_advance"
        (ao.RUN_DIR / task).mkdir(parents=True, exist_ok=True)
        try:
            (ao.RUN_DIR / task / "coverage.json").write_text(
                json.dumps({"totals": {"percent_covered": 30.0}}), encoding="utf-8"
            )
            st = ao.new_state(task, "r")
            ao.set_dispatcher(lambda s, p: f"out-{s}")
            res = ao.next_instruction(st)
            # 驱动到 test 阶段（design 不跳过）
            while res.next_stage and res.next_stage != "test":
                res = ao.advance(st, res.next_stage,
                                 ao._call(ao.STAGE_AGENT[res.next_stage], res.next_prompt))
            self.assertEqual(res.next_stage, "test")
            ao.advance(st, "test", "tests out")
            self.assertTrue(any("[QUALITY]" in b for b in st.blocking_issues))
            self.assertTrue(st.needs_human)
        finally:
            shutil.rmtree(ao.RUN_DIR / task, ignore_errors=True)

    def test_granularity_warning(self) -> None:
        st = self.st
        scope = ao.RUN_DIR / self.task / "scope.md"
        scope.parent.mkdir(parents=True, exist_ok=True)
        scope.write_text(
            "需求：apps/order 与 apps/user 以及 apps/pay 都要改，另外加上限流。",
            encoding="utf-8",
        )
        st.scope_ref = str(scope)
        warns = ao.check_granularity(st)
        self.assertTrue(any("app" in w for w in warns))

    def test_idempotency_same_stage_retries(self) -> None:
        st = self.st
        st.phase = "fix"
        st.retries = 1
        r1 = ao.advance(st, "fix", "fix attempt 1")
        before = len(st.dispatched)
        r2 = ao.advance(st, "fix", "fix attempt 1 dup")  # 同 stage@retries
        self.assertEqual(len(st.dispatched), before)       # 不再重复归档
        self.assertEqual(r1.phase, r2.phase)

    def test_state_serialize_roundtrip_and_hydrate(self) -> None:
        st = ao.new_state(self.task2(), "r")
        ao.advance(st, "scope", "SCOPE CONTENT HERE")
        d = ao.state_to_dict(st)
        d2 = json.loads(json.dumps(d))                    # 模拟跨进程 JSON
        st2 = ao.state_from_dict(d2)
        self.assertEqual(st2.phase, "design")             # advance 后已前移到 design
        self.assertFalse(st2.artifacts["scope"].content)   # 反序列化后 content 空
        ao.hydrate_state(st2)
        self.assertEqual(st2.artifacts["scope"].content, "SCOPE CONTENT HERE")

    def task2(self) -> str:
        return "t_serialize"


class HookProtocolTest(unittest.TestCase):
    """hook 协议：零干预、stdout 形状、退出码、BOM、matcher、stop_forced 上限。"""

    def setUp(self) -> None:
        self.task = "t_hook_proto"
        _cleanup(self.task)

    def tearDown(self) -> None:
        _cleanup(self.task)

    # —— 零干预不变量：无活跃编排时所有钩子都必须 stdout 为空 ——
    def test_no_active_workflow_zero_intervention(self) -> None:
        self.assertIsNone(ostate.current_task())
        self.assertEqual(ohook.cmd_on_post_task({}), {})
        self.assertEqual(ohook.cmd_on_pre_task({}), {})
        self.assertEqual(ohook.cmd_on_stop({}), {})

    # —— stdin 解析 ——
    def test_invalid_json_returns_empty(self) -> None:
        saved = sys.stdin
        sys.stdin = io.TextIOWrapper(io.BytesIO(b"not json {{{"), encoding="utf-8")
        try:
            self.assertEqual(ohook._read_payload(), {})
        finally:
            sys.stdin = saved

    def test_bom_is_stripped(self) -> None:
        saved = sys.stdin
        sys.stdin = io.TextIOWrapper(
            io.BytesIO(b"\xef\xbb\xbf{\"tool_name\":\"Task\"}"), encoding="utf-8"
        )
        try:
            self.assertEqual(ohook._read_payload(), {"tool_name": "Task"})
        finally:
            sys.stdin = saved

    # —— tool_name matcher：非 Task 事件零干预 ——
    def test_non_task_tool_zero_intervention(self) -> None:
        ostate.set_current(self.task)
        env = {"schema": 1,
               "state": ao.state_to_dict(ao.new_state(self.task, "r")),
               "hook": {"stop_forced": 0}}
        ostate.write_new(self.task, env)
        self.assertEqual(ohook.cmd_on_post_task({"tool_name": "Write"}), {})
        self.assertEqual(ohook.cmd_on_pre_task({"tool_name": "Edit"}), {})

    # —— PostToolUse 给出 additionalContext ——
    def test_post_task_returns_next_instruction(self) -> None:
        ostate.set_current(self.task)
        env = {"schema": 1,
               "state": ao.state_to_dict(ao.new_state(self.task, "r")),
               "hook": {"stop_forced": 0}}
        ostate.write_new(self.task, env)
        out = ohook.cmd_on_post_task({
            "tool_name": "Task",
            "tool_input": {"subagent_type": "scope"},
            "tool_response": "scope 产出内容",
        })
        self.assertIn("hookSpecificOutput", out)
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "PostToolUse")
        self.assertIn("architecture-designer", out["hookSpecificOutput"]["additionalContext"])

    # —— Stop 未完成 → continue:false ——
    def test_stop_blocks_when_incomplete(self) -> None:
        ostate.set_current(self.task)
        env = {"schema": 1,
               "state": ao.state_to_dict(ao.new_state(self.task, "r")),
               "hook": {"stop_forced": 0}}
        ostate.write_new(self.task, env)
        out = ohook.cmd_on_stop({"stop_hook_active": False})
        self.assertIn("continue", out)
        self.assertFalse(out["continue"])

    # —— Stop 已完成 → 放行（空）——
    def test_stop_allows_when_done(self) -> None:
        st = ao.new_state(self.task, "r")
        st.phase = ao.PHASE_DONE
        ostate.set_current(self.task)
        ostate.write_new(self.task, {"schema": 1,
                                     "state": ao.state_to_dict(st),
                                     "hook": {"stop_forced": 0}})
        self.assertEqual(ohook.cmd_on_stop({}), {})

    # —— stop_forced 上限：连续拦截到上限后放行，不无限驱动（防死循环）——
    def test_stop_forced_upper_bound(self) -> None:
        ostate.set_current(self.task)
        env = {"schema": 1,
               "state": ao.state_to_dict(ao.new_state(self.task, "r")),
               "hook": {"stop_forced": ohook.STOP_FORCED_MAX}}
        ostate.write_new(self.task, env)
        out = ohook.cmd_on_stop({"stop_hook_active": False})
        # 已达上限 → 放行停止（无 continue 字段）+ 转人工提示
        self.assertNotIn("continue", out)
        self.assertIn("systemMessage", out)

    def test_stop_forced_increments_atomically(self) -> None:
        ostate.set_current(self.task)
        env = {"schema": 1,
               "state": ao.state_to_dict(ao.new_state(self.task, "r")),
               "hook": {"stop_forced": 0}}
        ostate.write_new(self.task, env)
        ohook.cmd_on_stop({})
        got = ostate.load(self.task)
        self.assertEqual(got["hook"]["stop_forced"], 1)
        # 完成一次实质推进后 stop_forced 重置
        ohook.cmd_on_post_task({"tool_name": "Task",
                                "tool_input": {"subagent_type": "scope"},
                                "tool_response": "x"})
        got = ostate.load(self.task)
        self.assertEqual(got["hook"]["stop_forced"], 0)


class DesignGateTest(unittest.TestCase):
    """方案 A：design 产出后拉起人工 review 门禁，approve 前拦截一切推进与停止。"""

    def setUp(self) -> None:
        self.task = "t_design_gate"
        _cleanup(self.task)
        ostate.set_current(self.task)
        st = ao.new_state(self.task, "为网关新增租户限流")
        ostate.write_new(self.task, {
            "schema": 1,
            "state": ao.state_to_dict(st),
            "hook": {"stop_forced": 0},
        })

    def tearDown(self) -> None:
        _cleanup(self.task)

    def _post(self, subagent: str, response: str) -> dict:
        return ohook.cmd_on_post_task({
            "tool_name": "Task",
            "tool_input": {"subagent_type": subagent},
            "tool_response": response,
        })

    def test_design_triggers_gate(self) -> None:
        self._post("scope", "scope 产出")
        env = ostate.load(self.task)
        self.assertEqual(env["state"]["phase"], "design")
        out = self._post("architecture-designer", "design 产出")
        env = ostate.load(self.task)
        self.assertTrue(env["hook"]["await_design_approval"])
        self.assertIn("hookSpecificOutput", out)
        self.assertIn("approve", out["hookSpecificOutput"]["additionalContext"])

    def test_stop_blocked_while_gate(self) -> None:
        self._post("scope", "scope 产出")
        self._post("architecture-designer", "design 产出")
        out = ohook.cmd_on_stop({"stop_hook_active": False})
        self.assertIn("continue", out)
        self.assertFalse(out["continue"])
        self.assertIn("approve", out["reason"])

    def test_post_blocked_while_gate(self) -> None:
        self._post("scope", "scope 产出")
        self._post("architecture-designer", "design 产出")
        out = self._post("python-coder", "code 产出")
        ctx = out.get("hookSpecificOutput", {}).get("additionalContext", "")
        self.assertIn("approve", ctx)
        env = ostate.load(self.task)
        self.assertEqual(env["state"]["phase"], "code")   # 未推进到 review
        self.assertNotIn("review", env["state"].get("artifacts", {}))

    def test_approve_then_code_proceeds(self) -> None:
        self._post("scope", "scope 产出")
        self._post("architecture-designer", "design 产出")
        self.assertEqual(ohook.cmd_approve(self.task), 0)
        env = ostate.load(self.task)
        self.assertFalse(env["hook"].get("await_design_approval", False))
        self._post("python-coder", "code 产出")
        env = ostate.load(self.task)
        self.assertEqual(env["state"]["phase"], "review")

    def test_approve_noop_without_pending_gate(self) -> None:
        # 门禁未拉起时 approve 应安全 no-op（不报错、不置位）
        self.assertEqual(ohook.cmd_approve(self.task), 0)
        env = ostate.load(self.task)
        self.assertFalse(env["hook"].get("await_design_approval", False))

    def test_approve_no_active_returns_zero(self) -> None:
        ostate.clear_current()
        self.assertIsNone(ostate.current_task())
        self.assertEqual(ohook.cmd_approve(None), 0)


class MergeGateTest(unittest.TestCase):
    """方案④：test 完成后拉起人工 merge 门禁，merge 前拦截一切推进与停止。

    为隔离 design 门禁，setUp 写入既有设计文档使 design 阶段被跳过，
    从而流程直达 code→review→test→merge。
    """

    def setUp(self) -> None:
        self.task = "t_merge_gate"
        _cleanup(self.task)
        # 既有设计文档：让 design 阶段跳过，直达 merge 门禁
        self.design_doc = ao.ARCH_DIR / f"{self.task}_design.md"
        self.design_doc.write_text("# 既有设计\n复用限流\n", encoding="utf-8")
        ostate.set_current(self.task)
        st = ao.new_state(self.task, "为网关新增租户限流")
        ostate.write_new(self.task, {
            "schema": 1,
            "state": ao.state_to_dict(st),
            "hook": {"stop_forced": 0},
        })

    def tearDown(self) -> None:
        self.design_doc.unlink(missing_ok=True)
        _cleanup(self.task)

    def _post(self, subagent: str, response: str) -> dict:
        return ohook.cmd_on_post_task({
            "tool_name": "Task",
            "tool_input": {"subagent_type": subagent},
            "tool_response": response,
        })

    def _drive_to_merge(self) -> None:
        self._post("scope", "scope 产出")
        self._post("python-coder", "code 产出")
        self._post("code-reviewer", "review 产出（无 BLOCKING）")
        self._post("test-generator", "test 产出")

    def test_test_completes_triggers_merge_gate(self) -> None:
        self._drive_to_merge()
        env = ostate.load(self.task)
        self.assertTrue(env["hook"]["await_merge_approval"])
        self.assertEqual(env["state"]["phase"], "merge")
        out = self._post("test-generator", "test 产出")
        self.assertIn("hookSpecificOutput", out)
        self.assertIn("merge", out["hookSpecificOutput"]["additionalContext"])

    def test_stop_blocked_while_merge_gate(self) -> None:
        self._drive_to_merge()
        out = ohook.cmd_on_stop({"stop_hook_active": False})
        self.assertIn("continue", out)
        self.assertFalse(out["continue"])
        self.assertIn("merge", out["reason"])

    def test_post_blocked_while_merge_gate(self) -> None:
        self._drive_to_merge()
        out = self._post("python-coder", "code 产出")
        ctx = out.get("hookSpecificOutput", {}).get("additionalContext", "")
        self.assertIn("merge", ctx)
        env = ostate.load(self.task)
        self.assertEqual(env["state"]["phase"], "merge")   # 未推进
        self.assertNotIn("fix", env["state"].get("artifacts", {}))  # 未产生新派发

    def test_merge_then_done(self) -> None:
        self._drive_to_merge()
        self.assertEqual(ohook.cmd_merge(self.task), 0)
        env = ostate.load(self.task)
        self.assertFalse(env["hook"].get("await_merge_approval", False))
        self.assertEqual(env["state"]["phase"], "done")
        self.assertIsNone(ostate.current_task())   # .current 已释放

    def test_merge_noop_without_pending_gate(self) -> None:
        # 门禁未拉起时 merge 应安全 no-op（不报错、不置位）
        self.assertEqual(ohook.cmd_merge(self.task), 0)
        env = ostate.load(self.task)
        self.assertFalse(env["hook"].get("await_merge_approval", False))

    def test_merge_no_active_returns_zero(self) -> None:
        ostate.clear_current()
        self.assertIsNone(ostate.current_task())
        self.assertEqual(ohook.cmd_merge(None), 0)


class GitIntegrationTest(unittest.TestCase):
    """方案① PR 模型：git 集成模块默认关闭、调用安全 no-op，不误触仓库。"""

    def test_branch_name(self) -> None:
        self.assertEqual(og.branch_name("order-service"), "orch/order-service")

    def test_disabled_is_safe_noop(self) -> None:
        # 默认未启用（ORCH_GIT 未设），commit/open_pr 不得触碰仓库
        self.assertFalse(og.enabled())
        self.assertFalse(og.commit_task("t_git", "msg"))
        self.assertIsNone(og.open_pr("t_git"))

    def test_git_probes_return_bool(self) -> None:
        self.assertIsInstance(og.git_available(), bool)
        self.assertIsInstance(og.in_git_repo(), bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark golden tasks 的「流程级回归」门禁（方案③）。

设计意图：每次改 orchestrator 的 advance / _next_phase / 各类门禁后，
把固定考题集 GOLDEN_TASKS 重新喂给真实状态机（agent_orchestrator.run_workflow），
校验「这套流程」没被改坏——而不是评某次产物的内容质量（那是 code_eval_harness
在 --real 模式下做的事）。

判据（流程级、确定性、无 LLM 依赖）：
  - 每个 task 必须跑到 phase=done；
  - 不得出现非预期的 needs_human（happy-path 流程应全自动收尾，
    除非故意注入 BLOCKING 的设计/质量门禁场景）；
  - 阶段序列稳定（scope → … → test），结构不被破坏。

与 check_workflow_drift 并列常态化：CI 每次 PR 跑一遍，和文档漂移自检一样，
成为「流程改动必过」的第二道客观凭证。

用法：
  python scripts/run_eval_regression.py                # 对比基线，退化则 exit 1
  python scripts/run_eval_regression.py --seed         # （重）生成基线后退出
  python scripts/run_eval_regression.py --baseline <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
EVAL_DIR = REPO_ROOT / "benchmark" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import agent_orchestrator as ao  # noqa: E402
from golden_tasks import GOLDEN_TASKS  # noqa: E402


def _snapshot() -> dict:
    """用确定性 dispatcher 跑完全部 golden task，返回流程级快照 + 把关效果指标。"""
    ao.set_dispatcher(lambda s, p: f"<out {s}>")
    snap: dict = {}
    for gt in GOLDEN_TASKS:
        st = ao.run_workflow(gt.task, gt.user_request)
        stages = [k for k in ("scope", "design", "code", "review", "fix", "test")
                  if any(d.split("@")[0] == k for d in st.dispatched)]

        # 各 gate 的 BLOCKING 计数（改造前除 quality 外均为 0）
        gate_counts: dict = {}
        for issue in st.blocking_issues:
            prefix = issue.split("]")[0] + "]" if "]" in issue else "OTHER"
            gate_counts[prefix] = gate_counts.get(prefix, 0) + 1

        snap[gt.task] = {
            "phase": st.phase,
            "needs_human": st.needs_human,
            "blocking_total": len(st.blocking_issues),
            "blocking_by_gate": gate_counts,
            "fix_retries": st.retries,
            "stages": stages,
        }
    return snap


def _diff(old: dict, new: dict) -> list:
    """返回退化项列表（空 = 无回归）。"""
    regressions: list = []
    for task, cur in new.items():
        base = old.get(task)
        if base is None:
            regressions.append(f"{task}: 基线缺失（新任务？请 --seed 更新基线）")
            continue
        if cur["phase"] != "done":
            regressions.append(
                f"{task}: phase={cur['phase']}（基线 {base['phase']}），流程未跑通"
            )
        if cur["needs_human"] and not base["needs_human"]:
            regressions.append(
                f"{task}: 出现非预期 needs_human（基线无），门禁/回退可能误触发"
            )
        if cur["stages"] != base["stages"]:
            regressions.append(
                f"{task}: 阶段序列变化 {base['stages']} -> {cur['stages']}"
            )
    return regressions


def _aggregate_metrics(snap: dict) -> dict:
    """汇总全量 golden tasks 的把关效果指标。"""
    total_fix_retries = 0
    gate_totals: dict = {}
    total_blocking = 0
    for task_data in snap.values():
        total_fix_retries += task_data.get("fix_retries", 0)
        total_blocking += task_data.get("blocking_total", 0)
        for gate, count in task_data.get("blocking_by_gate", {}).items():
            gate_totals[gate] = gate_totals.get(gate, 0) + count
    return {
        "total_tasks": len(snap),
        "total_blocking": total_blocking,
        "gate_breakdown": gate_totals,
        "total_fix_retries": total_fix_retries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="golden tasks 流程级回归门禁")
    ap.add_argument("--baseline",
                    default=str(REPO_ROOT / "benchmark_results" / "eval_baseline.json"))
    ap.add_argument("--seed", action="store_true",
                    help="强制重新生成基线并退出（不对比）")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    baseline_path = Path(args.baseline)
    snap = _snapshot()

    if args.seed or not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[eval-regression] 基线已{'更新' if args.seed else '初始化'}: {baseline_path}")
        print(f"[eval-regression] 覆盖 {len(snap)} 个 golden task")
        return 0

    try:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[eval-regression] 基线读取失败：{exc}（可加 --seed 重建）", file=sys.stderr)
        return 2

    regressions = _diff(base, snap)
    metrics = _aggregate_metrics(snap)
    if not regressions:
        print(f"[eval-regression] OK：{len(snap)} 个 golden task 流程无回归。")
        print(f"  把关效果指标（改造前对照值）：")
        print(f"    total_blocking={metrics['total_blocking']}")
        print(f"    gate_breakdown={metrics['gate_breakdown']}")
        print(f"    total_fix_retries={metrics['total_fix_retries']}")
        print(f"    human_rejections.design=0（eval 自动批准，待 hook 路径接入）")
        print(f"    human_rejections.merge=0（eval 自动批准，待 hook 路径接入）")
        return 0

    print("[eval-regression] FAIL：检测到流程回归：")
    for r in regressions:
        print(f"  - {r}")
    print(f"[eval-regression] 若属预期变更，运行 "
          f"`python scripts/run_eval_regression.py --seed` 更新基线。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

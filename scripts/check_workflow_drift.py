#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_workflow_drift.py — 文档↔执行漂移自检（L4 中段补齐项）

问题：`.codebuddy/agents/multi-agent-workflow.md` 与 `agent_driver.md` 描述的
阶段/agent 映射、阈值、工作流 job，靠人工与实际代码保持一致。人一改单边就漂移，
而漂移的文档比没有文档更危险（它会误导后续 Agent 与人）。

本脚本把"两边手动对齐"变成"漂移自动红"，在 CI 里跑，不碰任何控制边界。

校验项（每项独立，互不阻塞）：
  D1 阶段→agent 映射：agent_orchestrator.STAGE_AGENT vs agent_driver.md 表格
  D2 subagent 存在性：映射引用的 agent 必须有 .codebuddy/agents/<name>.md（scope/human 除外）
  D3 覆盖率门禁一致：ci_failure_triage.COVERAGE_GATE vs ci-tests.yml vs workflow.md
  D4 重试上限一致：agent_orchestrator.MAX_AUTO_RETRIES vs agent_driver.md
  D5 派单上限一致：ci_failure_triage.MAX_FIX_TARGETS vs workflow.md
  D6 工作流引用存在：workflow.md 提到的脚本/工作流文件必须真实存在
  D7 hooks 注册：.codebuddy/settings.json 的 hooks 块须注册
     PreToolUse→on-pre-task / PostToolUse→on-post-task / Stop→on-stop
  D8 子命令一致性：settings.json 与 agent_driver.md 引用的
     orchestrator_hook.py 子命令必须真实实现
  D9 合并门禁（方案④）：PHASE_MERGE 迁移 + hook 的 merge 子命令/Stop 拦截
     + agent_driver.md 合并门禁说明，必须真实落地且与设计门禁同构
  D10 硬性质量门（方案②）：QUALITY_COVERAGE_GATE 常量 + check_quality_gate 实现
     + advance(PHASE_TEST) 真实调用，覆盖率/SAST 门禁不得被静默移除
  D11 eval 回归（方案③）：ci-tests.yml 须跑 run_eval_regression.py，
     且 baseline 文件 benchmark_results/eval_baseline.json 必须存在，
     否则流程级回归门禁形同虚设
   D12 PR 模型（方案①）：orchestrator_git.py 须实现 ensure_branch/commit_task/open_pr，
      orchestrator_hook.py 须 import 并调用；agent_driver.md 须说明 PR 模型。
      默认关闭（ORCH_GIT=1 启用），但接线不得缺失。
   D13 PHASE_CONTRACT 与阶段链（方案①）：PHASE_CONTRACT 常量 + STAGE_AGENT[PHASE_CONTRACT]
      映射 + advance() 三处转换（SCOPE→CONTRACT、DESIGN→CONTRACT、CONTRACT→CODE）
      + contract_gen.py 存在。默认未启用时软跳过。
   D14 沙箱门禁（方案② T1-T3）：check_sandbox 实现存在 + advance(PHASE_TEST) 真实调用
      + sandbox_run.py 存在 + 软跳过约定已文档化（sandbox.json 缺失不报错）。
   D15 对抗式审查（方案③-3a，降级/可选）：PHASE_REDTEAM 常量 + STAGE_AGENT 映射
      + advance() 转换 + red-team-reviewer.md 存在 + 无复现用例不产生 BLOCKING。
      建议 T8 后启用，默认未启用时软跳过。
   D16 eval 回归（方案③-3b）：ci-tests.yml 须跑 run_eval_regression.py 且触发
      pull_request + baseline 文件 benchmark_results/eval_baseline.json 存在。
   D17 形式化接线（方案④ T13-T14）：critical_paths.yaml 存在 + check_critical_guarantees
      实现存在 + advance(PHASE_TEST) 真实调用 + 无 critical_paths.yaml 时软跳过。
   D18 元评测（方案⑤ T8）：eval_meta_check.py 存在 + 7 类注入用例全覆盖 + 
      semgrep 自定义规则文件存在 + 判据层映射一致性（反模式走 semgrep、伪绿接
      mutation_check.py、自洽接 sandbox_run.py）。

退出码：0 = 无漂移；1 = 存在漂移（CI 应标红）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / ".codebuddy" / "agents"
WORKFLOW_MD = AGENTS_DIR / "multi-agent-workflow.md"
DRIVER_MD = AGENTS_DIR / "agent_driver.md"
SETTINGS_JSON = REPO_ROOT / ".codebuddy" / "settings.json"
CI_TESTS_YML = REPO_ROOT / ".github" / "workflows" / "ci-tests.yml"
ORCH_PY = REPO_ROOT / "scripts" / "agent_orchestrator.py"
TRIAGE_PY = REPO_ROOT / "scripts" / "ci_failure_triage.py"

# orchestrator_hook.py 实际实现的子命令（真相源，与 argparse 保持一致）
IMPLEMENTED_SUBCOMMANDS = {
    "start", "on-pre-task", "on-post-task", "on-stop",
    "status", "reset", "approve", "merge", "dry-run",
}

# 映射中允许不存在对应 agent 文件的伪 agent（非 subagent，而是角色标记）
PSEUDO_AGENTS = {"scope", "human", "contract-gen"}


@dataclass
class Drift:
    code: str          # D1..D8
    item: str          # 漂移对象
    expected: str      # 代码/真相源侧
    actual: str        # 文档侧
    hint: str = ""


def _load_module(path: Path, name: str) -> Optional[Any]:
    """按文件路径加载模块（避免依赖 sys.path 布局）。"""
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── D1/D2：阶段→agent 映射 ──
def parse_driver_mapping(text: str) -> dict[str, str]:
    """解析 agent_driver.md 的映射表格：| stage | subagent | ... |"""
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        stage, agent = cells[0], cells[1]
        # 跳过表头与分隔行
        if stage in ("stage", "") or set(stage) <= set("-: "):
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", stage):
            continue
        mapping[stage] = agent
    return mapping


def check_stage_mapping(drifts: list[Drift]) -> None:
    # STAGE_AGENT 已提升为模块级常量（方案 P0）：hooks 改造删除了 Orchestrator
    # 手动驱动类，drift 检查必须读模块级 STAGE_AGENT 而非 Orchestrator.STAGE_AGENT。
    orch = _load_module(ORCH_PY, "_orch_drift")
    if orch is None or not hasattr(orch, "STAGE_AGENT"):
        drifts.append(Drift("D1", "agent_orchestrator.STAGE_AGENT",
                            "可导入且含 STAGE_AGENT", "导入失败",
                            "检查 scripts/agent_orchestrator.py 语法"))
        return
    code_map: dict[str, str] = dict(orch.STAGE_AGENT)
    doc_map = parse_driver_mapping(_read(DRIVER_MD))

    for stage, agent in code_map.items():
        if stage not in doc_map:
            drifts.append(Drift("D1", f"stage={stage}", agent, "文档缺失",
                                "在 agent_driver.md 映射表补此行"))
        elif doc_map[stage] != agent:
            drifts.append(Drift("D1", f"stage={stage}", agent, doc_map[stage],
                                "文档与 STAGE_AGENT 不一致"))
    for stage in doc_map:
        if stage not in code_map:
            drifts.append(Drift("D1", f"stage={stage}", "代码中不存在",
                                doc_map[stage], "文档描述了已移除的阶段"))

    # D2：引用的 agent 必须真实存在
    for stage, agent in code_map.items():
        if agent in PSEUDO_AGENTS:
            continue
        if not (AGENTS_DIR / f"{agent}.md").exists():
            drifts.append(Drift("D2", f"agent={agent}", "存在 .codebuddy/agents/*.md",
                                "文件不存在",
                                f"stage={stage} 指向不存在的 subagent"))


# ── D3/D4/D5：阈值一致性 ──
def _find_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def check_thresholds(drifts: list[Drift]) -> None:
    triage = _load_module(TRIAGE_PY, "_triage_drift")
    orch = _load_module(ORCH_PY, "_orch_drift2")
    wf_text = _read(WORKFLOW_MD)
    driver_text = _read(DRIVER_MD)

    # D3 覆盖率门禁：triage 常量 vs ci-tests.yml vs workflow.md
    if triage is not None:
        code_gate = int(getattr(triage, "COVERAGE_GATE", 0))
        yml_gate = _find_int(_read(CI_TESTS_YML), r"COVERAGE_GATE\s*=\s*(\d+)")
        md_gate = _find_int(wf_text, r"COVERAGE_GATE\s*=\s*(\d+)")
        if yml_gate is not None and yml_gate != code_gate:
            drifts.append(Drift("D3", "COVERAGE_GATE", str(code_gate), str(yml_gate),
                                "ci_failure_triage.py 与 ci-tests.yml 门禁不一致"))
        if md_gate is not None and md_gate != code_gate:
            drifts.append(Drift("D3", "COVERAGE_GATE(文档)", str(code_gate), str(md_gate),
                                "workflow.md 覆盖率门禁描述过期"))

        # D5 派单上限
        code_max = int(getattr(triage, "MAX_FIX_TARGETS", 0))
        md_max = _find_int(wf_text, r"MAX_FIX_TARGETS\s*=\s*(\d+)")
        if md_max is not None and md_max != code_max:
            drifts.append(Drift("D5", "MAX_FIX_TARGETS", str(code_max), str(md_max),
                                "workflow.md 派单上限描述过期"))
    else:
        drifts.append(Drift("D3", "ci_failure_triage.py", "可导入", "导入失败"))

    # D4 重试上限：orchestrator 常量 vs agent_driver.md
    if orch is not None:
        code_retry = int(getattr(orch, "MAX_AUTO_RETRIES", 0))
        md_retry = _find_int(driver_text, r"MAX_AUTO_RETRIES\s*=\s*(\d+)")
        if md_retry is not None and md_retry != code_retry:
            drifts.append(Drift("D4", "MAX_AUTO_RETRIES", str(code_retry), str(md_retry),
                                "agent_driver.md 重试上限描述过期"))


# ── D6：文档引用的文件必须存在 ──
_REF_RE = re.compile(r"`((?:scripts|\.github/workflows)/[\w./-]+)`")


def check_file_refs(drifts: list[Drift]) -> None:
    for md in (WORKFLOW_MD, DRIVER_MD):
        text = _read(md)
        for ref in sorted(set(_REF_RE.findall(text))):
            if not (REPO_ROOT / ref).exists():
                drifts.append(Drift("D6", ref, "文件存在", "不存在",
                                    f"{md.name} 引用了不存在的文件"))


# ── D7：hooks 注册一致性（settings.json ↔ orchestrator_hook.py）──
def _hook_cmd_has(entry: Any, sub: str) -> bool:
    if not isinstance(entry, dict):
        return False
    return f"orchestrator_hook.py {sub}" in str(entry.get("command", ""))


def check_hooks_registration(drifts: list[Drift]) -> None:
    cfg = _load_json(SETTINGS_JSON)
    if cfg is None:
        drifts.append(Drift("D7", ".codebuddy/settings.json", "存在且为合法 JSON",
                            "不存在/非法", "hooks 驱动依赖此文件注册事件"))
        return
    hooks = cfg.get("hooks")
    if not isinstance(hooks, dict):
        drifts.append(Drift("D7", "hooks", "存在 hooks 块", "缺失",
                            "需在 settings.json 注册 PreToolUse/PostToolUse/Stop"))
        return

    # 期望：每个事件都注册为指向正确子命令的 hook
    expected = {
        "PreToolUse": "on-pre-task",
        "PostToolUse": "on-post-task",
        "Stop": "on-stop",
    }
    for event, sub in expected.items():
        entries = hooks.get(event)
        if not isinstance(entries, list) or not entries:
            drifts.append(Drift("D7", f"hooks.{event}", f"注册 {sub}", "缺失",
                                f"在 settings.json 的 hooks.{event} 添加调用 "
                                f"orchestrator_hook.py {sub}"))
            continue
        if not any(_hook_cmd_has(e, sub) for e in entries):
            drifts.append(Drift("D7", f"hooks.{event}", f"指向 {sub}", "未指向",
                                f"command 应包含 'orchestrator_hook.py {sub}'"))


# ── D8：子命令清单一致（settings.json + agent_driver.md ↔ 实现）──
_HOOK_SUB_RE = re.compile(r"orchestrator_hook\.py\s+([A-Za-z0-9_-]+)")


def check_subcommand_consistency(drifts: list[Drift]) -> None:
    refs: set[str] = set()

    cfg = _load_json(SETTINGS_JSON)
    if isinstance(cfg, dict) and isinstance(cfg.get("hooks"), dict):
        for entries in cfg["hooks"].values():
            if isinstance(entries, list):
                for e in entries:
                    m = _HOOK_SUB_RE.search(str(e.get("command", "")))
                    if m:
                        refs.add(m.group(1))

    for m in _HOOK_SUB_RE.finditer(_read(DRIVER_MD)):
        refs.add(m.group(1))

    for sub in sorted(refs):
        if sub not in IMPLEMENTED_SUBCOMMANDS:
            drifts.append(Drift("D8", f"子命令 {sub}", "真实实现", "未实现",
                                f"settings.json / agent_driver.md 引用了不存在的子命令 {sub}"))


# ── D9：合并门禁（方案④）一致性（领域层 PHASE_MERGE + hook merge 子命令 + 文档）──
def check_merge_gate(drifts: list[Drift]) -> None:
    orch_src = _read(ORCH_PY)
    hook_src = _read(REPO_ROOT / "scripts" / "orchestrator_hook.py")
    doc = _read(DRIVER_MD)

    # D9.1 领域层：PHASE_MERGE 常量 + test→merge→done 迁移
    orch = _load_module(ORCH_PY, "_orch_drift_m")
    if orch is None or not hasattr(orch, "PHASE_MERGE"):
        drifts.append(Drift("D9", "PHASE_MERGE", "领域层定义", "缺失",
                            "agent_orchestrator.py 需定义 PHASE_MERGE"))
    if not re.search(r"if cur == PHASE_TEST:.*?return PHASE_MERGE", orch_src, re.S):
        drifts.append(Drift("D9", "_next_phase", "PHASE_TEST→PHASE_MERGE", "未迁移",
                            "test 完成后应进入合并门禁而非直接 done"))
    if not re.search(r"if cur == PHASE_MERGE:.*?return PHASE_DONE", orch_src, re.S):
        drifts.append(Drift("D9", "_next_phase", "PHASE_MERGE→PHASE_DONE", "未迁移",
                            "合并门禁放行后应收尾至 done"))

    # D9.2 hook 层：merge 子命令真实实现并在 IMPLEMENTED_SUBCOMMANDS 中
    if "merge" not in IMPLEMENTED_SUBCOMMANDS:
        drifts.append(Drift("D9", "子命令 merge", "实现", "未在 IMPLEMENTED_SUBCOMMANDS",
                            "orchestrator_hook.py 的 merge 子命令需登记"))
    if "def cmd_merge" not in hook_src:
        drifts.append(Drift("D9", "cmd_merge", "实现", "缺失",
                            "orchestrator_hook.py 需实现 cmd_merge 放行合并门禁"))

    # D9.3 Stop 拦截：合并门禁未过时 continue:false
    if "await_merge_approval" not in hook_src:
        drifts.append(Drift("D9", "cmd_on_stop", "await_merge_approval 拦截", "缺失",
                            "合并门禁未过时 Stop 必须 continue:false"))

    # D9.4 文档：agent_driver.md 须说明合并门禁
    if "合并门禁" not in doc:
        drifts.append(Drift("D9", "agent_driver.md", "合并门禁 说明", "缺失",
                            "文档需补充合并门禁（方案④）章节"))


# ── D10：硬性质量门（方案②）一致性（常量 + 实现 + 调用点）──
def check_quality_gate_drift(drifts: list[Drift]) -> None:
    orch_src = _read(ORCH_PY)
    orch = _load_module(ORCH_PY, "_orch_drift_q")
    if orch is None or not hasattr(orch, "QUALITY_COVERAGE_GATE"):
        drifts.append(Drift("D10", "QUALITY_COVERAGE_GATE", "领域层定义", "缺失",
                            "agent_orchestrator.py 需定义覆盖率门禁常量"))
    if "def check_quality_gate" not in orch_src:
        drifts.append(Drift("D10", "check_quality_gate", "实现", "缺失",
                            "agent_orchestrator.py 需实现 check_quality_gate"))
    if "check_quality_gate(state)" not in orch_src:
        drifts.append(Drift("D10", "advance(PHASE_TEST)", "调用 check_quality_gate", "未调用",
                            "test 阶段后必须真实执行质量门"))


# ── D11：eval 回归（方案③）常态化（CI 引用 + baseline 存在）──
def check_eval_regression(drifts: list[Drift]) -> None:
    ci = _read(CI_TESTS_YML)
    if "run_eval_regression.py" not in ci:
        drifts.append(Drift("D11", "ci-tests.yml", "引用 run_eval_regression.py", "未引用",
                            "需在 ci-tests.yml 增加 eval-regression job 跑回归门禁"))
    baseline = REPO_ROOT / "benchmark_results" / "eval_baseline.json"
    if not baseline.exists():
        drifts.append(Drift("D11", "benchmark_results/eval_baseline.json", "存在", "缺失",
                            "运行 `python scripts/run_eval_regression.py --seed` 生成基线"))


# ── D12：PR 模型（方案①）接线一致性 ──
def check_pr_model(drifts: list[Drift]) -> None:
    git_src = _read(REPO_ROOT / "scripts" / "orchestrator_git.py")
    hook_src = _read(REPO_ROOT / "scripts" / "orchestrator_hook.py")
    doc = _read(DRIVER_MD)
    if not all(fn in git_src for fn in
               ("def ensure_branch", "def commit_task", "def open_pr")):
        drifts.append(Drift("D12", "orchestrator_git.py",
                            "ensure_branch/commit_task/open_pr", "缺失",
                            "需实现方案① 的 git 集成函数"))
    if "import orchestrator_git" not in hook_src:
        drifts.append(Drift("D12", "orchestrator_hook.py", "import orchestrator_git", "缺失",
                            "hook 层需接入 git 集成"))
    if "og.commit_task" not in hook_src or "og.open_pr" not in hook_src:
        drifts.append(Drift("D12", "orchestrator_hook.py", "调用 commit_task/open_pr", "未调用",
                            "start/on-post-task/merge 需接线 git 集成"))
    if "方案①" not in doc or "PR" not in doc:
        drifts.append(Drift("D12", "agent_driver.md", "PR 模型说明", "缺失",
                            "文档需补充方案① PR 模型章节"))


# ── D13：PHASE_CONTRACT 与阶段链（方案①）──
def check_phase_contract(drifts: list[Drift]) -> None:
    orch_src = _read(ORCH_PY)
    if "PHASE_CONTRACT" not in orch_src:
        drifts.append(Drift("D13", "PHASE_CONTRACT", "领域层定义", "缺失",
                            "agent_orchestrator.py 需定义 PHASE_CONTRACT"))
    if not (re.search(r'STAGE_AGENT\[PHASE_CONTRACT\]', orch_src) or
            re.search(r'"contract"\s*:', orch_src)):
        drifts.append(Drift("D13", "STAGE_AGENT[PHASE_CONTRACT]", "映射存在", "缺失",
                            "STAGE_AGENT 需包含 contract 映射"))
    # advance() 三处转换
    patterns = [
        r"if cur == PHASE_SCOPE:.*?return PHASE_CONTRACT",
        r"if cur == PHASE_DESIGN:.*?return PHASE_CONTRACT",
        r"if cur == PHASE_CONTRACT:.*?return PHASE_CODE",
    ]
    for pat in patterns:
        if not re.search(pat, orch_src, re.S):
            drifts.append(Drift("D13", f"advance({pat[:20]}...)", "转换存在", "缺失",
                                "advance() 需实现指定阶段转换"))
    cg = REPO_ROOT / "scripts" / "contract_gen.py"
    if not cg.exists():
        drifts.append(Drift("D13", "contract_gen.py", "存在", "缺失",
                            "需新建 scripts/contract_gen.py"))


# ── D14：沙箱门禁（方案② T1-T3）──
def check_sandbox_gate(drifts: list[Drift]) -> None:
    orch_src = _read(ORCH_PY)
    if "def check_sandbox" not in orch_src:
        drifts.append(Drift("D14", "check_sandbox", "实现", "缺失",
                            "agent_orchestrator.py 需实现 check_sandbox"))
    if "check_sandbox(state)" not in orch_src:
        drifts.append(Drift("D14", "advance(PHASE_TEST)", "调用 check_sandbox", "未调用",
                            "test 阶段后必须真实执行沙箱门禁"))
    sr = REPO_ROOT / "scripts" / "sandbox_run.py"
    if not sr.exists():
        drifts.append(Drift("D14", "sandbox_run.py", "存在", "缺失",
                            "需新建 scripts/sandbox_run.py"))
    # 软跳过约定：sandbox.json 缺失返回 []
    if 'return []' not in orch_src or 'sandbox.json' not in orch_src:
        drifts.append(Drift("D14", "check_sandbox 软跳过", "sandbox.json 缺失返回 []", "未实现",
                            "缺产物时必须软跳过，不卡离线 eval"))


# ── D15：对抗式审查（方案③-3a，降级/可选）──
def check_red_team(drifts: list[Drift]) -> None:
    orch_src = _read(ORCH_PY)
    if "PHASE_REDTEAM" not in orch_src:
        drifts.append(Drift("D15", "PHASE_REDTEAM", "领域层定义", "缺失（可选，建议 T8 后启用）",
                            "agent_orchestrator.py 需定义 PHASE_REDTEAM"))
    rt = REPO_ROOT / ".codebuddy" / "agents" / "red-team-reviewer.md"
    if not rt.exists():
        drifts.append(Drift("D15", "red-team-reviewer.md", "存在", "缺失（可选）",
                            "需新建 .codebuddy/agents/red-team-reviewer.md"))


# ── D16：每 PR eval 回归（方案③-3b）──
def check_eval_pr_regression(drifts: list[Drift]) -> None:
    ci = _read(CI_TESTS_YML)
    if "run_eval_regression.py" not in ci:
        drifts.append(Drift("D16", "ci-tests.yml", "引用 run_eval_regression.py", "未引用",
                            "需在 ci-tests.yml 增加 eval-regression job 并触发 PR"))
    if "pull_request" not in ci:
        drifts.append(Drift("D16", "ci-tests.yml on:", "pull_request 触发", "缺失",
                            "eval-regression job 须在 PR 时触发"))
    baseline = REPO_ROOT / "benchmark_results" / "eval_baseline.json"
    if not baseline.exists():
        drifts.append(Drift("D16", "benchmark_results/eval_baseline.json", "存在", "缺失",
                            "运行 `python scripts/run_eval_regression.py --seed` 生成基线"))


# ── D17：形式化接线（方案④ T13-T14）──
def check_formal_gate(drifts: list[Drift]) -> None:
    orch_src = _read(ORCH_PY)
    if "def check_critical_guarantees" not in orch_src:
        drifts.append(Drift("D17", "check_critical_guarantees", "实现", "缺失（试点）",
                            "agent_orchestrator.py 需实现 check_critical_guarantees"))
    if "check_critical_guarantees(state)" not in orch_src:
        drifts.append(Drift("D17", "advance(PHASE_TEST)", "调用 check_critical_guarantees", "未调用",
                            "test 阶段后需并列调用形式化门禁"))
    cp = REPO_ROOT / "critical_paths.yaml"
    if not cp.exists():
        drifts.append(Drift("D17", "critical_paths.yaml", "存在", "缺失（试点，可暂缓）",
                            "需新建 critical_paths.yaml 定义关键路径"))


# ── D18：元评测（方案⑤ T8）──
def check_meta_eval(drifts: list[Drift]) -> None:
    meta = REPO_ROOT / "scripts" / "eval_meta_check.py"
    if not meta.exists():
        drifts.append(Drift("D18", "eval_meta_check.py", "存在", "缺失",
                            "需新建 scripts/eval_meta_check.py"))
    semgrep_rules = REPO_ROOT / ".semgrep" / "rules"
    if not semgrep_rules.exists() or not any(semgrep_rules.iterdir()):
        drifts.append(Drift("D18", ".semgrep/rules/", "自定义规则存在", "缺失",
                            "需新建 .semgrep/rules/*.yml"))
    # 判据层映射一致性
    orch_src = _read(ORCH_PY)
    harness_src = _read(REPO_ROOT / "benchmark" / "eval" / "code_eval_harness" / "checks.py")
    if "semgrep" not in orch_src and "semgrep" not in harness_src:
        drifts.append(Drift("D18", "判据层映射", "反模式走 semgrep", "未接入",
                            "harness 反模式判据需调用 semgrep"))
    if "mutation_check" not in orch_src and "mutation" not in harness_src:
        drifts.append(Drift("D18", "判据层映射", "伪绿接 mutation_check.py", "未接入",
                            "harness 伪绿判据需接 mutation_check.py"))
    if "sandbox_run" not in orch_src:
        drifts.append(Drift("D18", "判据层映射", "自洽接 sandbox_run.py", "未接入",
                            "harness tests_pass 需接 sandbox_run.py"))


TITLE = "## 文档-执行漂移自检"


def render(drifts: list[Drift]) -> str:
    if not drifts:
        return f"{TITLE}\n\n[OK] 无漂移：文档与实际代码/工作流一致。"
    lines = [TITLE, "",
             f"[FAIL] 检测到 {len(drifts)} 处漂移：", "",
             "| 检查项 | 对象 | 代码侧(真相) | 文档侧 | 提示 |",
             "|--------|------|-------------|--------|------|"]
    for d in drifts:
        lines.append(f"| {d.code} | {d.item} | {d.expected} | {d.actual} | {d.hint} |")
    lines += ["", "> 真相源以代码/工作流为准，请更新文档或修正代码后重跑。"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="文档↔执行漂移自检")
    ap.add_argument("--summary-out", default="", help="Markdown 摘要输出路径")
    args = ap.parse_args()

    # Windows 控制台默认 GBK，输出中文/符号会崩，强制 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    drifts: list[Drift] = []
    check_stage_mapping(drifts)
    check_thresholds(drifts)
    check_file_refs(drifts)
    check_hooks_registration(drifts)
    check_subcommand_consistency(drifts)
    check_merge_gate(drifts)
    check_quality_gate_drift(drifts)
    check_eval_regression(drifts)
    check_pr_model(drifts)
    check_phase_contract(drifts)
    check_sandbox_gate(drifts)
    check_red_team(drifts)
    check_eval_pr_regression(drifts)
    check_formal_gate(drifts)
    check_meta_eval(drifts)

    out = render(drifts)
    print(out)
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(out, encoding="utf-8")
    return 1 if drifts else 0


if __name__ == "__main__":
    sys.exit(main())

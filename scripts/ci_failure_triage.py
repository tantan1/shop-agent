#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci_failure_triage.py — CI 失败结构化分诊（档 3 的输入端）

职责：把 CI 产出的原始报告（junit xml / coverage.json）解析成
"可直接派给修复 Agent 的结构化任务"，而不是把整段日志丢给模型。

输出：
  1) stdout 打印人读摘要
  2) --out 指定路径写 JSON（供 workflow 后续步骤 / Orchestrator 消费）
  3) 退出码：0 = 无需修复；1 = 存在可自动修复的失败

分诊分类（决定派哪个 subagent）：
  test_failure     用例失败/报错        → python-coder（修实现或修用例）
  coverage_gate    覆盖率低于门禁       → test-generator（补测试）
  missing_report   报告缺失/收集失败    → 转人工（不自动修，避免瞎猜）

设计原则（与 multi-agent-workflow.md 一致）：
  - 只在"有明确失败信号"时才派单，避免无意义烧 token
  - 单次最多派 MAX_FIX_TARGETS 个目标，防止一次改动面过大
  - 无法归因的失败（missing_report）直接转人工
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# 覆盖率门禁需与 .github/workflows/ci-tests.yml 的 COVERAGE_GATE 保持一致
COVERAGE_GATE = 60.0

# 单次自动修复最多处理的目标数（防止改动面过大、review 不可控）
MAX_FIX_TARGETS = 5

APPS = ["gateway", "monitoring-agent", "shop-agent"]


@dataclass
class FixTarget:
    """一个可派发的修复目标。"""
    kind: str                       # test_failure | coverage_gate | missing_report
    app: str
    detail: str                     # 失败摘要（用例名 / 覆盖率数值）
    test_id: Optional[str] = None   # 形如 tests/test_x.py::TestA::test_b
    message: str = ""               # 失败首行信息，供定位
    suggested_agent: str = "python-coder"


@dataclass
class TriageResult:
    total_failures: int = 0
    overall_coverage: Optional[float] = None
    coverage_gate: float = COVERAGE_GATE
    auto_fixable: bool = False
    needs_human: bool = False
    reasons: list = field(default_factory=list)
    targets: list = field(default_factory=list)


# ── junit 解析（不引第三方依赖，正则足够稳）──
_TESTCASE_RE = re.compile(
    r'<testcase\b[^>]*?\bclassname="(?P<cls>[^"]*)"[^>]*?\bname="(?P<name>[^"]*)"[^>]*?'
    r'(?:/>|>(?P<body>.*?)</testcase>)',
    re.DOTALL,
)
_FAIL_RE = re.compile(r'<(?:failure|error)\b[^>]*?message="(?P<msg>[^"]*)"', re.DOTALL)
_FAIL_TAG_RE = re.compile(r'<(?:failure|error)\b')


def parse_junit_failures(path: Path, app: str) -> list[FixTarget]:
    """从 junit xml 提取失败/报错用例。"""
    if not path.exists():
        return []
    xml = path.read_text(encoding="utf-8", errors="replace")
    targets: list[FixTarget] = []
    for m in _TESTCASE_RE.finditer(xml):
        body = m.group("body") or ""
        if not _FAIL_TAG_RE.search(body):
            continue
        cls = m.group("cls") or ""
        name = m.group("name") or ""
        msg_m = _FAIL_RE.search(body)
        msg = (msg_m.group("msg") if msg_m else "").strip()
        # classname 形如 tests.test_foo.TestBar → tests/test_foo.py::TestBar
        parts = cls.split(".")
        if len(parts) >= 2:
            file_part = "/".join(parts[:-1]) + ".py"
            test_id = f"{file_part}::{parts[-1]}::{name}"
        else:
            test_id = f"{cls}::{name}" if cls else name
        targets.append(FixTarget(
            kind="test_failure",
            app=app,
            detail=f"{cls}::{name}",
            test_id=test_id,
            message=_unescape(msg)[:500],
            suggested_agent="python-coder",
        ))
    return targets


def _unescape(s: str) -> str:
    return (s.replace("&quot;", '"').replace("&apos;", "'")
             .replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))


def parse_coverage(path: Path) -> Optional[tuple[int, int]]:
    """返回 (covered_lines, num_statements)，解析失败返回 None。"""
    if not path.exists():
        return None
    try:
        cov = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    totals = cov.get("totals") or cov.get("summary") or {}
    covered = totals.get("covered_lines")
    stmts = totals.get("num_statements")
    if covered is None or stmts is None:
        return None
    return int(covered), int(stmts)


def lowest_covered_files(path: Path, app: str, top: int = 3) -> list[FixTarget]:
    """挑覆盖率最低的文件作为补测试目标（比"全项目补测试"精准得多）。"""
    if not path.exists():
        return []
    try:
        cov = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    files = cov.get("files") or {}
    scored: list[tuple[float, str, int]] = []
    for fname, info in files.items():
        s = info.get("summary") or {}
        stmts = s.get("num_statements") or 0
        pct = s.get("percent_covered")
        if not stmts or pct is None:
            continue
        if stmts < 10:      # 太小的文件补测收益低，跳过
            continue
        scored.append((float(pct), fname, int(stmts)))
    # 语句多且覆盖低的优先（收益最大）
    scored.sort(key=lambda x: (x[0], -x[2]))
    return [
        FixTarget(
            kind="coverage_gate",
            app=app,
            detail=f"{fname} 行覆盖率 {pct:.1f}%（{stmts} 语句）",
            suggested_agent="test-generator",
        )
        for pct, fname, stmts in scored[:top]
    ]


def triage(artifacts_dir: Path) -> TriageResult:
    res = TriageResult()
    sum_covered = sum_stmts = 0
    saw_any_report = False

    for app in APPS:
        base = artifacts_dir / f"pytest-results-{app}"
        junit = base / f"pytest-results-{app}.xml"
        covjson = base / "coverage.json"

        if junit.exists():
            saw_any_report = True
            fails = parse_junit_failures(junit, app)
            res.total_failures += len(fails)
            res.targets.extend(fails)
        else:
            res.needs_human = True
            res.reasons.append(f"{app}: 缺少 junit 报告（疑似收集阶段就失败）")
            res.targets.append(FixTarget(
                kind="missing_report", app=app,
                detail="junit 报告缺失，无法归因",
                suggested_agent="human",
            ))

        cov = parse_coverage(covjson)
        if cov:
            saw_any_report = True
            sum_covered += cov[0]
            sum_stmts += cov[1]

    if sum_stmts > 0:
        res.overall_coverage = round(sum_covered / sum_stmts * 100, 1)
        if res.overall_coverage < COVERAGE_GATE:
            res.reasons.append(
                f"总体覆盖率 {res.overall_coverage}% < 门禁 {COVERAGE_GATE}%"
            )
            # 覆盖率不达标 → 定位最低覆盖文件补测试
            for app in APPS:
                covjson = artifacts_dir / f"pytest-results-{app}" / "coverage.json"
                res.targets.extend(lowest_covered_files(covjson, app))

    if res.total_failures:
        res.reasons.append(f"存在 {res.total_failures} 项用例失败/报错")

    if not saw_any_report:
        res.needs_human = True
        res.reasons.append("未找到任何 CI 报告，无法分诊")

    # 截断到上限，并判定是否可自动修复
    res.targets = res.targets[:MAX_FIX_TARGETS]
    res.auto_fixable = any(
        t.kind in ("test_failure", "coverage_gate") for t in res.targets
    )
    return res


def render_summary(res: TriageResult) -> str:
    lines = ["## CI 失败分诊报告", ""]
    lines.append(f"- 失败用例数：{res.total_failures}")
    cov = f"{res.overall_coverage}%" if res.overall_coverage is not None else "N/A"
    lines.append(f"- 总体覆盖率：{cov}（门禁 {res.coverage_gate}%）")
    lines.append(f"- 可自动修复：{'是' if res.auto_fixable else '否'}")
    lines.append(f"- 需人工介入：{'是' if res.needs_human else '否'}")
    if res.reasons:
        lines += ["", "### 判定依据"] + [f"- {r}" for r in res.reasons]
    if res.targets:
        lines += ["", "### 派单目标", "", "| 类型 | App | 明细 | 建议 Agent |",
                  "|------|-----|------|-----------|"]
        for t in res.targets:
            detail = t.detail.replace("|", "\\|")[:120]
            lines.append(f"| {t.kind} | {t.app} | {detail} | {t.suggested_agent} |")
    else:
        lines += ["", "✅ 无需修复。"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="CI 失败结构化分诊")
    ap.add_argument("--artifacts", default="artifacts",
                    help="CI 报告下载目录（含 pytest-results-*/ 子目录）")
    ap.add_argument("--out", default="", help="分诊结果 JSON 输出路径")
    ap.add_argument("--summary-out", default="", help="人读 Markdown 摘要输出路径")
    args = ap.parse_args()

    res = triage(Path(args.artifacts))
    summary = render_summary(res)
    print(summary)

    payload = asdict(res)
    payload["targets"] = [asdict(t) if not isinstance(t, dict) else t
                          for t in res.targets]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(summary, encoding="utf-8")

    # 退出码：有可自动修复项 → 1（供 workflow 判断是否派单）
    return 1 if res.auto_fixable else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""改测试放宽检测（文章 23 第三层：堵"非法消红"最隐蔽形态）。

背景：
  "非法消红"最严重的不是改测试让 CI 变绿（那已被 check_test_not_fakegreen 抓弱断言），
  而是 **放宽已有断言** —— 把 `==` 改 `in`、删边界检查、阈值放大、注释掉 assert。
  这种改动不引入假绿，而是让原本能抓 bug 的测试变钝，悄无声息地撤防。

本脚本基于 git diff 检测 PR 对 tests/ 下 *已有文件* 的断言改动：
  1. 删除断言行（含 assert / 注释掉的 assert）
  2. 断言放宽：
     - 严格比较 `==` / `is` → 宽松 `in` / `is not None` / `>=`
     - 边界检查被删（如 `len(x) == 0` 改 `len(x) < 10`）
     - 数值阈值放大（如 `> 100` 改 `> 1000`）
  3. 区分"新增测试文件"（合法，免审）vs "改已有测试"（需声明 + 分离 PR）

输出：
  - 人类可读报告（每条可疑改动标注位置 + 风险类型）
  - --json 导出（供 CI：发现可疑放宽则阻断，除非带豁免标记）
  - 退出码：发现未豁免的可疑放宽 → 1

豁免机制：
  在 PR 描述或 commit 中含 `[relax-test: <理由>]` 可豁免（仍需分离 PR 说明），
  但本脚本只负责 *检出*，豁免判断交给 CI / 人审。

用法：
  python scripts/check_test_relax.py --base origin/main
  python scripts/check_test_relax.py --base HEAD~3 --json artifacts/relax.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 断言放宽模式：(正则, 风险描述)
RELAX_PATTERNS = [
    (re.compile(r"^\s*#.*assert\b", re.I), "断言被注释掉"),
    (re.compile(r"assert\b"), "断言行被删除(出现在 - 行)"),
    (re.compile(r"(==|is\b)\s", re.I), "严格比较可能改为宽松"),
    (re.compile(r"(>=|is not None|in\b)", re.I), "出现宽松判定(需人工确认是否放宽)"),
]

# 阈值放大检测：同一被测文件内 `> N` / `< N` 的 N 变大
THRESHOLD_RE = re.compile(r"(>=?|<=?)\s*(\d+)")


def git_diff_files(base: str) -> list:
    """返回相对 base 改动的文件列表（status + path）。

    兼容两种调用场景：
      - CI/PR 环境：base=origin/main，用 `base...HEAD` 取合并基到 HEAD 的改动
      - 本地未提交改动：退回 `git diff base`（含工作区/暂存区）
    优先尝试 ...HEAD，失败则用普通 diff（含工作区）。
    """
    out = subprocess.run(
        ["git", "diff", "--name-status", base + "...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0 or not out.stdout.strip():
        out = subprocess.run(
            ["git", "diff", "--name-status", base],
            cwd=REPO_ROOT,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    files = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        files.append((status, path))
    return files


def get_diff_for(path: str, base: str) -> str:
    out = subprocess.run(
        ["git", "diff", base, "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return out.stdout


def analyze_test_file(path: str, base: str) -> list:
    """对一个测试文件 diff 做放宽分析，返回可疑项列表。"""
    diff = get_diff_for(path, base)
    if not diff:
        return []
    findings = []
    for line in diff.splitlines():
        if not line.startswith("-") or line.startswith("---"):
            continue  # 只看删除/改动前的行（原断言消失）
        content = line[1:].strip()
        if not content or content.startswith(("import", "from", "# pylint")):
            continue
        # 删除的断言行
        if re.search(r"\bassert\b", content, re.I):
            findings.append({"type": "assert_deleted", "line": content[:120]})
            continue
        # 宽松判定出现（在删除行里意味着原测试可能用了更严格形式，此处仅提示）
        for pat, desc in RELAX_PATTERNS:
            if pat.search(content):
                findings.append({"type": "possible_relax", "line": content[:120], "desc": desc})
                break
    # 阈值放大：对比同一 diff 内 + / - 行的数值
    findings += _threshold_swell(diff)
    return findings


def _threshold_swell(diff: str) -> list:
    """检测同一 hunks 内阈值被放大（如 >100 改 >1000）。"""
    out = []
    # 简单启发：找出 - 行和 + 行里同运算符不同数值
    mins = {}  # op -> set of values in deleted lines
    plus = {}  # op -> set of values in added lines
    for line in diff.splitlines():
        if line.startswith("@@"):
            continue
        m_op = re.search(r"(>=?|<=?)\s*(\d+)", line)
        if not m_op:
            continue
        op, val = m_op.group(1), int(m_op.group(2))
        if line.startswith("-"):
            mins.setdefault(op, set()).add(val)
        elif line.startswith("+"):
            plus.setdefault(op, set()).add(val)
    for op, dels in mins.items():
        adds = plus.get(op, set())
        for d in dels:
            for a in adds:
                if a > d * 2:  # 阈值翻倍以上视为可疑放大
                    out.append({
                        "type": "threshold_swell",
                        "line": f"{op}{d} -> {op}{a}",
                        "desc": "阈值被放大，可能放宽断言严格度",
                    })
    return out


def main():
    ap = argparse.ArgumentParser(description="改测试放宽检测（堵文章23非法消红）")
    ap.add_argument("--base", default="origin/main", help="对比基准分支/提交")
    ap.add_argument("--tests-dir", default="apps/gateway/tests", help="测试目录（相对仓库根）")
    ap.add_argument("--json", help="导出结果 JSON")
    ap.add_argument("--strict", action="store_true", help="出现任何可疑项即阻断（含 possible_relax）")
    args = ap.parse_args()

    files = git_diff_files(args.base)
    test_files = [
        (status, path) for status, path in files
        if path.startswith(args.tests_dir) and path.endswith(".py")
    ]

    print(f"\n=== 改测试放宽检测 ===")
    print(f"基准: {args.base}")
    print(f"测试目录: {args.tests_dir}")
    print(f"改动测试文件数: {len(test_files)}\n")

    all_findings = []
    new_files = []
    modified = []
    for status, path in test_files:
        if status.startswith("A"):
            new_files.append(path)
            print(f"  [新增] {path}  (合法，免审)")
            continue
        modified.append(path)
        findings = analyze_test_file(path, args.base)
        if findings:
            print(f"  [改动] {path}  -> {len(findings)} 处可疑:")
            for f in findings:
                print(f"        - {f['type']}: {f.get('desc','')} | {f['line']}")
            all_findings.append({"file": path, "findings": findings})
        else:
            print(f"  [改动] {path}  (无可疑放宽)")

    # 判定
    blocking = []
    for item in all_findings:
        for f in item["findings"]:
            if f["type"] == "assert_deleted" or f["type"] == "threshold_swell":
                blocking.append({"file": item["file"], **f})
            elif args.strict and f["type"] == "possible_relax":
                blocking.append({"file": item["file"], **f})

    print(f"\n--- 汇总 ---")
    print(f"  新增测试文件: {len(new_files)}（免审）")
    print(f"  改动测试文件: {len(modified)}")
    print(f"  可疑放宽项:   {sum(len(i['findings']) for i in all_findings)}")
    print(f"  阻断项:       {len(blocking)}")
    if blocking:
        print(f"\n  阻断项清单（需分类声明 + 分离 PR，或在 PR 描述加 [relax-test:理由] 豁免）:")
        for b in blocking:
            print(f"    - {b['file']}: {b.get('desc','')} | {b['line']}")

    blocked = len(blocking) > 0
    print(f"\n  门禁: {'❌ 阻断' if blocked else '✅ 通过'}")

    if args.json:
        payload = {
            "base": args.base,
            "new_files": new_files,
            "modified": modified,
            "blocking": blocking,
            "blocked": blocked,
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果已导出: {args.json}")

    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()

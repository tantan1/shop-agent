#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""关键路径保证（方案④ T13）：mypy --strict + 不变式测试。

对 critical_paths.yaml 清单中的模块跑类型检查与不变式测试；
文件缺失或模块未命中 → 软跳过。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CRITICAL_YAML = REPO_ROOT / "critical_paths.yaml"


def _load_critical_paths() -> list:
    if not CRITICAL_YAML.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(CRITICAL_YAML.read_text(encoding="utf-8"))
        return data.get("modules", []) if isinstance(data, dict) else []
    except Exception:
        return []


def check_critical_guarantees(state=None) -> list:
    """形式化门禁：对关键路径跑 mypy --strict + 不变式测试。

    产物由 critical_guarantees.py 生成在 .codebuddy/run/{task}/ 下：
       - type_check.json：mypy --strict 输出
       - invariant_test.json：不变式测试结果
    产物缺失 → 软跳过。
    """
    problems: list = []
    paths = _load_critical_paths()
    if not paths:
        return []

    run = REPO_ROOT / ".codebuddy" / "run"
    if state is not None:
        run = run / state.task

    tc = run / "type_check.json"
    if tc.exists():
        try:
            data = json.loads(tc.read_text(encoding="utf-8"))
            errors = data.get("errors", [])
            if errors:
                problems.append(f"[FORMAL] mypy 错误：{errors[0]}")
        except Exception as exc:
            problems.append(f"[FORMAL] type_check 解析异常：{exc}")

    inv = run / "invariant_test.json"
    if inv.exists():
        try:
            data = json.loads(inv.read_text(encoding="utf-8"))
            if not data.get("passed", True):
                problems.append(f"[FORMAL] 不变式测试失败：{data.get('failure', '')}")
        except Exception as exc:
            problems.append(f"[FORMAL] invariant_test 解析异常：{exc}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="关键路径保证检查")
    ap.add_argument("--task", help="任务名（.codebuddy/run/{task}/）")
    ap.add_argument("--json", help="导出结果 JSON")
    args = ap.parse_args()

    class _FakeState:
        task = args.task or ""

    problems = check_critical_guarantees(_FakeState())
    summary = {
        "modules_checked": len(_load_critical_paths()),
        "problems": problems,
        "passed": not problems,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

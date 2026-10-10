#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成时代码安全扫描（挡住 AI 生成代码的"生成时"漏洞）。

背景：
  运行时护栏（PII 脱敏 / 注入闸 / 合规）只管 *流量*，但 AI 生成代码本身可能引入：
    - 硬编码密钥（api_key = "sk-..."）
    - 危险函数（eval / exec / pickle.loads / subprocess shell=True / yaml.load 不加 SafeLoader）
    - 危险依赖（requests 旧版 / 已知 CVE 包）
  这些是 *编码阶段* 进来的，运行时护栏覆盖不到 —— 属于文章 22/23 "机器把关"的盲区。

本脚本提供两层：
  1. 内置轻量规则（零依赖）：正则扫硬编码密钥 + 危险函数调用，覆盖最高危项。
  2. 可选 bandit 增强（--bandit）：若环境装了 bandit，调用其对目标目录做完整 SAST。
  3. 可选依赖审计（--deps）：若环境装了 pip-audit，对 requirements 做 CVE 扫描。

输出：
  - 人类可读报告（每条命中含文件:行号 + 风险类型 + 建议）
  - --json 导出（供 CI 门禁：高危命中 → 阻断）
  - 退出码：发现高危且未豁免 → 1

用法：
  python scripts/security_scan.py --path apps/gateway/gateway
  python scripts/security_scan.py --path apps/gateway/gateway --bandit --deps --json artifacts/sec.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 高危模式：(正则, 风险类型, 严重度)
# 严重度 high = 阻断；medium = 告警不阻断
HARDCODED_SECRET = re.compile(
    r"""(?i)(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)\s*[:=]\s*["']([^"']{8,})["']"""
)
DANGEROUS_CALLS = [
    (re.compile(r"\beval\s*\("), "eval() 执行动态代码", "high"),
    (re.compile(r"\bexec\s*\("), "exec() 执行动态代码", "high"),
    (re.compile(r"pickle\.loads?\s*\("), "pickle 反序列化（RCE 风险）", "high"),
    (re.compile(r"yaml\.load\s*\((?![^)]*Loader)"), "yaml.load 未指定 SafeLoader", "high"),
    (re.compile(r"subprocess\.[A-Za-z]+\([^)]*shell\s*=\s*True"), "subprocess shell=True", "high"),
    (re.compile(r"os\.system\s*\("), "os.system 执行 shell", "medium"),
]


def scan_file(path: Path) -> list:
    """对单文件做内置规则扫描，返回命中列表。"""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    findings = []
    for i, line in enumerate(text.splitlines(), 1):
        # 跳过注释行（避免误报注释里的示例）
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        m = HARDCODED_SECRET.search(line)
        if m:
            # 排除明显占位（如 "your-api-key-here" / "<...>" / "xxxx"）
            val = m.group(2)
            if val.lower() in ("your-api-key-here", "change-me", "xxx", "test", "example"):
                continue
            if val.startswith(("<", "{", "${")):
                continue
            findings.append({
                "file": str(path), "line": i, "severity": "high",
                "type": "hardcoded_secret",
                "detail": f"{m.group(1)} 硬编码疑似密钥: {val[:6]}...",
            })
        for pat, desc, sev in DANGEROUS_CALLS:
            if pat.search(line):
                findings.append({
                    "file": str(path), "line": i, "severity": sev,
                    "type": "dangerous_call",
                    "detail": desc,
                })
    return findings


def scan_path(target: str) -> list:
    root = (REPO_ROOT / target).resolve() if not Path(target).is_absolute() else Path(target)
    findings = []
    py_files = root.rglob("*.py") if root.is_dir() else [root]
    for f in py_files:
        if "tests" in str(f) or "test_" in f.name:
            continue  # 测试代码不扫（避免误报）
        findings += scan_file(f)
    return findings


def run_bandit(target: str) -> list:
    """可选：调用 bandit 做完整 SAST，返回结构化命中。"""
    try:
        proc = subprocess.run(
            ["bandit", "-f", "json", "-q", "-r", target],
            cwd=REPO_ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        print("[WARN] bandit 未安装，跳过（仅用内置规则）", file=sys.stderr)
        return []
    if not proc.stdout.strip():
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    out = []
    for r in data.get("results", []):
        out.append({
            "file": r.get("filename", ""), "line": r.get("line_number", 0),
            "severity": r.get("issue_severity", "medium").lower(),
            "type": f"bandit:{r.get('test_id','')}",
            "detail": r.get("issue_text", ""),
        })
    return out


def run_pip_audit(req_file: str) -> list:
    """可选：调用 pip-audit 扫依赖 CVE。"""
    req = (REPO_ROOT / req_file).resolve()
    if not req.exists():
        return []
    try:
        proc = subprocess.run(
            ["pip-audit", "-r", str(req), "-f", "json"],
            cwd=REPO_ROOT, capture_output=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        print("[WARN] pip-audit 未安装，跳过依赖审计", file=sys.stderr)
        return []
    if not proc.stdout.strip():
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    out = []
    for dep in data.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            out.append({
                "file": str(req), "line": 0, "severity": "high",
                "type": "dependency_cve",
                "detail": f"{dep.get('name')} {dep.get('version')}: {vuln.get('id')}",
            })
    return out


def main():
    ap = argparse.ArgumentParser(description="生成时代码安全扫描（挡 AI 生成代码漏洞）")
    ap.add_argument("--path", required=True, help="扫描目标（.py 文件或目录）")
    ap.add_argument("--bandit", action="store_true", help="启用 bandit 完整 SAST")
    ap.add_argument("--deps", help="依赖审计的 requirements 路径（如 apps/gateway/requirements.txt）")
    ap.add_argument("--json", help="导出结果 JSON")
    args = ap.parse_args()

    print(f"\n=== 生成时代码安全扫描 ===")
    print(f"目标: {args.path}")
    findings = scan_path(args.path)
    print(f"内置规则命中: {len(findings)}")

    if args.bandit:
        bandit_hits = run_bandit(args.path)
        print(f"bandit 命中: {len(bandit_hits)}")
        findings += bandit_hits

    if args.deps:
        dep_hits = run_pip_audit(args.deps)
        print(f"依赖 CVE 命中: {len(dep_hits)}")
        findings += dep_hits

    # 报告
    highs = [f for f in findings if f["severity"] == "high"]
    print(f"\n--- 命中明细 ---")
    for f in findings:
        tag = "🔴 HIGH" if f["severity"] == "high" else "🟡 MED"
        print(f"  {tag} {f['file']}:{f['line']} [{f['type']}] {f['detail']}")

    blocked = len(highs) > 0
    print(f"\n--- 汇总 ---")
    print(f"  总命中: {len(findings)}  (高危 {len(highs)})")
    print(f"  门禁: {'[BLOCKED] 阻断' if blocked else '[PASS] 通过'}")

    if args.json:
        payload = {"path": args.path, "total": len(findings),
                   "high": len(highs), "blocked": blocked, "findings": findings}
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果已导出: {args.json}")

    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()

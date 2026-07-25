"""
unified 1.7B 一键评测：merge 产物就绪后自动跑三档（S / M8 / 参数抽取）并汇总。

流程：
  1. 检查 merge 产物 ./models/Qwen3-1.7B-unified 是否存在（缺则提示先 train --merge）
  2. S 档工具选择（lscale_S.json，5 工具全集）
  3. M8 档工具选择（lscale_M.json，8 候选）
  4. 参数抽取字段级（shop_param_test.json，base vs unified 对照）
  5. 汇总 markdown 报告 + JSON 到 benchmark_results/

用法：
  python scripts/eval_unified_all.py                          # cuda
  python scripts/eval_unified_all.py --device cpu --runs 1    # CPU 冒烟
  python scripts/eval_unified_all.py --merge-out ./models/Qwen3-1.7B-unified
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

DEFAULT_MODEL = os.path.join(ROOT, "models/Qwen3-1.7B-unified")
DEFAULT_BASE = os.path.join(ROOT, "models/Qwen3-1.7B")
DATA_S = os.path.join(ROOT, "data/lscale_S.json")
DATA_M = os.path.join(ROOT, "data/lscale_M.json")
DATA_PARAM = os.path.join(ROOT, "data/llamafactory/shop_param_test.json")
OUT_DIR = os.path.join(ROOT, "benchmark_results")


def run(cmd: list[str], tag: str) -> dict:
    print(f"\n{'='*70}\n[{tag}]\n  {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.monotonic()
    r = subprocess.run(cmd, capture_output=False)
    return {"cmd": cmd, "tag": tag, "returncode": r.returncode,
            "elapsed_min": round((time.monotonic() - t0) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="unified merge 后模型路径")
    ap.add_argument("--base", default=DEFAULT_BASE, help="参数抽取对照的 base 模型")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--runs", type=int, default=3, help="工具选择每条采样次数")
    ap.add_argument("--batch-size", type=int, default=8, help="参数抽取 batch")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "eval_unified_all.json"))
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        sys.exit(f"[FATAL] merge 产物不存在: {args.model}\n"
                 f"  请先跑训练并 merge：python scripts/train_unified_sft.py --merge")
    if not os.path.exists(DATA_S) or not os.path.exists(DATA_M) or not os.path.exists(DATA_PARAM):
        sys.exit(f"[FATAL] 评测数据缺失: {DATA_S} / {DATA_M} / {DATA_PARAM}")

    py = sys.executable
    eval_tool = os.path.join(SCRIPT_DIR, "eval_tool_select_sft.py")
    eval_field = os.path.join(SCRIPT_DIR, "eval_field_level.py")

    steps = []

    # 1) S 档（M=0 -> 全集 5 工具，默认 lscale_S.json）
    out_s = os.path.join(OUT_DIR, "eval_unified_S.json")
    steps.append(run([py, eval_tool, "--model", args.model, "--data", DATA_S,
                      "--device", args.device, "--runs", str(args.runs), "--out", out_s],
                     "S 档工具选择"))

    # 2) M8 档（--M 8 显式传 --data，避免脚本默认路径切到 apps/shop-agent/data）
    out_m = os.path.join(OUT_DIR, "eval_unified_M8.json")
    steps.append(run([py, eval_tool, "--model", args.model, "--data", DATA_M,
                      "--M", "8",
                      "--device", args.device, "--runs", str(args.runs), "--out", out_m],
                     "M8 档工具选择"))

    # 3) 参数抽取字段级（base vs unified）
    out_p = os.path.join(OUT_DIR, "eval_unified_param.json")
    steps.append(run([py, eval_field, "--base", args.base, "--sft", args.model,
                      "--data", DATA_PARAM, "--device", args.device,
                      "--batch-size", str(args.batch_size), "--out", out_p],
                     "参数抽取字段级"))

    # ── 汇总 ──
    try:
        s = json.load(open(out_s, encoding="utf-8"))
        m = json.load(open(out_m, encoding="utf-8"))
        p = json.load(open(out_p, encoding="utf-8"))
    except FileNotFoundError:
        sys.exit("[FATAL] 部分评测产物缺失，无法汇总（查看上方各步骤输出）")

    s_sum, m_sum = s["summary"], m["summary"]
    sft_fields = p["sft"]

    # 验收（对齐 TEST_PLAN §5）
    s_ok = s_sum["group5_free_gen"]["overall_acc"] >= 0.95 and \
           s_sum["ambiguous_subset"]["free_acc"] >= 0.95
    m_ok = m_sum["group5_free_gen"]["overall_acc"] >= 0.95
    p_overall = p.get("overall")  # eval_field_level 未聚合整体，用字段命中率近似
    high_ok = all(v["hit_rate"] is not None and v["hit_rate"] >= 0.95
                  for k, v in sft_fields.items() if v["risk"] == "high")

    lines = []
    lines.append("# Unified 1.7B 评测结果汇总")
    lines.append("")
    lines.append(f"> 模型: `{args.model}`  | 设备: {args.device}  | 时间: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## S 档（5 工具全集）")
    lines.append("")
    g5 = s_sum["group5_free_gen"]; g6 = s_sum["group6_constrained"]; amb = s_sum["ambiguous_subset"]
    lines.append(f"- 自由生成: 整体 {g5['overall_acc']:.1%} | 格式合规 {g5['format_compliance']:.1%} | "
                 f"p95 {g5['latency_ms_p95']}ms")
    lines.append(f"- 约束解码: 整体 {g6['overall_acc']:.1%} | 格式合规 {g6['format_compliance']:.1%} | "
                 f"p95 {g6['latency_ms_p95']}ms")
    lines.append(f"- ambiguous 子集: free {amb['free_acc']:.1%} | con {amb['con_acc']:.1%}")
    lines.append(f"- H2 PASS={s_sum['h2_pass']} | H3 PASS={s_sum['h3_pass']}")
    lines.append("")
    lines.append("## M8 档（8 候选）")
    lines.append("")
    g5 = m_sum["group5_free_gen"]; g6 = m_sum["group6_constrained"]
    lines.append(f"- 自由生成: 整体 {g5['overall_acc']:.1%} | 格式合规 {g5['format_compliance']:.1%} | "
                 f"p95 {g5['latency_ms_p95']}ms")
    lines.append(f"- 约束解码: 整体 {g6['overall_acc']:.1%} | 格式合规 {g6['format_compliance']:.1%} | "
                 f"p95 {g6['latency_ms_p95']}ms")
    lines.append("")
    lines.append("## 参数抽取（字段级，unified）")
    lines.append("")
    lines.append("| 字段 | 风险 | 样本数 | 命中率 |")
    lines.append("|---|---|---|---|")
    for k in sorted(sft_fields, key=lambda x: (sft_fields[x]["risk"] != "high", x)):
        v = sft_fields[k]
        hr = f"{v['hit_rate']:.2%}" if v["hit_rate"] is not None else "-"
        lines.append(f"| {k} | {v['risk']} | {v['sample_with_value']} | {hr} |")
    lines.append("")
    lines.append("## 验收判定")
    lines.append("")
    lines.append(f"- S 档（≥95% 且 amb≥95%）: **{'PASS' if s_ok else 'FAIL'}**")
    lines.append(f"- M8 档（≥95%）: **{'PASS' if m_ok else 'FAIL'}**")
    lines.append(f"- 高风险字段命中率（≥95%）: **{'PASS' if high_ok else 'FAIL'}**")
    lines.append("")

    report_path = os.path.join(OUT_DIR, "eval_unified_all.md")
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    result = {"steps": steps,
              "S": s_sum, "M8": m_sum, "param_fields": sft_fields,
              "verdict": {"S": s_ok, "M8": m_ok, "high_risk": high_ok}}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n[DONE] 汇总报告 -> {report_path}\n       JSON -> {args.out}")


if __name__ == "__main__":
    main()

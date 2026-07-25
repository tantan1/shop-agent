"""工具选择优化前后对比聚合器（对应 23-工具选择优化效果对比方案 §3）。

输入：
  --base       BASE 变体跑出的 JSON（如 base_p0p1.json，含 p0p1.per_case）
  --optimized  OPTIMIZED 变体跑出的 JSON（opt_p0p1.json，含 fc.per_case 或 p0p1.per_case）

输出：
  compare_tool_selection_before_after.json
    {
      base, optimized,                # 两份原始汇总
      diff: {total, base_hits, opt_hits, net_salvage, salvaged[], regressions[]},
      regressions: [...],             # 与 doc 同名字段，便于复用 eval 工具
      per_level_diff: {level: {base_hits, opt_hits, salvage, regression}},
      config: {latency_baseline_ms, ...}
    }

验收口径（§4）：D1/D2 用"净救回条数"(救回-回归)，不看百分比。
"""
import argparse
import json
import statistics
from collections import defaultdict


def _pctl(values: list, p: float) -> float:
    """线性插值百分位数，values 无需预排序。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def extract_per_case(summary: dict):
    """返回 (per_case_list, mode)。mode 决定 selected/hit 字段来源。
    优先级：p2（BASE 旧三阶段终态）> fc（合并 FC）> p0p1（embedding/重排）> soft_filter（软预过滤）。"""
    if "p2" in summary and summary["p2"] and "per_case" in summary["p2"]:
        return summary["p2"]["per_case"], "p2"
    if "fc" in summary and summary["fc"] and "per_case" in summary["fc"]:
        return summary["fc"]["per_case"], "fc"
    if "p0p1" in summary and summary["p0p1"] and "per_case" in summary["p0p1"]:
        return summary["p0p1"]["per_case"], "p0p1"
    if "soft_filter" in summary and summary["soft_filter"] and "per_case" in summary["soft_filter"]:
        return summary["soft_filter"]["per_case"], "soft_filter"
    raise KeyError("找不到带 per_case 的结果段（需要 p2 / fc / p0p1 / soft_filter）")


def norm(pc: dict, mode: str):
    """归一化为 (idx, correct, selected, hit, level)。"""
    if mode == "fc":
        selected = pc.get("selected")
        hit = bool(pc.get("fc_hit"))
    elif mode == "p2":
        selected = pc.get("selected")
        hit = bool(pc.get("p2_hit"))
    elif mode == "soft_filter":
        selected = pc.get("p1_top1")
        hit = bool(pc.get("p1_top1_hit"))
    else:  # p0p1
        selected = pc.get("p1_top1")
        hit = bool(pc.get("p1_top1_hit"))
    return pc["idx"], pc.get("correct_tool"), selected, hit, pc.get("level", "exact")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="BASE 变体 JSON")
    ap.add_argument("--optimized", required=True, help="OPTIMIZED 变体 JSON")
    ap.add_argument("--output", default="compare_tool_selection_before_after.json")
    ap.add_argument("--latency-baseline", type=float, default=182.0,
                    help="D3 延迟基线 p95 (ms)，文档 22 引用值")
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(args.optimized, "r", encoding="utf-8") as f:
        opt = json.load(f)

    base_cases, base_mode = extract_per_case(base)
    opt_cases, opt_mode = extract_per_case(opt)

    base_map = {idx: (correct, selected, hit, level)
                for idx, correct, selected, hit, level in (norm(pc, base_mode) for pc in base_cases)}
    opt_map = {idx: (correct, selected, hit, level)
               for idx, correct, selected, hit, level in (norm(pc, opt_mode) for pc in opt_cases)}

    common = sorted(set(base_map) & set(opt_map))
    base_hits = opt_hits = 0
    salvaged, regressions = [], []
    per_level = defaultdict(lambda: {"base_hits": 0, "opt_hits": 0, "salvage": 0, "regression": 0})
    per_intent = defaultdict(lambda: {"base_hits": 0, "opt_hits": 0, "salvage": 0, "regression": 0})

    for idx in common:
        b_correct, b_sel, b_hit, level = base_map[idx]
        o_correct, o_sel, o_hit, _ = opt_map[idx]
        if b_hit:
            base_hits += 1
        if o_hit:
            opt_hits += 1
        per_level[level]["base_hits"] += 1 if b_hit else 0
        per_level[level]["opt_hits"] += 1 if o_hit else 0

        if (not b_hit) and o_hit:
            salvaged.append({"idx": idx, "message": "", "correct_tool": o_correct,
                             "base_selected": b_sel, "opt_selected": o_sel, "level": level})
        elif b_hit and (not o_hit):
            regressions.append({"idx": idx, "message": "", "correct_tool": b_correct,
                                "base_selected": b_sel, "opt_selected": o_sel, "level": level})
            per_level[level]["regression"] += 1
        if (not b_hit) and o_hit:
            per_level[level]["salvage"] += 1

        # 按工具维度（correct_tool）统计，覆盖文档 23 §3 per_intent_diff
        tool = o_correct or b_correct
        per_intent[tool]["base_hits"] += 1 if b_hit else 0
        per_intent[tool]["opt_hits"] += 1 if o_hit else 0
        if (not b_hit) and o_hit:
            per_intent[tool]["salvage"] += 1
        elif b_hit and (not o_hit):
            per_intent[tool]["regression"] += 1

    # 回填 message 便于人工复核
    msg_of = {pc["idx"]: pc.get("message", "") for pc in base_cases}
    for lst in (salvaged, regressions):
        for item in lst:
            item["message"] = msg_of.get(item["idx"], "")

    total = len(common)
    net_salvage = len(salvaged) - len(regressions)

    # D1: ambiguous 子集净救回（验收看这里，不看百分比）
    amb = per_level.get("ambiguous", {"base_hits": 0, "opt_hits": 0, "salvage": 0, "regression": 0})
    amb_net = amb["salvage"] - amb["regression"]

    # D3: 延迟（文档 25 §4 要求 p95；基线优先取 BASE 实测 p95，否则用文档引用 182ms）
    # 若两侧都未测延迟（如 embedding 级别对比，未跑 LLM），标记 N/A，避免误用 182 基线。
    opt_latency = opt.get("fc", {}) or {}
    # 若 optimized 是 soft_filter（FC-on-TopK 结果仍写在 soft_filter key 下，无顶层
    # p95_latency_ms），从 per_case 聚合延迟，确保 D3 能拿到 V3 实测延迟。
    if not opt_latency.get("p95_latency_ms") and opt.get("soft_filter"):
        sf_pc = (opt["soft_filter"] or {}).get("per_case", []) or []
        lats = [pc.get("latency_ms") for pc in sf_pc
                if isinstance(pc.get("latency_ms"), (int, float)) and pc["latency_ms"] > 0]
        if lats:
            opt_latency = {
                "p50_latency_ms": round(statistics.median(lats), 1),
                "p95_latency_ms": round(_pctl(lats, 0.95), 1),
            }
    base_p2 = base.get("p2", {}) or {}
    base_p95 = base_p2.get("p95_latency_ms")
    opt_p95 = opt_latency.get("p95_latency_ms")
    if base_p95 is None and opt_p95 is None:
        d3 = {
            "optimized_p50_ms": None, "optimized_p95_ms": None,
            "baseline_p95_ms": None, "baseline_source": "no_latency_data",
            "p95_within_baseline": None, "verdict": "N/A",
        }
    else:
        baseline_source = "base_p2_measured" if base_p95 is not None else "doc_reference_182"
        baseline_p95 = base_p95 if base_p95 is not None else args.latency_baseline
        within = opt_p95 is not None and opt_p95 <= baseline_p95
        d3 = {
            "optimized_p50_ms": opt_latency.get("p50_latency_ms"),
            "optimized_p95_ms": opt_p95,
            "baseline_p95_ms": baseline_p95,
            "baseline_source": baseline_source,
            "p95_within_baseline": within,
            "verdict": "PASS" if within else "FAIL",
        }

    result = {
        "base": {"variant": base.get("variant"), "summary": base},
        "optimized": {"variant": opt.get("variant"), "summary": opt},
        "diff": {
            "total_compared": total,
            "base_hits": base_hits,
            "opt_hits": opt_hits,
            "net_salvage": net_salvage,
            "salvaged_count": len(salvaged),
            "regression_count": len(regressions),
            "d1_ambiguous": {
                "base_hits": amb["base_hits"], "opt_hits": amb["opt_hits"],
                "salvage": amb["salvage"], "regression": amb["regression"],
                "net_salvage": amb_net,
                "verdict": "PASS" if amb_net > 0 else "FAIL",
            },
            "d2_p0_salvage": {
                "narrowed_miss_in_base": base.get("p0", {}).get("narrowed_misses"),
                "net_salvage": net_salvage,
                "verdict": "PASS" if net_salvage > 0 else "FAIL",
            },
            "d3_latency": d3,
            "d4_overall": {
                "opt_hit_rate": round(opt_hits / total, 4) if total else None,
                "verdict": "PASS" if opt_hits >= base_hits else "FAIL",
            },
            "salvaged": salvaged,
        },
        "regressions": regressions,
        "per_level_diff": {lv: dict(v) for lv, v in per_level.items()},
        "per_intent_diff": {name: dict(v) for name, v in per_intent.items()},
        "config": {
            "base_mode": base_mode, "opt_mode": opt_mode,
            "latency_baseline_ms": args.latency_baseline,
            "note": "D1/D2 以净救回条数为判定（见 23 文档 §4），不依赖百分比。",
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"对比完成 -> {args.output}")
    print(f"  可比条数: {total}")
    print(f"  BASE 命中: {base_hits}  OPT 命中: {opt_hits}  净救回: {net_salvage}")
    print(f"  D1(ambiguous) 净救回: {amb_net}  -> {result['diff']['d1_ambiguous']['verdict']}")
    print(f"  D2(P0救回)   净救回: {net_salvage}  -> {result['diff']['d2_p0_salvage']['verdict']}")
    print(f"  D3 延迟 p95={d3['optimized_p95_ms']}ms (基线 {d3['baseline_p95_ms']}ms, {d3['baseline_source']}) -> {d3.get('verdict', d3['p95_within_baseline'])}")
    print(f"  D4 综合命中率: {result['diff']['d4_overall']['opt_hit_rate']} -> {result['diff']['d4_overall']['verdict']}")


if __name__ == "__main__":
    main()

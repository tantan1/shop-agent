"""
对比 零样本(zero-shot) / few-shot / 微调(sft) 在参数抽取上的准确性与成本。

复用 eval_sft_before_after.py 的推理与 prompt 逻辑（坑④ 一致性已内置），
仅新增「few-shot 构造器」和「三条件同测试集对比」。

三种条件：
  - zero_shot : base 模型 + 仅 system prompt（无样例）          —— 现有 base 评测
  - few_shot  : base 模型 + system + K 个同意图样例             —— 新增
  - sft       : 微调后模型 + 仅 system prompt（无样例）         —— 现有 sft 评测

公平性原则：
  - 三者共用同一份测试集与同一套 eval_extraction（字段级 P/R/F1 + 整条达标率）。
  - few-shot 样例池默认从测试集按 query 去重切出（测试集剔除这些问句）；也可通过
    --fewshot-pool-data 指定训练集作为样例来源，测试集保持完整（复现简历基线口径）。
  - 额外统计 avg_input_tokens（样例税的成本代理）与 avg_latency_ms。

用法：
  # 先校验数据/样例拆分（无需 GPU）
  python scripts/compare_prompt_vs_sft.py --dry-run

  # 跑三条件对比（需 GPU + 权重）
  python scripts/compare_prompt_vs_sft.py \
      --base ./models/Qwen2.5-1.5B-Instruct \
      --sft  ./models/Qwen2.5-1.5B-Instruct-sft \
      --data data/llamafactory/shop_param_v1.json \
      --device cuda --k-fewshot 4 --batch-size 8 \
      --out compare_prompt_vs_sft.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import eval_sft_before_after as E  # 复用 load_model/generate_batch/extract_json_from_text/eval_extraction/load_testset

# 与 eval_sft_before_after 保持一致的评测 prompt（坑④）
PARAM_EXTRACTION_PROMPTS = E.PARAM_EXTRACTION_PROMPTS


# ── 数据加载：切分 few-shot 池与测试集（不交叠） ───────────────────────────
def _parse_records(data_path: str) -> Dict[str, List[dict]]:
    """把 ShareGPT json 解析成 {intent: [ {query, gold}, ... ]}。"""
    raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
    by_intent: Dict[str, List[dict]] = defaultdict(list)
    for rec in raw:
        intent = rec.get("intent")
        convs = rec.get("conversations", [])
        query = gold_raw = None
        for c in convs:
            if c["role"] == "user":
                query = c["content"]
            elif c["role"] == "assistant":
                gold_raw = c["content"]
        if query is None or gold_raw is None or intent not in PARAM_EXTRACTION_PROMPTS:
            continue
        try:
            gold = json.loads(gold_raw)
        except Exception:
            continue
        if not isinstance(gold, dict):
            continue
        by_intent[intent].append({"intent": intent, "query": query, "gold": gold})
    return by_intent


def _build_pool(by_intent: Dict[str, List[dict]], k_fewshot: int,
                exclude: Optional[set] = None) -> Dict[str, List[dict]]:
    """每意图取 K 条 *不同问句* 作为 few-shot 样例；exclude 中的 query 跳过。"""
    pool: Dict[str, List[dict]] = {}
    for intent, items in by_intent.items():
        picked: List[dict] = []
        seen: set = set()
        for it in items:
            if it["query"] in seen:
                continue
            if exclude is not None and it["query"] in exclude:
                continue
            if len(picked) >= k_fewshot:
                break
            picked.append(it)
            seen.add(it["query"])
        pool[intent] = picked
    return pool


def load_split(data_path: str, k_fewshot: int, max_samples: Optional[int],
               pool_data_path: Optional[str] = None) -> tuple:
    """返回 (fewshot_pool, testset)。

    两种模式：
      A) pool_data_path 省略（默认）：样例池与测试集来自同一文件，按 query 去重
         切分，测试集剔除样例问句（避免泄漏虚高）。适合全量 v1 数据自比。
      B) pool_data_path 指定（推荐对齐简历基线）：测试集 = data_path 完整样本
         （含 check-balance 送分题，复现简历 23%→98% 的 360 条口径），样例池从
         pool_data_path（训练集）抽取且排除与测试集同问句，三者共用同一 360 测试集。

    模式 B 下测试集不被削减，可直接对齐 eval_sft_before_after.json 的 360 子集。
    """
    test_by = _parse_records(data_path)
    testset: List[dict] = []
    for items in test_by.values():
        testset.extend(items)
    if max_samples:
        testset = testset[:max_samples]

    if pool_data_path and pool_data_path != data_path:
        # 模式 B：样例池来自训练集，排除任何出现在测试集的问句
        test_queries = {t["query"] for t in testset}
        pool_by = _parse_records(pool_data_path)
        fewshot_pool = _build_pool(pool_by, k_fewshot, exclude=test_queries)
    else:
        # 模式 A：同文件去重切分
        fewshot_pool = _build_pool(test_by, k_fewshot)
        pool_q = {q for v in fewshot_pool.values() for q in (x["query"] for x in v)}
        testset = [t for t in testset if t["query"] not in pool_q]
    return fewshot_pool, testset


def dry_run(data_path: str, k_fewshot: int, pool_data_path: Optional[str] = None):
    few, test = load_split(data_path, k_fewshot, None, pool_data_path)
    print(f"[DRY-RUN] few-shot 池: {sum(len(v) for v in few.values())} 条 "
          f"(每意图 K={k_fewshot})")
    print(f"[DRY-RUN] 测试集: {len(test)} 条")
    print(f"[DRY-RUN] 意图分布(pool): { {k: len(v) for k, v in few.items()} }")
    print(f"[DRY-RUN] 意图分布(test): "
          f"{ {i: sum(1 for t in test if t['intent']==i) for i in few} }")
    pool_q = {q for v in few.values() for q in (x["query"] for x in v)}
    overlap = sum(1 for t in test if t["query"] in pool_q)
    print(f"[DRY-RUN] 样例-测试重叠数(应为0): {overlap}")
    return


# ── 消息构造器 ─────────────────────────────────────────────────────────────
def builder_zero_shot(case: dict) -> list:
    """base 现状：system + query，无样例。"""
    return E.build_messages(case["intent"], case["query"])


def make_builder_few_shot(fewshot_pool: Dict[str, List[dict]], k: int) -> Callable:
    def builder(case: dict) -> list:
        intent = case["intent"]
        sys = PARAM_EXTRACTION_PROMPTS[intent]
        msgs = [{"role": "system", "content": sys}]
        for ex in fewshot_pool.get(intent, [])[:k]:
            msgs.append({"role": "user", "content": ex["query"]})
            msgs.append({"role": "assistant",
                         "content": json.dumps(ex["gold"], ensure_ascii=False)})
        msgs.append({"role": "user", "content": case["query"]})
        return msgs
    return builder


# ── 单条件评测（支持自定义消息构造器） ─────────────────────────────────────
def run_eval(model_path: str, device: str, testset: List[dict],
             builder: Callable[[dict], list],
             batch_size: int = 8, max_new_tokens: int = 128) -> tuple:
    model, tokenizer = E.load_model(model_path, device)
    enable_thinking = "Qwen3" not in model_path

    per_case: List[dict] = []
    latencies: List[float] = []
    input_tokens: List[int] = []
    total_tp = total_fp = total_fn = 0
    value_matches = 0
    intent_stat: Dict[str, List[int]] = {}
    err_counter = defaultdict(int)

    total = len(testset)
    tag = Path(model_path).name
    print(f"  [eval] {tag}: {total} 条 (builder={builder.__name__})", flush=True)

    def _accumulate(i: int, case: dict, raw: str, elapsed: float, in_tok: int):
        nonlocal total_tp, total_fp, total_fn, value_matches
        latencies.append(elapsed)
        input_tokens.append(in_tok)
        data = E.extract_json_from_text(raw) or {}
        if not isinstance(data, dict):
            data = {}
        predicted = {k: v for k, v in data.items() if v is not None}
        m = E.eval_extraction(predicted, case["gold"])
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        if m["value_exact_match"]:
            value_matches += 1
        istat = intent_stat.setdefault(case["intent"], [0, 0])
        istat[1] += 1
        if m["value_exact_match"]:
            istat[0] += 1
        err_counter[m["error_type"]] += 1
        per_case.append({
            "idx": i + 1, "intent": case["intent"], "query": case["query"],
            "predicted": predicted, "gold": case["gold"],
            "latency_ms": round(elapsed, 1), "input_tokens": in_tok, **m,
        })

    starts = range(0, total, max(1, batch_size))
    for start in starts:
        chunk = testset[start:start + max(1, batch_size)]
        batch_messages = [builder(c) for c in chunk]
        texts, per_lat = E.generate_batch(
            model, tokenizer, batch_messages, max_new_tokens, device, enable_thinking
        )
        for j, (case, raw) in enumerate(zip(chunk, texts)):
            in_tok = len(tokenizer.encode(
                tokenizer.apply_chat_template(
                    batch_messages[j], tokenize=False, add_generation_prompt=True
                )
            ))
            _accumulate(start + j, case, raw, per_lat, in_tok)

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tok = sum(input_tokens) / len(input_tokens) if input_tokens else 0.0
    intent_rate = {k: round(v[0] / v[1], 4) if v[1] else 0.0
                   for k, v in intent_stat.items()}
    summary = {
        "model": model_path,
        "condition": builder.__name__,
        "total_samples": total,
        "field_precision": round(macro_p, 4),
        "field_recall": round(macro_r, 4),
        "field_f1": round(macro_f1, 4),
        "value_exact_match_rate": round(value_matches / total, 4) if total else 0.0,
        "value_matches": value_matches,
        "avg_latency_ms": round(avg_lat, 1),
        "avg_input_tokens": round(avg_tok, 1),
        "per_intent_value_match_rate": intent_rate,
        "error_breakdown": dict(err_counter),
    }
    return summary, per_case


# ── 主流程 ─────────────────────────────────────────────────────────────────
METRIC_DOC = {
    "value_exact_match_rate": "整条达标率（最严格核心指标）：所有 gold 字段键和值完全相等才算对",
    "field_f1": "字段级 F1：键名维度的精确率/召回率调和平均",
    "avg_latency_ms": "单条生成平均延迟（毫秒）",
    "avg_input_tokens": "单条平均输入 token 数（样例税的成本代理：few-shot 显著高于 zero-shot）",
}


def main():
    ap = argparse.ArgumentParser(description="对比 zero-shot / few-shot / sft 参数抽取")
    ap.add_argument("--base", default=None, help="微调前模型目录（zero-shot/few-shot 用）")
    ap.add_argument("--sft", default=None, help="微调后模型目录（sft 用）")
    ap.add_argument("--data", default="data/llamafactory/shop_param_v1.json")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--k-fewshot", type=int, default=4,
                    help="每个意图塞入的 few-shot 样例数")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--fewshot-pool-data", default=None,
                    help="few-shot 样例来源文件（默认=测试集同文件去重切分；"
                         "指定训练集可让测试集保持完整、复现简历基线口径）")
    ap.add_argument("--out", default="compare_prompt_vs_sft.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.data, args.k_fewshot, args.fewshot_pool_data)
        return

    if not args.base or not args.sft:
        ap.error("--base 和 --sft 都必填（除非 --dry-run）")

    fewshot_pool, testset = load_split(args.data, args.k_fewshot, args.max_samples,
                                       args.fewshot_pool_data)
    print(f"[INFO] few-shot 池 {sum(len(v) for v in fewshot_pool.values())} 条；"
          f"测试集 {len(testset)} 条")

    builder_fs = make_builder_few_shot(fewshot_pool, args.k_fewshot)

    print(f"\n[1/3] zero-shot (base)")
    zs_sum, _ = run_eval(args.base, args.device, testset,
                          builder_zero_shot, batch_size=args.batch_size)
    print(f"      value_match={zs_sum['value_exact_match_rate']:.4f} "
          f"f1={zs_sum['field_f1']:.4f} lat={zs_sum['avg_latency_ms']}ms "
          f"tok={zs_sum['avg_input_tokens']}")

    print(f"\n[2/3] few-shot (base, K={args.k_fewshot})")
    fs_sum, _ = run_eval(args.base, args.device, testset,
                          builder_fs, batch_size=args.batch_size)
    print(f"      value_match={fs_sum['value_exact_match_rate']:.4f} "
          f"f1={fs_sum['field_f1']:.4f} lat={fs_sum['avg_latency_ms']}ms "
          f"tok={fs_sum['avg_input_tokens']}")

    print(f"\n[3/3] sft (微调后)")
    sft_sum, _ = run_eval(args.sft, args.device, testset,
                           builder_zero_shot, batch_size=args.batch_size)
    print(f"      value_match={sft_sum['value_exact_match_rate']:.4f} "
          f"f1={sft_sum['field_f1']:.4f} lat={sft_sum['avg_latency_ms']}ms "
          f"tok={sft_sum['avg_input_tokens']}")

    # ── 对比表 ──
    conds = {"zero-shot": zs_sum, "few-shot": fs_sum, "sft": sft_sum}
    print("\n" + "=" * 78)
    print("三条件对比（同测试集 / 同口径）")
    print("=" * 78)
    hdr = f"{'指标':<24}{'zero-shot':<14}{'few-shot':<14}{'sft':<14}"
    print(hdr)
    print("-" * 66)
    for k in ("value_exact_match_rate", "field_f1",
              "avg_latency_ms", "avg_input_tokens"):
        row = "".join(f"{conds[c][k]:<14}" for c in conds)
        if k in ("avg_latency_ms", "avg_input_tokens"):
            # 整数展示
            row = (f"{conds['zero-shot'][k]:<14.0f}"
                   f"{conds['few-shot'][k]:<14.0f}"
                   f"{conds['sft'][k]:<14.0f}")
        print(f"{k:<24}{row}")
    print("\n逐意图 value_exact_match_rate:")
    intents = sorted({i for c in conds.values() for i in c["per_intent_value_match_rate"]})
    for it in intents:
        line = "  " + f"{it:<18}"
        for c in conds:
            line += f"{conds[c]['per_intent_value_match_rate'].get(it, 0):.3f}    "
        print(line)

    result = {
        "指标说明": METRIC_DOC,
        "config": {"base": args.base, "sft": args.sft, "data": args.data,
                   "fewshot_pool_data": args.fewshot_pool_data,
                   "k_fewshot": args.k_fewshot, "device": args.device,
                   "batch_size": args.batch_size,
                   "testset_size": len(testset),
                   "fewshot_pool_size": sum(len(v) for v in fewshot_pool.values())},
        "zero-shot": zs_sum,
        "few-shot": fs_sum,
        "sft": sft_sum,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n[DONE] 详细结果 -> {args.out}")


if __name__ == "__main__":
    main()

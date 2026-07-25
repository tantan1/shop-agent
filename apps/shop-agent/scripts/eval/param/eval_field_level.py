"""
字段级参数抽取评测：按字段名统计命中率/精确率/召回率/F1。

目的（对应博客 G 行「评测口径业务化」）：
  抽错订单号(order_id)是资损级风险，远比抽错优惠券(coupon_type)严重。
  但总体 97.8% 是未加权均值，掩盖了字段间差异。本脚本把命中率拆到
  每个字段，直接对比「高风险字段」与「低风险字段」的表现。

复用 eval_sft_before_after.py 的推理/prompt 逻辑（坑④ 一致性已内置），
仅新增字段级累计。

用法：
  python scripts/eval_field_level.py \
      --base ./models/Qwen2.5-1.5B-Instruct \
      --sft  ./models/Qwen2.5-1.5B-Instruct-sft \
      --data data/llamafactory/shop_param_test.json \
      --device cuda --batch-size 16 --out eval_field_level.json
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import eval_sft_before_after as E  # 复用 load_model/generate_batch/load_testset

HIGH_RISK_FIELDS = {"order_id", "tracking_number"}  # 资损级
LOW_RISK_FIELDS = {"coupon_type"}                    # 低风险


def run_field_eval(model_path: str, device: str, testset: List[dict],
                   max_new_tokens: int = 128, batch_size: int = 8):
    """对单个模型跑字段级评测，返回字段统计。"""
    model, tokenizer = E.load_model(model_path, device)
    enable_thinking = "Qwen3" not in model_path

    # 字段级累计（仅统计 gold 中「该字段有值」的样本，即模型应当抽出的样本）
    f_present = defaultdict(int)   # 该字段在 gold 中有值的样本数
    f_correct = defaultdict(int)   # 其中 predicted 值 == gold 值的样本数
    # 字段级 P/R/F1（键名维度）
    f_tp = defaultdict(int)
    f_fp = defaultdict(int)
    f_fn = defaultdict(int)

    total = len(testset)
    tag = Path(model_path).name
    print(f"  [field-eval] {tag}: {total} 条 (batch_size={batch_size})", flush=True)

    starts = range(0, total, max(1, batch_size))
    for start in starts:
        chunk = testset[start:start + max(1, batch_size)]
        msgs = [E.build_messages(c["intent"], c["query"]) for c in chunk]
        texts, _ = E.generate_batch(model, tokenizer, msgs, max_new_tokens,
                                     device, enable_thinking)
        for case, raw in zip(chunk, texts):
            data = E.extract_json_from_text(raw) or {}
            if not isinstance(data, dict):
                data = {}
            pred = {k: v for k, v in data.items() if v is not None}
            gold = case["gold"]
            pk, gk = set(pred), set(gold)
            for k in (pk | gk):
                if k in pk and k in gk:
                    f_tp[k] += 1
                elif k in pk:
                    f_fp[k] += 1
                else:
                    f_fn[k] += 1
            # 命中率口径：gold 该字段有值 -> 模型需抽到且值相等
            for k, gv in gold.items():
                if gv in (None, ""):
                    continue
                f_present[k] += 1
                if pred.get(k) == gv:
                    f_correct[k] += 1

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

    fields = sorted(set(f_present) | set(f_tp))
    out = {}
    for k in fields:
        tp, fp, fn = f_tp[k], f_fp[k], f_fn[k]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        present = f_present[k]
        hit = f_correct[k] / present if present else None
        out[k] = {
            "sample_with_value": present,
            "hit_rate": round(hit, 4) if hit is not None else None,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "risk": ("high" if k in HIGH_RISK_FIELDS else
                     "low" if k in LOW_RISK_FIELDS else "neutral"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="字段级参数抽取评测（对比高风险/低风险字段）")
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--data", default="data/llamafactory/shop_param_test.json")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default="eval_field_level.json")
    args = ap.parse_args()

    testset = E.load_testset(args.data, args.max_samples)
    print(f"[INFO] 测试集: {len(testset)} 条 (from {args.data})")

    print(f"\n[1/2] BASE: {args.base}")
    base_field = run_field_eval(args.base, args.device, testset,
                                batch_size=args.batch_size)
    print(f"\n[2/2] SFT: {args.sft}")
    sft_field = run_field_eval(args.sft, args.device, testset,
                               batch_size=args.batch_size)

    result = {
        "说明": "命中率=该字段在gold中有值的样本中，预测值==gold值的比例；"
                "high=资损级(订单号/物流号)，low=低风险(优惠券)。",
        "base": base_field,
        "sft": sft_field,
        "config": {"base": args.base, "sft": args.sft,
                   "data": args.data, "device": args.device,
                   "batch_size": args.batch_size},
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n[DONE] 字段级结果 -> {args.out}")

    # 控制台速览：高风险 vs 低风险
    print("\n字段命中率速览 (SFT):")
    print(f"  {'字段':<16}{'风险':<8}{'样本数':<8}{'命中率':<10}")
    for k in sorted(sft_field, key=lambda x: (sft_field[x]['risk'] != 'high', x)):
        v = sft_field[k]
        hr = f"{v['hit_rate']:.2%}" if v['hit_rate'] is not None else "-"
        print(f"  {k:<16}{v['risk']:<8}{v['sample_with_value']:<8}{hr:<10}")


if __name__ == "__main__":
    main()

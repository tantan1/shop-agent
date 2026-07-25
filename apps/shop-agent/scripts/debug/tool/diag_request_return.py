"""
诊断脚本：只针对 request-return 意图，跑 SFT 模型，把 error_type == over_extract
的 case（预测多出 gold 没有的字段）逐条打印出来，确认 over_extract 的根因。

复用 eval_sft_before_after.py 的全部推理逻辑（prompt / 解码 / 评分），保证与
正式评测同口径。

用法：
  python scripts/diag_request_return.py --sft ./models/Qwen2.5-1.5B-Instruct-sft \
      --data data/llamafactory/shop_param_v1.json --device cuda
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sft_before_after import (  # type: ignore
    PARAM_EXTRACTION_PROMPTS,
    load_model,
    generate,
    load_testset,
    extract_json_from_text,
    eval_extraction,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="SFT merge 后的模型目录")
    ap.add_argument("--data", default="data/llamafactory/shop_param_v1.json")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--only", default="request-return",
                    help="只看该意图（默认 request-return）")
    args = ap.parse_args()

    testset = load_testset(os.path.join(ROOT, args.data), None)
    target = [c for c in testset if c["intent"] == args.only]
    print(f"[INFO] {args.only} 测试样本: {len(target)} 条")

    model, tokenizer = load_model(args.sft, args.device)
    enable_thinking = "Qwen3" not in args.sft

    over_cases = []
    cnt = Counter()
    for i, case in enumerate(target):
        messages = [
            {"role": "system", "content": PARAM_EXTRACTION_PROMPTS[case["intent"]]},
            {"role": "user", "content": case["query"]},
        ]
        raw, _ = generate(model, tokenizer, messages, args.max_new_tokens,
                          args.device, enable_thinking)
        data = extract_json_from_text(raw) or {}
        if not isinstance(data, dict):
            data = {}
        pred = {k: v for k, v in data.items()
                if v is not None}
        m = eval_extraction(pred, case["gold"])
        cnt[m["error_type"]] += 1
        if m["error_type"] == "over_extract":
            over_cases.append(case | {"predicted": pred, "extra": sorted(
                set(pred.keys()) - set(case["gold"].keys()))})

    print(f"\n[SUMMARY] {args.only} 错误分布: {dict(cnt)}")
    print(f"[OVER_EXTRACT] 共 {len(over_cases)} 条\n")

    for c in over_cases:
        print("=" * 70)
        print(f"query : {c['query']}")
        print(f"gold  : {c['gold']}")
        print(f"pred  : {c['predicted']}")
        print(f"extra (gold 没有但被抽出的字段): {c['extra']}")

    if over_cases:
        # 简单归类：gold 是否为空（缺参）vs 给了部分信息
        empty_gold = sum(1 for c in over_cases if not c["gold"])
        print("\n" + "=" * 70)
        print(f"gold 为空(用户没给任何参数)导致 over 的: {empty_gold} / {len(over_cases)}")
        print(f"gold 非空(给了部分信息仍被多抽)的:        {len(over_cases) - empty_gold} / {len(over_cases)}")


if __name__ == "__main__":
    main()

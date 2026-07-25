"""
诊断某意图在 SFT 模型上的逐条错误（query / gold / pred / error_type）。

复用 eval_sft_before_after.py 的推理逻辑，仅打印 value_exact_match 不达标
的样本，用于确认错误到底是「训练覆盖不足」还是「测试集标注 bug」。

用法：
  python scripts/diag_intent_errors.py --intent check-shipping --device cuda
  python scripts/diag_intent_errors.py --intent request-return --sft ./models/Qwen2.5-1.5B-Instruct-sft
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_sft_before_after import (  # type: ignore
    load_model,
    generate,
    extract_json_from_text,
    load_testset,
    eval_extraction,
    build_messages,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", required=True, help="要诊断的意图，如 check-shipping")
    ap.add_argument("--sft", default="./models/Qwen2.5-1.5B-Instruct-sft")
    ap.add_argument("--data", default="data/llamafactory/shop_param_test.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-show", type=int, default=999, help="最多打印多少条错误")
    args = ap.parse_args()

    testset = [c for c in load_testset(args.data, None) if c["intent"] == args.intent]
    if not testset:
        print(f"[WARN] 测试集里没有意图 {args.intent} 的样本")
        return

    model, tok = load_model(args.sft, args.device)
    enable_thinking = "Qwen3" not in args.sft

    errs = []
    for c in testset:
        msgs = build_messages(c["intent"], c["query"])
        raw, _ = generate(model, tok, msgs, 128, args.device, enable_thinking)
        pred = extract_json_from_text(raw) or {}
        if not isinstance(pred, dict):
            pred = {}
        pred = {k: v for k, v in pred.items() if v is not None}
        m = eval_extraction(pred, c["gold"])
        if not m["value_exact_match"]:
            errs.append((c["query"], c["gold"], pred, m["error_type"]))

    print(f"\n[intent={args.intent}] 样本={len(testset)}  错误={len(errs)}")
    print("=" * 70)
    for i, (q, g, p, et) in enumerate(errs[: args.max_show], 1):
        print(f"[{et}] ({i}/{len(errs)}) query={q!r}")
        print(f"      gold={g}")
        print(f"      pred={p}")
    if len(errs) > args.max_show:
        print(f"... 还有 {len(errs) - args.max_show} 条未打印")


if __name__ == "__main__":
    main()

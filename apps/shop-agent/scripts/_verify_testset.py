"""在 HELD-OUT 测试集 (shop_param_test.json, 360) 上复算：

1) value_exact_match_rate（命中率）—— 对应博客 23.3% / 97.8%
2) 格式合规率（输出可被 JSON 解析）—— 对应博客表格「格式合规」列

两个模型都加载一次，避免重复 load。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sft_before_after import (
    load_model, load_testset, build_messages, generate_batch, extract_json_from_text,
    eval_extraction,
)

MODELS = {
    "base": "./models/Qwen2.5-1.5B-Instruct",
    "sft": "./models/Qwen2.5-1.5B-Instruct-sft",
}
DATA = "data/llamafactory/shop_param_test.json"


def main():
    testset = load_testset(DATA, None)
    print(f"[INFO] 测试集 {len(testset)} 条")
    batches = [build_messages(c["intent"], c["query"]) for c in testset]
    golds = [c["gold"] for c in testset]
    for name, path in MODELS.items():
        model, tok = load_model(path, "cuda")
        enable_thinking = "Qwen3" not in path
        hits = 0
        parses = 0
        for i in range(0, len(batches), 16):
            chunk_b = batches[i:i + 16]
            chunk_g = golds[i:i + 16]
            texts, _ = generate_batch(model, tok, chunk_b, 128, "cuda", enable_thinking)
            for t, g in zip(texts, chunk_g):
                pred = extract_json_from_text(t) or {}
                if not isinstance(pred, dict):
                    pred = {}
                pred = {k: v for k, v in pred.items() if v is not None}
                if extract_json_from_text(t) is not None:
                    parses += 1
                m = eval_extraction(pred, g)
                if m["value_exact_match"]:
                    hits += 1
        hr = hits / len(batches)
        fc = parses / len(batches)
        print(f"[RESULT] {name}: 命中率={hr:.4f} ({hr:.1%}), 格式合规={fc:.4f} ({fc:.1%})")


if __name__ == "__main__":
    main()

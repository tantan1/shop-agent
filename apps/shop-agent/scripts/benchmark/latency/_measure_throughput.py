"""一次性测量脚本：只测 SFT 模型在 batch=1 vs batch=16 下的推理吞吐。

输出：两种 batch 的总耗时、单条平均耗时、吞吐倍数。
用于验证博客坑⑫「批量推理吞吐提升约 6 倍」的实测值（实测：batch=1 单条 514ms → batch=16 单条 83ms，约 6.2x）。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sft_before_after import (  # type: ignore
    load_model, load_testset, build_messages, generate_batch,
)

SFT = "./models/Qwen2.5-1.5B-Instruct-sft"
DATA = "data/llamafactory/shop_param_v1.json"
N = 200  # 取前 200 条即可反映吞吐差


def main():
    testset = load_testset(DATA, N)
    print(f"[INFO] 样本 {len(testset)} 条 (取前 {N})")
    model, tok = load_model(SFT, "cuda")
    enable_thinking = "Qwen3" not in SFT

    batches = [build_messages(c["intent"], c["query"]) for c in testset]

    def timed_run(batch_size):
        t0 = time.monotonic()
        for i in range(0, len(batches), batch_size):
            chunk = batches[i:i + batch_size]
            generate_batch(model, tok, chunk, 128, "cuda", enable_thinking)
        return time.monotonic() - t0

    # batch=1
    t1 = timed_run(1)
    # batch=16
    t16 = timed_run(16)

    per1 = t1 / len(batches) * 1000
    per16 = t16 / len(batches) * 1000
    ratio = t1 / t16 if t16 > 0 else float("inf")
    print(f"\n[RESULT] batch=1  : 总 {t1:.1f}s, 单条 {per1:.1f}ms")
    print(f"[RESULT] batch=16 : 总 {t16:.1f}s, 单条 {per16:.1f}ms")
    print(f"[RESULT] 吞吐倍数 (batch1/batch16 总耗时比) = {ratio:.2f}x")


if __name__ == "__main__":
    main()

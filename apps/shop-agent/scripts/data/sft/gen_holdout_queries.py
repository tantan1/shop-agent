"""
gen_holdout_queries.py — 生成「模型从未见过」的 M 规模 holdout 测试集
=====================================================================
目的：现有 SFT-M8 模型在 lscale_M.json（seed=20260727）上评到 100%，
      因训练 query == 评测 query，数字偏乐观（记忆效应）。
      本脚本复用 lscale_M.json 的同一套 40 工具池（模型训练时见过的工具名），
      仅用不同 seed 重新合成 query 文本，生成模型从未背过的新测试集。
      -> 直接拿现有 SFT-M8 模型评，无需重训，即得干净泛化数字。

用法：
  python scripts/gen_holdout_queries.py --src data/lscale_M.json \
      --out data/lscale_M_holdout.json --seed 99 --n 480
"""
from __future__ import annotations
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_synthetic_lscale import build_queries  # 复用同一套 query 合成逻辑


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/lscale_M.json")
    ap.add_argument("--out", default="data/lscale_M_holdout.json")
    ap.add_argument("--seed", type=int, default=99, help="与训练 seed(20260727)不同的新 seed")
    ap.add_argument("--n", type=int, default=480, help="目标 holdout query 数（<= 工具数×每工具 query 数）")
    args = ap.parse_args()

    src = json.load(open(args.src, encoding="utf-8"))
    tools = src["tools"]
    train_msgs = {q["message"] for q in src["queries"]}
    print(f"[holdout] 载入 tools={len(tools)}，训练 query={len(train_msgs)}")

    new_queries = []
    seen = set(train_msgs)
    seed = args.seed
    attempts = 0
    while len(new_queries) < args.n and attempts < 50:
        rng = random.Random(seed + hash("M") % 1000)
        # build_queries 会产出 len(tools)*qpt 条（M: 40*12=480）
        cand = build_queries(tools, "M", rng)
        for q in cand:
            if q["message"] not in seen:
                seen.add(q["message"])
                new_queries.append(q)
                if len(new_queries) >= args.n:
                    break
        seed += 1
        attempts += 1

    if len(new_queries) < args.n:
        print(f"[holdout] WARN: 仅生成 {len(new_queries)} 条（目标 {args.n}），可能 seed 空间耗尽")
    new_queries = new_queries[: args.n]

    # 结构保持与 lscale_M.json 兼容（tools 不变，queries 换新）
    out = dict(src)
    out["queries"] = new_queries
    out["_holdout"] = {
        "src_seed": 20260727,
        "holdout_seed": args.seed,
        "n_train_queries": len(train_msgs),
        "n_holdout_queries": len(new_queries),
        "overlap_with_train": len(set(q["message"] for q in new_queries) & train_msgs),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[holdout] 写出 {args.out}：queries={len(new_queries)}，与训练重叠={out['_holdout']['overlap_with_train']}")


if __name__ == "__main__":
    main()

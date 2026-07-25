"""
按评测错误类型自动补训练数据（误差驱动增强 / error-driven augmentation）。

思路：
  读 eval_sft_before_after.json 里 sft 的 per_intent_error_breakdown，
  找出 sft 还做错的意图与错误类型，针对性生成新样本追加进训练集：
    - under_extract / value_wrong / mixed 的部分  -> 生成更多「正常模板」样本（覆盖槽位+随机取值）
    - over_extract / mixed 的部分                 -> 生成更多「缺参难例」（gold={} 或局部槽位），
                                                     治「该空时不空、硬补字段」
  生成逻辑完全复用训练集生成器（同模板/同 schema/同 system），不引入分布漂移。

输出：默认写 data/llamafactory/shop_param_v1_aug.json（原训练集 + 新增），不动原文件。
      重新训练时把 dataset_info 指向这个 aug 文件即可。

用法：
  # 默认：读 eval 结果，补出数据写到 _aug.json
  python scripts/augment_by_errors.py

  # 自定义条数策略
  python scripts/augment_by_errors.py --multiplier 4 --max-per-intent 150

  # 直接追加回原训练集（覆盖写入 shop_param_v1.json）
  python scripts/augment_by_errors.py --in-place
"""
import sys
import os
import json
import argparse
import random
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 复用训练集生成器的模板与工具，保证同分布
from scripts.generate_param_sft_data import (  # type: ignore
    SYNTH_TEMPLATES,
    PARAM_EXTRACTION_PROMPTS,
    _coerce_str,
    _to_sharegpt,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 1. 读取评测结果，汇总每意图的失败构成
# =============================================================================
def load_sft_errors(eval_path: str) -> Dict[str, Dict[str, int]]:
    """返回 {intent: {error_type: count}}，只含 sft 的失败分布。"""
    with open(eval_path, encoding="utf-8") as f:
        res = json.load(f)
    if "per_intent_error_breakdown" not in res.get("sft", {}):
        raise SystemExit(
            "[ERR] eval 结果里没有 per_intent_error_breakdown，"
            "请先跑带错误细分的 eval_sft_before_after.py（已支持）"
        )
    return res["sft"]["per_intent_error_breakdown"]


# =============================================================================
# 2. 生成单条样本（复用训练器模板）
# =============================================================================
def _random_fill(spec: dict, rng: random.Random) -> dict:
    fill = {}
    if "order_id" in spec.get("slots", []):
        fill["order_id"] = rng.choice(spec.get("order_ids", ["WB202405270001"]))
    if "phone" in spec.get("slots", []):
        fill["phone"] = rng.choice(spec.get("phones", ["1234"]))
    if "status_filter" in spec.get("slots", []):
        fill["status_filter"] = rng.choice(spec.get("status", ["已发货"]))
    if "tracking_number" in spec.get("slots", []):
        fill["tracking_number"] = rng.choice(spec.get("trackings", ["SF1234567890"]))
    if "reason" in spec.get("slots", []):
        fill["reason"] = rng.choice(spec.get("reasons", ["其他"]))
    if "coupon_type" in spec.get("slots", []):
        fill["coupon_type"] = rng.choice(spec.get("types", ["通用"]))
    return fill


# 开放格式槽位（随机数字串）：必须训练≠测试，否则模型靠背号蒙对，需与测试集取值避撞。
# 类别槽位（reason/coupon_type/status_filter）是闭集，训练/测试共用合法值属正常，不需避开。
OPEN_SLOTS = {"order_id", "tracking_number", "phone"}


def _gen_intent_samples(intent: str, kind: str, n: int, rng: random.Random,
                        test_values: set) -> List[dict]:
    """生成某意图的 n 条样本。kind='normal' 用模板，kind='hard' 用缺参难例。"""
    spec = SYNTH_TEMPLATES[intent]
    prompt = PARAM_EXTRACTION_PROMPTS[intent]
    out = []
    for _ in range(n):
        if kind == "hard":
            query, params = rng.choice(spec["hard_cases"])
            params = _coerce_str(dict(params))
        else:
            # 正常样本：随机模板 + 随机取值；仅开放格式槽位避开测试集取值（保持 holdout 纯净）
            for _try in range(20):
                tmpl, label_fn = rng.choice(spec["templates"])
                fill = _random_fill(spec, rng)
                open_vals = [v for k, v in fill.items() if k in OPEN_SLOTS]
                if not any(v in test_values for v in open_vals):
                    break
            params = _coerce_str(label_fn(fill))
            query = tmpl.format(**fill)
        out.append(_to_sharegpt(intent, prompt, query, params))
    return out


def collect_test_values(test_path: str) -> set:
    """收集测试集 gold 里出现过的所有字符串取值，用于避免训练/测试取值撞车。"""
    s: set = set()
    if not os.path.exists(test_path):
        return s
    data = json.load(open(test_path, encoding="utf-8"))
    for r in data:
        for c in r.get("conversations", []):
            if c["role"] != "assistant":
                continue
            try:
                obj = json.loads(c["content"])
            except Exception:
                continue
            for v in obj.values():
                if isinstance(v, str):
                    s.add(v)
    return s


# =============================================================================
# 3. 按失败分布决定每意图补多少、正常/难例各多少
# =============================================================================
def plan_augmentation(errors: Dict[str, Dict[str, int]], multiplier: int,
                      min_per: int, max_per: int) -> Dict[str, Dict[str, int]]:
    """
    返回 {intent: {"normal": n, "hard": m}}。
      - 总补条数 = max(min_per, 失败总数 * multiplier)，上限 max_per
      - hard 占比 = 真正的 over_extract 在失败中的份额（不含 mixed / value_wrong /
        under_extract）。这些「多抽字段」才该用缺参难例（gold={}）去压；
        mixed / 取值错 / 漏抽 一律靠更多「正常、正确标注」样本来覆盖。

    经验（见诊断）：把 mixed 算半个 over_extract 会让最弱意图(check-shipping)
    被塞一堆缺参难例，反而变差。故 hard 只由 over_extract 真值驱动。
    在修正后的基准上 over_extract 全局为 0，因此本轮增强全部是 normal。
    """
    plan: Dict[str, Dict[str, int]] = {}
    for intent, et in errors.items():
        fails = {k: v for k, v in et.items() if k != "correct"}
        total_fail = sum(fails.values())
        if total_fail == 0:
            continue
        add = max(min_per, total_fail * multiplier)
        add = min(add, max_per)
        over = fails.get("over_extract", 0)   # 仅真正多抽字段才给缺参难例
        hard_ratio = (over / total_fail) if total_fail else 0.0
        hard_n = int(round(add * hard_ratio))
        normal_n = add - hard_n
        plan[intent] = {"normal": normal_n, "hard": hard_n}
    return plan


# =============================================================================
# 4. 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_sft_before_after.json",
                    help="带错误细分的评测结果 json")
    ap.add_argument("--train", default="data/llamafactory/shop_param_v1.json",
                    help="原训练集（仅读取，不覆盖，除非 --in-place）")
    ap.add_argument("--test", default="data/llamafactory/shop_param_test.json",
                    help="独立测试集，用于避开其取值")
    ap.add_argument("--out", default="data/llamafactory/shop_param_v1_aug.json",
                    help="增强后输出文件")
    ap.add_argument("--multiplier", type=int, default=3,
                    help="每意图补条数 = 失败数 × 该系数（下限 min，上限 max）")
    ap.add_argument("--min-per-intent", type=int, default=30)
    ap.add_argument("--max-per-intent", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--in-place", action="store_true",
                    help="直接把新增样本追加回 --train 指向的文件（覆盖写入）")
    args = ap.parse_args()

    errors = load_sft_errors(os.path.join(ROOT, args.eval))
    plan = plan_augmentation(
        errors, args.multiplier, args.min_per_intent, args.max_per_intent
    )
    if not plan:
        print("[INFO] sft 在测试集上零失败，无需补数据。")
        return

    rng = random.Random(args.seed)
    test_values = collect_test_values(os.path.join(ROOT, args.test))

    new_samples: List[dict] = []
    print("\n[PLAN] 按错误类型补数据：")
    for intent, nm in plan.items():
        et = errors[intent]
        gen = _gen_intent_samples(intent, "normal", nm["normal"], rng, test_values)
        gen += _gen_intent_samples(intent, "hard", nm["hard"], rng, test_values)
        new_samples.extend(gen)
        print(f"  - {intent:14s} 失败={et}  -> 新增 normal={nm['normal']} hard={nm['hard']} "
              f"(合计 {len(gen)})")

    # 读取原训练集
    train_path = os.path.join(ROOT, args.train)
    train_data = json.load(open(train_path, encoding="utf-8"))
    out_path = train_path if args.in_place else os.path.join(ROOT, args.out)

    # 防自重复：若已含本次生成标记则先剥离（支持重复运行 --in-place）
    train_data = [s for s in train_data if not s.get("_aug")]

    merged = train_data + [
        {**s, "_aug": True} for s in new_samples  # 标记，便于后续剔除/统计
    ]
    rng.shuffle(merged)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 原训练集 {len(train_data)} 条 + 新增 {len(new_samples)} 条 "
          f"= {len(merged)} 条 -> {out_path}")
    if not args.in_place:
        print("[NEXT] 重新训练时把 dataset_info.json 的 file_name 指向该 aug 文件，例如：")
        print(json.dumps({
            "shop_param_v1_aug": {
                "file_name": os.path.basename(out_path),
                "formatting": "sharegpt",
                "tags": {"system": "system", "user": "user", "assistant": "assistant"},
            }
        }, ensure_ascii=False, indent=2))
    else:
        print("[NEXT] 已就地追加，直接重训 --data 指向原训练集即可（原文件已含 _aug 样本）。")


if __name__ == "__main__":
    main()

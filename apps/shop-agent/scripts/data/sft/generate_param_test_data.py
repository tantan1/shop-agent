"""
生成「与训练集不重叠」的参数抽取测试集（ShareGPT 格式）。

设计目标：
  - 同 schema：5 类意图、字段集合、system prompt 与训练集完全一致（避免坑④不公平）
  - 不同数据：query 措辞全部换写、取值由程序随机生成，确保与训练集的
              query 文本 / 订单号 / 手机尾号 / 快递单号 等完全不重叠
  - 用途：喂给 eval_sft_before_after.py --data 做「泛化能力」评测，
          而不是在训练集上测（那会虚高，见对话上下文）

与训练脚本差异：
  - 训练：固定值池 + 固定模板（值就那么几个，容易背下来）
  - 本脚本：随机值生成器（随机订单号/尾号/快递单号）+ 全新措辞模板

用法：
  python scripts/generate_param_test_data.py
  python scripts/generate_param_test_data.py --per-intent 60 --hard-each 12
"""

import sys
import os
import json
import random
from typing import Dict, List, Any

# ── 允许从项目根运行脚本时 import src（与训练脚本同源，保证 system prompt 一致）──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.chat.schemas import PARAM_EXTRACTION_PROMPTS


# =============================================================================
# 随机值生成器（全新取值，确保与训练集固定值池不重叠）
# =============================================================================
_STATUS = ["待付款", "已发货", "派送中", "已签收"]
_REASONS = ["质量问题", "不想要", "发错货", "与描述不符", "其他"]
_TYPES = ["满减券", "折扣券", "运费券", "通用"]


def _gen_order_id(rng: random.Random) -> str:
    prefix = rng.choice(["WB", "DD", "SO", "EC", "MY"])
    return f"{prefix}{rng.randint(10 ** 11, 10 ** 12 - 1)}"  # 12 位数字，随机


def _gen_tracking(rng: random.Random) -> str:
    prefix = rng.choice(["SF", "YT", "JD", "ZT", "EMS", "HT", "YD"])
    return f"{prefix}{rng.randint(10 ** 9, 10 ** 10 - 1)}"  # 10 位数字，随机


def _gen_phone(rng: random.Random) -> str:
    return f"{rng.randint(0, 9999):04d}"  # 4 位尾号，随机


def _coerce_str(params: Dict[str, Any]) -> Dict[str, Any]:
    """字段类型强约束（博客坑②）：订单号/单号强制字符串。"""
    for k in ("order_id", "tracking_number", "phone"):
        if k in params and params[k] is not None:
            params[k] = str(params[k])
    return params


# =============================================================================
# 测试集模板（全新措辞，与训练脚本 TEMPLATES 不重复）
# 每个意图：make_fill 生成随机槽值；templates=正常样本；hard_cases=缺参难例
# =============================================================================
TEST_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "query-order": {
        "make_fill": lambda r: {
            "order_id": _gen_order_id(r),
            "phone": _gen_phone(r),
            "status_filter": r.choice(_STATUS),
        },
        "templates": [
            ("我想知道订单 {order_id} 现在到哪个节点了", lambda d: {"order_id": d["order_id"]}),
            ("{order_id} 这单目前走到哪一步了", lambda d: {"order_id": d["order_id"]}),
            ("帮我看看 {order_id} 的物流进度怎么样", lambda d: {"order_id": d["order_id"]}),
            ("手机号后四位 {phone} 名下都有什么订单", lambda d: {"phone": d["phone"]}),
            ("把目前{status_filter}的订单都列给我", lambda d: {"status_filter": d["status_filter"]}),
            ("{order_id} 发货了没有，帮我确认下", lambda d: {"order_id": d["order_id"]}),
            ("麻烦查一下单号 {order_id} 的状态", lambda d: {"order_id": d["order_id"]}),
            ("{status_filter}的包裹我还有几个没收到", lambda d: {"status_filter": d["status_filter"]}),
        ],
        "hard_cases": [
            ("查下我买的东西到没到", lambda d: {}),
            ("我的包裹现在在哪呢", lambda d: {}),
            ("帮我看下订单情况", lambda d: {}),
            ("我想查订单但不知道单号", lambda d: {}),
        ],
    },
    "check-shipping": {
        "make_fill": lambda r: {
            "order_id": _gen_order_id(r),
            "tracking_number": _gen_tracking(r),
        },
        "templates": [
            ("帮我追踪一下快递 {tracking_number}", lambda d: {"tracking_number": d["tracking_number"]}),
            ("{tracking_number} 这个运单到哪了", lambda d: {"tracking_number": d["tracking_number"]}),
            ("我想看订单 {order_id} 的配送轨迹", lambda d: {"order_id": d["order_id"]}),
            ("快递 {tracking_number} 是不是已经发出去了", lambda d: {"tracking_number": d["tracking_number"]}),
            ("单号 {order_id} 的物流到哪一步了", lambda d: {"order_id": d["order_id"]}),
            ("{tracking_number} 什么时候能送到", lambda d: {"tracking_number": d["tracking_number"]}),
        ],
        "hard_cases": [
            ("{order_id} 的物流到哪了", lambda d: {"order_id": d["order_id"]}),
            ("我用订单号 {order_id} 查下快递", lambda d: {"order_id": d["order_id"]}),
        ],
    },
    "request-return": {
        "make_fill": lambda r: {
            "order_id": _gen_order_id(r),
            "reason": r.choice(_REASONS),
        },
        "templates": [
            ("我打算把订单 {order_id} 退掉", lambda d: {"order_id": d["order_id"]}),
            ("{order_id} 这单我想退货，因为{reason}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            ("{reason}，所以想退掉 {order_id}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            ("退个货，订单号 {order_id}，{reason}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            # 注意：query 里已显式含"不想要"（合法 reason 值），gold 必须带 reason，
            # 否则测试集内部自相矛盾（其余"不想要"模板都标了 reason），会误记 over_extract。
            ("{order_id} 不想要了能退吗", lambda d: {"order_id": d["order_id"], "reason": "不想要"}),
            ("{order_id} 这个我申请下退货", lambda d: {"order_id": d["order_id"]}),
        ],
        "hard_cases": [
            ("先帮我查 {order_id} 然后申请退货", lambda d: {"order_id": d["order_id"]}),
            ("{order_id} 到哪了，不对我要退了", lambda d: {"order_id": d["order_id"]}),
        ],
    },
    "check-balance": {
        "make_fill": lambda r: {},
        "templates": [
            ("我账户里还有多少余额", lambda d: {}),
            ("帮我看下积分剩多少", lambda d: {}),
            ("钱包里现在还有多少钱", lambda d: {}),
            ("帮我查下账户余额", lambda d: {}),
            ("我的可用额度还有多少", lambda d: {}),
        ],
        "hard_cases": [
            ("券和钱还有多少", lambda d: {}),
            ("余额和积分分别多少", lambda d: {}),
        ],
    },
    "coupon-inquiry": {
        "make_fill": lambda r: {"coupon_type": r.choice(_TYPES)},
        "templates": [
            ("我想领一张{coupon_type}", lambda d: {"coupon_type": d["coupon_type"]}),
            ("{coupon_type}去哪领啊", lambda d: {"coupon_type": d["coupon_type"]}),
            ("有没有{coupon_type}可以用", lambda d: {"coupon_type": d["coupon_type"]}),
            ("帮我看看{coupon_type}的领取入口", lambda d: {"coupon_type": d["coupon_type"]}),
            ("{coupon_type}还能不能领", lambda d: {"coupon_type": d["coupon_type"]}),
        ],
        "hard_cases": [
            ("有没有红包可以拿", lambda d: {"coupon_type": "满减券"}),
            ("打折的券还有没有", lambda d: {"coupon_type": "折扣券"}),
            ("免邮券还能领吗", lambda d: {"coupon_type": "运费券"}),
        ],
    },
}


def _to_sharegpt(intent: str, prompt: str, query: str, params: Dict) -> Dict:
    return {
        "intent": intent,
        "conversations": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(params, ensure_ascii=False)},
        ],
    }


def build_test_samples(per_intent: int = 60, hard_each: int = 12,
                       seed: int = 20260721) -> List[Dict]:
    """生成与训练集不重叠的测试样本。

    per_intent：每个意图的正常样本条数
    hard_each ：每个意图的缺参难例条数
    """
    samples: List[Dict] = []
    rng = random.Random(seed)

    for intent, spec in TEST_TEMPLATES.items():
        prompt = PARAM_EXTRACTION_PROMPTS[intent]
        normal = spec["templates"]
        hard = spec["hard_cases"]

        for _ in range(per_intent):
            tmpl, label_fn = rng.choice(normal)
            fill = spec["make_fill"](rng)
            params = _coerce_str(label_fn(fill))
            query = tmpl.format(**fill)
            samples.append(_to_sharegpt(intent, prompt, query, params))

        for _ in range(hard_each):
            tmpl, label_fn = rng.choice(hard)
            fill = spec["make_fill"](rng)
            params = _coerce_str(label_fn(fill))
            query = tmpl.format(**fill)
            samples.append(_to_sharegpt(intent, prompt, query, params))

    rng.shuffle(samples)
    return samples


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/llamafactory/shop_param_test.json")
    ap.add_argument("--per-intent", type=int, default=60)
    ap.add_argument("--hard-each", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260721)
    args = ap.parse_args()

    samples = build_test_samples(
        per_intent=args.per_intent, hard_each=args.hard_each, seed=args.seed
    )
    print(f"[INFO] 测试样本: {len(samples)} 条")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 写出 {len(samples)} 条 -> {args.out}")
    print("[NOTE] 用该文件跑泛化评测：")
    print(f"  python scripts/eval_sft_before_after.py --base <base> --sft <sft> "
          f"--data {args.out} --device cuda")


if __name__ == "__main__":
    main()

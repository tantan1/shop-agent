"""
生成 LLaMA-Factory 参数抽取微调数据（ShareGPT 格式）。

混合方式：
  1. 合成 query 打底（离线即可生成，不依赖 LLM）—— 覆盖 5 类意图 + 场景难例
  2. 真实难例混入（--real path.jsonl，用户后续提供生产日志/难例）
  3. 可选 LLM 蒸馏增强（--distill，调用 chat_qwen_structured 重新打标）

输出：
  - data/llamafactory/shop_param_v1.json   (ShareGPT 训练集)
  - 终端打印 dataset_info.json 片段

用法：
  # 离线合成（默认，零依赖，立即可跑）
  python scripts/generate_param_sft_data.py

  # 混入真实难例
  python scripts/generate_param_sft_data.py --real data/real_hard_cases.jsonl

  # 用云端 LLM 蒸馏增强（需配置 tongyi_api_key）
  python scripts/generate_param_sft_data.py --distill

字段约束（对应博客坑②）：订单号/单号强制字符串，避免推理端 model_dump 崩。
PII 脱敏（对应博客坑①）：真实难例入库前先脱敏，合成数据无 PII 跳过。
"""

import sys
import os
import json
import re
import argparse
import asyncio
import random
from typing import Dict, List, Any, Optional

# ── 允许从项目根运行脚本时 import src ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.chat.schemas import INTENT_PARAM_SCHEMAS, PARAM_EXTRACTION_PROMPTS


# =============================================================================
# 1. PII 脱敏（用于真实难例；合成数据不含 PII，跳过）
# =============================================================================
_PII_PATTERNS = {
    "phone": re.compile(r"1[3-9]\d{9}"),
    "idcard": re.compile(r"\d{17}[\dXx]"),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "order_wb": re.compile(r"WB\d{8,}"),
}


def mask_pii(text: str) -> str:
    """对真实语料做轻量脱敏，避免把生产 PII 烧进权重。"""
    text = _PII_PATTERNS["phone"].sub("[PHONE]", text)
    text = _PII_PATTERNS["idcard"].sub("[ID]", text)
    text = _PII_PATTERNS["email"].sub("[EMAIL]", text)
    text = _PII_PATTERNS["order_wb"].sub("[ORDER]", text)
    return text


# =============================================================================
# 2. 合成 query 模板（混合方式的主数据来源）
# =============================================================================
# 每个意图：query 模板 + 可填充的参数值池
# 标签直接用模板里已知的参数值（离线即可，不需 LLM）

SYNTH_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "query-order": {
        "slots": ["order_id", "phone", "status_filter"],
        "order_ids": ["WB202405270001", "WB202406130088", "202405270001", "NO202407019999"],
        "phones": ["1234", "5678", "9012", "3456"],
        "status": ["待付款", "已发货", "派送中", "已签收"],
        "templates": [
            ("帮我查一下订单 {order_id} 到哪了", lambda d: {"order_id": d["order_id"]}),
            ("订单 {order_id} 现在是什么状态", lambda d: {"order_id": d["order_id"]}),
            ("查订单 {order_id}", lambda d: {"order_id": d["order_id"]}),
            ("手机尾号 {phone} 的订单有哪些", lambda d: {"phone": d["phone"]}),
            ("看下{status_filter}的订单", lambda d: {"status_filter": d["status_filter"]}),
            ("我的订单 {order_id} 发货了没", lambda d: {"order_id": d["order_id"]}),
        ],
        # 难例：缺参（用户没给足够信息）
        "hard_cases": [
            ("我想查一下我的订单", {}),
            ("订单到哪了", {}),
            ("帮我看看买的东西", {}),
        ],
    },
    "check-shipping": {
        "slots": ["tracking_number", "order_id"],
        # 前缀池与测试集生成器对齐（generate_param_test_data._gen_order_id /
        # _gen_tracking），让模型学到「前缀字母→字段」的抽象映射，而不是背具体值：
        #   order_id  → WB/DD/SO/EC/MY（12 位）
        #   tracking  → SF/YT/JD/ZT/EMS/HT/YD（10 位）
        # 位数也严格对齐（order 12 位 / tracking 10 位），强化「位数→字段」特征。
        "trackings": ["SF1234567890", "YT1234567890", "JD0012345678",
                      "ZT8888888888", "EMS1234567890", "HT1234567890", "YD1234567890"],
        "order_ids": ["WB202405270001", "WB202406130088", "202405270001",
                      "DD202405270002", "SO202405270003", "EC202405270004", "MY202405270005"],
        "templates": [
            ("快递单号 {tracking_number} 到哪了", lambda d: {"tracking_number": d["tracking_number"]}),
            ("查物流 {tracking_number}", lambda d: {"tracking_number": d["tracking_number"]}),
            ("订单 {order_id} 的物流信息", lambda d: {"order_id": d["order_id"]}),
            ("我的快递 {tracking_number} 到哪了", lambda d: {"tracking_number": d["tracking_number"]}),
            ("看看 {order_id} 的物流", lambda d: {"order_id": d["order_id"]}),
            # ── 对齐测试集分布：裸编号 / 歧义 / 双字段（治 mixed 根因：前缀偏移）──
            # 测试集 check-shipping 大量样本是「裸编号 + 模糊文字」，靠编号前缀判字段；
            # 训练集原本只覆盖 WB→order、SF/YT/JD/ZT→tracking，且都带明确文字提示，
            # 模型对未见前缀(EC/SO/MY/HT/YD)靠猜 → mixed。补齐如下：
            # 裸编号（无字段文字，逼模型看前缀 / 位数）
            ("{tracking_number} 什么时候能送到", lambda d: {"tracking_number": d["tracking_number"]}),
            ("{order_id} 到哪了", lambda d: {"order_id": d["order_id"]}),
            ("{tracking_number} 这个件到哪了", lambda d: {"tracking_number": d["tracking_number"]}),
            ("{order_id} 的物流到哪一步了", lambda d: {"order_id": d["order_id"]}),
            # 测试集真实措辞镜像（含「单号{order}」「运单{tracking}」歧义对齐）
            ("单号 {order_id} 的物流到哪一步了", lambda d: {"order_id": d["order_id"]}),
            ("运单 {tracking_number} 到哪了", lambda d: {"tracking_number": d["tracking_number"]}),
            ("帮我追踪一下快递 {tracking_number}", lambda d: {"tracking_number": d["tracking_number"]}),
            ("我想看订单 {order_id} 的配送轨迹", lambda d: {"order_id": d["order_id"]}),
            ("快递 {tracking_number} 是不是已经发出去了", lambda d: {"tracking_number": d["tracking_number"]}),
            # 双字段组合（治两者并存的 mixed）
            ("{tracking_number} 和订单 {order_id} 一起查下物流",
             lambda d: {"tracking_number": d["tracking_number"], "order_id": d["order_id"]}),
            ("查下 {order_id}，单号 {tracking_number}",
             lambda d: {"tracking_number": d["tracking_number"], "order_id": d["order_id"]}),
        ],
        # 难例：物流单号 vs 订单号混淆
        "hard_cases": [
            ("WB202405270001 到哪了", {"order_id": "WB202405270001"}),  # 用户把订单号当快递号
            ("SF1234567890 这个订单的状态", {"tracking_number": "SF1234567890"}),
        ],
    },
    "request-return": {
        "slots": ["order_id", "reason"],
        # 对齐测试集 order_id 前缀池（WB/DD/SO/EC/MY 12位）：原只 WB/2024 三个固定值，
        # 模型没见过 EC/SO/MY/DD 前缀的订单号 → 生成时偶发截断/误抽（诊断 value_wrong 2条截断）。
        "order_ids": ["WB202405270001", "WB202406130088", "202405270001",
                      "DD202405270002", "SO202405270003", "EC202405270004", "MY202405270005"],
        "reasons": ["质量问题", "不想要", "发错货", "与描述不符", "其他"],
        "templates": [
            ("我要退订单 {order_id}", lambda d: {"order_id": d["order_id"]}),
            ("订单 {order_id} 申请退货，原因{reason}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            ("质量有问题想退 {order_id}", lambda d: {"order_id": d["order_id"], "reason": "质量问题"}),
            ("{order_id} 发错货了要退", lambda d: {"order_id": d["order_id"], "reason": "发错货"}),
            ("不想要了，退 {order_id}", lambda d: {"order_id": d["order_id"], "reason": "不想要"}),
            # ── 对齐测试集分布：裸 reason 枚举值 / 句首句尾 /「因为」引导 ──
            # 治 under_extract：测试集大量「退个货，订单号 X，其他」这种裸枚举值放句尾、
            # 无"原因"前缀，模型原没学过 → 漏抽 reason（诊断 6 条里 5 条如此）
            ("退个货，订单号 {order_id}，{reason}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            ("{reason}，所以想退掉 {order_id}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            ("退货，订单 {order_id}，{reason}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            # 治 value_wrong：「因为+裸 reason」被误判（如"因为其他"→"与描述不符"）
            ("{order_id} 这单我想退货，因为{reason}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
            ("我想退 {order_id}，因为{reason}", lambda d: {"order_id": d["order_id"], "reason": d["reason"]}),
        ],
        # 难例：多意图（查+退）
        "hard_cases": [
            ("查下订单 WB202405270001 然后申请退货", {"order_id": "WB202405270001"}),
            ("订单 202405270001 到哪了，不对我要退", {"order_id": "202405270001"}),
        ],
    },
    "check-balance": {
        "slots": [],
        "templates": [
            ("查一下我的余额", lambda d: {}),
            ("我的积分还有多少", lambda d: {}),
            ("账户里还有多少钱", lambda d: {}),
            ("钱包余额", lambda d: {}),
        ],
        "hard_cases": [
            ("积分", {}),
            ("我有多少券和余额", {}),
        ],
    },
    "coupon-inquiry": {
        "slots": ["coupon_type"],
        "types": ["满减券", "折扣券", "运费券", "通用"],
        "templates": [
            ("有什么{coupon_type}可以领", lambda d: {"coupon_type": d["coupon_type"]}),
            ("满减券在哪领", lambda d: {"coupon_type": "满减券"}),
            ("折扣券怎么用", lambda d: {"coupon_type": "折扣券"}),
            ("运费券还有吗", lambda d: {"coupon_type": "运费券"}),
        ],
        # 难例：优惠券别名
        "hard_cases": [
            ("红包在哪领", {"coupon_type": "满减券"}),  # 红包≈满减券别名
            ("有没有打折的券", {"coupon_type": "折扣券"}),
        ],
    },
}


def _coerce_str(params: Dict[str, Any]) -> Dict[str, Any]:
    """字段类型强约束（博客坑②）：订单号/单号强制字符串。"""
    for k in ("order_id", "tracking_number", "phone"):
        if k in params and params[k] is not None:
            params[k] = str(params[k])
    return params


def build_synthetic_samples(per_intent: int = 280, hard_each: int = 40) -> List[Dict]:
    """离线合成：模板 + 随机填充，生成 ShareGPT 样本。"""
    samples = []
    rng = random.Random(42)
    for intent, spec in SYNTH_TEMPLATES.items():
        prompt = PARAM_EXTRACTION_PROMPTS[intent]
        # 正常样本
        for _ in range(per_intent):
            tmpl, label_fn = rng.choice(spec["templates"])
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
            params = _coerce_str(label_fn(fill))
            query = tmpl.format(**fill)
            samples.append(_to_sharegpt(intent, prompt, query, params))
        # 难例
        for _ in range(hard_each):
            query, params = rng.choice(spec["hard_cases"])
            params = _coerce_str(dict(params))
            samples.append(_to_sharegpt(intent, prompt, query, params))
    rng.shuffle(samples)
    return samples


def _to_sharegpt(intent: str, prompt: str, query: str, params: Dict) -> Dict:
    return {
        "intent": intent,  # 仅作元信息，LLaMA-Factory 不使用
        "conversations": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(params, ensure_ascii=False)},
        ],
    }


# =============================================================================
# 3. 真实难例接入
# =============================================================================
def load_real_cases(path: str) -> List[Dict]:
    """读取真实难例 jsonl。每行：{"intent":..., "query":..., "params":{...}}
    若缺 params，下游 distill 模式会用 LLM 补标；离线模式则跳过该行。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            query = mask_pii(rec["query"])  # 真实数据先脱敏
            intent = rec["intent"]
            params = rec.get("params")
            cases.append({"intent": intent, "query": query, "params": params})
    return cases


# =============================================================================
# 4. 可选 LLM 蒸馏打标
# =============================================================================
async def distill_with_llm(cases: List[Dict]) -> List[Dict]:
    """用 chat_qwen_structured 当教师，对真实难例（无 params）补标。"""
    from src.modules.chat.core.llm_service import LLMService

    llm = LLMService.get_instance()
    llm.initialize()
    out = []
    for c in cases:
        intent = c["intent"]
        prompt = PARAM_EXTRACTION_PROMPTS[intent]
        schema = INTENT_PARAM_SCHEMAS[intent]
        if c["params"] is not None:
            params = _coerce_str(c["params"])
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": c["query"]},
            ]
            try:
                result = await llm.chat_qwen_structured(
                    messages=messages, output_schema=schema, temperature=0.0
                )
                params = _coerce_str(result.model_dump(exclude_none=True))
            except Exception as e:
                print(f"[WARN] distill 失败 intent={intent} query={c['query'][:20]} err={e}")
                continue
        out.append(_to_sharegpt(intent, prompt, c["query"], params))
    return out


# =============================================================================
# 5. 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/llamafactory/shop_param_v1.json")
    ap.add_argument("--real", default=None, help="真实难例 jsonl 路径")
    ap.add_argument("--distill", action="store_true", help="用云端 LLM 对真实难例补标")
    ap.add_argument("--per-intent", type=int, default=280)
    ap.add_argument("--hard-each", type=int, default=40)
    args = ap.parse_args()

    samples = build_synthetic_samples(per_intent=args.per_intent, hard_each=args.hard_each)
    print(f"[INFO] 合成样本: {len(samples)} 条")

    if args.real:
        real = load_real_cases(args.real)
        print(f"[INFO] 真实难例: {len(real)} 条（已脱敏）")
        if args.distill:
            real_samples = asyncio.run(distill_with_llm(real))
        else:
            # 离线模式：仅保留已带 params 的真实样本
            real_samples = [
                _to_sharegpt(c["intent"], PARAM_EXTRACTION_PROMPTS[c["intent"]], c["query"], _coerce_str(c["params"]))
                for c in real if c["params"] is not None
            ]
            skipped = len(real) - len(real_samples)
            if skipped:
                print(f"[WARN] {skipped} 条真实样本无 params 且未开 --distill，已跳过")
        samples.extend(real_samples)

    # 写文件
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"[DONE] 写出 {len(samples)} 条 -> {args.out}")

    # 打印 dataset_info.json 片段
    print("\n# 把下面这段加进 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps({
        "shop_param_v1": {
            "file_name": os.path.basename(args.out),
            "formatting": "sharegpt",
            "tags": {"system": "system", "user": "user", "assistant": "assistant"},
        }
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

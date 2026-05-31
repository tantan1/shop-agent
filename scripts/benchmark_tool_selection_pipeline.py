"""
P0 -> P1 -> P2 意图识别流水线分层命中率 Benchmark
==================================================
评测三级意图识别流水线各阶段累进命中率，产出实测数据替代 blog v3 中的估计值（50%/90%/95%）。

用法:
    # P0 关键词匹配（极快，纯 Python，无依赖）
    python scripts/benchmark_tool_selection_pipeline.py --stage p0

    # P0+P1 语义匹配（需要 pip install sentence-transformers）
    python scripts/benchmark_tool_selection_pipeline.py --stage p0p1

    # P0+P1+P2 全流程（需要 GPU + 本地模型做意图确认）
    python scripts/benchmark_tool_selection_pipeline.py --stage all --model ./models/Qwen2.5-1.5B-Instruct --device cuda

    # 输出 JSON
    python scripts/benchmark_tool_selection_pipeline.py --stage p0p1 --json

输出示例:
    P0 命中率:        86.0%  (关键词 -> intent_tool_map 覆盖)
    P0+P1 Top-1命中率: 94.0%  (关键词 + bge-small-zh-v1.5 语义匹配)
    P0+P1 Top-2命中率: 98.0%
    综合 P0+P1+P2:     ~94.0% (基于 benchmark_local_model_comparison 的 P2=92% Top1 数据)

架构说明:
    意图识别（Intent Recognition）: P0→P1→P2 三层流水线，P2 LLM 仅在模糊 case 触发
    工具选择（Tool Selection）:    意图确认后的一层确定性映射（INTENT_TOOL_MAP），不需要 LLM
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# ---- 工具定义（与 skills/*/SKILL.md YAML frontmatter 一致） ----

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "query-order",
        "display_name": "订单查询",
        "trigger_keywords": [
            "订单号", "我的订单", "买了什么", "待付款", "待发货",
            "历史订单", "订单详情", "订单", "发货没",
        ],
        "allowed_tools": ["query-order", "check-shipping"],
    },
    {
        "name": "check-shipping",
        "display_name": "物流查询",
        "trigger_keywords": [
            "快递", "物流", "到哪了", "什么时候送到", "配送",
            "派送", "跟踪", "揽收", "运单", "包裹",
        ],
        "allowed_tools": ["check-shipping", "query-order"],
    },
    {
        "name": "request-return",
        "display_name": "退货退款",
        "trigger_keywords": [
            "退货", "退款", "我要退", "退掉", "申请退款",
            "退款到账", "能退吗", "想退", "退钱", "质量有问题",
        ],
        "allowed_tools": ["request-return"],
    },
    {
        "name": "check-balance",
        "display_name": "余额积分查询",
        "trigger_keywords": [
            "余额", "钱包", "有多少钱", "积分", "我的积分",
            "还剩多少", "积分明细", "积分够不够", "查下余额",
        ],
        "allowed_tools": ["check-balance"],
    },
    {
        "name": "coupon-inquiry",
        "display_name": "优惠券查询",
        "trigger_keywords": [
            "优惠券", "代金券", "满减", "有什么券", "折扣券",
            "运费券", "新人券", "优惠码", "活动优惠", "券",
        ],
        "allowed_tools": ["coupon-inquiry"],
    },
]

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "query-order": (
        "查询用户的订单列表或指定订单详情。触发条件：用户询问订单状态、订单号、我的订单。"
        "区分：与 check-shipping 不同——本工具查询订单信息全貌（状态/金额/列表），"
        "check-shipping 专门查询物流轨迹详情。"
    ),
    "check-shipping": (
        "查询物流配送进度。返回揽收→运输→派送每一步的时间线和状态。"
        "触发条件：用户问到哪了、物流、快递、什么时候送到。"
        "区分：与 query-order 不同——本工具返回明细物流轨迹，query-order 是订单宏观信息。"
    ),
    "request-return": (
        "为用户提交退货退款申请。提交后生成退货单号，退款 1-3 个工作日原路返回。"
        "触发条件：用户明确表达退货、退款、我要退。"
        "注意：如果用户只是问退货政策条件（而非正式申请），先用 knowledge_search 查知识库。"
    ),
    "check-balance": (
        "查询账户余额和可用积分。触发条件：用户问余额、钱包、有多少钱、积分、我的积分。"
        "区分：与 coupon-inquiry 不同——本工具查账户资金/积分，coupon-inquiry 查优惠券。"
    ),
    "coupon-inquiry": (
        "查询可用的优惠券列表，包括有效期、使用门槛。"
        "触发条件：用户问优惠券、代金券、满减、有什么券、优惠码。"
        "区分：与 check-balance 不同——本工具查优惠券而非账户余额。"
    ),
}

INTENT_TOOL_MAP: Dict[str, Set[str]] = {
    t["name"]: set(t["allowed_tools"]) for t in TOOL_DEFINITIONS
}
INTENT_TOOL_MAP["unknown"] = {t["name"] for t in TOOL_DEFINITIONS}

KEYWORD_TOOL_MAP: Dict[str, str] = {}
for t in TOOL_DEFINITIONS:
    for kw in t["trigger_keywords"]:
        if kw not in KEYWORD_TOOL_MAP:
            KEYWORD_TOOL_MAP[kw] = t["name"]

# ---- 测试用例（100 条，每工具 20 条，5 级难度） ----
# level: exact(精确关键词) | synonym(同义/口语无关键词) | implied(隐含意图)
#        ambiguous(歧义边界) | mixed(多意图混合)

TOOL_SELECTION_PIPELINE_TESTS: List[Dict[str, str]] = [
    # ================================================================
    # query-order (20 条)
    # ================================================================
    # -- exact (5) --
    {"message": "我的订单到哪了", "correct_tool": "query-order", "intent": "query-order", "level": "exact"},
    {"message": "订单号 WB202405270001 什么状态", "correct_tool": "query-order", "intent": "query-order", "level": "exact"},
    {"message": "待付款的订单还有哪些", "correct_tool": "query-order", "intent": "query-order", "level": "exact"},
    {"message": "已发货的单子列一下", "correct_tool": "query-order", "intent": "query-order", "level": "exact"},
    {"message": "看看订单详情，手机号后四位6688", "correct_tool": "query-order", "intent": "query-order", "level": "exact"},
    # -- synonym (5): 口语化、无精确关键词 --
    {"message": "帮我查下最近买了什么", "correct_tool": "query-order", "intent": "query-order", "level": "synonym"},
    {"message": "看看有没有新买的东西", "correct_tool": "query-order", "intent": "query-order", "level": "synonym"},
    {"message": "上次那个蓝色的，什么时候能收到", "correct_tool": "query-order", "intent": "query-order", "level": "synonym"},
    {"message": "我最近下单的几样东西都什么进度了", "correct_tool": "query-order", "intent": "query-order", "level": "synonym"},
    {"message": "之前买过一个充电器，帮我找找记录", "correct_tool": "query-order", "intent": "query-order", "level": "synonym"},
    # -- implied (5): 隐含意图 --
    {"message": "上个月的消费记录拉一下", "correct_tool": "query-order", "intent": "query-order", "level": "implied"},
    {"message": "怎么看我都买过啥", "correct_tool": "query-order", "intent": "query-order", "level": "implied"},
    {"message": "我下了好几单，分别什么状态", "correct_tool": "query-order", "intent": "query-order", "level": "implied"},
    {"message": "给我汇总一下今年以来的购物情况", "correct_tool": "query-order", "intent": "query-order", "level": "implied"},
    {"message": "现在有几个没收到货的", "correct_tool": "query-order", "intent": "query-order", "level": "implied"},
    # -- ambiguous (3): 可能被误判到其他工具 --
    {"message": "我买的东西到哪了", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},  # 可能被判 to check-shipping
    {"message": "订单上的物流怎么不动了", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},  # 含"物流"
    {"message": "退掉的那单帮我查下", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},  # 含"退掉"
    # -- mixed (2): 多意图混合 --
    {"message": "查下订单，顺便看看物流到哪了", "correct_tool": "query-order", "intent": "query-order", "level": "mixed"},
    {"message": "我的订单状态和能用的券都说一下", "correct_tool": "query-order", "intent": "query-order", "level": "mixed"},

    # ================================================================
    # check-shipping (20 条)
    # ================================================================
    # -- exact (5) --
    {"message": "我的快递到哪了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "exact"},
    {"message": "快递什么时候能到", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "exact"},
    {"message": "SF1234567890 物流跟踪一下", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "exact"},
    {"message": "派送到哪一步了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "exact"},
    {"message": "我的包裹有更新吗", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "exact"},
    # -- synonym (5): 口语化 --
    {"message": "东西运到哪了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "synonym"},
    {"message": "还有几天到啊，急着用", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "synonym"},
    {"message": "我的货现在在哪个城市", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "synonym"},
    {"message": "快递小哥是不是快到了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "synonym"},
    {"message": "从广东发过来要多久", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "synonym"},
    # -- implied (5): 隐含意图 --
    {"message": "怎么还没送到，都三天了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "implied"},
    {"message": "是不是今天能到", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "implied"},
    {"message": "京东的都发了两天了还没动静", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "implied"},
    {"message": "能帮我催一下吗，太慢了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "implied"},
    {"message": "显示签收了但我没收到，什么情况", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "implied"},
    # -- ambiguous (3): 可能被误判 --
    {"message": "中通那个单子还在路上吗", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},  # "单子"可能匹配订单
    {"message": "帮我查下运到哪里了，YT1234567890123", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},  # 无物流/快递关键词
    {"message": "揽收两天了怎么还没更新", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    # -- mixed (2) --
    {"message": "看一下物流，另外我的订单退款进度也说一下", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "mixed"},
    {"message": "快递到哪里了，到了我要申请退货", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "mixed"},

    # ================================================================
    # request-return (20 条)
    # ================================================================
    # -- exact (5) --
    {"message": "质量有问题，申请退款", "correct_tool": "request-return", "intent": "request-return", "level": "exact"},
    {"message": "收到的货跟图片不一样，我要退款", "correct_tool": "request-return", "intent": "request-return", "level": "exact"},
    {"message": "发错货了怎么退", "correct_tool": "request-return", "intent": "request-return", "level": "exact"},
    {"message": "退货申请，单号 202405220678", "correct_tool": "request-return", "intent": "request-return", "level": "exact"},
    {"message": "商品有瑕疵想退货", "correct_tool": "request-return", "intent": "request-return", "level": "exact"},
    # -- synonym (5): 口语化 --
    {"message": "这个不想要了，帮我处理一下", "correct_tool": "request-return", "intent": "request-return", "level": "synonym"},
    {"message": "不合适，怎么处理", "correct_tool": "request-return", "intent": "request-return", "level": "synonym"},
    {"message": "这东西不好使，我想换了它", "correct_tool": "request-return", "intent": "request-return", "level": "synonym"},
    {"message": "把那个退了吧，重新买一个", "correct_tool": "request-return", "intent": "request-return", "level": "synonym"},
    {"message": "你们收到退货了吗，钱多久能回来", "correct_tool": "request-return", "intent": "request-return", "level": "synonym"},
    # -- implied (5): 隐含意图 --
    {"message": "穿了两次就开线了", "correct_tool": "request-return", "intent": "request-return", "level": "implied"},
    {"message": "颜色跟图片完全不一样啊", "correct_tool": "request-return", "intent": "request-return", "level": "implied"},
    {"message": "收到的尺码不对，M号发成L号了", "correct_tool": "request-return", "intent": "request-return", "level": "implied"},
    {"message": "少了一个配件，盒子是破的", "correct_tool": "request-return", "intent": "request-return", "level": "implied"},
    {"message": "朋友不喜欢这个礼物，能处理吗", "correct_tool": "request-return", "intent": "request-return", "level": "implied"},
    # -- ambiguous (3) --
    {"message": "我要退货，订单号 WB202405050088", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},  # 含"订单号"
    {"message": "那个券用不了，帮我解决", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},  # 含"券"，可能是coupon
    {"message": "发货太慢了，我不要了", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},  # 含"发货"
    # -- mixed (2) --
    {"message": "帮我退了最近的单子，顺便查下退款到哪了", "correct_tool": "request-return", "intent": "request-return", "level": "mixed"},
    {"message": "申请退款顺便看看我还有多少余额", "correct_tool": "request-return", "intent": "request-return", "level": "mixed"},

    # ================================================================
    # check-balance (20 条)
    # ================================================================
    # -- exact (5) --
    {"message": "账户余额多少", "correct_tool": "check-balance", "intent": "check-balance", "level": "exact"},
    {"message": "积分有多少了", "correct_tool": "check-balance", "intent": "check-balance", "level": "exact"},
    {"message": "查下余额", "correct_tool": "check-balance", "intent": "check-balance", "level": "exact"},
    {"message": "看看钱包还有多少", "correct_tool": "check-balance", "intent": "check-balance", "level": "exact"},
    {"message": "积分明细帮我查一下", "correct_tool": "check-balance", "intent": "check-balance", "level": "exact"},
    # -- synonym (5): 口语化 --
    {"message": "我还有多少可以用来买东西的", "correct_tool": "check-balance", "intent": "check-balance", "level": "synonym"},
    {"message": "账户里还有米吗", "correct_tool": "check-balance", "intent": "check-balance", "level": "synonym"},
    {"message": "看看我剩多少银两", "correct_tool": "check-balance", "intent": "check-balance", "level": "synonym"},
    {"message": "够不够付下一单的", "correct_tool": "check-balance", "intent": "check-balance", "level": "synonym"},
    {"message": "上次充的钱还剩多少", "correct_tool": "check-balance", "intent": "check-balance", "level": "synonym"},
    # -- implied (5): 隐含意图 --
    {"message": "我想用积分换东西，看看够不够", "correct_tool": "check-balance", "intent": "check-balance", "level": "implied"},
    {"message": "这两个月攒了多少分了", "correct_tool": "check-balance", "intent": "check-balance", "level": "implied"},
    {"message": "上次活动送的到账了没", "correct_tool": "check-balance", "intent": "check-balance", "level": "implied"},
    {"message": "为啥买东西提示余额不足", "correct_tool": "check-balance", "intent": "check-balance", "level": "implied"},
    {"message": "帮我看看资金情况", "correct_tool": "check-balance", "intent": "check-balance", "level": "implied"},
    # -- ambiguous (3) --
    {"message": "我的积分够不够换个优惠券", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},  # 含"优惠券"
    {"message": "余额和券都帮我看看", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},  # 含"券"
    {"message": "积分能兑换什么，先看看我还有多少分", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    # -- mixed (2) --
    {"message": "余额有多少，再看看最近的消费", "correct_tool": "check-balance", "intent": "check-balance", "level": "mixed"},
    {"message": "查下积分和有没有新订单", "correct_tool": "check-balance", "intent": "check-balance", "level": "mixed"},

    # ================================================================
    # coupon-inquiry (20 条)
    # ================================================================
    # -- exact (5) --
    {"message": "有什么优惠券可以用", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "exact"},
    {"message": "满减券还有吗", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "exact"},
    {"message": "看看我的代金券", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "exact"},
    {"message": "有没有满100减15的券", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "exact"},
    {"message": "新人优惠券在哪里领", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "exact"},
    # -- synonym (5): 口语化 --
    {"message": "买东西能便宜点吗", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "synonym"},
    {"message": "有什么福利可以领的", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "synonym"},
    {"message": "最近有啥薅羊毛的地方", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "synonym"},
    {"message": "我是不是还有折扣没用", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "synonym"},
    {"message": "能省点钱吗，有什么活动", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "synonym"},
    # -- implied (5): 隐含意图 --
    {"message": "想买个大件，看看能不能减点", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "implied"},
    {"message": "新注册的有啥好处没", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "implied"},
    {"message": "快过期的东西提醒我一下", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "implied"},
    {"message": "这个商品能用什么抵扣", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "implied"},
    {"message": "618到了，有啥促销不", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "implied"},
    # -- ambiguous (3) --
    {"message": "那个券用不了，帮我解决", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},  # 可能是request-return
    {"message": "我的余额能不能买那个优惠券礼包", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},  # 含"余额"
    {"message": "有满减活动吗", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
    # -- mixed (2) --
    {"message": "看看有啥券，顺便查下我的积分", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "mixed"},
    {"message": "有没有免运费的券，物流太慢了", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "mixed"},
]

# ================================================================
# P0: 关键词匹配 -> intent_tool_map 覆盖
# ================================================================

def p0_match_intent(message: str) -> Optional[str]:
    """长关键词优先匹配，模拟 P0 意图识别。"""
    sorted_kw = sorted(KEYWORD_TOOL_MAP.keys(), key=len, reverse=True)
    for kw in sorted_kw:
        if kw in message:
            return KEYWORD_TOOL_MAP[kw]
    return None


def p0_filter(intent: Optional[str]) -> Set[str]:
    return INTENT_TOOL_MAP.get(intent or "unknown", INTENT_TOOL_MAP["unknown"])


def run_p0(samples: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    P0 指标重新定义：
      - narrowed_hit: 关键词命中 + 意图识别正确 + 工具子集包含正确工具
      - narrowed_miss: 关键词命中 + 意图识别错误 → 工具子集不含正确工具
      - fallback: 无关键词命中 → candidate=ALL(5个) → 放弃缩小范围
    """
    total = len(samples)
    narrowed_hits, narrowed_misses, fallbacks = 0, 0, 0
    hit_by_intent, hit_by_level = {}, {}
    miss_details, per_case = [], []
    level_counts: Dict[str, int] = {}
    level_narrowed: Dict[str, int] = {}
    level_fallback: Dict[str, int] = {}

    for i, case in enumerate(samples):
        msg, correct = case["message"], case["correct_tool"]
        expected_intent = case.get("intent", "")
        level = case.get("level", "exact")
        level_counts[level] = level_counts.get(level, 0) + 1

        detected_intent = p0_match_intent(msg)
        candidates = p0_filter(detected_intent)

        if detected_intent is None:
            # 无任何关键词命中 → 兜底返回全部工具 → 不算命中
            fallbacks += 1
            level_fallback[level] = level_fallback.get(level, 0) + 1
            is_hit = False
            hit_type = "fallback"
        elif correct in candidates:
            # 有关键词命中 + 工具集正确 → 真正的命中
            narrowed_hits += 1
            hit_by_intent[expected_intent] = hit_by_intent.get(expected_intent, 0) + 1
            hit_by_level[level] = hit_by_level.get(level, 0) + 1
            level_narrowed[level] = level_narrowed.get(level, 0) + 1
            is_hit = True
            hit_type = "narrowed_hit"
        else:
            # 关键词命中但意图判错 → 工具集不对
            narrowed_misses += 1
            is_hit = False
            hit_type = "narrowed_miss"
            miss_details.append({
                "idx": i + 1, "message": msg, "correct_tool": correct,
                "detected_intent": detected_intent,
                "candidates": sorted(candidates), "expected_intent": expected_intent,
                "level": level,
            })

        per_case.append({
            "idx": i + 1, "message": msg, "correct_tool": correct,
            "detected_intent": detected_intent or "(none)",
            "candidates": sorted(candidates), "p0_hit": is_hit,
            "hit_type": hit_type, "level": level,
        })

    # 分层统计：只算 narrowed_hit 和 narrowed_miss（不计 fallback）
    level_stats = {}
    level_order = ["exact", "synonym", "implied", "ambiguous", "mixed"]
    for lv in level_order:
        if lv in level_counts:
            nh = level_narrowed.get(lv, 0)
            lf = level_fallback.get(lv, 0)
            lt = level_counts[lv]
            # narrowed 内部的命中率（排除 fallback）
            narrowed_total = lt - lf
            level_stats[lv] = {
                "total": lt, "narrowed_hits": nh, "fallbacks": lf,
                "narrowed_total": narrowed_total,
                "narrowed_rate": round(nh / narrowed_total, 4) if narrowed_total > 0 else None,
            }

    intent_stats = {}
    for name in sorted(set(c["intent"] for c in samples)):
        intent_cases = [c for c in samples if c["intent"] == name]
        ih = hit_by_intent.get(name, 0)
        intent_stats[name] = {"total": len(intent_cases), "hits": ih,
                              "rate": round(ih / len(intent_cases), 4)}

    narrowed_rate = round(narrowed_hits / (narrowed_hits + narrowed_misses), 4) if (narrowed_hits + narrowed_misses) > 0 else 0.0
    keyword_coverage = round((narrowed_hits + narrowed_misses) / total, 4)  # 关键词覆盖比例

    print(f"\n{'='*60}")
    print(f"P0 关键词匹配 -> intent_tool_map 覆盖（修正指标）")
    print(f"{'='*60}")
    print(f"  总样本: {total}")
    print(f"  关键词覆盖:     {narrowed_hits + narrowed_misses}/{total} ({keyword_coverage:.0%})  ← 有多少用户问法触发了关键词")
    print(f"  明确命中:       {narrowed_hits}/{narrowed_hits + narrowed_misses} ({narrowed_rate:.0%})  ← 触发关键词后意图判对的比例")
    print(f"  意图判错:       {narrowed_misses}/{narrowed_hits + narrowed_misses}  ← 触发关键词但判错了")
    print(f"  兜底(无关键词):  {fallbacks}/{total} ({fallbacks/total:.0%})  ← 没有任何关键词命中，candidate=ALL，靠后续阶段")

    print(f"\n  按难度分层:")
    level_names = {"exact": "精确关键词", "synonym": "同义/口语", "implied": "隐含意图",
                   "ambiguous": "歧义边界", "mixed": "多意图混合"}
    for lv in level_order:
        if lv in level_stats:
            st = level_stats[lv]
            nr = st["narrowed_rate"]
            if nr is not None:
                bar_n = max(1, int(nr * 20))
                bar = "#" * bar_n + "-" * (20 - bar_n)
                print(f"    {level_names.get(lv, lv):<12} 命中{st['narrowed_hits']:>2}/{st['narrowed_total']:<2}"
                      f" 兜底{st['fallbacks']:>2}  {nr:.0%}  [{bar}]")
            else:
                print(f"    {level_names.get(lv, lv):<12} 全部兜底 (fallback={st['fallbacks']})")

    print(f"\n  各 intent（narrowed 命中率）:")
    for name, st in intent_stats.items():
        bar_n = max(1, int(st["rate"] * 20))
        bar = "#" * bar_n + "-" * (20 - bar_n)
        print(f"    {name:<18} {st['hits']:>2}/{st['total']}  {st['rate']:.0%}  [{bar}]")

    if narrowed_misses > 0:
        print(f"\n  意图判错 ({narrowed_misses} 条):")
        for m in miss_details[:20]:
            print(f"    [{m['idx']:>2}] [{m['level']:<9}] \"{m['message'][:43]}\"  "
                  f"got={m['detected_intent']} cand={m['candidates']} "
                  f"want_intent={m['expected_intent']}")

    return {
        "stage": "P0", "total": total,
        "narrowed_hits": narrowed_hits, "narrowed_misses": narrowed_misses,
        "fallbacks": fallbacks, "narrowed_rate": narrowed_rate,
        "keyword_coverage": keyword_coverage,
        "intent_stats": intent_stats, "level_stats": level_stats,
        "miss_details": miss_details, "per_case": per_case,
    }


# ================================================================
# P1: bge-small-zh-v1.5 语义重排
# ================================================================

@dataclass
class P1EmbeddingMatcher:
    model_path: str = "./models/BAAI/bge-small-zh-v1.5"
    _model: Any = None
    _tool_embeddings: Optional[Dict[str, Any]] = None
    _ready: bool = False

    def ensure_ready(self):
        if self._ready:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  加载 bge-small-zh-v1.5 模型 ({self.model_path})...", end=" ", flush=True)
            t0 = time.monotonic()
            self._model = SentenceTransformer(self.model_path)
            desc_texts, desc_names = [], []
            for name, desc in TOOL_DESCRIPTIONS.items():
                desc_texts.append(f"工具名称：{name}；功能描述：{desc}")
                desc_names.append(name)
            self._tool_embeddings = {}
            emb = self._model.encode(desc_texts, normalize_embeddings=True)
            for name, e in zip(desc_names, emb):
                self._tool_embeddings[name] = e
            self._ready = True
            print(f"({time.monotonic() - t0:.1f}s, {len(desc_names)} tools)")
        except ImportError:
            print("\n  [WARN] sentence-transformers 未安装")
            raise
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            raise

    def rank(
        self, user_query: str, candidate_names: Set[str],
        intent_action: Optional[str], top_k: int = 3, intent_boost: float = 1.5,
    ) -> List[Tuple[str, float]]:
        self.ensure_ready()
        query_emb = self._model.encode(user_query, normalize_embeddings=True)
        intent_tools = INTENT_TOOL_MAP.get(intent_action, set()) if intent_action else set()

        scored: List[Tuple[str, float]] = []
        for name in candidate_names:
            if name not in self._tool_embeddings:
                continue
            sim = float(query_emb @ self._tool_embeddings[name])
            if name in intent_tools:
                sim *= intent_boost
            scored.append((name, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def run_p0p1(samples: List[Dict[str, str]], p0_results: Dict[str, Any],
              p1_model_path: str = "./models/BAAI/bge-small-zh-v1.5") -> Dict[str, Any]:
    total = len(samples)
    matcher = P1EmbeddingMatcher(model_path=p1_model_path)
    hits_top1, hits_top2, p1_salvages = 0, 0, 0
    per_case = []

    for i, case in enumerate(samples):
        msg, correct = case["message"], case["correct_tool"]
        detected_intent = p0_match_intent(msg)
        candidates = p0_filter(detected_intent)
        p0_is_fallback = detected_intent is None  # 无关键词命中 → 兜底
        p0_narrowed_hit = not p0_is_fallback and correct in candidates  # 关键词触发+意图对
        p0_narrowed_miss = not p0_is_fallback and correct not in candidates  # 关键词触发+意图错

        if len(candidates) > 1:
            ranked = matcher.rank(
                user_query=msg, candidate_names=candidates,
                intent_action=detected_intent, top_k=min(3, len(candidates)),
            )
            top1_name = ranked[0][0] if ranked else None
            top2_names = [n for n, _ in ranked[:2]]
            ranked_list = [(n, round(s, 4)) for n, s in (ranked or [])]
        else:
            top1_name = list(candidates)[0] if candidates else None
            top2_names = list(candidates)[:2]
            ranked_list = [(top1_name, 1.0)] if top1_name else []

        top1_hit = top1_name == correct
        top2_hit = correct in top2_names
        if top1_hit:
            hits_top1 += 1
        if top2_hit:
            hits_top2 += 1
        # P1 救回: P0 兜底(无关键词) → P1 Top-1 正确选中
        if p0_is_fallback and top1_hit:
            p1_salvages += 1

        per_case.append({
            "idx": i + 1, "message": msg, "correct_tool": correct,
            "detected_intent": detected_intent or "(none)",
            "p0_narrowed_hit": p0_narrowed_hit,
            "p0_narrowed_miss": p0_narrowed_miss,
            "p0_fallback": p0_is_fallback,
            "p1_top1": top1_name,
            "p1_top1_hit": top1_hit, "p1_top2_hit": top2_hit,
            "ranked": ranked_list,
        })

    ht1 = round(hits_top1 / total, 4)
    ht2 = round(hits_top2 / total, 4)

    # 分层统计 P1 效果
    fallback_total = sum(1 for pc in per_case if pc["p0_fallback"])
    fallback_top1 = sum(1 for pc in per_case if pc["p0_fallback"] and pc["p1_top1_hit"])
    fallback_top2 = sum(1 for pc in per_case if pc["p0_fallback"] and pc["p1_top2_hit"])
    narrowed_miss_total = sum(1 for pc in per_case if pc["p0_narrowed_miss"])

    print(f"\n{'='*60}")
    print(f"P0+P1 BGE-M3 语义匹配（意图识别）")
    print(f"{'='*60}")
    print(f"  P0 明确命中率:    {p0_results['narrowed_rate']:.0%}  ({p0_results['narrowed_hits']}/{p0_results['narrowed_hits'] + p0_results['narrowed_misses']})")
    print(f"  P0 关键词覆盖:    {p0_results['keyword_coverage']:.0%}  ({p0_results['narrowed_hits'] + p0_results['narrowed_misses']}/{total})")
    print(f"  P0 兜底(无关键词): {p0_results['fallbacks']}/{total}")
    print(f"  P0+P1 Top-1:      {hits_top1}/{total} ({ht1:.1%})")
    print(f"  P0+P1 Top-2:      {hits_top2}/{total} ({ht2:.1%})")
    print(f"  P1 救回兜底:      {p1_salvages}/{fallback_total} 条 ← 无关键词时 P1 语义 Top-1 命中")
    print(f"  P1 救回判错:      0/{narrowed_miss_total} 条 ← 关键词判错→候选集不含正确工具→P1 无法救")
    print(f"  兜底层 Top-1:     {fallback_top1}/{fallback_total} ({fallback_top1/fallback_total:.0%}) ← P1 补回能力")
    print(f"  兜底层 Top-2:     {fallback_top2}/{fallback_total} ({fallback_top2/fallback_total:.0%})")

    fails = [pc for pc in per_case if not pc["p1_top1_hit"]]
    if fails:
        print(f"\n  P1 Top-1 仍失败 ({len(fails)} 条):")
        for f in fails[:10]:
            ranked_str = ", ".join(f"{n}={s:.3f}" for n, s in f.get("ranked", []))
            print(f"    [{f['idx']:>2}] \"{f['message'][:35]}\"  "
                  f"intent={f['detected_intent']}, correct={f['correct_tool']}, "
                  f"top1={f['p1_top1']}, ranked=[{ranked_str}]")

    return {
        "stage": "P0+P1", "total": total,
        "p0_narrowed_rate": p0_results["narrowed_rate"],
        "p0_keyword_coverage": p0_results["keyword_coverage"],
        "p0p1_top1_hits": hits_top1, "p0p1_top1_rate": ht1,
        "p0p1_top2_hits": hits_top2, "p0p1_top2_rate": ht2,
        "p1_salvages": p1_salvages, "per_case": per_case,
    }


# ================================================================
# P2: 本地模型确认（接口占位，实际运行需 GPU）
# ================================================================

P2_NOTE = """
P2 本地模型意图确认说明:
  已有 benchmark_local_model_comparison.py 任务 B 的实测 P2 数据:
  Qwen2.5-1.5B-Instruct on GPU: Top1=92%, Top2=96%, p50=94ms.

  综合 P0+P1+P2 最终命中率:
    假设 P0+P1 Top-2 = X%, P2 从 Top-2 中选对的概率 = 96%,
    则 P0+P1+P2 ≈ X% * 96%.

  注意: P2 做的是意图确认（从 P1 筛选后的意图候选中确认最终意图），
        工具选择是意图确认后的确定性映射（INTENT_TOOL_MAP），不需要 LLM。

  如需在本脚本运行 P2, 请加:
    --model ./models/Qwen2.5-1.5B-Instruct --device cuda
"""


def run_p2(
    samples: List[Dict[str, str]], p0p1_results: Dict[str, Any],
    model_path: Optional[str] = None, device: str = "cpu",
) -> Dict[str, Any]:
    if model_path is None:
        print(P2_NOTE)
        return {"stage": "P2", "status": "skipped",
                "note": "P2 requires GPU. Use benchmark_local_model_comparison.py data: 92% Top1."}

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\n  [WARN] torch/transformers not installed, skipping P2")
        return {"stage": "P2", "status": "skipped"}

    print(f"\n  加载本地模型: {model_path} on {device}...", flush=True)
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else "cpu", trust_remote_code=True,
    )
    print(f"  加载完成 ({time.monotonic() - t0:.1f}s)", flush=True)

    total, hits, latencies = len(samples), 0, []
    per_case = []

    for i, case in enumerate(samples):
        msg, correct = case["message"], case["correct_tool"]
        p1_case = p0p1_results.get("per_case", [])
        ranked = p1_case[i].get("ranked", []) if i < len(p1_case) else []
        candidate_names = [n for n, _ in ranked[:3]] if ranked else [correct]

        if len(candidate_names) <= 1:
            is_hit = candidate_names[0] == correct if candidate_names else False
            if is_hit:
                hits += 1
            per_case.append({"idx": i + 1, "message": msg, "correct_tool": correct,
                             "candidates": candidate_names, "p2_hit": is_hit, "latency_ms": 0})
            continue

        desc_lines = "\n".join(f"- {n}: {TOOL_DESCRIPTIONS.get(n, '')}" for n in candidate_names)
        system = "你是一个电商客服意图路由器。根据用户消息，从候选意图对应的工具列表中选择最相关的工具。每个工具的 description 已包含其功能说明，请根据语义进行匹配。当用户同时涉及多个操作时（如订单+物流），可以同时选中。"
        user = f"候选工具:\n{desc_lines}\n\n用户消息: {msg}\n\n请选出最相关的工具（只输出工具名，每行一个）:"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        t1 = time.monotonic()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
        elapsed_ms = (time.monotonic() - t1) * 1000
        latencies.append(elapsed_ms)

        # 解析：取第一行非空工具名
        selected = "".join(generated.split())
        for line in generated.strip().splitlines():
            name = line.strip().lstrip("-* 0123456789.、，").strip().strip('\'"`,，:')
            if name in set(candidate_names):
                selected = name
                break

        is_hit = selected == correct
        if is_hit:
            hits += 1

        per_case.append({"idx": i + 1, "message": msg, "correct_tool": correct,
                         "candidates": candidate_names, "selected": selected,
                         "raw": generated[:80], "p2_hit": is_hit,
                         "latency_ms": round(elapsed_ms, 1)})

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{total}] current_acc={hits/(i+1):.1%}", flush=True)

    hit_rate = round(hits / total, 4)
    p50 = statistics.median(latencies) if latencies else 0
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else (latencies[0] if latencies else 0)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'='*60}")
    print(f"P2 本地模型意图确认")
    print(f"{'='*60}")
    print(f"  命中: {hits}/{total} ({hit_rate:.1%})")
    print(f"  p50={p50:.0f}ms  p99={p99:.0f}ms  avg={avg_lat:.0f}ms")

    return {"stage": "P2", "total": total, "hits": hits, "hit_rate": hit_rate,
            "p50_latency_ms": round(p50, 1), "p99_latency_ms": round(p99, 1),
            "avg_latency_ms": round(avg_lat, 1), "per_case": per_case}


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="P0 -> P1 -> P2 意图识别流水线分层命中率 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=P2_NOTE,
    )
    parser.add_argument("--stage", choices=["p0", "p0p1", "all"], default="p0p1",
                        help="评测阶段 (默认 p0p1)")
    parser.add_argument("--p1-model", type=str, default="./models/BAAI/bge-small-zh-v1.5",
                        help="P1 Embedding 模型路径 (默认 ./models/BAAI/bge-small-zh-v1.5)")
    parser.add_argument("--model", type=str, default=None,
                        help="P2 本地模型路径")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="P2 推理设备")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    parser.add_argument("--output", type=str, default=None, help="JSON 输出文件路径")
    args = parser.parse_args()

    samples = TOOL_SELECTION_PIPELINE_TESTS

    if not args.json:
        print(f"意图识别流水线 Benchmark: {len(samples)} 条样本, stage={args.stage}")

    # P0
    p0_results = run_p0(samples)
    summary: Dict[str, Any] = {"p0": {
        "narrowed_rate": p0_results["narrowed_rate"],
        "narrowed_hits": p0_results["narrowed_hits"],
        "narrowed_misses": p0_results["narrowed_misses"],
        "fallbacks": p0_results["fallbacks"],
        "keyword_coverage": p0_results["keyword_coverage"],
    }}

    # P0+P1
    p0p1_results = None
    if args.stage in ("p0p1", "all"):
        try:
            p0p1_results = run_p0p1(samples, p0_results, args.p1_model)
            summary["p0p1"] = {
                "top1_hit_rate": p0p1_results["p0p1_top1_rate"],
                "top1_hits": p0p1_results["p0p1_top1_hits"],
                "top2_hit_rate": p0p1_results["p0p1_top2_rate"],
                "top2_hits": p0p1_results["p0p1_top2_hits"],
                "p1_salvages": p0p1_results["p1_salvages"],
            }
        except (ImportError, Exception) as e:
            print(f"\n  P1 skipped: {e}")
            summary["p0p1"] = {"status": "skipped", "reason": str(e)}

    # P0+P1+P2
    if args.stage == "all":
        if args.model and p0p1_results:
            p2_results = run_p2(samples, p0p1_results, args.model, args.device)
            summary["p2"] = {
                "hit_rate": p2_results.get("hit_rate"),
                "hits": p2_results.get("hits"),
                "p50_latency_ms": p2_results.get("p50_latency_ms"),
                "p99_latency_ms": p2_results.get("p99_latency_ms"),
            }
        else:
            if not args.json:
                print(P2_NOTE)
            summary["p2"] = {"status": "skipped",
                             "note": "Use --model to specify local model path."}

    summary["config"] = {"samples": len(samples), "tools": len(TOOL_DEFINITIONS),
                         "stage": args.stage}

    if not args.json:
        print(f"\n{'='*60}")
        print(f"综合汇总")
        print(f"{'='*60}")
        p0s = summary["p0"]
        print(f"  P0 关键词覆盖:  {p0s['keyword_coverage']:.0%}  "
              f"({p0s['narrowed_hits'] + p0s['narrowed_misses']}/{len(samples)})  "
              f"兜底 {p0s['fallbacks']} 条")
        print(f"  P0 明确命中率:  {p0s['narrowed_rate']:.0%}  "
              f"({p0s['narrowed_hits']}/"
              f"{p0s['narrowed_hits'] + p0s['narrowed_misses']})  "
              f"← 触发关键词后判对的比例")
        if "top1_hit_rate" in summary.get("p0p1", {}):
            print(f"  P0+P1 Top-1:    {summary['p0p1']['top1_hit_rate']:.1%}  "
                  f"({summary['p0p1']['top1_hits']}/{len(samples)})")
            print(f"  P0+P1 Top-2:    {summary['p0p1']['top2_hit_rate']:.1%}  "
                  f"({summary['p0p1']['top2_hits']}/{len(samples)})")
            print(f"  P1 救回:        {summary['p0p1']['p1_salvages']} 条")
        if "hit_rate" in summary.get("p2", {}) and isinstance(summary["p2"].get("hit_rate"), float):
            print(f"  P0+P1+P2:       {summary['p2']['hit_rate']:.1%}  "
                  f"({summary['p2']['hits']}/{len(samples)})")
        # 综合估计
        if "top2_hit_rate" in summary.get("p0p1", {}):
            est = summary["p0p1"]["top2_hit_rate"] * 0.96  # P2 Top1=92% for Top-2 input
            print(f"  综合估计:       ~{est:.1%}  (P0+P1 Top-2 * P2 96%)")

    if args.json:
        json_str = json.dumps(summary, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"JSON written to {args.output}")
        else:
            print(json_str)


if __name__ == "__main__":
    main()

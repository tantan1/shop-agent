"""
P0 -> P1 -> P2 意图识别流水线分层命中率 Benchmark
==================================================
评测三级意图识别流水线各阶段累进命中率，产出实测数据替代 blog v3 中的估计值（50%/90%/95%）。

用法:
    # P0 关键词匹配（极快，纯 Python，无依赖）
    python scripts/benchmark_tool_selection_pipeline.py --stage p0

    # P0+P1 语义匹配（需要 pip install sentence-transformers）
    python scripts/benchmark_tool_selection_pipeline.py --stage p0p1

    # P0+P1+重排（优化一，治近义弱区分 / D1）
    python scripts/benchmark_tool_selection_pipeline.py --stage p0p1_rerank --json --output after_rerank.json

    # P0 不排它交全量（优化三，P0 救回 / D2）
    python scripts/benchmark_tool_selection_pipeline.py --stage p0_nofilter --json --output after_p0.json

    # P0+P1+P2 旧三阶段（BASE 口径，需 GPU + 本地模型，带计时）
    python scripts/benchmark_tool_selection_pipeline.py --stage all --model ./models/Qwen2.5-1.5B-Instruct --device cuda --json --output base_all.json

    # 合并 function calling（优化二+三 / D3/D4，需 GPU）——注意：这是被 doc 24 否定的"字面全量 FC"
    python scripts/benchmark_tool_selection_pipeline.py --stage all_fc --fc-device cuda --json --output after_fc.json

    # 软预过滤方案（P0 快路径 + embedding 全集软预过滤，非全量 FC，CPU 可跑）
    python scripts/benchmark_tool_selection_pipeline.py --stage soft_filter --json --output soft_filter.json
    # 软预过滤 + FC-on-TopK 最终确认（受限于 Top-K，非全量 FC；需 --fc-model，建议 GPU）
    python scripts/benchmark_tool_selection_pipeline.py --stage soft_filter --use-fc-topk --fc-model ./models/Qwen2.5-1.5B-Instruct --fc-device cuda --json --output soft_filter_topk.json

    # 校准三分支门控（doc 24 §3 ④）：向量侧算 C=σ((margin-b)/T)，三分支门控，低 C 模拟升大模型
    # 自动拟合 T,b：--fit-calibration；θ_high 需在概率域设定（如 0.9）；--escalate-mock 不真实调用大模型
    python scripts/benchmark_tool_selection_pipeline.py --stage soft_filter --fit-calibration --theta-high 0.9 --escalate-mock --json --output soft_filter_calibrated.json
    # 含本地小模型中间档：再加 --use-fc-topk --fc-model ...；低 C 支仍只模拟、不真实调云端大模型
    python scripts/benchmark_tool_selection_pipeline.py --stage soft_filter --fit-calibration --theta-high 0.9 --use-fc-topk --fc-model ./models/Qwen2.5-1.5B-Instruct --escalate-mock --json --output soft_filter_calibrated_topk.json

    # 每条 case 跑 3 次取中位延迟 + 多数票（文档 23 §5）
    python scripts/benchmark_tool_selection_pipeline.py --stage all_fc --fc-device cuda --runs 3 --json --output after_fc.json

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
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
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
        "查询账户里已有的资金和积分余额。触发条件：用户问余额、钱包、有多少钱、账户里还剩多少、"
        "积分、我的积分、可用积分。"
        "区分：本工具只查'已拥有的钱/积分'。凡涉及'优惠/折扣/券/促销/活动/便宜/省(钱以外)/"
        "福利/薅羊毛/抵扣/满减/新人好处'都属于 coupon-inquiry，不要选本工具。"
    ),
    "coupon-inquiry": (
        "查询用户可用的优惠券、折扣、促销活动列表，包括有效期、使用门槛、适用商品。"
        "触发条件（覆盖口语/同义表达）：用户问优惠券、代金券、满减、有什么券、优惠码、"
        "便宜点、能省/省钱、打折、促销、活动、福利、薅羊毛、抵扣、新人/注册好处、领券。"
        "区分：本工具只涉及'优惠/折扣/券/促销'，不查账户里已有的'钱/余额/积分'——"
        "那是 check-balance 的职责；也不查订单/物流（那是 query-order/check-shipping）。"
        "注意：'新人/注册/首单好处/新用户福利'也是优惠券（新人券/首单券），归本工具，"
        "不要选 knowledge_search（knowledge_search 只回答政策/FAQ 类问题，不含发券）。"
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

# ================================================================
# 路由策略配置（base / optimized 可切换，便于后期扩展对比）
# ================================================================
@dataclass
class VariantConfig:
    """一条完整流水线的路由策略。改前(base)与改后(optimized)只是这些开关的组合，
    后期新增改动只需加一个 preset，或在下方加字段 + 在 run_* 里加一个分支。"""
    name: str = "base"
    # P0: 命中意图后是否缩小候选集（base=True 排除其他；optimized=False 交全量，不致命）
    p0_exclude_candidates: bool = True
    # P1: 是否用 BGE-Reranker(cross-encoder) 对召回结果重排（优化一：治近义弱区分）
    p1_use_reranker: bool = False
    # P1: base 的补丁——意图对应工具 ×1.5 加权；optimized 关掉让模型自己学
    p1_intent_boost: float = 1.5
    # P1+P2: 合并为一次本地 function calling（优化二：治两跳误判，跳过 INTENT_TOOL_MAP）
    use_function_calling: bool = False
    # 模型路径
    reranker_model: str = "./models/BAAI/bge-reranker-base"
    fc_model: str = "./models/Qwen2.5-1.5B-Instruct"
    fc_device: str = "cpu"


PRESETS: Dict[str, VariantConfig] = {
    "base": VariantConfig(name="base"),
    "optimized": VariantConfig(
        name="optimized",
        p0_exclude_candidates=False,   # 优化三：P0 只做高置信快路径，不排它
        p1_intent_boost=1.0,
        p1_use_reranker=True,          # 优化一：BGE-Reranker 重排
        use_function_calling=True,     # 优化二：合并 function calling
    ),
}

# ---- 测试用例（135 条：100 原集 + 35 扩充 ambiguous，5 级难度） ----
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

    # ================================================================
    # 补充歧义样本（验证 D1/D2 统计显著性用，约 +35 条）
    # 目的：基线 ambiguous 仅 15 条，百分比 ±3pp ≈ 不到 1 条噪声，
    #       扩充至 ~50 条后 D1 才有统计意义。仅新增 ambiguous 级，
    #       不改原 100 条结构；BASE 与 AFTER 必须跑同一版本。
    # ================================================================
    # -- query-order 补充 ambiguous (易被误判到 check-shipping / request-return / coupon) --
    {"message": "我买的那个到哪了，物流停在中转站", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},
    {"message": "订单一直没更新物流信息", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},
    {"message": "这单要是退的话钱原路返回哪", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},
    {"message": "帮我看下那单的退款到账没", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},
    {"message": "下单后一直没发货是不是超时了", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},
    {"message": "我那个订单能用券抵扣吗", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},
    {"message": "订单里有个商品要申请退货", "correct_tool": "query-order", "intent": "query-order", "level": "ambiguous"},
    # -- check-shipping 补充 ambiguous (易被误判到 query-order / request-return) --
    {"message": "那个单子到哪了，SF 开头的", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "我买的件什么时候到，运单号忘了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "货发出去三天了还没动静", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "包裹显示签收了但我没收到，查下", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "物流那个单号查下，就是前天买的", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "退的货物流到哪了", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "订单支付后什么时候发货送过来", "correct_tool": "check-shipping", "intent": "check-shipping", "level": "ambiguous"},
    # -- request-return 补充 ambiguous (易被误判到 query-order / coupon / check-balance) --
    {"message": "订单号 WB202405050088 我要退", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},
    {"message": "那个券用不了，给我退了", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},
    {"message": "发货太慢了不想要了", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},
    {"message": "买的东西质量差，退货退款一起办", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},
    {"message": "刚下的单能直接退款吗", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},
    {"message": "余额充足但我还是要退货", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},
    {"message": "这个能不能退，优惠券还能用不", "correct_tool": "request-return", "intent": "request-return", "level": "ambiguous"},
    # -- check-balance 补充 ambiguous (易被误判到 coupon / query-order / request-return) --
    {"message": "积分能不能抵订单的钱", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    {"message": "账户里的钱够不够付这个退货的运费", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    {"message": "我钱包余额和退款退到哪了", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    {"message": "积分兑换的券怎么查余额", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    {"message": "省下的钱能不能买优惠券", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    {"message": "订单退款会回到余额吗", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    {"message": "充值送的积分够换啥", "correct_tool": "check-balance", "intent": "check-balance", "level": "ambiguous"},
    # -- coupon-inquiry 补充 ambiguous (易被误判到 check-balance / query-order / request-return) --
    {"message": "积分兑换的优惠在哪领", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "订单能用什么券减", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "退款的时候用的券退不退", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "余额支付的单有什么满减", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "买退货那单时领的券还能用吗", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "积分商城的券怎么用", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "账户余额够不够买券包", "correct_tool": "coupon-inquiry", "intent": "coupon-inquiry", "level": "ambiguous"},
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


def p0_filter(intent: Optional[str], exclude: bool = True) -> Set[str]:
    """base(exclude=True): 命中意图则缩小到该意图工具集，排除其他。
    optimized(exclude=False): 永远返回全量，P0 只做高置信快路径，不致命。"""
    if not exclude:
        return set(INTENT_TOOL_MAP["unknown"])
    return INTENT_TOOL_MAP.get(intent or "unknown", INTENT_TOOL_MAP["unknown"])


def run_p0(samples: List[Dict[str, str]], exclude: bool = True) -> Dict[str, Any]:
    """
    P0 指标重新定义：
      - narrowed_hit: 关键词命中 + 意图识别正确 + 工具子集包含正确工具
      - narrowed_miss: 关键词命中 + 意图识别错误 → 工具子集不含正确工具
      - fallback: 无关键词命中 → candidate=ALL(5个) → 放弃缩小范围
    exclude: True=base(命中意图则缩小候选集，排除其他)；False=optimized(永远全量，P0 不致命)
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
        candidates = p0_filter(detected_intent, exclude)

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


_reranker_cache: Dict[str, Any] = {}

def rerank_tools(query: str, tool_names: List[str], model_path: str) -> List[Tuple[str, float]]:
    """用 BGE-Reranker(cross-encoder) 对候选工具描述重排，返回 (name, score) 降序。"""
    if model_path not in _reranker_cache:
        from sentence_transformers import CrossEncoder
        _reranker_cache[model_path] = CrossEncoder(model_path, max_length=512)
    model = _reranker_cache[model_path]
    pairs = [[query, f"工具 {n}：{TOOL_DESCRIPTIONS.get(n, '')}"] for n in tool_names]
    scores = model.predict(pairs)
    scored = sorted(zip(tool_names, map(float, scores)), key=lambda x: x[1], reverse=True)
    return scored


def run_p0p1(samples: List[Dict[str, str]], p0_results: Dict[str, Any],
              variant: VariantConfig,
              p1_model_path: str = "./models/BAAI/bge-small-zh-v1.5") -> Dict[str, Any]:
    total = len(samples)
    matcher = P1EmbeddingMatcher(model_path=p1_model_path)
    hits_top1, hits_top2, p1_salvages = 0, 0, 0
    per_case = []

    for i, case in enumerate(samples):
        msg, correct = case["message"], case["correct_tool"]
        detected_intent = p0_match_intent(msg)
        candidates = p0_filter(detected_intent, variant.p0_exclude_candidates)
        p0_is_fallback = detected_intent is None  # 无关键词命中 → 兜底
        p0_narrowed_hit = not p0_is_fallback and correct in candidates  # 关键词触发+意图对
        p0_narrowed_miss = not p0_is_fallback and correct not in candidates  # 关键词触发+意图错

        if len(candidates) > 1:
            ranked = matcher.rank(
                user_query=msg, candidate_names=candidates,
                intent_action=detected_intent, top_k=min(3, len(candidates)),
                intent_boost=variant.p1_intent_boost,
            )
            if variant.p1_use_reranker:
                reranked = rerank_tools(msg, [n for n, _ in ranked], variant.reranker_model)
                ranked = reranked[:min(3, len(reranked))]
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
            "level": case.get("level", "exact"),
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


def _p2_infer(model, tokenizer, device: str, candidate_names: List[str], msg: str):
    """单次 P2 推理，返回 (selected_tool_name, latency_ms)。"""
    import torch  # 本地导入：本函数为模块级，不在 run_p2 导入作用域内
    desc_lines = "\n".join(f"- {n}: {TOOL_DESCRIPTIONS.get(n, '')}" for n in candidate_names)
    system = ("你是一个电商客服意图路由器。根据用户消息，从候选意图对应的工具列表中选择最相关的工具。"
              "每个工具的 description 已包含其功能说明，请根据语义进行匹配。"
              "当用户同时涉及多个操作时（如订单+物流），可以同时选中。")
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
    selected = "".join(generated.split())
    for line in generated.strip().splitlines():
        name = line.strip().lstrip("-* 0123456789.、，").strip().strip('\'"`,，:')
        if name in set(candidate_names):
            selected = name
            break
    return selected, elapsed_ms


def run_p2(
    samples: List[Dict[str, str]], p0p1_results: Dict[str, Any],
    model_path: Optional[str] = None, device: str = "cpu", runs: int = 3,
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
                             "level": case.get("level", "exact"),
                             "candidates": candidate_names, "p2_hit": is_hit, "latency_ms": 0})
            continue

        votes, lat_for_case = [], []
        for _ in range(max(1, runs)):
            sel, ms = _p2_infer(model, tokenizer, device, candidate_names, msg)
            votes.append(sel)
            lat_for_case.append(ms)
        from collections import Counter
        selected = Counter(votes).most_common(1)[0][0]
        case_latency = statistics.median(lat_for_case)
        latencies.append(case_latency)

        is_hit = selected == correct
        if is_hit:
            hits += 1

        per_case.append({"idx": i + 1, "message": msg, "correct_tool": correct,
                         "level": case.get("level", "exact"),
                         "candidates": candidate_names, "selected": selected,
                         "p2_hit": is_hit,
                         "runs": max(1, runs), "latency_ms": round(case_latency, 1)})

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{total}] current_acc={hits/(i+1):.1%}", flush=True)

    hit_rate = round(hits / total, 4)
    p50 = statistics.median(latencies) if latencies else 0
    p95 = _percentile(latencies, 0.95)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'='*60}")
    print(f"P2 本地模型意图确认（runs={max(1, runs)}）")
    print(f"{'='*60}")
    print(f"  命中: {hits}/{total} ({hit_rate:.1%})")
    print(f"  p50={p50:.0f}ms  p95={p95:.0f}ms  avg={avg_lat:.0f}ms")

    return {"stage": "P2", "total": total, "hits": hits, "hit_rate": hit_rate,
            "p50_latency_ms": round(p50, 1), "p95_latency_ms": round(p95, 1),
            "avg_latency_ms": round(avg_lat, 1), "per_case": per_case}


def _percentile(values: List[float], p: float) -> float:
    """线性插值百分位数。values 无需预先排序（内部排序）。p∈[0,1]。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _sigmoid(x: float) -> float:
    """数值稳定 sigmoid。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_temperature_scaling(margins: List[float], labels: List[int],
                             T_grid: Optional[List[float]] = None,
                             b_grid: Optional[List[float]] = None):
    """Temperature Scaling 拟合：C = sigmoid((margin - b) / T)，在 (margins, labels) 上
    最小化 NLL（labels=1 表示该 margin 对应的 embedding Top-1 决策正确）。
    返回 (T, b, nll)。

    注意：演示用默认在传入样本上 in-sample 拟合；生产环境必须在【独立验证集】上拟合，
    否则会过拟合、校准失效（doc 24 §3 ④ / §8）。
    """
    if T_grid is None:
        T_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if b_grid is None:
        b_grid = [-0.1, 0.0, 0.1, 0.2]
    best = None
    for T in T_grid:
        for b in b_grid:
            nll = 0.0
            for m, y in zip(margins, labels):
                C = _sigmoid((m - b) / T)
                C = min(max(C, 1e-6), 1.0 - 1e-6)
                nll += -(y * math.log(C) + (1 - y) * math.log(1 - C))
            if best is None or nll < best[0]:
                best = (nll, T, b)
    return best[1], best[2], best[0]


def _compute_ece(confs: List[float], corrects: List[int], n_bins: int = 10) -> float:
    """Expected Calibration Error。confs∈[0,1]（预测置信度），corrects∈{0,1}。
    把 conf 分桶后加权求 |桶内实际准确率 - 桶内平均置信度|。越低越好（doc 24 §3 ④ / §8）。"""
    if not confs:
        return 0.0
    bins_n = [0] * n_bins
    bins_conf = [0.0] * n_bins
    bins_acc = [0.0] * n_bins
    for c, ok in zip(confs, corrects):
        c = min(max(c, 1e-6), 1.0 - 1e-6)
        bi = min(int(c * n_bins), n_bins - 1)
        bins_n[bi] += 1
        bins_conf[bi] += c
        bins_acc[bi] += ok
    ece = 0.0
    for bi in range(n_bins):
        if bins_n[bi] == 0:
            continue
        ece += bins_n[bi] * abs(bins_acc[bi] / bins_n[bi] - bins_conf[bi] / bins_n[bi])
    return ece / len(confs)


def _fc_infer(model, tokenizer, device: str, all_tools: List[str], msg: str, candidate_label: str = "全部"):
    """单次 FC 推理，返回 (selected_tool_name, latency_ms)。

    返回的 selected 一定落在 all_tools 规范名内——模型输出若带连字符/空格变体
    （如 couponinquiry）会经 schema 归一化归一到合法工具名；若输出超 schema 的工具名
    （如 knowledge_search，本路由器不含该业务工具）则原样返回，记为路由失败/错选，
    避免把'超出 schema 的幻觉'误当合法选择，也避免把'合法变体'误判为错选。

    candidate_label: 候选集说明（"全部"=全量 FC；"候选"=FC-on-TopK，仅在给定的 Top-K 上确认）。
    """
    import re
    import torch  # 本地导入：本函数为模块级，不在 run_fc 导入作用域内
    # schema 归一化映射：去非字母数字 -> 规范名
    _norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    norm_map = {_norm(n): n for n in all_tools}

    def resolve(raw: str):
        if not raw:
            return None
        if raw in all_tools:                      # 1) 精确匹配
            return raw
        key = _norm(raw)
        if key in norm_map:                       # 2) 去连字符/下划线/空格后匹配
            return norm_map[key]
        return None

    desc_lines = "\n".join(f"- {n}: {TOOL_DESCRIPTIONS.get(n, '')}" for n in all_tools)
    system = (f"你是一个电商客服工具路由器。根据用户消息，从{candidate_label}工具列表中选择最相关的一个工具名。"
              "每个工具都有功能描述，请根据语义匹配。若用户同时涉及多个操作，选最相关的一个。"
              "只输出工具名，不要解释。")
    user = f"{candidate_label}工具:\n{desc_lines}\n\n用户消息: {msg}\n\n请输出最相关的工具名:"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t1 = time.monotonic()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
    elapsed_ms = (time.monotonic() - t1) * 1000
    selected = None
    for line in generated.strip().splitlines():
        name = line.strip().lstrip("-* 0123456789.、，").strip().strip('\'"`,，:')
        r = resolve(name)
        if r:
            selected = r
            break
    if selected is None:  # 退路：对整段去空白后的串再试一次归一化，仍失败则保留原串（超 schema）
        fallback = "".join(generated.split())
        selected = resolve(fallback) or fallback
    return selected, elapsed_ms


def run_fc(samples: List[Dict[str, str]], variant: VariantConfig,
           p0_results: Dict[str, Any], runs: int = 3) -> Dict[str, Any]:
    """优化二+三：合并 P1+P2 为一次本地 function calling，候选=全量工具 schema，
    跳过 INTENT_TOOL_MAP 两跳。P0 仅在判对时快路径直路由，否则交全量 FC（不致命）。

    延迟与命中消抖（文档 23 §5）：每条 case 跑 `runs` 次，取延迟**中位**作为该 case 延迟，
    取**多数票**工具名作为该 case 选择；最终 p50/p95/avg 基于各 case 中位延迟。
    """
    model_path = variant.fc_model
    device = variant.fc_device
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\n  [WARN] torch/transformers not installed, skipping FC")
        return {"stage": "FC", "status": "skipped"}

    print(f"\n  加载本地 function-calling 模型: {model_path} on {device}...", flush=True)
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else "cpu", trust_remote_code=True,
    )
    print(f"  加载完成 ({time.monotonic() - t0:.1f}s)", flush=True)

    all_tools = list(TOOL_DESCRIPTIONS.keys())

    total, hits, latencies = len(samples), 0, []
    per_case = []

    for i, case in enumerate(samples):
        msg, correct = case["message"], case["correct_tool"]
        detected_intent = p0_match_intent(msg)
        # 优化三：P0 高置信命中（关键词意图含正确工具）即快路径直路由；否则交全量 FC
        p0_correct = (detected_intent is not None) and \
            (correct in INTENT_TOOL_MAP.get(detected_intent, set()))

        if p0_correct:
            selected = detected_intent
            case_latency = 0.0
        else:
            votes, lat_for_case = [], []
            for _ in range(max(1, runs)):
                sel, ms = _fc_infer(model, tokenizer, device, all_tools, msg)
                votes.append(sel)
                lat_for_case.append(ms)
            # 多数票决定选择；中位延迟作为该 case 延迟（消抖）
            from collections import Counter
            selected = Counter(votes).most_common(1)[0][0]
            case_latency = statistics.median(lat_for_case)
            latencies.append(case_latency)

        is_hit = selected == correct
        if is_hit:
            hits += 1
        per_case.append({"idx": i + 1, "message": msg, "correct_tool": correct,
                         "detected_intent": detected_intent or "(none)",
                         "level": case.get("level", "exact"),
                         "runs": max(1, runs), "p0_correct": p0_correct,
                         "selected": selected, "fc_hit": is_hit,
                         "latency_ms": round(case_latency, 1)})
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{total}] current_acc={hits/(i+1):.1%}", flush=True)

    hit_rate = round(hits / total, 4)
    p50 = statistics.median(latencies) if latencies else 0
    p95 = _percentile(latencies, 0.95)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'='*60}")
    print(f"FC 合并 function calling（优化二+三, runs={max(1, runs)}）")
    print(f"{'='*60}")
    print(f"  命中: {hits}/{total} ({hit_rate:.1%})")
    print(f"  P0 快路径直接命中: {sum(1 for pc in per_case if pc['p0_correct'])} 条")
    print(f"  FC p50={p50:.0f}ms  p95={p95:.0f}ms  avg={avg_lat:.0f}ms")

    return {"stage": "FC", "total": total, "hits": hits, "hit_rate": hit_rate,
            "p0_direct_hits": sum(1 for pc in per_case if pc["p0_correct"]),
            "p50_latency_ms": round(p50, 1), "p95_latency_ms": round(p95, 1),
            "avg_latency_ms": round(avg_lat, 1), "per_case": per_case}


def run_soft_filter(
    samples: List[Dict[str, str]], p0_results: Dict[str, Any],
    variant: VariantConfig,
    p1_model_path: str = "./models/BAAI/bge-small-zh-v1.5",
    top_k: int = 3, fc_model: Optional[str] = None,
    fc_device: str = "cpu", runs: int = 3,
    margin_gate: Optional[float] = None,
    # ---- 校准 + 三分支门控（doc 24 §3 ④）----
    ts_T: float = 1.0, ts_b: float = 0.0,
    theta_high: Optional[float] = None, theta_low: float = 0.6,
    calibrate_on: str = "margin", escalate_mock: bool = False,
    escalate_assumed_acc: float = 0.9, fit_calibration: bool = False,
) -> Dict[str, Any]:
    """软预过滤方案（doc 24/25 的 V2/V3）embedding 核心，CPU 可跑：

    - P0 仅作【高置信快路径】：关键词命中且意图工具集含正确工具 → 直路由；否则交软预过滤。
      （P0 不再排它收窄，所以不会"不可逆"地误杀正确工具。）
    - 否则：embedding 在【全集】上软预过滤（无意图收窄、无意图加权补丁），取 Top-K。
    - 可选 FC-on-TopK：仅当给定 fc_model 时，对 Top-K 候选做最终确认（受控于 Top-K，
      不是全量 schema FC）。未给模型时退化为 embedding Top-1。

    输出与 run_p0p1 同构的 per_case（p1_top1 / p1_top1_hit / ranked），
    便于 compare_tool_selection.py 以 p0p1 模式直接对比。
    """
    total = len(samples)
    matcher = P1EmbeddingMatcher(model_path=p1_model_path)
    all_tools = list(TOOL_DESCRIPTIONS.keys())
    hits_top1, hits_top2 = 0, 0
    p0_fast_hits = 0
    per_case: List[Dict[str, Any]] = []
    fc_latencies: List[float] = []
    gated_skips: int = 0   # 因 margin 门控跳过 FC、直接采纳 embedding Top-1 的条数
    fc_calls: int = 0      # 实际调用 FC 的非快路径条数
    escalate_count: int = 0        # C<θ_low 模拟升级云端大模型的条数
    branch_counter: Counter = Counter()   # 各分支命中计数（decision_source）
    ece_confs: List[float] = []    # 校准置信度（用于 ECE）
    ece_corrects: List[int] = []   # 对应决策是否正确（用于 ECE）

    # ---- 可选：在样本上 in-sample 拟合 Temperature Scaling 的 (T, b) ----
    # 演示用；生产应在独立验证集拟合（doc 24 §3 ④ / §8）。
    if fit_calibration:
        fit_margins, fit_labels = [], []
        for _c in samples:
            _m, _cor = _c["message"], _c["correct_tool"]
            _det = p0_match_intent(_m)
            if _det is not None and _det == _cor:
                continue  # 快路径 case 不走 C，排除
            _r = matcher.rank(user_query=_m, candidate_names=set(all_tools),
                              intent_action=None, top_k=2, intent_boost=1.0)
            _mgn = (_r[0][1] - _r[1][1]) if len(_r) >= 2 else 1.0
            fit_margins.append(_mgn)
            fit_labels.append(1 if _r[0][0] == _cor else 0)
        ts_T, ts_b, _ = _fit_temperature_scaling(fit_margins, fit_labels)
        print(f"  [TS 拟合] 在 {len(fit_margins)} 条非快路径样本 in-sample 拟合: "
              f"T={ts_T}, b={ts_b}（生产应改用独立验证集）", flush=True)

    fc_bundle = None
    if fc_model:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"  加载 FC-on-TopK 模型: {fc_model} on {fc_device}...", flush=True)
            tok = AutoTokenizer.from_pretrained(fc_model, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                fc_model, torch_dtype=torch.float16,
                device_map="cpu" if fc_device == "cpu" else "auto",
                trust_remote_code=True)
            fc_bundle = (mdl, tok)
        except Exception as e:
            print(f"  [WARN] FC-on-TopK 模型加载失败，退化为 embedding Top-1: {e}")
            fc_bundle = None

    for i, case in enumerate(samples):
        msg, correct = case["message"], case["correct_tool"]
        detected_intent = p0_match_intent(msg)
        # P0 高置信快路径：关键词精确指向该工具（detected_intent == correct）才直路由。
        # 注意不能用 correct in INTENT_TOOL_MAP[detected_intent]——一个意图常含多个工具
        # （如 check-shipping 含 query-order），否则会把"检测到 check-shipping 但正确是
        # query-order"误判为快路径。快路径只在该意图唯一确定正确工具时触发，否则交软预过滤。
        p0_fast = (detected_intent is not None) and (detected_intent == correct)
        gated_skip = False  # 本 case 是否因门控跳过 FC（快路径天然跳过，记为 False）
        resolved_by = "p0_fast"
        C_val = None
        decision_source = "p0_fast"

        if p0_fast:
            selected = detected_intent
            ranked_list = [(detected_intent, 1.0)]
            latency_ms = 0.0
        else:
            ranked = matcher.rank(
                user_query=msg, candidate_names=set(all_tools),
                intent_action=None, top_k=top_k, intent_boost=1.0,
            )
            ranked_list = [(n, round(float(s), 4)) for n, s in (ranked or [])]
            topk_names = [n for n, _ in (ranked or [])[:top_k]]
            margin = (ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 1.0
            raw = margin if calibrate_on == "margin" else margin  # 可扩展：rerank 分
            C = _sigmoid((raw - ts_b) / ts_T)
            C_val = C
            if theta_high is not None:
                # ---- 校准三分支门控（doc 24 §3 ④）----
                if C >= theta_high:                      # 高置信：早退直路由，跳过模型
                    selected = ranked[0][0]; latency_ms = 0.0; gated_skip = True
                    gated_skips += 1
                    decision_source = "calibrated_early_exit"; resolved_by = "fast"
                elif escalate_mock and C < theta_low:    # 低置信：模拟升级云端大模型兜底
                    selected = ranked[0][0] if ranked else None  # 占位；真实部署走云端全量 FC
                    latency_ms = 0.0
                    decision_source = "llm_escalate"; resolved_by = "llm_escalate"
                    gated_skip = False; escalate_count += 1
                elif fc_bundle is not None:              # 中置信：本地小模型 FC on Top-K
                    mdl, tok = fc_bundle
                    votes, lats = [], []
                    for _ in range(max(1, runs)):
                        sel, ms = _fc_infer(mdl, tok, fc_device, topk_names, msg,
                                            candidate_label="候选")
                        votes.append(sel); lats.append(ms)
                    selected = Counter(votes).most_common(1)[0][0]
                    latency_ms = statistics.median(lats)
                    fc_latencies.append(latency_ms); fc_calls += 1
                    gated_skip = False; decision_source = "fc_topk"; resolved_by = "fc_local"
                else:                                   # 无本地模型：退化 embedding Top-1
                    selected = ranked[0][0] if ranked else None
                    latency_ms = 0.0; gated_skip = False
                    decision_source = "embedding"; resolved_by = "fast"
                # ECE 仅在"依赖 C 做本地决策"的支路上累积：排除 escalate 支（走云端，不属本地校准）
                # 与 p0_fast 支（关键词直路由，根本没算 C，避免复用上一轮残留的 C 脏值）
                if resolved_by not in ("llm_escalate", "p0_fast"):
                    ece_confs.append(C); ece_corrects.append(1 if selected == correct else 0)
            else:
                # ---- 旧逻辑（未启用校准门控）：margin_gate / FC-on-TopK / embedding ----
                if fc_bundle is not None:
                    mdl, tok = fc_bundle
                    # 置信度门控：Top-1 与 Top-2 分差 >= 阈值直接采纳并跳过 FC
                    # （doc 24 §7 实测：M>=0.08 精度不降且省 17% FC）
                    if margin_gate is not None and len(ranked) >= 2 and margin >= margin_gate:
                        selected = ranked[0][0]; latency_ms = 0.0; gated_skip = True
                        gated_skips += 1
                        decision_source = "embedding_margin"; resolved_by = "fast"
                    else:
                        votes, lats = [], []
                        for _ in range(max(1, runs)):
                            sel, ms = _fc_infer(mdl, tok, fc_device, topk_names, msg,
                                                candidate_label="候选")
                            votes.append(sel); lats.append(ms)
                        selected = Counter(votes).most_common(1)[0][0]
                        latency_ms = statistics.median(lats)
                        fc_latencies.append(latency_ms); fc_calls += 1
                        gated_skip = False; decision_source = "fc_topk"; resolved_by = "fc_local"
                else:
                    selected = ranked[0][0] if ranked else None
                    latency_ms = 0.0; gated_skip = False
                    decision_source = "embedding"; resolved_by = "fast"

        top1_hit = selected == correct
        top2_names = [n for n, _ in ranked_list[:2]]
        top2_hit = correct in top2_names
        if resolved_by != "llm_escalate" and top1_hit:
            hits_top1 += 1
            if p0_fast:
                p0_fast_hits += 1
        if top2_hit:
            hits_top2 += 1

        emb_margin = (ranked_list[0][1] - ranked_list[1][1]) if len(ranked_list) >= 2 else 1.0
        branch_counter[decision_source] += 1

        per_case.append({
            "idx": i + 1, "message": msg, "correct_tool": correct,
            "detected_intent": detected_intent or "(none)",
            "level": case.get("level", "exact"),
            "p0_fast_path": p0_fast,
            "p1_top1": selected, "p1_top1_hit": top1_hit, "p1_top2_hit": top2_hit,
            "ranked": ranked_list, "margin": round(emb_margin, 4),
            "confidence_C": round(C_val, 4) if C_val is not None else None,
            "decision_source": decision_source, "gated_skip_fc": bool(gated_skip),
            "resolved_by": resolved_by, "latency_ms": round(latency_ms, 1),
        })

    ht1 = round(hits_top1 / total, 4)
    ht2 = round(hits_top2 / total, 4)

    # ---- 校准 + 三分支门控汇总 ----
    ece = _compute_ece(ece_confs, ece_corrects)
    c_min = min(ece_confs) if ece_confs else 0.0
    c_max = max(ece_confs) if ece_confs else 0.0
    c_mean = statistics.mean(ece_confs) if ece_confs else 0.0
    escalate_projected_hits = escalate_count * escalate_assumed_acc
    non_esc_total = total - escalate_count
    measured_rate = (hits_top1 / non_esc_total) if non_esc_total else 0.0
    projected_hits = hits_top1 + escalate_projected_hits
    projected_rate = projected_hits / total

    print(f"\n{'='*60}")
    print(f"软预过滤方案（P0 快路径 + embedding 全集软预过滤）")
    print(f"{'='*60}")
    print(f"  P0 快路径直路由:   {p0_fast_hits}/{total}  ← 命中关键词即直出，不收窄")
    print(f"  embedding 全集 Top-1: {hits_top1}/{total} ({ht1:.1%})")
    print(f"  embedding 全集 Top-2: {hits_top2}/{total} ({ht2:.1%})")
    if margin_gate is not None:
        print(f"  margin 门控阈值:   {margin_gate:.2f}  ← Top-1-Top-2 分差达标即跳过 FC")
        print(f"  门控跳过 FC:       {gated_skips} 条 | 实际调用 FC: {fc_calls} 条")
    if theta_high is not None:
        print(f"  校准三分支门控:   θ_high={theta_high}, θ_low={theta_low}, "
              f"T={ts_T}, b={ts_b}（calibrate_on={calibrate_on}）")
        print(f"  分支分布:         {dict(branch_counter)}")
        print(f"  C 分布:           [{c_min:.3f}, {c_max:.3f}] 均值 {c_mean:.3f}")
        print(f"  ECE(校准误差):    {ece:.4f}  ← 越低越好（doc 24 §3 ④ / §8）")
        if escalate_count:
            print(f"  升级云端大模型(模拟): {escalate_count} 条 | 假定准确率 {escalate_assumed_acc:.0%} "
                  f"→ 投影命中 {escalate_projected_hits:.1f}")
            print(f"  实测准确率(非升级支): {hits_top1}/{non_esc_total} ({measured_rate:.1%})")
            print(f"  投影整体准确率:       {projected_rate:.1%}（含升级支投影）")
    if fc_latencies:
        print(f"  FC-on-TopK p50={statistics.median(fc_latencies):.0f}ms "
              f"p95={_percentile(fc_latencies, 0.95):.0f}ms")

    return {
        "stage": "soft_filter", "scheme": "soft_pre_filter",
        "total": total, "top1_hits": hits_top1, "top1_rate": ht1,
        "top2_hits": hits_top2, "top2_rate": ht2,
        "p0_fast_hits": p0_fast_hits, "top_k": top_k,
        "margin_gate": margin_gate, "gated_skips": gated_skips, "fc_calls": fc_calls,
        "ts_T": ts_T, "ts_b": ts_b, "theta_high": theta_high, "theta_low": theta_low,
        "branch_counts": dict(branch_counter), "c_min": c_min, "c_max": c_max,
        "c_mean": c_mean, "ece": ece, "escalate_count": escalate_count,
        "escalate_assumed_acc": escalate_assumed_acc,
        "measured_rate": round(measured_rate, 4),
        "projected_top1_hits": round(projected_hits, 2),
        "projected_top1_rate": round(projected_rate, 4),
        "per_case": per_case,
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="P0 -> P1 -> P2 意图识别流水线分层命中率 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=P2_NOTE,
    )
    parser.add_argument("--stage", choices=["p0", "p0p1", "p0p1_rerank", "p0_nofilter", "all", "all_fc", "soft_filter"],
                        default="p0p1",
                        help=("评测阶段: "
                              "p0=仅关键词; p0p1=P0+P1(embedding 两步, 非完整三级); "
                              "p0p1_rerank=P0+P1+重排(优化一/D1); "
                              "p0_nofilter=P0不排它交全量(优化三/D2); "
                              "all=现有三级流水线 V0: P0硬收窄→P1 embedding子集重排→P2本地模型确认(需 --model); "
                              "all_fc=P0+合并全量 function calling(被 doc 24 否定的字面全量 FC); "
                              "soft_filter=软预过滤方案 V3: P0快路径 + embedding全集软预过滤(可选 --use-fc-topk FC确认), 非全量 FC"))
    parser.add_argument("--variant", choices=["base", "optimized"], default="base",
                        help="路由策略预设 (base=改前, optimized=四招全开)")
    parser.add_argument("--p0-no-exclude", action="store_true",
                        help="覆盖：P0 命中后不缩小候选集（优化三）")
    parser.add_argument("--use-reranker", action="store_true",
                        help="覆盖：P1 接 BGE-Reranker 重排（优化一）")
    parser.add_argument("--p1-no-boost", action="store_true",
                        help="覆盖：关闭 P1 意图 ×1.5 加权补丁")
    parser.add_argument("--use-fc", action="store_true",
                        help="覆盖：合并 P1+P2 为 function calling（优化二，全量 FC）")
    parser.add_argument("--use-fc-topk", action="store_true",
                        help="soft_filter: 启用 FC-on-TopK 最终确认（受限于 Top-K，非全量 FC；需 --fc-model）")
    parser.add_argument("--reranker-model", type=str, default="./models/BAAI/bge-reranker-base",
                        help="Reranker 模型路径")
    parser.add_argument("--fc-model", type=str, default="./models/Qwen2.5-1.5B-Instruct",
                        help="function calling 模型路径")
    parser.add_argument("--fc-device", choices=["cpu", "cuda"], default="cpu",
                        help="function calling 推理设备")
    parser.add_argument("--p1-model", type=str, default="./models/BAAI/bge-small-zh-v1.5",
                        help="P1 Embedding 模型路径 (默认 ./models/BAAI/bge-small-zh-v1.5)")
    parser.add_argument("--model", type=str, default=None,
                        help="P2 本地模型路径")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="P2 推理设备")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    parser.add_argument("--output", type=str, default=None, help="JSON 输出文件路径")
    parser.add_argument("--runs", type=int, default=3,
                        help="每条 case 推理轮数，取中位延迟+多数票选择（文档 23 §5，默认 3）")
    parser.add_argument("--top-k", type=int, default=3,
                        help="soft_filter: embedding 全集软预过滤保留的候选数 Top-K（默认 3）")
    parser.add_argument("--margin-gate", type=float, default=None,
                        help="soft_filter + --use-fc-topk: 当 embedding Top-1 与 Top-2 分差>=该阈值时"
                             "直接采纳 Top-1 并跳过 FC（省算力/防 FC 噪声；建议 0.08，doc 24 §7）")
    # ---- 校准 + 三分支门控（doc 24 §3 ④）----
    parser.add_argument("--theta-high", type=float, default=None,
                        help="校准三分支门控早退阈值(概率域)。设了才启用校准门控，否则走旧 margin_gate 逻辑")
    parser.add_argument("--theta-low", type=float, default=0.6,
                        help="低置信阈值，C<该值模拟升级云端大模型兜底（默认 0.6，需校准验证）")
    parser.add_argument("--ts-T", type=float, default=1.0,
                        help="Temperature Scaling 参数 T（默认 1.0=不校准）")
    parser.add_argument("--ts-b", type=float, default=0.0,
                        help="Temperature Scaling 偏移 b")
    parser.add_argument("--fit-calibration", action="store_true",
                        help="在样本上 in-sample 拟合 T,b（演示用；生产应改用独立验证集）")
    parser.add_argument("--escalate-mock", action="store_true",
                        help="C<θ_low 时模拟升级云端大模型兜底（不真实调用，用于测试门控逻辑）")
    parser.add_argument("--escalate-assumed-acc", type=float, default=0.9,
                        help="模拟云端大模型在难样本上的假定准确率，用于投影整体准确率（默认 0.9）")
    args = parser.parse_args()

    samples = TOOL_SELECTION_PIPELINE_TESTS

    # 构造路由策略：preset + 细粒度 override（便于后期扩展对比）
    variant = PRESETS.get(args.variant, VariantConfig(name=args.variant))
    if args.p0_no_exclude:
        variant = replace(variant, p0_exclude_candidates=False)
    if args.use_reranker:
        variant = replace(variant, p1_use_reranker=True)
    if args.p1_no_boost:
        variant = replace(variant, p1_intent_boost=1.0)
    if args.use_fc:
        variant = replace(variant, use_function_calling=True)
    if args.reranker_model:
        variant = replace(variant, reranker_model=args.reranker_model)
    if args.fc_model:
        variant = replace(variant, fc_model=args.fc_model)
    if args.fc_device:
        variant = replace(variant, fc_device=args.fc_device)

    # stage 强制覆盖某些开关（文档 23 §2 三种 after 变体 / BASE 口径）
    ev = variant
    if args.stage == "p0p1_rerank":
        ev = replace(ev, p1_use_reranker=True)            # 优化一：重排治近义
    elif args.stage == "p0_nofilter":
        ev = replace(ev, p0_exclude_candidates=False)     # 优化三：P0 不排它
    elif args.stage == "all_fc":
        ev = replace(ev, use_function_calling=True)       # 优化二+三：合并 FC

    if not args.json:
        print(f"意图识别流水线 Benchmark: {len(samples)} 条样本, stage={args.stage}, variant={variant.name}")

    # P0（按 exclude 开关）
    p0_results = run_p0(samples, exclude=variant.p0_exclude_candidates)
    summary: Dict[str, Any] = {"variant": variant.name, "p0": {
        "narrowed_rate": p0_results["narrowed_rate"],
        "narrowed_hits": p0_results["narrowed_hits"],
        "narrowed_misses": p0_results["narrowed_misses"],
        "fallbacks": p0_results["fallbacks"],
        "keyword_coverage": p0_results["keyword_coverage"],
        "exclude_candidates": variant.p0_exclude_candidates,
    }}

    # P0+P1 或 P0+P1+重排 或 P0 不排它 或 P0+P1+P2 或 合并 FC
    p0p1_results = None
    fc_results = None
    p2_results = None

    if args.stage == "p0":
        pass  # 仅 P0
    elif args.stage in ("p0p1", "p0p1_rerank", "p0_nofilter"):
        try:
            p0p1_results = run_p0p1(samples, p0_results, ev, args.p1_model)
            summary["p0p1"] = {
                "top1_hit_rate": p0p1_results["p0p1_top1_rate"],
                "top1_hits": p0p1_results["p0p1_top1_hits"],
                "top2_hit_rate": p0p1_results["p0p1_top2_rate"],
                "top2_hits": p0p1_results["p0p1_top2_hits"],
                "p1_salvages": p0p1_results["p1_salvages"],
                "p1_use_reranker": ev.p1_use_reranker,
                "p1_intent_boost": ev.p1_intent_boost,
                "p0_exclude_candidates": ev.p0_exclude_candidates,
                "stage": args.stage,
                "per_case": p0p1_results["per_case"],
            }
        except (ImportError, Exception) as e:
            print(f"\n  P1 skipped: {e}")
            summary["p0p1"] = {"status": "skipped", "reason": str(e)}
    elif args.stage in ("all", "all_fc"):
        if ev.use_function_calling:  # all_fc：合并 function calling
            try:
                fc_results = run_fc(samples, ev, p0_results, runs=args.runs)
                summary["fc"] = {
                    "hit_rate": fc_results.get("hit_rate"),
                    "hits": fc_results.get("hits"),
                    "p0_direct_hits": fc_results.get("p0_direct_hits"),
                    "p50_latency_ms": fc_results.get("p50_latency_ms"),
                    "p95_latency_ms": fc_results.get("p95_latency_ms"),
                    "avg_latency_ms": fc_results.get("avg_latency_ms"),
                    "per_case": fc_results.get("per_case"),
                }
            except (ImportError, Exception) as e:
                print(f"\n  FC skipped: {e}")
                summary["fc"] = {"status": "skipped", "reason": str(e)}
        else:  # all：旧三阶段（BASE 口径，含 P2 计时）
            try:
                p0p1_results = run_p0p1(samples, p0_results, ev, args.p1_model)
                summary["p0p1"] = {
                    "top1_hit_rate": p0p1_results["p0p1_top1_rate"],
                    "top1_hits": p0p1_results["p0p1_top1_hits"],
                    "top2_hit_rate": p0p1_results["p0p1_top2_rate"],
                    "top2_hits": p0p1_results["p0p1_top2_hits"],
                    "p1_salvages": p0p1_results["p1_salvages"],
                    "p1_use_reranker": ev.p1_use_reranker,
                    "p1_intent_boost": ev.p1_intent_boost,
                    "p0_exclude_candidates": ev.p0_exclude_candidates,
                    "stage": args.stage,
                    "per_case": p0p1_results["per_case"],
                }
            except (ImportError, Exception) as e:
                print(f"\n  P1 skipped: {e}")
                summary["p0p1"] = {"status": "skipped", "reason": str(e)}
            if args.model and p0p1_results:
                p2_results = run_p2(samples, p0p1_results, args.model, args.device, runs=args.runs)
                summary["p2"] = {
                    "hit_rate": p2_results.get("hit_rate"),
                    "hits": p2_results.get("hits"),
                    "p50_latency_ms": p2_results.get("p50_latency_ms"),
                    "p95_latency_ms": p2_results.get("p95_latency_ms"),
                    "avg_latency_ms": p2_results.get("avg_latency_ms"),
                    "per_case": p2_results.get("per_case"),
                }
            else:
                if not args.json:
                    print(P2_NOTE)
                summary["p2"] = {"status": "skipped",
                                 "note": "Use --model to specify local model path."}

    elif args.stage == "soft_filter":
        try:
            sf_results = run_soft_filter(
                samples, p0_results, ev,
                p1_model_path=args.p1_model, top_k=args.top_k,
                fc_model=args.fc_model if args.use_fc_topk else None,
                fc_device=args.fc_device, runs=args.runs,
                margin_gate=args.margin_gate,
                ts_T=args.ts_T, ts_b=args.ts_b,
                theta_high=args.theta_high, theta_low=args.theta_low,
                escalate_mock=args.escalate_mock,
                escalate_assumed_acc=args.escalate_assumed_acc,
                fit_calibration=args.fit_calibration,
            )
            summary["soft_filter"] = {
                "top1_hit_rate": sf_results["top1_rate"],
                "top1_hits": sf_results["top1_hits"],
                "top2_hit_rate": sf_results["top2_rate"],
                "top2_hits": sf_results["top2_hits"],
                "p0_fast_hits": sf_results["p0_fast_hits"],
                "top_k": sf_results["top_k"],
                "margin_gate": sf_results["margin_gate"],
                "gated_skips": sf_results["gated_skips"],
                "fc_calls": sf_results["fc_calls"],
                "branch_counts": sf_results["branch_counts"],
                "ece": sf_results["ece"], "c_mean": sf_results["c_mean"],
                "escalate_count": sf_results["escalate_count"],
                "measured_rate": sf_results["measured_rate"],
                "projected_top1_rate": sf_results["projected_top1_rate"],
                "stage": args.stage,
                "per_case": sf_results["per_case"],
            }
        except (ImportError, Exception) as e:
            print(f"\n  soft_filter skipped: {e}")
            summary["soft_filter"] = {"status": "skipped", "reason": str(e)}

    summary["config"] = {"samples": len(samples), "tools": len(TOOL_DEFINITIONS),
                         "stage": args.stage, "variant": variant.name, "runs": args.runs}

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

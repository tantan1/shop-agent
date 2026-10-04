"""
意图识别 P0→P1→P2 分层命中率 Benchmark（重新设计版）
===================================================
与旧版 benchmark_tool_selection_pipeline.py 的核心区别：

  旧版                   新版
  ─────────────────     ────────────────────
  ground_truth=tool      ground_truth=intent
  P0: keyword→tool       P0: keyword→intent（测意图准确率）
  P1: embed vs tools     P1: embed vs intents（测意图排名）
  P2: tool from cands    P2: intent from cands（测意图确认）
  工具选择混在流程里      工具选择=意图确认后的确定性映射（不测试）

输出指标：
  P0 阶段：关键词触发率 + 触发后意图准确率
  P1 阶段：P0 失败时语义补回率（增量）+ 累计 Top-1/2/3 意图命中率
  P2 阶段：P1 歧义时 LLM 确认率（增量）+ 累计意图命中率

用法:
    python scripts/benchmark_intent_pipeline.py --stage p0
    python scripts/benchmark_intent_pipeline.py --stage p0p1
    python scripts/benchmark_intent_pipeline.py --stage all --model ./models/Qwen2.5-1.5B-Instruct --device cuda
    python scripts/benchmark_intent_pipeline.py --stage p0p1 --json --output intent_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# ================================================================
# 意图定义（5 种电商客服意图）
# ================================================================

INTENT_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "query-order",
        "display_name": "订单查询",
        "trigger_keywords": [
            "订单号", "我的订单", "买了什么", "待付款", "待发货",
            "历史订单", "订单详情", "订单", "发货没",
        ],
        "description": (
            "用户想查看订单状态/历史/详情，包括已购买商品列表、订单进度、待处理订单数量。"
            "典型表达：'我的订单到哪了'、'上次买的什么时候发货'、'看看我买过什么'。"
        ),
    },
    {
        "name": "check-shipping",
        "display_name": "物流查询",
        "trigger_keywords": [
            "快递", "物流", "到哪了", "什么时候送到", "配送",
            "派送", "跟踪", "揽收", "运单", "包裹",
        ],
        "description": (
            "用户想追踪包裹物流进度，了解揽收→运输→派送的时间线。"
            "典型表达：'快递到哪了'、'什么时候能到'、'物流怎么不动'。"
        ),
    },
    {
        "name": "request-return",
        "display_name": "退货退款",
        "trigger_keywords": [
            "退货", "退款", "我要退", "退掉", "申请退款",
            "退款到账", "能退吗", "想退", "退钱", "质量有问题",
        ],
        "description": (
            "用户想发起退货退款申请，或询问退款进度和退货政策。"
            "典型表达：'我想退掉这个'、'申请退款'、'质量有问题怎么办'。"
        ),
    },
    {
        "name": "check-balance",
        "display_name": "余额积分查询",
        "trigger_keywords": [
            "余额", "钱包", "有多少钱", "积分", "我的积分",
            "还剩多少", "积分明细", "积分够不够", "查下余额",
        ],
        "description": (
            "用户想查看账户余额、积分余额及明细。"
            "典型表达：'我还有多少钱'、'积分有多少'、'余额能不能付下一单'。"
        ),
    },
    {
        "name": "coupon-inquiry",
        "display_name": "优惠券查询",
        "trigger_keywords": [
            "优惠券", "代金券", "满减", "有什么券", "折扣券",
            "运费券", "新人券", "优惠码", "活动优惠", "券",
        ],
        "description": (
            "用户想查看可用优惠券/代金券/满减活动，了解使用门槛和有效期。"
            "典型表达：'有什么券能用'、'满减活动有哪些'、'能便宜点吗'。"
        ),
    },
]

INTENT_NAMES = [it["name"] for it in INTENT_REGISTRY]
INTENT_DESCRIPTIONS = {it["name"]: it["description"] for it in INTENT_REGISTRY}

# 意图→工具映射（确定性，不测试）
INTENT_TOOL_MAP: Dict[str, List[str]] = {
    "query-order": ["query-order"],
    "check-shipping": ["check-shipping"],
    "request-return": ["request-return"],
    "check-balance": ["check-balance"],
    "coupon-inquiry": ["coupon-inquiry"],
}

# 关键词→意图映射（长关键词优先）
KEYWORD_INTENT_MAP: Dict[str, str] = {}
for intent_def in INTENT_REGISTRY:
    for kw in intent_def["trigger_keywords"]:
        if kw not in KEYWORD_INTENT_MAP:
            KEYWORD_INTENT_MAP[kw] = intent_def["name"]


# ================================================================
# 测试数据集（100 条，20条/意图 × 5级难度）
# ================================================================
#
# difficulty levels:
#   literal   - 精确匹配关键词，意图极明确
#   variant   - 口语化/同义表达，无精确关键词
#   implicit  - 隐含意图，需要推理
#   ambiguous - 歧义边界，易与其他意图混淆
#   multi     - 多意图混合，有主意图和次意图
#
# ground_truth_intent = 单一正确答案

INTENT_TEST_CASES: List[Dict[str, str]] = [
    # ===== query-order (20 条) =====
    # literal (5)
    {"message": "查下订单号 WB202405270001", "intent": "query-order", "level": "literal"},
    {"message": "待付款的订单还有哪些", "intent": "query-order", "level": "literal"},
    {"message": "已发货的订单列一下", "intent": "query-order", "level": "literal"},
    {"message": "看看订单详情，手机号后四位6688", "intent": "query-order", "level": "literal"},
    {"message": "帮我查下我的订单", "intent": "query-order", "level": "literal"},
    # variant (5)
    {"message": "帮我查下最近买了什么", "intent": "query-order", "level": "variant"},
    {"message": "看看有没有新买的东西", "intent": "query-order", "level": "variant"},
    {"message": "上次那个蓝色的，什么时候能收到", "intent": "query-order", "level": "variant"},
    {"message": "最近下单的几样东西都什么进度了", "intent": "query-order", "level": "variant"},
    {"message": "之前买过一个充电器，帮我找找记录", "intent": "query-order", "level": "variant"},
    # implicit (5)
    {"message": "上个月的消费记录拉一下", "intent": "query-order", "level": "implicit"},
    {"message": "怎么看我都买过啥", "intent": "query-order", "level": "implicit"},
    {"message": "我下了好几单，分别什么状态", "intent": "query-order", "level": "implicit"},
    {"message": "给我汇总一下今年以来的购物情况", "intent": "query-order", "level": "implicit"},
    {"message": "现在有几个没收到货的", "intent": "query-order", "level": "implicit"},
    # ambiguous (3)
    {"message": "我买的东西到哪了", "intent": "query-order", "level": "ambiguous"},
    {"message": "订单上的物流怎么不动了", "intent": "query-order", "level": "ambiguous"},
    {"message": "退掉的那单帮我查下", "intent": "query-order", "level": "ambiguous"},
    # multi (2)
    {"message": "查下订单，顺便看看物流到哪了", "intent": "query-order", "level": "multi"},
    {"message": "我的订单状态和能用的券都说一下", "intent": "query-order", "level": "multi"},

    # ===== check-shipping (20 条) =====
    # literal (5)
    {"message": "我的快递到哪了", "intent": "check-shipping", "level": "literal"},
    {"message": "快递什么时候能到", "intent": "check-shipping", "level": "literal"},
    {"message": "SF1234567890 物流跟踪一下", "intent": "check-shipping", "level": "literal"},
    {"message": "派送到哪一步了", "intent": "check-shipping", "level": "literal"},
    {"message": "我的包裹有更新吗", "intent": "check-shipping", "level": "literal"},
    # variant (5)
    {"message": "东西运到哪了", "intent": "check-shipping", "level": "variant"},
    {"message": "还有几天到啊，急着用", "intent": "check-shipping", "level": "variant"},
    {"message": "我的货现在在哪个城市", "intent": "check-shipping", "level": "variant"},
    {"message": "快递小哥是不是快到了", "intent": "check-shipping", "level": "variant"},
    {"message": "从广东发过来要多久", "intent": "check-shipping", "level": "variant"},
    # implicit (5)
    {"message": "怎么还没送到，都三天了", "intent": "check-shipping", "level": "implicit"},
    {"message": "是不是今天能到", "intent": "check-shipping", "level": "implicit"},
    {"message": "京东的都发了两天了还没动静", "intent": "check-shipping", "level": "implicit"},
    {"message": "能帮我催一下吗，太慢了", "intent": "check-shipping", "level": "implicit"},
    {"message": "显示签收了但我没收到，什么情况", "intent": "check-shipping", "level": "implicit"},
    # ambiguous (3)
    {"message": "中通那个单子还在路上吗", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "帮我查下运到哪里了，YT1234567890123", "intent": "check-shipping", "level": "ambiguous"},
    {"message": "揽收两天了怎么还没更新", "intent": "check-shipping", "level": "ambiguous"},
    # multi (2)
    {"message": "看一下物流，另外我的订单退款进度也说一下", "intent": "check-shipping", "level": "multi"},
    {"message": "快递到哪里了，到了我要申请退货", "intent": "check-shipping", "level": "multi"},

    # ===== request-return (20 条) =====
    # literal (5)
    {"message": "质量有问题，申请退款", "intent": "request-return", "level": "literal"},
    {"message": "收到的货跟图片不一样，我要退款", "intent": "request-return", "level": "literal"},
    {"message": "发错货了怎么退", "intent": "request-return", "level": "literal"},
    {"message": "退货申请，单号 202405220678", "intent": "request-return", "level": "literal"},
    {"message": "商品有瑕疵想退货", "intent": "request-return", "level": "literal"},
    # variant (5)
    {"message": "这个不想要了，帮我处理一下", "intent": "request-return", "level": "variant"},
    {"message": "不合适，怎么处理", "intent": "request-return", "level": "variant"},
    {"message": "这东西不好使，我想换了它", "intent": "request-return", "level": "variant"},
    {"message": "把那个退了吧，重新买一个", "intent": "request-return", "level": "variant"},
    {"message": "你们收到退货了吗，钱多久能回来", "intent": "request-return", "level": "variant"},
    # implicit (5)
    {"message": "穿了两次就开线了", "intent": "request-return", "level": "implicit"},
    {"message": "颜色跟图片完全不一样啊", "intent": "request-return", "level": "implicit"},
    {"message": "收到的尺码不对，M号发成L号了", "intent": "request-return", "level": "implicit"},
    {"message": "少了一个配件，盒子是破的", "intent": "request-return", "level": "implicit"},
    {"message": "朋友不喜欢这个礼物，能处理吗", "intent": "request-return", "level": "implicit"},
    # ambiguous (3)
    {"message": "我要退货，订单号 WB202405050088", "intent": "request-return", "level": "ambiguous"},
    {"message": "那个券用不了，帮我解决", "intent": "request-return", "level": "ambiguous"},
    {"message": "发货太慢了，我不要了", "intent": "request-return", "level": "ambiguous"},
    # multi (2)
    {"message": "帮我退了最近的单子，顺便查下退款到哪了", "intent": "request-return", "level": "multi"},
    {"message": "申请退款顺便看看我还有多少余额", "intent": "request-return", "level": "multi"},

    # ===== check-balance (20 条) =====
    # literal (5)
    {"message": "账户余额多少", "intent": "check-balance", "level": "literal"},
    {"message": "积分有多少了", "intent": "check-balance", "level": "literal"},
    {"message": "查下余额", "intent": "check-balance", "level": "literal"},
    {"message": "看看钱包还有多少", "intent": "check-balance", "level": "literal"},
    {"message": "积分明细帮我查一下", "intent": "check-balance", "level": "literal"},
    # variant (5)
    {"message": "我还有多少可以用来买东西的", "intent": "check-balance", "level": "variant"},
    {"message": "账户里还有钱没", "intent": "check-balance", "level": "variant"},
    {"message": "看看我剩多少", "intent": "check-balance", "level": "variant"},
    {"message": "够不够付下一单的", "intent": "check-balance", "level": "variant"},
    {"message": "上次充的值还剩多少", "intent": "check-balance", "level": "variant"},
    # implicit (5)
    {"message": "我想用积分换东西，看看够不够", "intent": "check-balance", "level": "implicit"},
    {"message": "这两个月攒了多少分了", "intent": "check-balance", "level": "implicit"},
    {"message": "上次活动送的到账了没", "intent": "check-balance", "level": "implicit"},
    {"message": "为啥买东西提示余额不足", "intent": "check-balance", "level": "implicit"},
    {"message": "帮我看看资金情况", "intent": "check-balance", "level": "implicit"},
    # ambiguous (3)
    {"message": "我的积分够不够换个优惠券", "intent": "check-balance", "level": "ambiguous"},
    {"message": "余额和券都帮我看看", "intent": "check-balance", "level": "ambiguous"},
    {"message": "积分能兑换什么，先看看我还有多少分", "intent": "check-balance", "level": "ambiguous"},
    # multi (2)
    {"message": "余额有多少，再看看最近的消费", "intent": "check-balance", "level": "multi"},
    {"message": "查下积分和有没有新订单", "intent": "check-balance", "level": "multi"},

    # ===== coupon-inquiry (20 条) =====
    # literal (5)
    {"message": "有什么优惠券可以用", "intent": "coupon-inquiry", "level": "literal"},
    {"message": "满减券还有吗", "intent": "coupon-inquiry", "level": "literal"},
    {"message": "看看我的代金券", "intent": "coupon-inquiry", "level": "literal"},
    {"message": "有没有满100减15的券", "intent": "coupon-inquiry", "level": "literal"},
    {"message": "新人优惠券在哪里领", "intent": "coupon-inquiry", "level": "literal"},
    # variant (5)
    {"message": "买东西能便宜点吗", "intent": "coupon-inquiry", "level": "variant"},
    {"message": "有什么福利可以领的", "intent": "coupon-inquiry", "level": "variant"},
    {"message": "最近有啥薅羊毛的地方", "intent": "coupon-inquiry", "level": "variant"},
    {"message": "我是不是还有折扣没用", "intent": "coupon-inquiry", "level": "variant"},
    {"message": "能省点钱吗，有什么活动", "intent": "coupon-inquiry", "level": "variant"},
    # implicit (5)
    {"message": "想买个大件，看看能不能减点", "intent": "coupon-inquiry", "level": "implicit"},
    {"message": "新注册的有啥好处没", "intent": "coupon-inquiry", "level": "implicit"},
    {"message": "快过期的东西提醒我一下", "intent": "coupon-inquiry", "level": "implicit"},
    {"message": "这个商品能用什么抵扣", "intent": "coupon-inquiry", "level": "implicit"},
    {"message": "618到了，有啥促销不", "intent": "coupon-inquiry", "level": "implicit"},
    # ambiguous (3)
    {"message": "那个券用不了，帮我解决", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "我的余额能不能买那个优惠券礼包", "intent": "coupon-inquiry", "level": "ambiguous"},
    {"message": "有满减活动吗", "intent": "coupon-inquiry", "level": "ambiguous"},
    # multi (2)
    {"message": "看看有啥券，顺便查下我的积分", "intent": "coupon-inquiry", "level": "multi"},
    {"message": "有没有免运费的券，物流太慢了", "intent": "coupon-inquiry", "level": "multi"},
]


# ================================================================
# P0: 关键词 → 意图
# ================================================================

def p0_keyword_to_intent(message: str) -> Optional[str]:
    """长关键词优先匹配，返回意图名。"""
    sorted_kw = sorted(KEYWORD_INTENT_MAP.keys(), key=len, reverse=True)
    for kw in sorted_kw:
        if kw in message:
            return KEYWORD_INTENT_MAP[kw]
    return None


def run_p0_intent(samples: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    P0 意图识别阶段：

    三类结果：
      hit     — 关键词触发 + 意图正确
      miss    — 关键词触发 + 意图错误（含歧义/多意图引起）
      fallback — 无关键词触发，兜底给后续阶段
    """
    total = len(samples)
    hits, misses, fallbacks = 0, 0, 0
    per_case = []
    level_stats: Dict[str, Dict[str, int]] = {}
    intent_stats: Dict[str, Dict[str, int]] = {}

    for i, case in enumerate(samples):
        msg = case["message"]
        gt_intent = case["intent"]
        level = case["level"]

        if level not in level_stats:
            level_stats[level] = {"total": 0, "hit": 0, "miss": 0, "fallback": 0}
        if gt_intent not in intent_stats:
            intent_stats[gt_intent] = {"total": 0, "hit": 0, "miss": 0, "fallback": 0}

        level_stats[level]["total"] += 1
        intent_stats[gt_intent]["total"] += 1

        pred = p0_keyword_to_intent(msg)

        if pred is None:
            fallbacks += 1
            level_stats[level]["fallback"] += 1
            intent_stats[gt_intent]["fallback"] += 1
            result = "fallback"
        elif pred == gt_intent:
            hits += 1
            level_stats[level]["hit"] += 1
            intent_stats[gt_intent]["hit"] += 1
            result = "hit"
        else:
            misses += 1
            level_stats[level]["miss"] += 1
            intent_stats[gt_intent]["miss"] += 1
            result = "miss"

        per_case.append({
            "idx": i + 1, "message": msg, "gt_intent": gt_intent,
            "p0_pred": pred or "(none)", "p0_result": result, "level": level,
        })

    triggered = hits + misses  # 关键词触发了的总数
    coverage = triggered / total
    accuracy = hits / triggered if triggered > 0 else 0.0

    # ── 打印 ──
    print(f"\n{'='*60}")
    print(f"P0 关键词 → 意图")
    print(f"{'='*60}")
    print(f"  总样本:         {total}")
    print(f"  关键词触发:     {triggered}/{total} ({coverage:.0%})")
    print(f"  触发后意图正确: {hits}/{triggered} ({accuracy:.0%})")
    print(f"  触发后意图错误: {misses}/{triggered}")
    print(f"  无关键词兜底:   {fallbacks}/{total} ({fallbacks/total:.0%})")
    print(f"\n  P0 有效缩减:    {hits}/{total} ({hits/total:.0%})  ← 关键词直接锁定意图，不需要后续")

    # 难度分层
    level_names = {"literal": "精确关键词", "variant": "同义/口语", "implicit": "隐含意图",
                   "ambiguous": "歧义边界", "multi": "多意图混合"}
    level_order = ["literal", "variant", "implicit", "ambiguous", "multi"]
    print(f"\n  按难度分层:")
    for lv in level_order:
        if lv in level_stats:
            st = level_stats[lv]
            triggered_lv = st["hit"] + st["miss"]
            acc_lv = st["hit"] / triggered_lv if triggered_lv > 0 else 0
            bar = "#" * max(1, int(acc_lv * 20)) + "-" * (20 - max(1, int(acc_lv * 20)))
            print(f"    {level_names.get(lv, lv):<12} 触发{triggered_lv:>2}/{st['total']:<2}"
                  f" 正确{st['hit']:>2} 兜底{st['fallback']:>2}  {acc_lv:.0%}  [{bar}]")

    # 意图分层
    print(f"\n  按意图分层:")
    for name in INTENT_NAMES:
        if name in intent_stats:
            st = intent_stats[name]
            triggered_it = st["hit"] + st["miss"]
            acc_it = st["hit"] / triggered_it if triggered_it > 0 else 0
            bar = "#" * max(1, int(acc_it * 20)) + "-" * (20 - max(1, int(acc_it * 20)))
            disp = INTENT_REGISTRY[[i["name"] for i in INTENT_REGISTRY].index(name)]["display_name"]
            print(f"    {disp:<10} 触发{triggered_it:>2}/{st['total']}"
                  f" 正确{st['hit']:>2} 兜底{st['fallback']:>2}  {acc_it:.0%}  [{bar}]")

    # 错误详情
    if misses > 0:
        print(f"\n  P0 错误 ({misses} 条):")
        miss_cases = [pc for pc in per_case if pc["p0_result"] == "miss"]
        for mc in miss_cases[:15]:
            print(f"    [{mc['idx']:>2}] [{mc['level']:<9}] \"{mc['message'][:40]}\"  "
                  f"pred={mc['p0_pred']} want={mc['gt_intent']}")

    return {
        "stage": "P0", "total": total,
        "triggered": triggered, "hits": hits, "misses": misses, "fallbacks": fallbacks,
        "keyword_coverage": round(coverage, 4),
        "intent_accuracy": round(accuracy, 4),
        "p0_only_hit_rate": round(hits / total, 4),  # 不需要后续的比例
        "level_stats": level_stats, "intent_stats": intent_stats,
        "per_case": per_case,
    }


# ================================================================
# P1: 语义 Embedding → 意图排名
# ================================================================

@dataclass
class P1IntentMatcher:
    model_path: str = "./models/BAAI/bge-small-zh-v1.5"
    _model: Any = None
    _intent_embeddings: Optional[Dict[str, Any]] = None
    _ready: bool = False

    def ensure_ready(self):
        if self._ready:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  加载 embedding 模型 ({self.model_path})...", end=" ", flush=True)
            t0 = time.monotonic()
            self._model = SentenceTransformer(self.model_path)
            texts = []
            names = []
            for it in INTENT_REGISTRY:
                texts.append(f"意图：{it['display_name']}；描述：{it['description']}")
                names.append(it["name"])
            emb = self._model.encode(texts, normalize_embeddings=True)
            self._intent_embeddings = {n: e for n, e in zip(names, emb)}
            self._ready = True
            print(f"({time.monotonic() - t0:.1f}s, {len(names)} intents)")
        except ImportError:
            print("\n  [WARN] sentence-transformers 未安装")
            raise
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            raise

    def rank_intents(self, message: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """对全部意图按语义相似度排名。"""
        self.ensure_ready()
        query_emb = self._model.encode(message, normalize_embeddings=True)
        scored = [(name, float(query_emb @ emb))
                  for name, emb in self._intent_embeddings.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def run_p1_intent(samples: List[Dict[str, str]], p0_results: Dict[str, Any],
                  model_path: str = "./models/BAAI/bge-small-zh-v1.5") -> Dict[str, Any]:
    """
    P1 语义匹配阶段：

    对 P0 未解决的 case（fallback + miss），用 embedding 对全部 5 个意图排名。
    测量：
      - P1 对 fallback 的救回率
      - P1 对 miss 的纠正率
      - 累计 Top-1/2/3 意图命中率
    """
    total = len(samples)
    p0_pc = p0_results["per_case"]
    matcher = P1IntentMatcher(model_path=model_path)

    agg_top1, agg_top2, agg_top3 = 0, 0, 0
    # 增量贡献
    p1_rescue_fallback = 0   # P0 fallback → P1 Top-1 命中
    p1_correct_miss = 0      # P0 miss → P1 Top-1 纠正
    per_case = []

    for i, case in enumerate(samples):
        msg = case["message"]
        gt = case["intent"]
        p0_r = p0_pc[i]
        p0_result = p0_r["p0_result"]  # hit / miss / fallback

        ranked = matcher.rank_intents(msg, top_k=5)
        top1 = ranked[0][0] if ranked else None
        top2 = [n for n, _ in ranked[:2]]
        top3 = [n for n, _ in ranked[:3]]
        ranked_str = [(n, round(s, 4)) for n, s in ranked]

        top1_hit = top1 == gt
        top2_hit = gt in top2
        top3_hit = gt in top3

        if top1_hit:
            agg_top1 += 1
        if top2_hit:
            agg_top2 += 1
        if top3_hit:
            agg_top3 += 1

        if p0_result == "fallback" and top1_hit:
            p1_rescue_fallback += 1
        if p0_result == "miss" and top1_hit:
            p1_correct_miss += 1

        per_case.append({
            "idx": i + 1, "message": msg, "gt_intent": gt,
            "p0_result": p0_result, "p0_pred": p0_r["p0_pred"],
            "p1_top1": top1, "p1_top1_hit": top1_hit,
            "p1_top2_hit": top2_hit, "p1_top3_hit": top3_hit,
            "p1_ranked": ranked_str,
            "level": case["level"],
        })

    p1_top1_rate = agg_top1 / total
    p1_top2_rate = agg_top2 / total
    p1_top3_rate = agg_top3 / total

    fallback_count = p0_results["fallbacks"]
    miss_count = p0_results["misses"]
    p1_fallback_rescue_rate = p1_rescue_fallback / fallback_count if fallback_count > 0 else 0
    p1_miss_correct_rate = p1_correct_miss / miss_count if miss_count > 0 else 0

    # ── 打印 ──
    print(f"\n{'='*60}")
    print(f"P1 语义匹配 → 意图排名")
    print(f"{'='*60}")
    print(f"  P0 已解决:       {p0_results['hits']}/{total} ({p0_results['hits']/total:.0%})  ← 不需 P1")
    print(f"  ──────────────────────────────────────────────")
    print(f"  P0+P1 Top-1:     {agg_top1}/{total} ({p1_top1_rate:.0%})  △+{agg_top1 - p0_results['hits']}")
    print(f"  P0+P1 Top-2:     {agg_top2}/{total} ({p1_top2_rate:.0%})  △+{agg_top2 - p0_results['hits']}")
    print(f"  P0+P1 Top-3:     {agg_top3}/{total} ({p1_top3_rate:.0%})  △+{agg_top3 - p0_results['hits']}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  P1 救回兜底:     {p1_rescue_fallback}/{fallback_count} ({p1_fallback_rescue_rate:.0%})  ← P0 无关键词 → P1 语义补回")
    print(f"  P1 纠正判错:     {p1_correct_miss}/{miss_count} ({p1_miss_correct_rate:.0%})  ← P0 关键词判错 → P1 语义纠正")

    # 失败分析
    fails = [pc for pc in per_case if not pc["p1_top1_hit"]]
    if fails:
        print(f"\n  P1 Top-1 仍失败 ({len(fails)} 条):")
        for f in fails[:12]:
            r = ", ".join(f"{n}={s:.3f}" for n, s in f["p1_ranked"][:3])
            print(f"    [{f['idx']:>2}] [{f['level']:<9}] \"{f['message'][:35]}\"  "
                  f"p0={f['p0_result']} top1={f['p1_top1']} want={f['gt_intent']} "
                  f"ranked=[{r}]")

    return {
        "stage": "P1", "total": total,
        "p1_top1_hits": agg_top1, "p1_top1_rate": round(p1_top1_rate, 4),
        "p1_top2_hits": agg_top2, "p1_top2_rate": round(p1_top2_rate, 4),
        "p1_top3_hits": agg_top3, "p1_top3_rate": round(p1_top3_rate, 4),
        "p1_rescue_fallback": p1_rescue_fallback,
        "p1_correct_miss": p1_correct_miss,
        "p1_fallback_rescue_rate": round(p1_fallback_rescue_rate, 4),
        "p1_miss_correct_rate": round(p1_miss_correct_rate, 4),
        "per_case": per_case,
    }


# ================================================================
# P2: LLM 意图确认
# ================================================================

P2_NOTE = """
P2 LLM 意图确认说明:
  已有 benchmark_local_model_comparison.py 任务 B 实测数据:
  Qwen2.5-1.5B-Instruct on GPU: Top1=92%, p50=94ms (50样本预筛选候选集)

  注意: 旧 task B 的候选集是人工预制的（保证正确答案在内），与 P0+P1
        实际产出的候选集不同。P2 局部精度 92% 可作为上限参考。

  本阶段从 P1 的 Top-2/3 候选中确认最终意图。
  
  如需运行 P2: --model ./models/Qwen2.5-1.5B-Instruct --device cuda
"""


def run_p2_intent(samples: List[Dict[str, str]], p1_results: Dict[str, Any],
                  model_path: Optional[str] = None, device: str = "cpu") -> Dict[str, Any]:
    """P2 LLM 意图确认。从 P1 Top-2/3 候选中由 LLM 选出最终意图。"""

    if model_path is None:
        print(P2_NOTE)
        return {"stage": "P2", "status": "skipped",
                "note": "P2 requires GPU. Use --model to specify model path."}

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

    total = len(samples)
    p0_hits = sum(1 for pc in p1_results["per_case"] if pc["p0_result"] == "hit")
    p1_pc = p1_results["per_case"]
    hits, latencies = 0, []
    p2_rescued, p2_corrected = 0, 0
    per_case = []

    for i, case in enumerate(samples):
        msg = case["message"]
        gt = case["intent"]
        pc = p1_pc[i]
        ranked = pc.get("p1_ranked", [])
        p0_result = pc["p0_result"]

        # 候选：P1 Top-3 中的意图
        candidates = [n for n, _ in ranked[:3]]
        if gt not in candidates and len(candidates) < 3:
            candidates.append(gt)  # 兜底：确保正确答案一定在候选里（仅评测用）
        candidates = list(dict.fromkeys(candidates))[:3]

        if len(candidates) <= 1:
            is_hit = candidates[0] == gt if candidates else False
            if is_hit:
                hits += 1
            per_case.append({"idx": i + 1, "message": msg, "gt_intent": gt,
                             "candidates": candidates, "p2_hit": is_hit,
                             "p0_result": p0_result, "latency_ms": 0})
            continue

        # 构造 prompt
        cand_desc = "\n".join(
            f"- {n}: {INTENT_DESCRIPTIONS.get(n, '')}" for n in candidates
        )
        system = (
            "你是一个电商客服意图识别器。根据用户消息，从候选意图中选择"
            "最匹配的一个意图。只输出意图名称，不要解释。"
        )
        user = f"候选意图:\n{cand_desc}\n\n用户消息: {msg}\n\n请选出最匹配的意图（只输出意图名）:"
        messages = [{"role": "system", "content": system},
                     {"role": "user", "content": user}]

        t1 = time.monotonic()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        elapsed_ms = (time.monotonic() - t1) * 1000
        latencies.append(elapsed_ms)

        # 解析：从生成结果中提取第一个匹配的意图名
        selected = None
        clean = generated.strip().splitlines()[0].strip().lstrip("-* 0123456789.、，").strip('\'"`,，:')
        for cand in candidates:
            if cand in clean:
                selected = cand
                break

        is_hit = selected == gt
        if is_hit:
            hits += 1
            if p0_result == "fallback":
                p2_rescued += 1
            elif p0_result == "miss":
                p2_corrected += 1

        per_case.append({
            "idx": i + 1, "message": msg, "gt_intent": gt,
            "candidates": candidates, "selected": selected,
            "p2_hit": is_hit, "p0_result": p0_result, "latency_ms": round(elapsed_ms, 1),
        })

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{total}] p2_acc={hits/(i+1):.1%}", flush=True)

    hit_rate = hits / total
    p50 = statistics.median(latencies) if latencies else 0
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else 0

    print(f"\n{'='*60}")
    print(f"P2 LLM 意图确认")
    print(f"{'='*60}")
    print(f"  P0 已解决:       {p0_hits}/{total} ({p0_hits/total:.0%})  ← 不需 P1/P2")
    print(f"  P2 最终命中:     {hits}/{total} ({hit_rate:.0%})  △+{hits - p0_hits}")
    print(f"  P2 救回兜底:     {p2_rescued} 条  ← P0 fallback + P1→P2 链路")
    print(f"  P2 纠正判错:     {p2_corrected} 条  ← P0 miss + P1→P2 链路")
    print(f"  p50={p50:.0f}ms  p99={p99:.0f}ms")

    return {
        "stage": "P2", "total": total,
        "hits": hits, "hit_rate": round(hit_rate, 4),
        "p2_rescued": p2_rescued, "p2_corrected": p2_corrected,
        "p50_latency_ms": round(p50, 1),
        "p99_latency_ms": round(p99, 1),
        "per_case": per_case,
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="意图识别 P0→P1→P2 分层命中率 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=P2_NOTE,
    )
    parser.add_argument("--stage", choices=["p0", "p0p1", "all"], default="p0p1",
                        help="评测阶段 (默认 p0p1)")
    parser.add_argument("--p1-model", type=str, default="./models/BAAI/bge-small-zh-v1.5",
                        help="P1 Embedding 模型路径")
    parser.add_argument("--model", type=str, default=None,
                        help="P2 本地 LLM 路径")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="P2 推理设备")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    parser.add_argument("--output", type=str, default=None, help="JSON 输出文件路径")
    args = parser.parse_args()

    samples = INTENT_TEST_CASES

    if not args.json:
        print(f"意图识别 P0→P1→P2 Benchmark: {len(samples)} 条样本, stage={args.stage}")

    # P0
    p0_results = run_p0_intent(samples)
    summary: Dict[str, Any] = {
        "p0": {
            "total": p0_results["total"],
            "keyword_coverage": p0_results["keyword_coverage"],
            "intent_accuracy": p0_results["intent_accuracy"],
            "hits": p0_results["hits"],
            "misses": p0_results["misses"],
            "fallbacks": p0_results["fallbacks"],
            "p0_only_hit_rate": p0_results["p0_only_hit_rate"],
        }
    }

    # P0+P1
    p1_results = None
    if args.stage in ("p0p1", "all"):
        try:
            p1_results = run_p1_intent(samples, p0_results, args.p1_model)
            summary["p1"] = {
                "top1_hit_rate": p1_results["p1_top1_rate"],
                "top1_hits": p1_results["p1_top1_hits"],
                "top2_hit_rate": p1_results["p1_top2_rate"],
                "top2_hits": p1_results["p1_top2_hits"],
                "top3_hit_rate": p1_results["p1_top3_rate"],
                "top3_hits": p1_results["p1_top3_hits"],
                "rescue_fallback": p1_results["p1_rescue_fallback"],
                "correct_miss": p1_results["p1_correct_miss"],
            }
        except (ImportError, Exception) as e:
            print(f"\n  P1 skipped: {e}")
            summary["p1"] = {"status": "skipped", "reason": str(e)}

    # P0+P1+P2
    if args.stage == "all":
        if args.model and p1_results:
            p2_results = run_p2_intent(samples, p1_results, args.model, args.device)
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

    summary["config"] = {"samples": len(samples), "intents": len(INTENT_REGISTRY),
                         "stage": args.stage}

    # ── 综合汇总 ──
    if not args.json:
        print(f"\n{'='*60}")
        print(f"综合汇总：P0 → P1 → P2 意图识别流水线")
        print(f"{'='*60}")
        p0 = summary["p0"]
        print(f"  P0 关键词:    {p0['keyword_coverage']:.0%} 触发 → "
              f"{p0['intent_accuracy']:.0%} 准确 → "
              f"有效解决 {p0['hits']}/{p0['total']} ({p0['p0_only_hit_rate']:.0%})")

        if p1_results:
            p1 = summary["p1"]
            print(f"  P0+P1 Top-1:  {p1['top1_hit_rate']:.0%}  (+{p1['top1_hits'] - p0['hits']} 条增量)")
            print(f"  P0+P1 Top-2:  {p1['top2_hit_rate']:.0%}  (+{p1['top2_hits'] - p0['hits']} 条增量)")
            print(f"  P0+P1 Top-3:  {p1['top3_hit_rate']:.0%}  (+{p1['top3_hits'] - p0['hits']} 条增量)")

        if "hit_rate" in summary.get("p2", {}) and isinstance(summary["p2"].get("hit_rate"), float):
            p2 = summary["p2"]
            print(f"  P0+P1+P2:     {p2['hit_rate']:.0%}  (+{p2['hits'] - p0['hits']} 条增量)")
        elif p1_results:
            # 估计 P2 贡献
            top3_rate = p1_results["p1_top3_rate"]
            remain = 1.0 - top3_rate
            est_p2 = top3_rate + remain * 0.92  # 假设 P2 能救回 92% 剩余错误
            print(f"  估算 P0+P1+P2: ~{est_p2:.0%}  (P1 Top-3 + P2 92% 救回剩余 {remain:.1%})")

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

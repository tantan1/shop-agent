"""
Benchmark: DashScope Qwen Function Calling 延迟 & 准确率
=========================================================
测试 qwen-turbo / qwen-plus 在工具选择（意图识别）任务上的表现，
用真实数据替换 Blog 大纲中的估算数字。

输出指标：
  - 延迟：p50, p95, p99, avg, min, max
  - 准确率：Top-1 命中率、Top-2 命中率
  - token 消耗统计

用法：
  python scripts/benchmark_dashscope_function_calling.py

成本估算（免费额度够用）：
  - qwen-turbo: ¥0.3/百万 input + ¥0.6/百万 output
  - 50 条 query × 10 repeats × ~2100 tokens ≈ 105 万 tokens ≈ ¥0.3
  - 新用户 DashScope 免费赠送 100 万 tokens，基本覆盖
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

# ── 项目根目录 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 加载 .env ──────────────────────────────────────────────────
def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()

API_KEY = os.getenv("TONGYI_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REPEATS = 10  # 每条 query 重复次数

if not API_KEY:
    print("❌ 未找到 TONGYI_API_KEY，请在 .env 中配置")
    exit(1)


# ══════════════════════════════════════════════════════════════════
# 工具定义 (OpenAI Function Calling 格式)
# 对应 Skills 目录下的 5 个核心工具
# ══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query-order",
            "description": "查询用户的订单列表或指定订单详情。支持按订单号、手机号、状态筛选。触发条件：用户询问订单状态、订单号、我的订单、买了什么。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，如 WB202405270001"},
                    "phone": {"type": "string", "description": "手机号后四位"},
                    "status_filter": {
                        "type": "string",
                        "enum": ["待付款", "已发货", "派送中", "已签收"],
                        "description": "按订单状态筛选",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check-shipping",
            "description": "查询物流配送进度。返回揽收→运输→派送每一步的时间线和当前位置。触发条件：用户问到哪了、物流、快递、什么时候送到。",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {"type": "string", "description": "快递单号，如 SF1234567890"},
                    "order_id": {"type": "string", "description": "订单号，用于反查物流单号"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request-return",
            "description": "为用户提交退货退款申请，生成退款明细并等待人工审批。触发条件：用户明确表达退货/退款/我要退/不想要了。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "要退款的订单号"},
                    "reason": {
                        "type": "string",
                        "enum": ["质量问题", "与描述不符", "发错货", "不想要", "其他"],
                        "description": "退货原因",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check-balance",
            "description": "查询账户余额和可用积分。触发条件：用户问余额、钱包、有多少钱、积分。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "coupon-inquiry",
            "description": "查询可用的优惠券列表，包括有效期、使用门槛和适用范围。触发条件：用户问优惠券、代金券、满减券、折扣券、有什么券。",
            "parameters": {
                "type": "object",
                "properties": {
                    "coupon_type": {
                        "type": "string",
                        "enum": ["满减券", "折扣券", "运费券", "全部"],
                        "description": "优惠券类型筛选",
                    },
                },
            },
        },
    },
]

# ══════════════════════════════════════════════════════════════════
# 测试数据: 50 条标注 query（复用 benchmark_local_model_comparison.py 的数据）
# ══════════════════════════════════════════════════════════════════

TOOL_SELECTION_TESTS: List[dict] = [
    # ── query-order (10 条) ──
    {"message": "我的订单到哪了", "correct": "query-order"},
    {"message": "帮我查下最近买了什么", "correct": "query-order"},
    {"message": "订单号 WB202405270001 什么状态", "correct": "query-order"},
    {"message": "看看有没有新订单", "correct": "query-order"},
    {"message": "已发货的单子列一下", "correct": "query-order"},
    {"message": "上周买的东西到哪了", "correct": "query-order"},
    {"message": "待付款的订单还有哪些", "correct": "query-order"},
    {"message": "我的历史订单帮我查查", "correct": "query-order"},
    {"message": "看看订单详情，手机号后四位6688", "correct": "query-order"},
    {"message": "我买的东西发货没", "correct": "query-order"},
    # ── check-shipping (10 条) ──
    {"message": "我的东西到哪了", "correct": "check-shipping"},
    {"message": "快递什么时候能到", "correct": "check-shipping"},
    {"message": "SF1234567890 帮我查下物流", "correct": "check-shipping"},
    {"message": "物流状态怎么三天没更新了", "correct": "check-shipping"},
    {"message": "YT123456 到哪了", "correct": "check-shipping"},
    {"message": "EMS1098765432 派送了吗", "correct": "check-shipping"},
    {"message": "快递现在在哪个中转站", "correct": "check-shipping"},
    {"message": "JD001234567 什么时候派送", "correct": "check-shipping"},
    {"message": "帮我跟踪一下中通 7512345678901", "correct": "check-shipping"},
    {"message": "这个物流怎么一直没动", "correct": "check-shipping"},
    # ── request-return (10 条) ──
    {"message": "我要退货，质量太差了", "correct": "request-return"},
    {"message": "申请退款，不想要了", "correct": "request-return"},
    {"message": "收到的货和描述不符，我要退", "correct": "request-return"},
    {"message": "发错了商品，帮我退货", "correct": "request-return"},
    {"message": "这个订单我想退掉", "correct": "request-return"},
    {"message": "质量有瑕疵，申请退货退款", "correct": "request-return"},
    {"message": "能退款吗，用了一个月就坏了", "correct": "request-return"},
    {"message": "退货地址是什么，我要寄回去", "correct": "request-return"},
    {"message": "收到空包裹，要求退款", "correct": "request-return"},
    {"message": "尺寸不合适，想换货或者退货", "correct": "request-return"},
    # ── check-balance (10 条) ──
    {"message": "我账户里还有多少钱", "correct": "check-balance"},
    {"message": "查一下我的积分有多少", "correct": "check-balance"},
    {"message": "余额和积分分别是多少", "correct": "check-balance"},
    {"message": "钱包余额麻烦查一下", "correct": "check-balance"},
    {"message": "我的积分够不够兑换", "correct": "check-balance"},
    {"message": "看看账户余额", "correct": "check-balance"},
    {"message": "还有多少积分可以用", "correct": "check-balance"},
    {"message": "余额不足的话怎么充值", "correct": "check-balance"},
    {"message": "积分要过期了，有多少", "correct": "check-balance"},
    {"message": "帮我看看钱包", "correct": "check-balance"},
    # ── coupon-inquiry (10 条) ──
    {"message": "有没有满减券可以用", "correct": "coupon-inquiry"},
    {"message": "我的优惠券有哪些", "correct": "coupon-inquiry"},
    {"message": "运费券还有吗", "correct": "coupon-inquiry"},
    {"message": "看看有什么优惠券", "correct": "coupon-inquiry"},
    {"message": "有没有满100减15的券", "correct": "coupon-inquiry"},
    {"message": "折扣券快到期了没", "correct": "coupon-inquiry"},
    {"message": "新用户有什么优惠", "correct": "coupon-inquiry"},
    {"message": "上次领的券还能用吗", "correct": "coupon-inquiry"},
    {"message": "有什么活动可以领券", "correct": "coupon-inquiry"},
    {"message": "会员专享券在哪领", "correct": "coupon-inquiry"},
]


# ══════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════

@dataclass
class SingleResult:
    """单次请求结果"""
    model: str
    query: str
    correct_tool: str
    predicted_tool: str = ""
    predicted_tools: List[str] = field(default_factory=list)  # Top-N
    hit: bool = False
    hit_top2: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


@dataclass
class ModelSummary:
    """单模型汇总"""
    model: str
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    top1_hits: int = 0
    top2_hits: int = 0
    top1_accuracy: float = 0.0
    top2_accuracy: float = 0.0
    latencies: List[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # 按意图分组的准确率
    per_intent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    details: List[dict] = field(default_factory=list)

    def compute(self, results: List[SingleResult]):
        self.total_requests = len(results)
        self.success_count = sum(1 for r in results if not r.error)
        self.error_count = self.total_requests - self.success_count

        valid = [r for r in results if not r.error]
        self.top1_hits = sum(1 for r in valid if r.hit)
        self.top2_hits = sum(1 for r in valid if r.hit_top2)
        self.top1_accuracy = self.top1_hits / len(valid) if valid else 0.0
        self.top2_accuracy = self.top2_hits / len(valid) if valid else 0.0

        self.latencies = [r.latency_ms for r in valid]
        if self.latencies:
            sorted_lat = sorted(self.latencies)
            self.p50_ms = sorted_lat[int(len(sorted_lat) * 0.5)]
            self.p95_ms = sorted_lat[int(len(sorted_lat) * 0.95)]
            self.p99_ms = sorted_lat[int(len(sorted_lat) * 0.99)]
            self.avg_ms = statistics.mean(self.latencies)
            self.min_ms = min(self.latencies)
            self.max_ms = max(self.latencies)

        self.total_input_tokens = sum(r.input_tokens for r in valid)
        self.total_output_tokens = sum(r.output_tokens for r in valid)

        # 按意图分组统计
        intent_groups: Dict[str, List[SingleResult]] = {}
        for r in valid:
            intent_groups.setdefault(r.correct_tool, []).append(r)
        for intent, items in intent_groups.items():
            hits = sum(1 for r in items if r.hit)
            self.per_intent[intent] = {
                "count": len(items),
                "hits": hits,
                "accuracy": hits / len(items),
                "avg_latency_ms": statistics.mean(r.latency_ms for r in items),
            }

        self.details = [asdict(r) for r in results]


# ══════════════════════════════════════════════════════════════════
# 核心 benchmark 逻辑
# ══════════════════════════════════════════════════════════════════

async def benchmark_model(
    model_name: str,
    test_cases: List[dict],
    repeats: int = REPEATS,
) -> List[SingleResult]:
    """对单个模型执行 function calling benchmark"""

    print(f"\n{'='*60}")
    print(f"  {model_name} — Function Calling Benchmark")
    print(f"  {len(test_cases)} 条 query × {repeats} repeats = {len(test_cases) * repeats} 次请求")
    print(f"{'='*60}")

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)
    # ★ No-Think: AsyncOpenAI 构造器不支持 extra_body，通过 monkey-patch 注入
    _original_create = client.chat.completions.create

    async def _no_think_create(**kwargs):
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {"enable_thinking": False}
        return await _original_create(**kwargs)

    client.chat.completions.create = _no_think_create
    results: List[SingleResult] = []
    total = len(test_cases) * repeats

    for idx, case in enumerate(test_cases):
        query = case["message"]
        correct = case["correct"]

        for rep in range(repeats):
            n = idx * repeats + rep + 1
            result = SingleResult(model=model_name, query=query, correct_tool=correct)

            try:
                t0 = time.perf_counter()
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是一个电商客服系统的意图识别模块。"
                                "根据用户输入，选择最合适的工具（function）。"
                                "只选一个最匹配的工具，不要选多个。"
                                "如果不确定，选最接近的那个。"
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.0,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                result.latency_ms = round(elapsed, 1)

                # 提取工具选择结果
                msg = response.choices[0].message
                if msg.tool_calls and len(msg.tool_calls) > 0:
                    result.predicted_tool = msg.tool_calls[0].function.name
                    result.predicted_tools = [
                        tc.function.name for tc in msg.tool_calls
                    ]
                else:
                    # LLM 没有调用工具（如当成普通对话回答了）
                    result.predicted_tool = "(无工具调用)"
                    result.predicted_tools = []

                result.hit = (result.predicted_tool == correct)
                result.hit_top2 = (correct in result.predicted_tools[:2])

                # Token 统计
                if response.usage:
                    result.input_tokens = response.usage.prompt_tokens or 0
                    result.output_tokens = response.usage.completion_tokens or 0

                status = "✅" if result.hit else "❌"
                print(
                    f"  [{n:4d}/{total}] {status} {result.latency_ms:6.0f}ms "
                    f"| pred={result.predicted_tool:16s} | true={correct:16s} "
                    f"| {query[:30]}..."
                )

            except Exception as e:
                result.error = str(e)[:200]
                print(f"  [{n:4d}/{total}] ⚠️ ERROR: {result.error[:80]}")

            results.append(result)

            # 小间隔避免触发限流
            if n % 20 == 0:
                await asyncio.sleep(0.1)

    return results


# ══════════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════════

def print_summary(s: ModelSummary, model_name: str):
    """打印模型汇总"""
    cost_input = s.total_input_tokens / 1_000_000
    cost_output = s.total_output_tokens / 1_000_000
    # qwen-turbo 价格
    prices = {
        "qwen-turbo": (0.3, 0.6),
        "qwen-plus": (0.8, 2.0),
        "qwen-max": (2.4, 9.6),
    }
    pi, po = prices.get(model_name, (0.3, 0.6))
    est_cost = cost_input * pi + cost_output * po

    print(f"\n{'─'*60}")
    print(f"  {model_name} 汇总")
    print(f"{'─'*60}")
    print(f"  请求总数:        {s.total_requests}")
    print(f"  成功:            {s.success_count}")
    print(f"  失败:            {s.error_count}")
    print(f"")
    print(f"  Top-1 准确率:    {s.top1_accuracy:.1%} ({s.top1_hits}/{s.success_count})")
    print(f"  Top-2 准确率:    {s.top2_accuracy:.1%} ({s.top2_hits}/{s.success_count})")
    print(f"")
    print(f"  延迟 p50:        {s.p50_ms:.0f} ms")
    print(f"  延迟 p95:        {s.p95_ms:.0f} ms")
    print(f"  延迟 p99:        {s.p99_ms:.0f} ms")
    print(f"  延迟 avg:        {s.avg_ms:.0f} ms")
    print(f"  延迟范围:        {s.min_ms:.0f} ~ {s.max_ms:.0f} ms")
    print(f"")
    print(f"  Token 消耗:      {s.total_input_tokens:,} input + {s.total_output_tokens:,} output")
    print(f"  估算成本:        ¥{est_cost:.4f}")
    print(f"")
    print(f"  按意图准确率:")
    for intent, info in sorted(s.per_intent.items()):
        bar = "█" * int(info["accuracy"] * 20) + "░" * (20 - int(info["accuracy"] * 20))
        print(
            f"    {intent:18s}  {info['accuracy']:.0%} {bar}  "
            f"({info['hits']}/{info['count']})  avg {info['avg_latency_ms']:.0f}ms"
        )


def _serialize(obj):
    """递归处理不可序列化的类型"""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return str(obj)


async def main():
    # 模型列表：从省到贵
    models = ["qwen-turbo"]
    # 如果额度充足，可以加 qwen-plus 对比
    # models = ["qwen-turbo", "qwen-plus"]

    all_summaries = []

    for model_name in models:
        results = await benchmark_model(model_name, TOOL_SELECTION_TESTS)
        summary = ModelSummary(model=model_name)
        summary.compute(results)
        all_summaries.append(summary)
        print_summary(summary, model_name)

    # ── 对比表 ──
    if len(all_summaries) >= 2:
        print(f"\n{'='*60}")
        print(f"  模型对比")
        print(f"{'='*60}")
        print(f"  {'模型':16s} {'Top1':>6s} {'Top2':>6s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'errors':>6s}")
        print(f"  {'─'*16} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
        for s in all_summaries:
            print(
                f"  {s.model:16s} {s.top1_accuracy:5.0%}  {s.top2_accuracy:5.0%}  "
                f"{s.p50_ms:4.0f}ms {s.p95_ms:4.0f}ms {s.p99_ms:4.0f}ms {s.error_count:6d}"
            )

    # ── 保存结果 ──
    output_path = PROJECT_ROOT / "scripts" / "dashscope_fc_benchmark_results.json"
    output_data = {
        "benchmark": "dashscope_function_calling",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_config": {
            "num_queries": len(TOOL_SELECTION_TESTS),
            "repeats_per_query": REPEATS,
            "tools_count": len(TOOLS),
        },
        "models": {},
    }
    for s in all_summaries:
        output_data["models"][s.model] = {
            "top1_accuracy": round(s.top1_accuracy, 4),
            "top2_accuracy": round(s.top2_accuracy, 4),
            "p50_ms": round(s.p50_ms, 1),
            "p95_ms": round(s.p95_ms, 1),
            "p99_ms": round(s.p99_ms, 1),
            "avg_ms": round(s.avg_ms, 1),
            "min_ms": round(s.min_ms, 1),
            "max_ms": round(s.max_ms, 1),
            "total_requests": s.total_requests,
            "success_count": s.success_count,
            "error_count": s.error_count,
            "total_input_tokens": s.total_input_tokens,
            "total_output_tokens": s.total_output_tokens,
            "per_intent": s.per_intent,
            "details": _serialize(s.details),
        }

    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n📁 结果已保存: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

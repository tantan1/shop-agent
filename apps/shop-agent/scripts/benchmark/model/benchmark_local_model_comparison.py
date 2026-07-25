"""
本地小模型多尺寸对比 Benchmark: 信息提取 & 工具选择
========================================================
三模型 × 两设备 × 两任务 = 12 组对比

模型: Qwen2.5-0.5B-Instruct / Qwen2.5-1.5B-Instruct / Qwen3-1.7B
设备: CPU (venv/) / GPU (venv_cuda/)
任务: A-信息提取(30条) / B-工具选择(50条)

用法:
  1. 驱动模式（在当前 venv 中运行，自动调度 CPU/GPU 测试）:
     python scripts/benchmark_local_model_comparison.py

  2. Worker 模式（由驱动通过 subprocess 调用，单独测一个模型×设备组合）:
     python scripts/benchmark_local_model_comparison.py \
       --model "Qwen/Qwen2.5-0.5B-Instruct" --device cpu --task all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import gc
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 项目根目录 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 模型列表 ────────────────────────────────────────────────────
MODELS = [
    "./models/Qwen2.5-0.5B-Instruct",
    "./models/Qwen2.5-1.5B-Instruct",
    "./models/Qwen3-1.7B",
    "./models/Qwen2.5-3B-Instruct",
]

# ── Venv 配置 ───────────────────────────────────────────────────
VENV_MAP = {
    "cpu": PROJECT_ROOT / "venv" / "Scripts" / "python.exe",
    "cuda": PROJECT_ROOT / "venv_cuda" / "Scripts" / "python.exe",
}

# ══════════════════════════════════════════════════════════════════
# 测试数据
# ══════════════════════════════════════════════════════════════════

# ── 任务 A: 信息提取测试样本 (30 条) ────────────────────────────
# 覆盖 5 个意图场景: query-order / check-shipping / request-return /
#                  check-balance / coupon-inquiry

EXTRACTION_TESTS: List[dict] = [
    # ── query-order (8 条) ──
    {
        "intent": "query-order",
        "message": "帮我查一下订单 WB202405270001",
        "ground_truth": {"order_id": "WB202405270001"},
    },
    {
        "intent": "query-order",
        "message": "我的订单到哪了，手机尾号6688",
        "ground_truth": {"phone": "6688"},
    },
    {
        "intent": "query-order",
        "message": "看看待付款的订单有哪些",
        "ground_truth": {"status_filter": "待付款"},
    },
    {
        "intent": "query-order",
        "message": "查一下 202405250016 这个订单的物流状态",
        "ground_truth": {"order_id": "202405250016"},
    },
    {
        "intent": "query-order",
        "message": "已发货的订单帮我列一下，手机号是9901",
        "ground_truth": {"status_filter": "已发货", "phone": "9901"},
    },
    {
        "intent": "query-order",
        "message": "我有个单号 202405010088 显示派送中，什么时候能到",
        "ground_truth": {"order_id": "202405010088", "status_filter": "派送中"},
    },
    {
        "intent": "query-order",
        "message": "最近买了什么，尾号1234",
        "ground_truth": {"phone": "1234"},
    },
    {
        "intent": "query-order",
        "message": "已签收的订单有吗",
        "ground_truth": {"status_filter": "已签收"},
    },

    # ── check-shipping (6 条) ──
    {
        "intent": "check-shipping",
        "message": "快递 SF1234567890 到哪了",
        "ground_truth": {"tracking_number": "SF1234567890"},
    },
    {
        "intent": "check-shipping",
        "message": "YT123456 的物流帮我查查",
        "ground_truth": {"tracking_number": "YT123456"},
    },
    {
        "intent": "check-shipping",
        "message": "JD001234567 什么时候派送",
        "ground_truth": {"tracking_number": "JD001234567"},
    },
    {
        "intent": "check-shipping",
        "message": "订单 WB202403150066 发货了吗，物流单号是多少",
        "ground_truth": {"order_id": "WB202403150066"},
    },
    {
        "intent": "check-shipping",
        "message": "中通快递 7512345678901 帮我跟踪一下",
        "ground_truth": {"tracking_number": "7512345678901"},
    },
    {
        "intent": "check-shipping",
        "message": "我的货怎么还没派送，订单是202403030099",
        "ground_truth": {"order_id": "202403030099"},
    },

    # ── request-return (6 条) ──
    {
        "intent": "request-return",
        "message": "我要退货，订单号 WB202405050088，质量太差了",
        "ground_truth": {"order_id": "WB202405050088", "reason": "质量问题"},
    },
    {
        "intent": "request-return",
        "message": "申请退款 202405200012 这个订单，不想要了",
        "ground_truth": {"order_id": "202405200012", "reason": "不想要"},
    },
    {
        "intent": "request-return",
        "message": "收到的货和描述不符，单号 WB202404110033 申请退款",
        "ground_truth": {"order_id": "WB202404110033", "reason": "与描述不符"},
    },
    {
        "intent": "request-return",
        "message": "发错了商品，202405180555 帮我退货",
        "ground_truth": {"order_id": "202405180555", "reason": "发错货"},
    },
    {
        "intent": "request-return",
        "message": "质量有瑕疵，WB202406010012 我要退货",
        "ground_truth": {"order_id": "WB202406010012", "reason": "质量问题"},
    },
    {
        "intent": "request-return",
        "message": "这个订单 202405220678 我想退掉",
        "ground_truth": {"order_id": "202405220678"},
    },

    # ── check-balance (5 条) ──
    {
        "intent": "check-balance",
        "message": "我账户里还有多少钱",
        "ground_truth": {},
    },
    {
        "intent": "check-balance",
        "message": "查一下我的积分有多少",
        "ground_truth": {},
    },
    {
        "intent": "check-balance",
        "message": "余额和积分分别是多少",
        "ground_truth": {},
    },
    {
        "intent": "check-balance",
        "message": "钱包余额麻烦查一下",
        "ground_truth": {},
    },
    {
        "intent": "check-balance",
        "message": "我的积分够不够兑换",
        "ground_truth": {},
    },

    # ── coupon-inquiry (5 条) ──
    {
        "intent": "coupon-inquiry",
        "message": "有没有满减券可以用",
        "ground_truth": {"coupon_type": "满减券"},
    },
    {
        "intent": "coupon-inquiry",
        "message": "我的优惠券有哪些，有没有折扣券",
        "ground_truth": {"coupon_type": "折扣券"},
    },
    {
        "intent": "coupon-inquiry",
        "message": "运费券还有吗",
        "ground_truth": {"coupon_type": "运费券"},
    },
    {
        "intent": "coupon-inquiry",
        "message": "看看有什么优惠券",
        "ground_truth": {},
    },
    {
        "intent": "coupon-inquiry",
        "message": "有没有满100减15的券",
        "ground_truth": {"coupon_type": "满减券"},
    },
]

# ── 任务 B: 工具选择测试样本 (50 条) ─────────────────────────────
# 每条样本: message + candidate_tools (P0+P1 过滤后的 3-5 个) + correct_tool

# 工具描述(与 skill SKILL.md 一致)
TOOL_DESCRIPTIONS = {
    "query-order": "查询用户的订单列表或指定订单详情。触发条件：用户询问订单状态、订单号、我的订单。",
    "check-shipping": "查询物流配送进度。返回揽收→运输→派送每一步的时间线和状态。触发条件：用户问到哪了/物流/快递/什么时候送到。",
    "request-return": "为用户提交退货退款申请。触发条件：用户明确表达退货/退款/我要退。",
    "check-balance": "查询账户余额和可用积分。触发条件：用户问余额/钱包/有多少钱/积分。",
    "coupon-inquiry": "查询可用的优惠券列表，包括有效期、使用门槛。触发条件：用户问优惠券/代金券/满减/有什么券。",
}

TOOL_SELECTION_TESTS: List[dict] = [
    # ── query-order (10 条) ──
    {
        "message": "我的订单到哪了",
        "candidates": ["query-order", "check-shipping"],
        "correct": "query-order",
    },
    {
        "message": "帮我查下最近买了什么",
        "candidates": ["query-order", "coupon-inquiry", "check-balance"],
        "correct": "query-order",
    },
    {
        "message": "订单号 WB202405270001 什么状态",
        "candidates": ["query-order", "check-shipping", "request-return"],
        "correct": "query-order",
    },
    {
        "message": "看看有没有新订单",
        "candidates": ["query-order", "coupon-inquiry"],
        "correct": "query-order",
    },
    {
        "message": "已发货的单子列一下",
        "candidates": ["query-order", "check-shipping", "request-return"],
        "correct": "query-order",
    },
    {
        "message": "上周买的东西到哪了",
        "candidates": ["query-order", "check-shipping"],
        "correct": "query-order",
    },
    {
        "message": "待付款的订单还有哪些",
        "candidates": ["query-order", "check-balance", "coupon-inquiry"],
        "correct": "query-order",
    },
    {
        "message": "我的历史订单帮我查查",
        "candidates": ["query-order", "coupon-inquiry", "request-return"],
        "correct": "query-order",
    },
    {
        "message": "看看订单详情，手机号后四位6688",
        "candidates": ["query-order", "check-balance", "check-shipping"],
        "correct": "query-order",
    },
    {
        "message": "我买的东西发货没",
        "candidates": ["query-order", "check-shipping", "request-return"],
        "correct": "query-order",
    },

    # ── check-shipping (10 条) ──
    {
        "message": "我的东西到哪了",
        "candidates": ["check-shipping", "query-order"],
        "correct": "check-shipping",
    },
    {
        "message": "快递什么时候能到",
        "candidates": ["check-shipping", "query-order", "request-return"],
        "correct": "check-shipping",
    },
    {
        "message": "SF1234567890 物流跟踪一下",
        "candidates": ["check-shipping", "query-order", "coupon-inquiry"],
        "correct": "check-shipping",
    },
    {
        "message": "物流信息帮我查查 YT123456",
        "candidates": ["check-shipping", "query-order"],
        "correct": "check-shipping",
    },
    {
        "message": "怎么还没送到，都三天了",
        "candidates": ["check-shipping", "query-order", "request-return"],
        "correct": "check-shipping",
    },
    {
        "message": "派送到哪一步了",
        "candidates": ["check-shipping", "query-order"],
        "correct": "check-shipping",
    },
    {
        "message": "中通快递单号 7512345678901 查下",
        "candidates": ["check-shipping", "query-order", "coupon-inquiry"],
        "correct": "check-shipping",
    },
    {
        "message": "快递有更新吗",
        "candidates": ["check-shipping", "query-order"],
        "correct": "check-shipping",
    },
    {
        "message": "帮我看看快递进度",
        "candidates": ["check-shipping", "query-order", "request-return"],
        "correct": "check-shipping",
    },
    {
        "message": "揽收了没有，JD001234567",
        "candidates": ["check-shipping", "query-order"],
        "correct": "check-shipping",
    },

    # ── request-return (10 条) ──
    {
        "message": "我要退货，订单号 WB202405050088",
        "candidates": ["request-return", "query-order", "check-shipping"],
        "correct": "request-return",
    },
    {
        "message": "这个不想要了，帮我退掉",
        "candidates": ["request-return", "query-order"],
        "correct": "request-return",
    },
    {
        "message": "质量有问题，申请退款",
        "candidates": ["request-return", "query-order", "check-balance"],
        "correct": "request-return",
    },
    {
        "message": "收到的货跟图片不一样，我要退款",
        "candidates": ["request-return", "query-order", "coupon-inquiry"],
        "correct": "request-return",
    },
    {
        "message": "发错货了怎么退",
        "candidates": ["request-return", "query-order", "check-shipping"],
        "correct": "request-return",
    },
    {
        "message": "退货申请，单号 202405220678",
        "candidates": ["request-return", "query-order", "check-shipping"],
        "correct": "request-return",
    },
    {
        "message": "退款什么时候到账，我已经申请了",
        "candidates": ["request-return", "check-balance", "query-order"],
        "correct": "request-return",
    },
    {
        "message": "大小不合适能退吗",
        "candidates": ["request-return", "query-order"],
        "correct": "request-return",
    },
    {
        "message": "商品有瑕疵想退货",
        "candidates": ["request-return", "check-shipping", "query-order"],
        "correct": "request-return",
    },
    {
        "message": "申请退款，都不想要了",
        "candidates": ["request-return", "coupon-inquiry", "query-order"],
        "correct": "request-return",
    },

    # ── check-balance (10 条) ──
    {
        "message": "账户余额多少",
        "candidates": ["check-balance", "coupon-inquiry", "query-order"],
        "correct": "check-balance",
    },
    {
        "message": "我还有多少钱在钱包里",
        "candidates": ["check-balance", "check-shipping"],
        "correct": "check-balance",
    },
    {
        "message": "积分有多少了",
        "candidates": ["check-balance", "coupon-inquiry", "query-order"],
        "correct": "check-balance",
    },
    {
        "message": "查下余额",
        "candidates": ["check-balance", "query-order"],
        "correct": "check-balance",
    },
    {
        "message": "我的积分够不够换个优惠券",
        "candidates": ["check-balance", "coupon-inquiry"],
        "correct": "check-balance",
    },
    {
        "message": "看看钱包还有多少",
        "candidates": ["check-balance", "coupon-inquiry", "request-return"],
        "correct": "check-balance",
    },
    {
        "message": "账户里余额和积分分别多少",
        "candidates": ["check-balance", "query-order", "coupon-inquiry"],
        "correct": "check-balance",
    },
    {
        "message": "积分明细帮我查一下",
        "candidates": ["check-balance", "query-order", "coupon-inquiry"],
        "correct": "check-balance",
    },
    {
        "message": "还剩多少钱",
        "candidates": ["check-balance", "query-order"],
        "correct": "check-balance",
    },
    {
        "message": "积分能兑换什么，先看看我还有多少分",
        "candidates": ["check-balance", "coupon-inquiry"],
        "correct": "check-balance",
    },

    # ── coupon-inquiry (10 条) ──
    {
        "message": "有什么优惠券可以用",
        "candidates": ["coupon-inquiry", "query-order", "check-balance"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "满减券还有吗",
        "candidates": ["coupon-inquiry", "check-balance", "request-return"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "看看我的代金券",
        "candidates": ["coupon-inquiry", "check-balance"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "有没有满100减15的券",
        "candidates": ["coupon-inquiry", "query-order"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "优惠券要过期了提醒我一下",
        "candidates": ["coupon-inquiry", "query-order", "check-shipping"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "运费券有没有",
        "candidates": ["coupon-inquiry", "check-shipping", "check-balance"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "新人优惠券在哪里领",
        "candidates": ["coupon-inquiry", "check-balance", "query-order"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "满200减30的券还能用吗",
        "candidates": ["coupon-inquiry", "query-order"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "折扣券有没有，想买大件",
        "candidates": ["coupon-inquiry", "request-return", "check-balance"],
        "correct": "coupon-inquiry",
    },
    {
        "message": "有什么活动优惠吗",
        "candidates": ["coupon-inquiry", "check-balance", "query-order"],
        "correct": "coupon-inquiry",
    },
]

# ── 参数提取的 Schema 定义（内联，不依赖项目导入） ──────────────

EXTRACTION_SCHEMAS: Dict[str, Dict[str, str]] = {
    "query-order": {
        "order_id": "string  // 订单号",
        "phone": "string  // 手机号后四位",
        "status_filter": "string  // 筛选状态: 待付款/已发货/派送中/已签收",
    },
    "check-shipping": {
        "tracking_number": "string  // 快递单号",
        "order_id": "string  // 订单号",
    },
    "request-return": {
        "order_id": "string  // 要退货的订单号",
        "reason": "string  // 退货原因: 质量问题/不想要/发错货/与描述不符/其他",
    },
    "check-balance": {},
    "coupon-inquiry": {
        "coupon_type": "string  // 券类型: 满减券/折扣券/运费券/通用",
    },
}

EXTRACTION_PROMPTS: Dict[str, str] = {
    "query-order": (
        "从用户消息中提取查询订单的参数。\n"
        "- order_id: 订单号通常是字母数字组合\n"
        "- phone: 手机号后四位\n"
        "- status_filter: 用户想看的订单状态(待付款/已发货/派送中/已签收)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "check-shipping": (
        "从用户消息中提取查询物流的参数。\n"
        "- tracking_number: 快递单号\n"
        "- order_id: 订单号\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "request-return": (
        "从用户消息中提取退货退款的参数。\n"
        "- order_id: 要退货的订单号\n"
        "- reason: 退货原因(质量问题/不想要/发错货/与描述不符/其他)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "check-balance": "用户查询账户余额或积分，当前无需额外参数。",
    "coupon-inquiry": (
        "从用户消息中提取查询优惠券的参数。\n"
        "- coupon_type: 券类型(满减券/折扣券/运费券/通用)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
}


# ══════════════════════════════════════════════════════════════════
# Worker 模式: 模型推理
# ══════════════════════════════════════════════════════════════════

# Qwen3 think 标签剥离正则
_STRIP_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    return _STRIP_THINK_RE.sub("", text).strip()


def extract_json_from_text(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON 对象（镜像 local_model_service 的逻辑）"""
    text = strip_think_tags(text)
    # 1. 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2. ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. 第一个 { ... } 对象
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    # 4. 最大括号范围
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def build_extraction_prompt(intent: str, message: str) -> list:
    """构建参数抽取 prompt，返回 messages 列表"""
    schema = EXTRACTION_SCHEMAS[intent]
    schema_lines = []
    for field_name, field_desc in schema.items():
        schema_lines.append(f'  "{field_name}": {field_desc}')
    schema_json = "{\n" + ",\n".join(schema_lines) + "\n}"

    system = (
        "你是一个参数提取助手。从用户消息中提取结构化参数。\n"
        "严格按照以下 JSON Schema 输出 JSON，不要输出任何额外的文字或 markdown 代码块标记。\n"
        "如果用户没有提到某个字段，该字段值设为 null。\n"
        "只输出纯 JSON 对象，以 { 开头，以 } 结尾。\n\n"
        "【重要】JSON Schema 中每个字段后面的 // 注释是该字段的含义说明，不是需要填入的值！"
        "请只从「用户消息」中提取真实数据，不要把注释内容当作参数值填入。"
    )
    extraction_prompt = EXTRACTION_PROMPTS[intent]
    user = (
        f"提取规则:\n{extraction_prompt}\n\n"
        f"输出 JSON Schema（字段及类型说明）:\n{schema_json}\n\n"
        f"用户消息: {message}\n\n"
        "请按 JSON Schema 输出提取结果（纯 JSON, 不要 ``` 等标记）:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_tool_selection_prompt(user_query: str, tool_names: List[str]) -> list:
    """构建工具选择 prompt，返回 messages 列表"""
    system = (
        "你是一个电商客服工具路由器。根据用户消息，从可用工具列表中选择最相关的工具。\n"
        "每个工具的 description 已包含其功能说明，请根据语义进行匹配。\n"
        "如果多个工具功能相似，选择最直接、最精准的。\n"
        "当用户同时涉及多个操作时（如订单+物流），可以同时选中。\n"
        "仅选择回答查询所直接需要的工具。"
    )
    tool_lines = "\n".join(
        f"- {n}: {TOOL_DESCRIPTIONS.get(n, '')}" for n in tool_names
    )
    user_msg = (
        f"候选工具:\n{tool_lines}\n\n"
        f"用户消息: {user_query}\n\n"
        f"请选出最相关的工具（只输出工具名，每行一个）:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def parse_tool_selection_output(raw: str, valid_names: set) -> List[str]:
    """从模型输出中解析工具名列表"""
    raw = strip_think_tags(raw)
    selected = []
    for line in raw.strip().splitlines() if raw else []:
        name = line.strip().lstrip("-* 0123456789.、，").strip()
        name = name.strip('\'"`,，:')
        if name and name in valid_names and name not in selected:
            selected.append(name)
    return selected


def generate(model, tokenizer, messages: list, max_new_tokens: int, device: str,
             enable_thinking: bool = True) -> Tuple[str, float]:
    """同步推理，使用 chat template 构建 prompt，返回 (输出文本, 推理耗时ms)"""
    import torch

    t0 = time.monotonic()
    # 构建 prompt：对 Qwen3 等 thinking 模型可通过 enable_thinking=False 关闭思考
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if not enable_thinking:
        chat_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    elapsed_ms = (time.monotonic() - t0) * 1000
    return result, elapsed_ms


# ══════════════════════════════════════════════════════════════════
# 任务 A: 信息提取评测
# ══════════════════════════════════════════════════════════════════

def eval_extraction(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> dict:
    """计算字段级精确率/召回率/F1 和值完全匹配率"""
    predict_keys = set(predicted.keys())
    truth_keys = set(ground_truth.keys())

    tp = len(predict_keys & truth_keys)
    fp = len(predict_keys - truth_keys)
    fn = len(truth_keys - predict_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if not truth_keys else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if not truth_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 值完全匹配: 所有 ground_truth 的 key 在 predicted 中值一致
    value_match = all(
        predicted.get(k) == v for k, v in ground_truth.items()
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "value_exact_match": value_match,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_task_a_extraction(
    model, tokenizer, device: str, max_new_tokens: int = 128, max_samples: int | None = None,
    enable_thinking: bool = True,
) -> dict:
    """运行任务 A: 信息提取"""
    samples = EXTRACTION_TESTS[:max_samples] if max_samples else EXTRACTION_TESTS
    latencies: List[float] = []
    f1_scores: List[float] = []
    value_matches: int = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    per_case: List[dict] = []

    total = len(samples)
    mode = "(debug, limited)" if max_samples else ""
    print(f"  [任务A] 信息提取 ({total} 条样本) {mode}...", flush=True)

    for i, case in enumerate(samples):
        print(f"    [{i+1}/{total}] intent={case['intent']} msg=\"{case['message'][:40]}...\"", flush=True)
        messages = build_extraction_prompt(case["intent"], case["message"])
        raw, elapsed_ms = generate(model, tokenizer, messages, max_new_tokens, device, enable_thinking=enable_thinking)
        print(f"      -> {elapsed_ms:.0f}ms  raw={repr(raw[:100])}", flush=True)
        latencies.append(elapsed_ms)

        data = extract_json_from_text(raw)
        if data is None:
            data = {}

        # 去掉 None 值
        predicted = {k: v for k, v in data.items() if v is not None}
        gt = case["ground_truth"]

        metrics = eval_extraction(predicted, gt)
        f1_scores.append(metrics["f1"])
        if metrics["value_exact_match"]:
            value_matches += 1
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        per_case.append({
            "idx": i + 1,
            "intent": case["intent"],
            "message": case["message"],
            "predicted": predicted,
            "ground_truth": gt,
            "latency_ms": round(elapsed_ms, 1),
            **metrics,
        })

    # 总体指标
    total_fields = total_tp + total_fp + total_fn
    macro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0 else 0.0
    )

    p50 = statistics.median(latencies) if latencies else 0
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else (latencies[0] if latencies else 0)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"    [OK] F1={macro_f1:.3f} | 值匹配率={value_matches}/{total} ({100*value_matches/total:.1f}%) | "
          f"p50={p50:.0f}ms | p99={p99:.0f}ms | avg={avg_latency:.0f}ms", flush=True)

    return {
        "total_samples": total,
        "field_precision": round(macro_precision, 4),
        "field_recall": round(macro_recall, 4),
        "field_f1": round(macro_f1, 4),
        "value_exact_match_rate": round(value_matches / total, 4),
        "value_matches": value_matches,
        "p50_latency_ms": round(p50, 1),
        "p99_latency_ms": round(p99, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "per_case": per_case,
    }


# ══════════════════════════════════════════════════════════════════
# 任务 B: 工具选择评测
# ══════════════════════════════════════════════════════════════════

def run_task_b_tool_selection(
    model, tokenizer, device: str, max_new_tokens: int = 64, max_samples: int | None = None,
    enable_thinking: bool = True,
) -> dict:
    """运行任务 B: 工具选择"""
    samples = TOOL_SELECTION_TESTS[:max_samples] if max_samples else TOOL_SELECTION_TESTS
    latencies: List[float] = []
    top1_hits = 0
    top2_hits = 0
    exact_matches = 0
    per_case: List[dict] = []
    confusion: Dict[str, Dict[str, int]] = {}  # correct → {predicted: count}

    total = len(samples)
    mode = "(debug, limited)" if max_samples else ""
    print(f"  [任务B] 工具选择 ({total} 条样本) {mode}...", flush=True)

    for i, case in enumerate(samples):
        candidate_names = list(case["candidates"])
        valid_set = set(candidate_names)
        correct = case["correct"]
        print(f"    [{i+1}/{total}] msg=\"{case['message'][:40]}...\" cands={candidate_names}", flush=True)
        messages = build_tool_selection_prompt(case["message"], candidate_names)
        raw, elapsed_ms = generate(model, tokenizer, messages, max_new_tokens, device, enable_thinking=enable_thinking)
        print(f"      -> {elapsed_ms:.0f}ms  raw={repr(raw[:100])}", flush=True)
        latencies.append(elapsed_ms)

        selected = parse_tool_selection_output(raw, valid_set)

        top1_match = selected[0] == correct if selected else False
        top2_match = correct in selected[:2] if selected else False
        exact_match = selected == [correct] if len(selected) == 1 else False

        if top1_match:
            top1_hits += 1
        if top2_match:
            top2_hits += 1
        if exact_match:
            exact_matches += 1

        # 混淆矩阵
        pred_top1 = selected[0] if selected else "(none)"
        if correct not in confusion:
            confusion[correct] = {}
        confusion[correct][pred_top1] = confusion[correct].get(pred_top1, 0) + 1

        per_case.append({
            "idx": i + 1,
            "message": case["message"],
            "candidates": candidate_names,
            "correct": correct,
            "selected": selected,
            "top1_match": top1_match,
            "top2_match": top2_match,
            "exact_match": exact_match,
            "latency_ms": round(elapsed_ms, 1),
        })

    p50 = statistics.median(latencies) if latencies else 0
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else (latencies[0] if latencies else 0)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"    [OK] Top1={top1_hits}/{total} ({100*top1_hits/total:.1f}%) | "
          f"Top2={top2_hits}/{total} ({100*top2_hits/total:.1f}%) | "
          f"Exact={exact_matches}/{total} ({100*exact_matches/total:.1f}%) | "
          f"p50={p50:.0f}ms | p99={p99:.0f}ms | avg={avg_latency:.0f}ms", flush=True)

    return {
        "total_samples": total,
        "top1_hit_rate": round(top1_hits / total, 4),
        "top2_hit_rate": round(top2_hits / total, 4),
        "exact_match_rate": round(exact_matches / total, 4),
        "top1_hits": top1_hits,
        "top2_hits": top2_hits,
        "exact_matches": exact_matches,
        "p50_latency_ms": round(p50, 1),
        "p99_latency_ms": round(p99, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "confusion_matrix": confusion,
        "per_case": per_case,
    }


# ══════════════════════════════════════════════════════════════════
# Worker 入口
# ══════════════════════════════════════════════════════════════════

def worker_main(args: argparse.Namespace):
    """Worker 模式：加载模型，执行评测，输出 JSON 结果到文件"""
    import torch

    model_id = args.model
    device = args.device
    task = args.task  # "extraction" | "tool_selection" | "all"
    output_path = Path(args.output)

    print(f"\n{'='*60}")
    print(f"Worker 启动: model={model_id}, device={device}, task={task}")
    print(f"torch version: {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n", flush=True)

    result: dict = {
        "model": model_id,
        "device": device,
        "task_a": None,
        "task_b": None,
    }

    try:
        # 加载模型
        from transformers import AutoTokenizer, AutoModelForCausalLM
        t_load_start = time.monotonic()

        print(f"  Loading tokenizer: {model_id}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs: dict = {"trust_remote_code": True}
        if device == "cpu":
            load_kwargs["dtype"] = torch.float32
        else:
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"

        print(f"  Loading model: {model_id} (dtype={load_kwargs.get('dtype')})...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()

        t_load_ms = (time.monotonic() - t_load_start) * 1000
        print(f"  [OK] 模型加载完成 ({t_load_ms:.0f}ms)\n", flush=True)

        # 内存/显存使用
        mem_info = {}
        if device == "cuda" and torch.cuda.is_available():
            mem_info["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            mem_info["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 2)
        else:
            try:
                import psutil
                proc = psutil.Process()
                mem_info["ram_rss_gb"] = round(proc.memory_info().rss / (1024**3), 2)
            except ImportError:
                mem_info["ram_rss_gb"] = -1

        result["load_time_ms"] = round(t_load_ms, 1)
        result["memory"] = mem_info

        max_samples = getattr(args, "max_samples", None)
        # Qwen3 为 thinking 模型，关闭 thinking 以获取直接输出
        enable_thinking = "Qwen3" not in model_id

        # 任务 A
        if task in ("extraction", "all"):
            result["task_a"] = run_task_a_extraction(model, tokenizer, device, max_samples=max_samples, enable_thinking=enable_thinking)

        # 任务 B
        if task in ("tool_selection", "all"):
            result["task_b"] = run_task_b_tool_selection(model, tokenizer, device, max_samples=max_samples, enable_thinking=enable_thinking)

        # 清理
        del model
        del tokenizer
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = str(e)[:500]
        print(f"  [FAIL] Worker 异常: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 写入结果（即使出错也写，driver 能从 error 字段识别）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  [Result] written to: {output_path}\n", flush=True)


# ══════════════════════════════════════════════════════════════════
# Driver 模式: 编排所有模型×设备组合
# ══════════════════════════════════════════════════════════════════

def run_subprocess(model_id: str, device: str, venv_python: str, max_samples: int | None = None) -> dict:
    """用指定 venv 的 Python 运行 worker 并解析结果。
    
    使用 Popen + 实时行读取避免管道缓冲死锁（capture_output=True 在大量输出时会卡死）。
    """
    import threading

    script_path = Path(__file__).resolve()
    output_dir = Path(__file__).resolve().parent / "benchmark_results"
    model_slug = model_id.replace("/", "_").replace(".", "_")
    output_file = output_dir / f"{model_slug}_{device}.json"

    print(f"\n{'─'*60}")
    print(f"[Launch] Worker: {model_id} @ {device}")
    print(f"   Python: {venv_python}")
    cmd = [
        str(venv_python),
        str(script_path),
        "--model", model_id,
        "--device", device,
        "--task", "all",
        "--output", str(output_file),
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    print(f"   Cmd: {' '.join(cmd)}")
    print(f"{'─'*60}\n", flush=True)

    t0 = time.monotonic()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
        encoding="utf-8",
        errors="replace",
    )

    # 收集输出（用于最后的日志）
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    def read_stream(stream, collector: list, prefix: str):
        """逐行读取并实时打印，同时收集到列表。"""
        console_enc = sys.stdout.encoding or "utf-8"
        for line in iter(stream.readline, ""):
            stripped = line.rstrip("\n\r")
            collector.append(stripped)
            if stripped:
                # 避免 Windows GBK 控制台编码错误
                try:
                    print(f"  {prefix}| {stripped}", flush=True)
                except UnicodeEncodeError:
                    safe = stripped.encode(console_enc, errors="replace").decode(console_enc)
                    print(f"  {prefix}| {safe}", flush=True)

    stdout_thread = threading.Thread(
        target=read_stream, args=(proc.stdout, stdout_lines, "OUT"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=read_stream, args=(proc.stderr, stderr_lines, "ERR"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        proc.wait(timeout=1800)  # 30 分钟超时
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"  [FAIL] Worker 超时（30 分钟）", flush=True)
        return {
            "model": model_id,
            "device": device,
            "error": "timeout after 30 minutes",
            "wall_time_s": round(time.monotonic() - t0, 1),
        }

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        stderr_tail = "\n".join(stderr_lines[-20:])
        print(f"  [FAIL] Worker failed (exit={proc.returncode})", flush=True)
        return {
            "model": model_id,
            "device": device,
            "error": stderr_tail if stderr_tail else f"exit code {proc.returncode}",
            "wall_time_s": round(elapsed, 1),
        }

    # 读取结果
    if output_file.exists():
        result = json.loads(output_file.read_text(encoding="utf-8"))
        result["wall_time_s"] = round(elapsed, 1)
        print(f"  [OK] Completed (wall_time={elapsed:.0f}s)", flush=True)
        return result
    else:
        print(f"  [WARN] Result file not found: {output_file}", flush=True)
        return {
            "model": model_id,
            "device": device,
            "error": "output file not found",
            "wall_time_s": round(elapsed, 1),
        }


def print_summary_table(results: List[dict]):
    """打印结果汇总表格"""
    print(f"\n{'='*80}")
    print("[Summary] Results")
    print(f"{'='*80}\n")

    # 信息提取
    print("【任务 A】信息提取（参数抽取）")
    print(f"{'模型':<30} {'设备':<6} {'字段F1':<8} {'值匹配率':<10} {'p50':<8} {'p99':<8} {'avg':<8} {'内存':<10}")
    print("-" * 88)
    for r in results:
        ta = r.get("task_a") or {}
        mem = r.get("memory", {})
        mem_str = (
            f"{mem.get('gpu_allocated_gb', '-')}G" if r.get("device") == "cuda"
            else f"{mem.get('ram_rss_gb', '-')}G"
        )
        if "error" in r:
            print(f"{r['model']:<30} {r['device']:<6} [FAIL] {r['error'][:50]}")
        elif ta:
            print(
                f"{r['model']:<30} {r['device']:<6} "
                f"{ta.get('field_f1', '-'):<8.3f} "
                f"{ta.get('value_exact_match_rate', '-'):<10.2%} "
                f"{ta.get('p50_latency_ms', '-'):<8.0f} "
                f"{ta.get('p99_latency_ms', '-'):<8.0f} "
                f"{ta.get('avg_latency_ms', '-'):<8.0f} "
                f"{mem_str}"
            )
    print()

    # 工具选择
    print("【任务 B】工具选择（P2 链路）")
    print(f"{'模型':<30} {'设备':<6} {'Top1命中':<10} {'Top2命中':<10} {'完全匹配':<10} {'p50':<8} {'p99':<8} {'avg':<8} {'显存':<10}")
    print("-" * 100)
    for r in results:
        tb = r.get("task_b") or {}
        mem = r.get("memory", {})
        mem_str = (
            f"{mem.get('gpu_allocated_gb', '-')}G" if r.get("device") == "cuda"
            else f"{mem.get('ram_rss_gb', '-')}G"
        )
        if "error" in r:
            print(f"{r['model']:<30} {r['device']:<6} [FAIL] {r['error'][:50]}")
        elif tb:
            print(
                f"{r['model']:<30} {r['device']:<6} "
                f"{tb.get('top1_hit_rate', '-'):<10.2%} "
                f"{tb.get('top2_hit_rate', '-'):<10.2%} "
                f"{tb.get('exact_match_rate', '-'):<10.2%} "
                f"{tb.get('p50_latency_ms', '-'):<8.0f} "
                f"{tb.get('p99_latency_ms', '-'):<8.0f} "
                f"{tb.get('avg_latency_ms', '-'):<8.0f} "
                f"{mem_str}"
            )
    print()

    # 保存完整结果
    summary_path = Path(__file__).resolve().parent / "benchmark_local_model_results.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[Result] full results saved to: {summary_path}\n")


def driver_main(args: argparse.Namespace):
    """Driver 模式：迭代所有模型×设备组合，收集结果并打印汇总"""
    # 确定要测试的组合
    if args.combo:
        models_to_test = [args.combo[0]]
    else:
        models_to_test = MODELS

    if args.device_filter:
        devices_to_test = [args.device_filter]
    else:
        devices_to_test = ["cpu", "cuda"]

    print("=" * 60)
    print("本地小模型多尺寸对比 Benchmark")
    print(f"模型 ({len(models_to_test)}): {', '.join(models_to_test)}")
    print(f"设备 ({len(devices_to_test)}): {', '.join(devices_to_test)}")
    print(f"组合数: {len(models_to_test) * len(devices_to_test)}")
    print("=" * 60)

    # 检查 venv 是否可用
    for dev in devices_to_test:
        venv_path = VENV_MAP.get(dev)
        if venv_path and not venv_path.exists():
            print(f"\n[WARN]️  {dev} venv 不存在: {venv_path}")
            print(f"   将跳过 {dev} 测试。请先创建虚拟环境。")

    results: List[dict] = []

    for model_id in models_to_test:
        for device in devices_to_test:
            venv_python = VENV_MAP.get(device)
            if not venv_python or not venv_python.exists():
                print(f"\n[SKIP]️  跳过 {model_id} @ {device} (venv 不存在)")
                results.append({
                    "model": model_id,
                    "device": device,
                    "error": f"venv not found: {venv_python}",
                })
                continue

            result = run_subprocess(model_id, device, str(venv_python), max_samples=getattr(args, "max_samples", None))
            results.append(result)

    print_summary_table(results)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="本地小模型多尺寸对比 Benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型 HF ID 或本地路径（worker 模式必需）",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="推理设备（worker 模式必需）",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["extraction", "tool_selection", "all"],
        default="all",
        help="评测任务（worker 模式）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出 JSON 路径（worker 模式）",
    )
    parser.add_argument(
        "--combo",
        type=str,
        default=None,
        nargs=2,
        metavar=("MODEL_ID", "DEVICE"),
        help="仅测试单个模型×设备组合（driver 模式），如: --combo 'Qwen/Qwen2.5-0.5B-Instruct' cpu",
    )
    parser.add_argument(
        "--device-filter",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="仅测试指定设备（driver 模式），如: --device-filter cpu",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制每个任务的测试样本数量（调试用，如 --max-samples 3）",
    )

    args = parser.parse_args()

    if args.model and args.device and args.output:
        # Worker 模式
        worker_main(args)
    else:
        # Driver 模式
        driver_main(args)


if __name__ == "__main__":
    main()

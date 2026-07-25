#!/usr/bin/env python3
"""
内容压缩测试 — 使用本地小模型 (Qwen3-1.7B) 对对话历史做摘要压缩。

验证：
  1. 压缩比（输入 tokens → 输出 tokens）
  2. 关键实体保留（订单号、金额、诉求）
  3. 推理耗时
  4. 极端场景（超长输入、无关键信息）
  5. 批量测试统计

前提条件：
  1. 本地模型已下载: python download.py qwen3-1.7b
  2. .env 中 LOCAL_PARAM_MODEL 指向模型路径
  3. 有 GPU 最好，CPU 也能跑（较慢）
"""

import os
import sys
import asyncio
import time
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# 测试用例 — 电商对话历史
# =============================================================================

# Case 1: 标准退货对话（含订单号、金额、多轮交互）
CASE_RETURN_CONVERSATION = [
    {"role": "user", "content": "你好，我买的苹果到了但有好几个都烂了，我要退货"},
    {"role": "assistant", "content": "非常抱歉给您带来不便！请您提供一下订单号，我帮您查询。"},
    {"role": "user", "content": "订单号是 GD20240527001"},
    {"role": "assistant", "content": "已查询到您的订单 GD20240527001，您购买了烟台红富士苹果 5斤装，金额 39.9 元，下单时间 2024-05-27。"},
    {"role": "user", "content": "对，就是这个。烂了大概有 3 个，其他看着也不太新鲜"},
    {"role": "assistant", "content": "了解。根据我们生鲜退货政策，签收后 24 小时内发现质量问题可以申请退货退款。您现在还在时效内吗？"},
    {"role": "user", "content": "我半小时前刚签收的，立马发现的"},
    {"role": "assistant", "content": "好的，已为您提交退货申请。退款金额 39.9 元预计在 1-3 个工作日内退回原支付方式。请您将损坏商品拍照保留，快递员上门取件时一并交回即可。"},
]

# Case 2: 跨意图对话（先查规格、再问退货、中间还问了优惠券）
CASE_MIXED_INTENT = [
    {"role": "user", "content": "你们那个戴森吸尘器 V12 和 V15 什么区别？"},
    {"role": "assistant", "content": "V12 吸力 150AW，续航 60 分钟；V15 吸力 240AW，续航 60 分钟，多了 LCD 屏幕和激光探测。V12 价格 3299，V15 价格 4499。"},
    {"role": "user", "content": "那 V15 现在有优惠吗？"},
    {"role": "assistant", "content": "V15 目前有满 3000 减 200 的券，到手价 4299。另外银联支付还可以再减 50。"},
    {"role": "user", "content": "好的谢谢。对了，我上次买的那个电饭煲 GD20240601008，煮饭的时候一直冒气，能退吗？"},
    {"role": "assistant", "content": "电饭煲有轻微冒气是正常的泄压设计。您能描述一下是冒大量蒸汽还是少量？使用多长时间了？"},
    {"role": "user", "content": "用了三天，煮饭的时候盖子一圈都在冒气，跟以前用的完全不一样。我觉得是质量问题"},
    {"role": "assistant", "content": "好的，电饭煲属于小家电品类，支持 7 天无理由退货。您的订单 GD20240601008 还在时效内。请您确认是否要提交退货申请？退款金额 299 元。"},
    {"role": "user", "content": "是的，帮我退货。那个 V15 的券也帮我领一下"},
    {"role": "assistant", "content": "已为您提交电饭煲的退货申请（退款 299 元），同时已领取 V15 吸尘器满 3000 减 200 优惠券。退货取件时间为明天上午 9:00-12:00。"},
]

# Case 3: 超长对话（模拟 20 轮后需要压缩的场景）
CASE_LONG_CONVERSATION = [
    {"role": "user", "content": "我的京东账户里还有多少余额？"},
    {"role": "assistant", "content": "您的账户余额为 520.00 元，积分 1280 分。"},
    {"role": "user", "content": "上个月买的那个美的空调 KFR-35GW 订单号 GD20240510005，制冷效果很差"},
    {"role": "assistant", "content": "已查询订单 GD20240510005，美的空调 KFR-35GW，购买日期 2024-05-10，价格 2999 元。已超过 7 天无理由时效，但仍在质保期内。需要我帮您预约售后检修吗？"},
    {"role": "user", "content": "行吧，帮我约一个。另外我还有一个快递 GD20240520012，到哪了？"},
    {"role": "assistant", "content": "已为您预约空调售后检修。订单 GD20240520012（海尔冰箱 BCD-500）当前物流状态：已到达北京分拣中心，预计明天 18:00 前派送。"},
    {"role": "user", "content": "那就等明天了。对了，冰箱要是也有问题的话，退换政策是什么？"},
    {"role": "assistant", "content": "冰箱属于大家电，支持 7 天无理由退货、15 天质量问题换货、整机保修 1 年。签收后请务必当面开箱验机，如有外观损坏可直接拒收。"},
    {"role": "user", "content": "明白了。上次买的洗衣粉到了没？订单 GD20240518003"},
    {"role": "assistant", "content": "GD20240518003 蓝月亮洗衣液 3kg 装已签收，5 月 19 日由您本人签收。"},
    {"role": "user", "content": "哦对，想起来了我记错了。那空调检修约的什么时候？"},
    {"role": "assistant", "content": "空调售后检修预约时间：5 月 23 日（周四）上午 10:00-12:00。工程师上门前 1 小时会电话确认。"},
    {"role": "user", "content": "行。再查一下我有没有什么优惠券快过期的"},
    {"role": "assistant", "content": "您有 1 张即将过期的优惠券：满 200 减 30 全场通用券，有效期至 2024-05-25。"},
]

# Case 4: 安全问题混合（测试小模型能否区分可保留的正常内容）
CASE_WITH_NOISE = [
    {"role": "user", "content": "你好，今天心情不太好"},
    {"role": "assistant", "content": "您好！有什么我可以帮到您的吗？"},
    {"role": "user", "content": "上次买的鞋子 GD20240515009 码数小了，想换大一码"},
    {"role": "assistant", "content": "好的，查到了订单 GD20240515009，Nike Air Max 270，42码。需要换成 43 码对吗？"},
    {"role": "user", "content": "对。你们这鞋子其实穿着挺舒服的，底很软，就是码数偏小"},
    {"role": "assistant", "content": "感谢您的反馈！已为您提交换货申请（42→43码），预计 3 天内发出。请保持原包装完整。"},
    {"role": "user", "content": "好的谢谢，我今天就是有点累，说话不太礼貌请你见谅"},
    {"role": "assistant", "content": "完全理解！您有任何需要随时找我。祝您生活愉快！"},
]


# =============================================================================
# 测试结果数据类
# =============================================================================

@dataclass
class CompressionResult:
    case_name: str
    input_messages: list
    input_text: str
    input_tokens: int
    summary: str
    output_tokens: int
    compression_ratio: float       # output / input
    duration_ms: float
    success: bool
    error: str = ""
    prompt_type: str = "structured"  # "structured" | "simple"
    # 质量检查
    entities_preserved: List[str] = field(default_factory=list)
    entities_missing: List[str] = field(default_factory=list)
    intents_preserved: List[str] = field(default_factory=list)
    intents_missing: List[str] = field(default_factory=list)
    noise_kept_count: int = 0        # 摘要中残留的噪音词数（寒暄、客套）


@dataclass
class PromptComparison:
    """同一 case 两种 prompt 的对比结果"""
    case_name: str
    input_tokens: int
    structured: CompressionResult
    simple: CompressionResult
    # delta: structured - simple (正 = structured 更好)
    delta_entity_rate: float = 0.0
    delta_intent_rate: float = 0.0
    delta_noise: int = 0            # 正 = simple 噪音更多
    delta_duration_ms: float = 0.0
    winner: str = ""                # "structured" | "simple" | "tie"


# =============================================================================
# 压缩 prompt 模板
# =============================================================================

# ── 结构化 Prompt（含明确规则约束，适用于小模型）──────────────
COMPRESSION_SYSTEM_PROMPT_STRUCTURED = (
    "你是一个对话摘要助手。将用户与客服之间的对话压缩成一段简洁的摘要。\n"
    "严格遵守以下规则：\n"
    "1. 必须保留所有订单号（格式：GD开头+数字）\n"
    "2. 必须保留所有金额（含单位：元）\n"
    "3. 必须保留用户的核心诉求（退货/退款/换货/查询/投诉）\n"
    "4. 必须保留客服的最终处理结果\n"
    "5. 丢弃寒暄、道歉、客套话\n"
    "6. 丢弃没有实际业务含义的闲聊\n"
    "7. 丢弃中间交互细节（如\"请提供订单号\"这类引导）\n"
    "8. 保留每个话题的切换点\n"
    "9. 输出纯文本，不要 markdown，不要解释，不要输出\"摘要：\"等前缀"
)

# ── 简单 Prompt（无规则约束，作为对照基线）────────────────
COMPRESSION_SYSTEM_PROMPT_SIMPLE = (
    "你是一个对话摘要助手。请将以下用户与客服之间的对话总结为一段简洁的摘要。"
)

# 向后兼容别名
COMPRESSION_SYSTEM_PROMPT = COMPRESSION_SYSTEM_PROMPT_STRUCTURED

COMPRESSION_USER_TEMPLATE = (
    "对话历史:\n"
    "{conversation}\n\n"
    "请按规则压缩为摘要："
)

COMPRESSION_USER_TEMPLATE_SIMPLE = (
    "对话历史:\n"
    "{conversation}\n\n"
    "请总结："
)

# =============================================================================
# 质量评估辅助：意图 / 噪音检测
# =============================================================================

# 核心意图关键词（电商客服场景）
INTENT_KEYWORDS = {
    "退货": ["退货", "退掉", "退换", "退", "退款"],
    "换货": ["换货", "换一", "更换", "换码", "换尺码"],
    "查询订单": ["查订单", "查询订单", "订单", "到哪了", "状态"],
    "查询物流": ["快递", "物流", "到哪", "派送", "运输"],
    "余额查询": ["余额", "积分", "还有多少"],
    "优惠券": ["优惠券", "券", "领券", "满减"],
    "咨询规格": ["区别", "规格", "参数", "耗电", "续航", "吸力"],
    "投诉": ["投诉", "不满", "问题", "质量"],
}


def _detect_intents(text: str) -> set:
    """从文本中检测核心意图类别"""
    found = set()
    for category, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found.add(category)
                break
    return found


# 噪音/寒暄词
NOISE_PATTERNS = [
    r"你好", r"您好", r"谢谢", r"感谢", r"抱歉", r"不好意思",
    r"对不起", r"请见谅", r"祝您", r"生活愉快", r"完全理解",
    r"有什么我可以帮", r"非常抱歉",
]


def _count_noise(text: str) -> int:
    """统计摘要中残留的噪音/寒暄词数"""
    count = 0
    for pat in NOISE_PATTERNS:
        count += len(re.findall(pat, text))
    return count


# =============================================================================
# 核心压缩函数
# =============================================================================

def _format_conversation(messages: list) -> str:
    """将 messages 列表格式化为纯文本"""
    lines = []
    for msg in messages:
        role_label = "用户" if msg["role"] == "user" else "客服"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)


def _build_compression_prompt(messages: list, prompt_type: str = "structured") -> tuple:
    """构建压缩 prompt，支持两种模式"""
    conversation = _format_conversation(messages)
    if prompt_type == "simple":
        system = COMPRESSION_SYSTEM_PROMPT_SIMPLE
        user_msg = COMPRESSION_USER_TEMPLATE_SIMPLE.format(conversation=conversation)
    else:
        system = COMPRESSION_SYSTEM_PROMPT_STRUCTURED
        user_msg = COMPRESSION_USER_TEMPLATE.format(conversation=conversation)
    return conversation, user_msg, system


def _extract_entities(text: str) -> List[str]:
    """从文本中提取订单号和金额"""
    entities = []
    # 订单号 GD开头+数字
    order_ids = re.findall(r'GD\d{8,}', text)
    entities.extend(order_ids)
    # 金额
    amounts = re.findall(r'\d+\.?\d*\s*元', text)
    entities.extend(amounts)
    return entities


async def compress_with_local_model(
    case_name: str,
    messages: list,
    local_service,
    token_estimator,
    prompt_type: str = "structured",
) -> CompressionResult:
    """使用本地小模型压缩对话历史

    Args:
        prompt_type: "structured" (带9条规则) | "simple" (无规则基线)
    """
    conversation_text, user_prompt, system_prompt = _build_compression_prompt(
        messages, prompt_type
    )

    # 计算输入 token
    try:
        input_tokens = token_estimator.estimate(conversation_text)
    except Exception:
        input_tokens = len(conversation_text)  # fallback: 字符数

    # 模型未加载
    if not local_service._ensure_loaded():
        return CompressionResult(
            case_name=case_name,
            input_messages=messages,
            input_text=conversation_text,
            input_tokens=input_tokens,
            summary="",
            output_tokens=0,
            compression_ratio=0,
            duration_ms=0,
            success=False,
            prompt_type=prompt_type,
            error="本地模型未加载",
        )

    # 构建 chat template prompt
    try:
        from transformers import AutoTokenizer
        tokenizer = local_service._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            chat_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    except Exception:
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # 执行推理
    try:
        t_start = time.perf_counter()
        raw_output = await asyncio.get_event_loop().run_in_executor(
            local_service._get_executor(),
            local_service._generate,
            prompt,
        )
        duration_ms = (time.perf_counter() - t_start) * 1000
    except Exception as e:
        return CompressionResult(
            case_name=case_name,
            input_messages=messages,
            input_text=conversation_text,
            input_tokens=input_tokens,
            summary="",
            output_tokens=0,
            compression_ratio=0,
            duration_ms=0,
            success=False,
            prompt_type=prompt_type,
            error=f"推理异常: {str(e)[:200]}",
        )

    # 清理输出
    if hasattr(local_service, '_strip_think_tags'):
        summary = local_service._strip_think_tags(raw_output)
    else:
        summary = raw_output.strip()

    # 计算输出 token
    try:
        output_tokens = token_estimator.estimate(summary)
    except Exception:
        output_tokens = len(summary)

    compression_ratio = output_tokens / max(input_tokens, 1)

    # 质量检查：实体保留
    original_entities = _extract_entities(conversation_text)
    summary_entities = _extract_entities(summary)
    entities_preserved = [e for e in original_entities if e in summary]
    entities_missing = [e for e in original_entities if e not in summary]

    # 质量检查：意图保留
    original_intents = _detect_intents(conversation_text)
    summary_intents = _detect_intents(summary)
    intents_preserved = sorted(original_intents & summary_intents)
    intents_missing = sorted(original_intents - summary_intents)

    # 质量检查：噪音残留
    noise_count = _count_noise(summary)

    return CompressionResult(
        case_name=case_name,
        input_messages=messages,
        input_text=conversation_text,
        input_tokens=input_tokens,
        summary=summary,
        output_tokens=output_tokens,
        compression_ratio=round(compression_ratio, 4),
        duration_ms=round(duration_ms, 1),
        success=True,
        prompt_type=prompt_type,
        entities_preserved=entities_preserved,
        entities_missing=entities_missing,
        intents_preserved=intents_preserved,
        intents_missing=intents_missing,
        noise_kept_count=noise_count,
    )


# =============================================================================
# 打印与统计
# =============================================================================

def _color(text: str, code: str) -> str:
    colors = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1"}
    return f"\033[{colors.get(code, '0')}m{text}\033[0m"


def print_result(result: CompressionResult, index: int):
    """打印单个压缩结果"""
    label = {"structured": "结构化 Prompt", "simple": "简单 Prompt"}.get(
        result.prompt_type, result.prompt_type
    )
    status = _color("✓ 成功", "green") if result.success else _color("✗ 失败", "red")
    ratio_pct = f"{result.compression_ratio * 100:.1f}%"

    print(f"\n{'─' * 70}")
    print(f" Case {index}: {_color(result.case_name, 'bold')}  [{label}]  {status}")
    print(f"{'─' * 70}")

    print(f"  📊 输入 tokens: {result.input_tokens}")
    print(f"  📊 输出 tokens: {result.output_tokens}")
    print(f"  📊 压缩比:     {ratio_pct}  ({_color('好', 'green') if result.compression_ratio < 0.3 else _color('偏高', 'yellow')})")
    print(f"  ⏱️  耗时:       {result.duration_ms:.1f} ms")

    if not result.success:
        print(f"  ❌ 错误: {result.error}")
        return

    # 摘要
    print(f"\n  📝 摘要:")
    summary = result.summary.replace('\n', '\n       ')
    print(f"       {summary[:500]}")
    if len(result.summary) > 500:
        print(f"       ... (共 {len(result.summary)} 字符)")

    # 实体保留
    if result.entities_preserved:
        print(f"\n  ✅ 保留实体: {', '.join(result.entities_preserved)}")
    if result.entities_missing:
        print(f"  ⚠️  丢失实体: {_color(', '.join(result.entities_missing), 'red')}")

    # 意图保留
    if result.intents_preserved:
        print(f"  🎯 保留意图: {', '.join(result.intents_preserved)}")
    if result.intents_missing:
        print(f"  ⚠️  丢失意图: {_color(', '.join(result.intents_missing), 'yellow')}")

    # 噪音
    if result.noise_kept_count > 0:
        print(f"  🔕 噪音残留: {_color(str(result.noise_kept_count), 'yellow')} 个寒暄/客套词")
    else:
        print(f"  🔕 噪音残留: 0 (干净)")


def print_comparison_table(comparisons: List[PromptComparison]):
    """打印结构化 vs 简单 prompt 的多维度对比表"""

    print(f"\n{'═' * 90}")
    print(f" {_color('📊 结构化 Prompt vs 简单 Prompt  多维度对比', 'bold')}")
    print(f"{'═' * 90}")

    H = "─" * 90
    HEADER = (
        f" {'Case':<20s} {'指标':<14s} {'结构化':>12s} {'简单':>12s} {'Δ':>10s} {'胜者':>10s}"
    )
    print(H)
    print(HEADER)
    print(H)

    for c in comparisons:
        s = c.structured
        si = c.simple
        short_name = c.case_name[:18]

        # 实体保留率
        s_ent_total = len(s.entities_preserved) + len(s.entities_missing)
        si_ent_total = len(si.entities_preserved) + len(si.entities_missing)
        s_ent_rate = f"{len(s.entities_preserved)}/{s_ent_total}" if s_ent_total else "-"
        si_ent_rate = f"{len(si.entities_preserved)}/{si_ent_total}" if si_ent_total else "-"

        ent_str = f"实体保留"
        s_str = f"{s_ent_rate} ({len(s.entities_preserved)}/{max(s_ent_total,1)})"
        si_str = f"{si_ent_rate} ({len(si.entities_preserved)}/{max(si_ent_total,1)})"
        delta_str = f"{c.delta_entity_rate:+.0%}"
        ent_winner = "结构化" if c.delta_entity_rate > 0 else ("简单" if c.delta_entity_rate < 0 else "平")
        print(f" {short_name:<20s} {_color(ent_str, 'cyan'):<14s} {s_str:>12s} {si_str:>12s} {delta_str:>10s} {_color(ent_winner, 'bold'):>10s}")

        # 意图保留率
        s_int_total = len(s.intents_preserved) + len(s.intents_missing)
        si_int_total = len(si.intents_preserved) + len(si.intents_missing)
        s_int_str = f"{len(s.intents_preserved)}/{s_int_total}" if s_int_total else "-"
        si_int_str = f"{len(si.intents_preserved)}/{si_int_total}" if si_int_total else "-"
        delta_int = f"{c.delta_intent_rate:+.0%}"
        int_winner = "结构化" if c.delta_intent_rate > 0 else ("简单" if c.delta_intent_rate < 0 else "平")
        print(f" {'':20s} {_color('意图保留', 'cyan'):<14s} {s_int_str:>12s} {si_int_str:>12s} {delta_int:>10s} {_color(int_winner, 'bold'):>10s}")

        # 压缩比
        s_ratio = f"{s.compression_ratio*100:.1f}%"
        si_ratio = f"{si.compression_ratio*100:.1f}%"
        delta_ratio = f"{(si.compression_ratio - s.compression_ratio)*100:+.1f}pp"
        ratio_winner = "结构化" if s.compression_ratio <= si.compression_ratio else "简单"
        print(f" {'':20s} {_color('压缩比', 'cyan'):<14s} {s_ratio:>12s} {si_ratio:>12s} {delta_ratio:>10s} {_color(ratio_winner, 'bold'):>10s}")

        # 噪音残留
        s_noise = str(s.noise_kept_count)
        si_noise = str(si.noise_kept_count)
        delta_noise = f"{si.noise_kept_count - s.noise_kept_count:+d}"
        noise_winner = "结构化" if s.noise_kept_count <= si.noise_kept_count else "简单"
        print(f" {'':20s} {_color('噪音残留', 'cyan'):<14s} {s_noise:>12s} {si_noise:>12s} {delta_noise:>10s} {_color(noise_winner, 'bold'):>10s}")

        # 耗时
        s_dur = f"{s.duration_ms:.0f}ms"
        si_dur = f"{si.duration_ms:.0f}ms"
        delta_dur = f"{c.delta_duration_ms:+.0f}ms"
        dur_winner = "结构化" if c.delta_duration_ms <= 0 else "简单"
        print(f" {'':20s} {_color('耗时', 'cyan'):<14s} {s_dur:>12s} {si_dur:>12s} {delta_dur:>10s} {_color(dur_winner, 'bold'):>10s}")

        # 综合判定
        print(f" {'':20s} {'─'*66}")
        w = c.winner
        w_color = "green" if w == "structured" else ("yellow" if w == "tie" else "red")
        print(f" {'':20s} {_color(f'综合胜者: {w}', w_color):>56s}")

        print(H)

    # 汇总行
    s_avg_ent = sum(
        len(c.structured.entities_preserved) / max(
            len(c.structured.entities_preserved) + len(c.structured.entities_missing), 1
        )
        for c in comparisons
    ) / max(len(comparisons), 1)
    si_avg_ent = sum(
        len(c.simple.entities_preserved) / max(
            len(c.simple.entities_preserved) + len(c.simple.entities_missing), 1
        )
        for c in comparisons
    ) / max(len(comparisons), 1)
    s_avg_int = sum(
        len(c.structured.intents_preserved) / max(
            len(c.structured.intents_preserved) + len(c.structured.intents_missing), 1
        )
        for c in comparisons
    ) / max(len(comparisons), 1)
    si_avg_int = sum(
        len(c.simple.intents_preserved) / max(
            len(c.simple.intents_preserved) + len(c.simple.intents_missing), 1
        )
        for c in comparisons
    ) / max(len(comparisons), 1)

    print(f"\n {_color('📈 汇总', 'bold')}")
    print(H)
    print(f" {'指标':<14s} {'结构化 (avg)':>22s} {'简单 (avg)':>22s} {'提升':>14s}")
    print(f" {'实体保留率':<14s} {f'{s_avg_ent:.0%}':>22s} {f'{si_avg_ent:.0%}':>22s} {_color(f'+{(s_avg_ent - si_avg_ent)*100:.1f}pp', 'green' if s_avg_ent >= si_avg_ent else 'red'):>14s}")
    print(f" {'意图保留率':<14s} {f'{s_avg_int:.0%}':>22s} {f'{si_avg_int:.0%}':>22s} {_color(f'+{(s_avg_int - si_avg_int)*100:.1f}pp', 'green' if s_avg_int >= si_avg_int else 'red'):>14s}")
    print(H)

    # 胜负统计
    structured_wins = sum(1 for c in comparisons if c.winner == "structured")
    simple_wins = sum(1 for c in comparisons if c.winner == "simple")
    ties = sum(1 for c in comparisons if c.winner == "tie")
    print(f"\n {_color('🏆 综合胜负', 'bold')}")
    print(f"   结构化胜: {_color(str(structured_wins), 'green')}  |  "
          f"简单胜: {_color(str(simple_wins), 'red')}  |  "
          f"平局: {_color(str(ties), 'yellow')}")
    print(f"{'═' * 90}")


def print_summary(results: List[CompressionResult]):
    """打印总体统计"""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n{'═' * 70}")
    print(f" {_color('📊 总体统计', 'bold')}")
    print(f"{'═' * 70}")

    print(f"  测试用例: {len(results)} 个")
    print(f"  成功:     {_color(str(len(successful)), 'green')} 个")
    print(f"  失败:     {_color(str(len(failed)), 'red')} 个")

    if not successful:
        print(f"\n  {_color('⚠️ 所有用例失败，请检查本地模型是否已加载', 'yellow')}")
        return

    avg_ratio = sum(r.compression_ratio for r in successful) / len(successful)
    avg_duration = sum(r.duration_ms for r in successful) / len(successful)
    total_preserved = sum(len(r.entities_preserved) for r in successful)
    total_entities = sum(len(r.entities_preserved) + len(r.entities_missing) for r in successful)
    entity_rate = total_preserved / max(total_entities, 1) * 100

    print(f"\n  平均压缩比:   {avg_ratio * 100:.1f}%")
    print(f"  平均耗时:     {avg_duration:.1f} ms")
    print(f"  实体保留率:   {entity_rate:.1f}% ({total_preserved}/{total_entities})")

    # 判断是否达标
    checks = []
    checks.append((avg_ratio < 0.30, f"压缩比 < 30%"))
    checks.append((avg_duration < 500, f"平均耗时 < 500ms"))
    checks.append((entity_rate >= 80, f"实体保留率 ≥ 80%"))

    print(f"\n  验收标准:")
    for passed, desc in checks:
        status = _color("✓", "green") if passed else _color("✗", "red")
        print(f"    {status} {desc}")

    if all(c[0] for c in checks):
        print(f"\n  {_color('✅ 全部达标 — 小模型压缩方案可行', 'green')}")
    else:
        print(f"\n  {_color('⚠️ 部分未达标 — 需优化 prompt 或考虑备选方案', 'yellow')}")


# =============================================================================
# 对比测试：结构化 vs 简单 Prompt
# =============================================================================

def _compute_comparison(structured: CompressionResult, simple: CompressionResult) -> PromptComparison:
    """计算两个结果的对比分数"""
    # 实体保留率 delta
    s_ent_total = max(len(structured.entities_preserved) + len(structured.entities_missing), 1)
    si_ent_total = max(len(simple.entities_preserved) + len(simple.entities_missing), 1)
    s_ent_rate = len(structured.entities_preserved) / s_ent_total
    si_ent_rate = len(simple.entities_preserved) / si_ent_total
    delta_entity = s_ent_rate - si_ent_rate

    # 意图保留率 delta
    s_int_total = max(len(structured.intents_preserved) + len(structured.intents_missing), 1)
    si_int_total = max(len(simple.intents_preserved) + len(simple.intents_missing), 1)
    s_int_rate = len(structured.intents_preserved) / s_int_total
    si_int_rate = len(simple.intents_preserved) / si_int_total
    delta_intent = s_int_rate - si_int_rate

    # 噪音 delta（正 = simple 更吵）
    delta_noise = simple.noise_kept_count - structured.noise_kept_count

    # 耗时 delta（正 = structured 更慢）
    delta_duration = structured.duration_ms - simple.duration_ms

    # 综合判定：按权重计分
    # 实体权重 35%，意图权重 35%，噪音权重 20%，耗时权重 10%
    s_score = (
        s_ent_rate * 0.35 +
        s_int_rate * 0.35 +
        (1.0 - min(structured.noise_kept_count / 5, 1.0)) * 0.20 +
        (1.0 - min(structured.duration_ms / 500, 1.0)) * 0.10
    )
    si_score = (
        si_ent_rate * 0.35 +
        si_int_rate * 0.35 +
        (1.0 - min(simple.noise_kept_count / 5, 1.0)) * 0.20 +
        (1.0 - min(simple.duration_ms / 500, 1.0)) * 0.10
    )

    if s_score - si_score > 0.03:
        winner = "structured"
    elif si_score - s_score > 0.03:
        winner = "simple"
    else:
        winner = "tie"

    return PromptComparison(
        case_name=structured.case_name,
        input_tokens=structured.input_tokens,
        structured=structured,
        simple=simple,
        delta_entity_rate=round(delta_entity, 4),
        delta_intent_rate=round(delta_intent, 4),
        delta_noise=delta_noise,
        delta_duration_ms=round(delta_duration, 1),
        winner=winner,
    )


async def run_comparison(local_service, token_estimator):
    """运行结构化 vs 简单 Prompt 的对比测试"""

    print(f"\n{'═' * 90}")
    print(f" 结构化 Prompt vs 简单 Prompt 对比测试")
    print(f" 模型: Qwen3-1.7B")
    print(f"{'═' * 90}")

    test_cases = [
        ("标准退货对话 (8轮)", CASE_RETURN_CONVERSATION),
        ("跨意图混合对话 (10轮)", CASE_MIXED_INTENT),
        ("超长多话题对话 (13轮)", CASE_LONG_CONVERSATION),
        ("含情绪/噪音的对话 (8轮)", CASE_WITH_NOISE),
    ]

    comparisons: List[PromptComparison] = []

    for i, (case_name, messages) in enumerate(test_cases, 1):
        print(f"\n  [{i}/{len(test_cases)}] {case_name}")

        # 运行结构化 prompt
        print(f"     结构化 Prompt ...", end=" ", flush=True)
        r_structured = await compress_with_local_model(
            case_name, messages, local_service, token_estimator,
            prompt_type="structured"
        )
        if r_structured.success:
            print(_color(f"OK ({r_structured.duration_ms:.0f}ms)  "
                         f"实体{len(r_structured.entities_preserved)}/{len(r_structured.entities_preserved)+len(r_structured.entities_missing)}  "
                         f"意图{len(r_structured.intents_preserved)}/{len(r_structured.intents_preserved)+len(r_structured.intents_missing)}",
                         "green"))
        else:
            print(_color(f"FAILED: {r_structured.error[:60]}", "red"))
            continue

        # 运行简单 prompt
        print(f"     简单 Prompt ...  ", end=" ", flush=True)
        r_simple = await compress_with_local_model(
            case_name, messages, local_service, token_estimator,
            prompt_type="simple"
        )
        if r_simple.success:
            print(_color(f"OK ({r_simple.duration_ms:.0f}ms)  "
                         f"实体{len(r_simple.entities_preserved)}/{len(r_simple.entities_preserved)+len(r_simple.entities_missing)}  "
                         f"意图{len(r_simple.intents_preserved)}/{len(r_simple.intents_preserved)+len(r_simple.intents_missing)}",
                         "green"))
        else:
            print(_color(f"FAILED: {r_simple.error[:60]}", "red"))
            continue

        comp = _compute_comparison(r_structured, r_simple)
        comparisons.append(comp)

    if not comparisons:
        print(f"\n  {_color('⚠️ 没有可展示的对比结果', 'yellow')}")
        return

    # ── 打印摘要对比 ──
    print(f"\n\n{'═' * 90}")
    print(f" {_color('📝 摘要内容逐条对比', 'bold')}")
    print(f"{'═' * 90}")

    for comp in comparisons:
        print(f"\n{'─' * 90}")
        print(f" {_color(comp.case_name, 'bold')}")
        print(f"{'─' * 90}")
        print(f"  {_color('结构化 Prompt', 'cyan')}:  {comp.structured.summary[:300]}")
        if len(comp.structured.summary) > 300:
            print(f"          ... (总 {len(comp.structured.summary)} 字)")
        print(f"  {_color('简单 Prompt', 'yellow')}:    {comp.simple.summary[:300]}")
        if len(comp.simple.summary) > 300:
            print(f"          ... (总 {len(comp.simple.summary)} 字)")

    # ── 打印对比表 ──
    print_comparison_table(comparisons)


# =============================================================================
# 主测试入口
# =============================================================================

async def main():
    print(f"{'═' * 70}")
    print(f" 本地小模型内容压缩测试")
    print(f" 模型: Qwen3-1.7B")
    print(f"{'═' * 70}")

    # ── Step 1: 初始化服务 ──
    print(f"\n[1/4] 初始化服务 ...")

    from src.modules.chat.core.local_model_service import LocalModelService
    from src.core.token_estimator import get_token_estimator

    local_service = LocalModelService.get_instance()

    # 尝试加载模型
    t_load = time.perf_counter()
    loaded = local_service._ensure_loaded()
    load_time = (time.perf_counter() - t_load) * 1000

    if not loaded:
        print(f"  {_color('✗ 本地模型加载失败', 'red')}")
        print(f"  原因: {local_service._load_failed_reason[:300] if local_service._load_failed_reason else '未知'}")
        print(f"\n  ⚠️  请确保模型已下载:")
        print(f"    python download.py qwen3-1.7b")
        print(f"  或在 .env 中设置 LOCAL_PARAM_MODEL 指向正确的路径")
        return
    else:
        print(f"  {_color('✓ 本地模型已加载', 'green')} ({load_time:.0f} ms)")

    # 初始化 token estimator
    try:
        token_estimator = get_token_estimator()
        print(f"  {_color('✓ Token Estimator 已就绪', 'green')}")
    except Exception as e:
        print(f"  {_color('⚠ Token Estimator 不可用', 'yellow')}: {e}")
        # 手动创建一个简易的
        class SimpleEstimator:
            def estimate(self, text: str) -> int:
                return len(text)
        token_estimator = SimpleEstimator()

    # ── Step 2: 执行测试 ──
    print(f"\n[2/4] 执行压缩测试 ...")

    test_cases = [
        ("标准退货对话 (8轮)", CASE_RETURN_CONVERSATION),
        ("跨意图混合对话 (10轮)", CASE_MIXED_INTENT),
        ("超长多话题对话 (13轮)", CASE_LONG_CONVERSATION),
        ("含情绪/噪音的对话 (8轮)", CASE_WITH_NOISE),
    ]

    results: List[CompressionResult] = []

    for i, (case_name, messages) in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] 测试: {case_name} ...", end=" ")
        result = await compress_with_local_model(
            case_name, messages, local_service, token_estimator
        )
        if result.success:
            print(_color(f"OK ({result.duration_ms:.0f}ms)", "green"))
        else:
            print(_color(f"FAILED - {result.error[:80]}", "red"))
        results.append(result)

    # ── Step 3: 打印详细结果 ──
    print(f"\n[3/4] 详细结果")
    for i, result in enumerate(results, 1):
        print_result(result, i)

    # ── Step 4: 总体统计 ──
    print(f"\n[4/4] 评估")
    print_summary(results)

    print(f"\n{'═' * 70}")


# =============================================================================
# 独立测试：使用纯文本构建 prompt（不依赖 tokenizer chat template）
# =============================================================================

async def test_with_manual_prompt():
    """备用测试：手动拼接 prompt，不依赖 tokenizer.apply_chat_template"""
    print(f"{'═' * 70}")
    print(f" 手动 Prompt 压缩测试（备选方案）")
    print(f"{'═' * 70}")

    from src.modules.chat.core.local_model_service import LocalModelService

    local_service = LocalModelService.get_instance()
    if not local_service._ensure_loaded():
        print(f"  本地模型未加载，跳过")
        return

    # 手动拼接 prompt
    conversation = _format_conversation(CASE_RETURN_CONVERSATION)
    prompt = (
        f"<|im_start|>system\n{COMPRESSION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"对话历史:\n{conversation}\n\n请按规则压缩为摘要：\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    t_start = time.perf_counter()
    summary = await asyncio.get_event_loop().run_in_executor(
        local_service._get_executor(),
        local_service._generate,
        prompt,
    )
    duration_ms = (time.perf_counter() - t_start) * 1000

    # 清理
    if hasattr(local_service, '_strip_think_tags'):
        summary = local_service._strip_think_tags(summary)
    else:
        summary = summary.strip()

    print(f"\n  输入: {len(conversation)} 字 / {len(conversation.split())} 词")
    print(f"  摘要: {summary[:300]}")
    print(f"  长度: {len(summary)} 字")
    print(f"  耗时: {duration_ms:.1f} ms")


# =============================================================================
# 对比测试入口辅助
# =============================================================================

async def _run_comparison_entry():
    """对比测试入口：初始化服务 → 运行对比"""
    from src.modules.chat.core.local_model_service import LocalModelService
    from src.core.token_estimator import get_token_estimator

    local_service = LocalModelService.get_instance()

    if not local_service._ensure_loaded():
        print(f"  {_color('✗ 本地模型加载失败，无法运行对比', 'red')}")
        return

    try:
        token_estimator = get_token_estimator()
    except Exception:
        class SimpleEstimator:
            def estimate(self, text: str) -> int:
                return len(text)
        token_estimator = SimpleEstimator()

    await run_comparison(local_service, token_estimator)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="本地小模型内容压缩测试")
    parser.add_argument(
        "--manual", action="store_true",
        help="使用手动拼接 prompt 模式（不依赖 tokenizer chat template）"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="运行结构化 Prompt vs 简单 Prompt 的 A/B 对比测试"
    )
    args = parser.parse_args()

    if args.compare:
        asyncio.run(_run_comparison_entry())
    elif args.manual:
        asyncio.run(test_with_manual_prompt())
    else:
        asyncio.run(main())

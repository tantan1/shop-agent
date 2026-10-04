"""
情绪检测服务 (Sentiment Detection Service)

职责：
- L1 规则关键词：极低成本关键词匹配（<1ms），覆盖明确情绪信号
- 返回 7 级情绪分类（EmotionLevel），兼容现有业务逻辑

简化原则：
- 去掉 L2 本地模型、L3 云端 LLM 兜底
- 去掉 SessionEmotionTracker 会话跟踪
- 保留 L1 规则关键词 + 否定词反转
- detect() 保持 async 签名，兼容现有调用方
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

from src.shared.logger import APILogger

logger = APILogger("sentiment_service")


# ═══════════════════════════════════════════════════════════════════════
# 情绪等级定义
# ═══════════════════════════════════════════════════════════════════════


class EmotionLevel(IntEnum):
    """情绪等级（值越大越危险，>=4 触发升级）"""

    GRATEFUL = 0  # 感激 — "太感谢了"
    SATISFIED = 1  # 满意 — "好的谢谢"
    NEUTRAL = 2  # 中立 — "帮我查下订单"
    ANXIOUS = 3  # 焦虑 — "怎么还没发货"
    DISAPPOINTED = 4  # 失望 — "等了三天了"
    ANGRY = 5  # 愤怒 — "你们是骗子吗"
    EMERGENCY = 6  # 舆情风险 — "我要打12315"


ESCALATE_THRESHOLD = EmotionLevel.DISAPPOINTED  # >=4 建议升级
EMERGENCY_THRESHOLD = EmotionLevel.EMERGENCY  # =6 强制升级


@dataclass
class EmotionResult:
    """单条消息的情绪检测结果"""

    level: EmotionLevel
    confidence: float  # 0.0~1.0
    escalate: bool  # 是否建议升级到人工
    is_emergency: bool  # 是否强制升级
    keywords: List[str] = field(default_factory=list)
    source: str = "L0:none"  # 来源标识


# ═══════════════════════════════════════════════════════════════════════
# L1: 规则关键词表（零成本，<1ms）
# ═══════════════════════════════════════════════════════════════════════

EMOTION_KEYWORDS: Dict[EmotionLevel, List[str]] = {
    EmotionLevel.EMERGENCY: [
        "12315",
        "消费者协会",
        "工商局",
        "消协",
        "举报你们",
        "起诉",
        "走法律程序",
        "我要曝光",
        "上热搜",
        "媒体曝光",
        "报警",
        "诈骗",
        "欺诈",
        "虚假宣传",
        "虚假广告",
        "人身安全",
        "威胁生命",
        "生命危险",
    ],
    EmotionLevel.ANGRY: [
        "骗子",
        "骗钱",
        "垃圾",
        "死全家",
        "日了狗",
        "操",
        "倒闭",
        "黑心",
        "奸商",
        "太过分了",
        "不可原谅",
        "再不处理我就",
        "一遍又一遍",
        "反复忽悠",
        "找你们领导",
        "投诉到底",
        "一直推脱",
        "拖了这么久",
    ],
    EmotionLevel.DISAPPOINTED: [
        "等了",
        "还没到",
        "又坏了",
        "怎么又",
        "失望",
        "上次就说",
        "说好的",
        "承诺",
        "不靠谱",
        "有问题",
        "不回复",
        "不处理",
        "客服态度",
        "无法接受",
    ],
    EmotionLevel.ANXIOUS: [
        "什么时候",
        "多久能",
        "还能到吗",
        "不会丢了吧",
        "怕",
        "担心",
        "着急",
        "急用",
        "催一下",
        "加急",
        "尽快",
        "麻烦快一点",
    ],
    EmotionLevel.SATISFIED: [
        "好的谢谢",
        "ok",
        "好的",
        "行",
        "可以",
        "明白了",
        "懂了",
        "了解了",
        "明白了谢谢",
    ],
    EmotionLevel.GRATEFUL: [
        "太感谢了",
        "谢谢你们",
        "很满意",
        "好评",
        "推荐给你们",
        "非常棒",
        "服务很好",
        "很到位",
        "帮忙解决了",
        "解决了",
        "感谢",
        "麻烦了",
    ],
}

_NEGATION_PATTERNS = re.compile(
    r"(不|没|非|别|无|本不是|没有|并非).{0,3}("
    r"骗子|骗钱|垃圾|曝光|举报|投诉|诈骗"
    r")"
)


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_PATTERNS.search(text))


# ═══════════════════════════════════════════════════════════════════════
# 情绪检测服务主体
# ═══════════════════════════════════════════════════════════════════════


class SentimentService:
    """简化的情绪检测器，仅基于 L1 规则关键词。

    使用方式：
        svc = SentimentService()
        result = await svc.detect("怎么还没发货，等了三天了")
        # → EmotionResult(level=DISAPPOINTED, escalate=True, source="L1:rule(2)")
    """

    def __init__(self) -> None:
        pass

    async def detect(
        self,
        text: str,
        *,
        session_id: str | None = None,
        skip_cloud: bool = True,
    ) -> EmotionResult:
        if not text or len(text.strip()) < 2:
            return EmotionResult(
                level=EmotionLevel.NEUTRAL,
                confidence=1.0,
                escalate=False,
                is_emergency=False,
                source="L0:short",
            )

        result = self._l1_classify(text)
        if result:
            return result

        if len(text) <= 5:
            return EmotionResult(
                level=EmotionLevel.NEUTRAL,
                confidence=0.7,
                escalate=False,
                is_emergency=False,
                source="L1:fallback_short",
            )

        return EmotionResult(
            level=EmotionLevel.NEUTRAL,
            confidence=0.5,
            escalate=False,
            is_emergency=False,
            source="L1:unclassified",
        )

    def _l1_classify(self, text: str) -> Optional[EmotionResult]:
        text_lower = text.lower()
        has_neg = _has_negation(text)

        for level in [
            EmotionLevel.EMERGENCY,
            EmotionLevel.ANGRY,
            EmotionLevel.DISAPPOINTED,
            EmotionLevel.ANXIOUS,
            EmotionLevel.SATISFIED,
            EmotionLevel.GRATEFUL,
        ]:
            keywords = EMOTION_KEYWORDS.get(level, [])
            hit_kw = [kw for kw in keywords if kw in text_lower]

            if hit_kw:
                if has_neg and level >= EmotionLevel.DISAPPOINTED:
                    continue

                return EmotionResult(
                    level=level,
                    confidence=min(0.95, 0.6 + 0.1 * len(hit_kw)),
                    escalate=level >= ESCALATE_THRESHOLD,
                    is_emergency=level == EmotionLevel.EMERGENCY,
                    keywords=hit_kw,
                    source=f"L1:rule({len(hit_kw)})",
                )

        return None


# ═══════════════════════════════════════════════════════════════════════
# 情绪驱动的 System Prompt 模板（供 react_agent 使用）
# ═══════════════════════════════════════════════════════════════════════

EMOTION_TONE_PROMPTS: Dict[str, str] = {
    "emergency": (
        "\n## 情绪感知\n"
        "用户情绪极为激动，可能涉及舆情风险。\n"
        "回复要求：\n"
        "1. 开头必须真诚道歉，承认问题\n"
        "2. 立即提供明确的升级渠道（人工客服/电话）\n"
        "3. 不要试图在对话中完全解决问题\n"
        "4. 不要使用'但是''不过'等转折词\n"
    ),
    "angry": (
        "\n## 情绪感知\n"
        "用户当前非常不满。\n"
        "回复要求：\n"
        "1. 先表示理解和歉意，再给方案\n"
        "2. 给出明确的行动步骤和时间承诺\n"
        "3. 避免推卸责任或解释过多流程细节\n"
        "4. 不要反问用户（如'您为什么不先看看说明书'）\n"
    ),
    "disappointed": (
        "\n## 情绪感知\n"
        "用户对服务体验感到失望。\n"
        "回复要求：\n"
        "1. 共情用户的等待/不便\n"
        "2. 主动让步（优惠券/加急/优先处理）若场景合适\n"
        "3. 用具体时间代替模糊承诺（如'今天18:00前'而非'尽快'）\n"
    ),
    "anxious": (
        "\n## 情绪感知\n"
        "用户较为焦急，希望快速得到结果。\n"
        "回复要求：\n"
        "1. 回复简洁高效，不绕圈子\n"
        "2. 优先给出核心信息（状态/时间节点）\n"
        "3. 结尾可以安抚一句（如'请放心，正在加急处理'）\n"
    ),
    "normal": "",
}

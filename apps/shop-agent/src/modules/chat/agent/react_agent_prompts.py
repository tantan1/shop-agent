"""
ReAct Agent 提示词与情绪映射。
"""
from __future__ import annotations

from src.modules.chat.core.sentiment_service import (
    EMOTION_TONE_PROMPTS,
    EmotionLevel,
)


def _emotion_to_tone_mode(level: EmotionLevel) -> str:
    """将情绪等级映射为 system prompt 语气模式。"""
    if level == EmotionLevel.EMERGENCY:
        return "emergency"
    if level == EmotionLevel.ANGRY:
        return "angry"
    if level == EmotionLevel.DISAPPOINTED:
        return "disappointed"
    if level == EmotionLevel.ANXIOUS:
        return "anxious"
    return "normal"


_REACT_SYSTEM_PROMPT = """你是客服助手。

## 工具使用
- 查询订单、物流、余额、优惠券或退货：**必须调用对应工具**
- 知识库查询：使用 knowledge_search
- 将工具结果整理为自然回复

## 规则
- **不要用相同参数重复调用同一个工具**
- knowledge_search 首次结果不相关时，换关键词重新搜索
- **request-return 是终端操作**：调用一次后直接告知结果
"""

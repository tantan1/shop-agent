"""对话历史摘要生成器。

长对话超过 token 预算时，使用 LLM 生成摘要保留核心上下文，
替代简单截断，减少关键信息丢失。
"""
from __future__ import annotations

from typing import List, Optional

from src.modules.chat.core.llm_service import LLMService
from src.shared.logger import APILogger

logger = APILogger("conversation_summarizer")

# 摘要 prompt：保留关键实体（订单号、商品名、用户偏好、问题状态）
_SUMMARIZE_PROMPT = """\
请将以下对话历史压缩为简洁摘要，严格保留以下关键信息：
- 订单号、快递单号、商品名称
- 用户已确认的偏好/地址/支付方式
- 当前问题状态（如"正在申请退款"、"已查询物流"）
- 未解决的诉求或待确认事项

要求：
1. 摘要长度控制在 200 字以内
2. 使用第三人称客观描述
3. 不要添加原文没有的信息
4. 如果对话很短或无实质内容，返回"无重要历史"

对话历史：
{chat_history}
"""


class ConversationSummarizer:
    """对话历史摘要器（LLM 驱动）。"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self._llm = llm_service or LLMService.get_instance()

    async def summarize_if_needed(
        self,
        messages: List[dict],
        max_tokens: int = 2000,
        token_budget_ratio: float = 0.3,
    ) -> str:
        """如果历史消息超过 token 预算，生成摘要；否则返回格式化历史。

        Args:
            messages: 对话消息列表 [{"role": "user/assistant", "content": "..."}]
            max_tokens: 允许的最大 token 数
            token_budget_ratio: 摘要目标长度占总预算的比例

        Returns:
            摘要字符串或格式化历史字符串
        """
        if not messages:
            return ""

        # 先估算原始历史的 token 数
        history_text = self._format_messages(messages)
        estimator = get_token_estimator()
        estimated_tokens = estimator.estimate(history_text)

        if estimated_tokens <= max_tokens:
            # 未超预算，直接返回格式化历史
            return history_text

        # 超预算：生成摘要
        logger.info(
            "对话历史超预算，生成摘要",
            total_messages=len(messages),
            estimated_tokens=estimated_tokens,
            budget=max_tokens,
        )

        # 保留最近 3 轮完整 + 更早的摘要
        recent_messages = messages[-6:]  # 最近 3 轮（user+assistant）
        older_messages = messages[:-6]

        summary = ""
        if older_messages:
            older_text = self._format_messages(older_messages)
            summary = await self._generate_summary(older_text)

        # 拼接：摘要 + 最近 3 轮
        recent_text = self._format_messages(recent_messages)
        combined = f"【历史摘要】\n{summary}\n\n【最近对话】\n{recent_text}"

        # 二次校验：如果摘要+近期仍超预算，截断近期
        combined_tokens = estimator.estimate(combined)
        if combined_tokens > max_tokens:
            # 按字符比例截断近期部分
            ratio = max_tokens / combined_tokens
            max_chars = int(len(recent_text) * ratio * 0.9)
            recent_text = recent_text[:max_chars] + "..."
            combined = f"【历史摘要】\n{summary}\n\n【最近对话】\n{recent_text}"

        return combined

    async def _generate_summary(self, history_text: str) -> str:
        """调用 LLM 生成摘要。"""
        prompt = _SUMMARIZE_PROMPT.format(chat_history=history_text)
        try:
            summary = await self._llm.chat_qwen(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return summary.strip() or "无重要历史"
        except Exception as e:
            logger.warning(f"对话摘要生成失败，回退到截断: {str(e)[:100]}")
            # 降级：直接截断
            return history_text[:500] + "..."

    def _format_messages(self, messages: List[dict]) -> str:
        """将消息列表格式化为可读文本。"""
        parts = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


def get_token_estimator():
    """获取 token 预估器（延迟加载，避免循环 import）。"""
    from src.core.token_estimator import get_token_estimator as _get
    return _get()

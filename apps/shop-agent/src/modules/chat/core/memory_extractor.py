"""
记忆提取器：从对话中提取结构化记忆信息
"""
import json
import re
from typing import Any, Dict, List

from src.modules.chat.core.llm_service import LLMService
from src.shared.logger import APILogger

logger = APILogger("memory_extractor")


MEMORY_EXTRACTION_PROMPT = """\
从以下客服对话中提取关键记忆信息，以 JSON 数组返回。

提取规则：
1. 只提取对未来对话有长期价值的信息
2. 不要提取一次性信息（如"好的"、"谢谢"）
3. 用户偏好（尺码、颜色、品牌、价格敏感度）→ type: preference
4. 订单信息（订单号、商品、物流状态）→ type: order
5. 投诉信息（问题描述、处理状态）→ type: complaint
6. 解决方案（用户接受的方案）→ type: resolution
7. 未解决事项（待确认、待处理）→ type: pending
8. 下一步行动（用户/客服需要后续执行的动作）→ type: next_action

每条记忆包含：
- type: 记忆类型
- label: 简短标签（如"尺码偏好"、"订单 12345"）
- value: 记忆内容
- importance: 1-5 分（5=必须记住，1=可遗忘）
- metadata: 扩展字段（如 {"order_id": "12345"}）

对话历史：
{chat_history}

当前用户输入：
{user_message}

返回格式：
[{{"type": "preference", "label": "尺码偏好", "value": "用户偏好 XL", "importance": 4, "metadata": {{}}}}]
"""


class MemoryExtractor:
    """LLM 记忆提取器"""

    def __init__(self, llm_service: LLMService):
        self._llm = llm_service

    async def extract(
        self, chat_history: List[Dict[str, str]], user_message: str
    ) -> List[Dict[str, Any]]:
        """从对话历史中提取结构化记忆

        Args:
            chat_history: 对话历史 [{role: "user"/"assistant", content: "..."}]
            user_message: 当前用户输入

        Returns:
            提取的记忆列表 [{type, label, value, importance, metadata}]
        """
        try:
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                chat_history=self._format_history(chat_history),
                user_message=user_message,
            )
            response = await self._llm.ainvoke(prompt)
            memories = self._parse_response(response)
            logger.debug(f"提取到 {len(memories)} 条记忆")
            return memories
        except Exception as e:
            logger.error(f"记忆提取失败: {e}")
            return []

    def _format_history(self, chat_history: List[Dict[str, str]]) -> str:
        """格式化对话历史"""
        lines = []
        for msg in chat_history[-20:]:  # 只取最近 20 轮
            role = "用户" if msg.get("role") == "user" else "客服"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的 JSON 数组"""
        try:
            # 提取 JSON 数组
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group(0))
            if not isinstance(data, list):
                return []
            # 验证每条记忆的必填字段
            valid_memories = []
            for item in data:
                if not all(k in item for k in ("type", "label", "value", "importance")):
                    continue
                valid_memories.append({
                    "type": item["type"],
                    "label": item["label"],
                    "value": item["value"],
                    "importance": int(item.get("importance", 3)),
                    "metadata": item.get("metadata", {}),
                })
            return valid_memories
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"解析记忆提取结果失败: {e}, raw: {response[:200]}")
            return []

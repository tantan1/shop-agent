"""
Mock LLM 适配器

用于测试和验证抽象协议，不依赖任何框架。
LLM_ADAPTER_TYPE=mock 时通过本适配器/LLMService 仿真延迟与错误率（压测 0 Token 消耗）。
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Dict, List, Type

from pydantic import BaseModel

from src.modules.chat.config import chat_config
from src.modules.chat.core.abstract.llm_client import LLMClient


class MockLLMAdapter(LLMClient):
    """Mock LLM 适配器（支持延迟/错误率仿真）"""

    def _simulate(self) -> None:
        """按配置仿真延迟与错误。"""
        latency_min = getattr(chat_config, "mock_llm_latency_min", 100) / 1000.0
        latency_max = getattr(chat_config, "mock_llm_latency_max", 500) / 1000.0
        error_rate = getattr(chat_config, "mock_llm_error_rate", 0.01)

        # 非事件循环上下文（如 ThreadPool）时退化为 time.sleep
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            import time

            time.sleep(random.uniform(latency_min, latency_max))
        else:
            # 仅简单同步路径用；异步场景由 LLMService._mock_simulate 处理
            pass

        if random.random() < error_rate:
            raise TimeoutError("Mock LLM timeout (simulated)")

    def _is_react_context(self, messages: List[Dict[str, str]]) -> bool:
        """检测是否为 ReAct Agent 上下文（包含工具定义的 system prompt）。"""
        if not messages:
            return False
        system_msg = messages[0].get("content", "") if messages[0].get("role") == "system" else ""
        # 检查英文和中文关键词
        keywords = ["tools", "function", "tool", "工具", "函数"]
        return any(kw in system_msg.lower() for kw in keywords)

    def _extract_order_id(self, text: str) -> str:
        """从用户消息中提取订单号。"""
        # 匹配常见订单号格式
        patterns = [
            r'WB\d{10,}',  # WB202409010077
            r'订单号[：:\s]*(\S+)',
            r'order[_\s]?id[：:\s]*(\S+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        return "WB202409010077"  # 默认测试订单号

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, tools: List[Dict] = None, **kwargs) -> str:
        await asyncio.sleep(
            random.uniform(
                getattr(chat_config, "mock_llm_latency_min", 100) / 1000.0,
                getattr(chat_config, "mock_llm_latency_max", 500) / 1000.0,
            )
        )
        # error_rate = getattr(chat_config, "mock_llm_error_rate", 0.01)
        # if random.random() < error_rate:
        #     raise TimeoutError("Mock LLM timeout (simulated)")

        last_user_message = messages[-1]["content"] if messages else ""

        # 检查是否已经有工具结果
        has_tool_result = any(msg.get("role") == "tool" for msg in messages)
        
        # ReAct Agent 上下文：返回结构化工具调用
        # 判断条件：有绑定的 tools 且没有工具结果，或者 system prompt 包含工具关键词
        has_tools = tools and len(tools) > 0
        is_react = (has_tools and not has_tool_result) or self._is_react_context(messages)
        
        if is_react:
            # 根据用户意图选择工具
            if any(kw in last_user_message for kw in ["退货", "退款", "申请退"]):
                order_id = self._extract_order_id(last_user_message)
                tool_call = {
                    "tool_calls": [
                        {
                            "name": "request-return",
                            "args": {"order_id": order_id, "reason": "用户申请退货"},
                            "id": "call_mock_1",
                            "type": "tool_call"
                        }
                    ]
                }
                return json.dumps(tool_call, ensure_ascii=False)
            elif any(kw in last_user_message for kw in ["查订单", "订单详情", "查单"]):
                order_id = self._extract_order_id(last_user_message)
                tool_call = {
                    "tool_calls": [
                        {
                            "name": "query-order",
                            "args": {"order_id": order_id},
                            "id": "call_mock_1",
                            "type": "tool_call"
                        }
                    ]
                }
                return json.dumps(tool_call, ensure_ascii=False)
            elif any(kw in last_user_message for kw in ["查物流", "物流", "发货"]):
                order_id = self._extract_order_id(last_user_message)
                tool_call = {
                    "tool_calls": [
                        {
                            "name": "check-shipping",
                            "args": {"order_id": order_id},
                            "id": "call_mock_1",
                            "type": "tool_call"
                        }
                    ]
                }
                return json.dumps(tool_call, ensure_ascii=False)
            elif any(kw in last_user_message for kw in ["查余额", "余额"]):
                tool_call = {
                    "tool_calls": [
                        {
                            "name": "check-balance",
                            "args": {},
                            "id": "call_mock_1",
                            "type": "tool_call"
                        }
                    ]
                }
                return json.dumps(tool_call, ensure_ascii=False)
            elif any(kw in last_user_message for kw in ["优惠券", "优惠码", "券"]):
                tool_call = {
                    "tool_calls": [
                        {
                            "name": "coupon-inquiry",
                            "args": {"coupon_code": "TEST"},
                            "id": "call_mock_1",
                            "type": "tool_call"
                        }
                    ]
                }
                return json.dumps(tool_call, ensure_ascii=False)
            elif any(kw in last_user_message for kw in ["政策", "规则", "怎么退", "退货条件"]):
                tool_call = {
                    "tool_calls": [
                        {
                            "name": "knowledge_search",
                            "args": {"query": last_user_message},
                            "id": "call_mock_1",
                            "type": "tool_call"
                        }
                    ]
                }
                return json.dumps(tool_call, ensure_ascii=False)
            elif has_tools:
                # 有 tools 但意图不明，调用第一个 tool
                # 尝试从用户消息中提取参数
                first_tool = tools[0]
                tool_name = first_tool.get("function", {}).get("name", "unknown")
                
                # 简单的参数提取：如果用户消息包含 query=xxx
                args = {}
                if "query=" in last_user_message:
                    # 简单提取 query= 后面的内容
                    import re
                    match = re.search(r'query=([^\s]+)', last_user_message)
                    if match:
                        args["query"] = match.group(1)
                
                tool_call = {
                    "tool_calls": [
                        {
                            "name": tool_name,
                            "args": args,
                            "id": "call_mock_1",
                            "type": "tool_call"
                        }
                    ]
                }
                return json.dumps(tool_call, ensure_ascii=False)

        # 普通对话：返回模拟回复
        return f"[Mock LLM] 收到消息：{last_user_message[:50]}..."

    async def chat_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[BaseModel],
        temperature: float = 0.0,
        **kwargs,
    ) -> BaseModel:
        return output_schema()

    async def embed(self, text: str) -> List[float]:
        return [0.1] * 768

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 768 for _ in texts]

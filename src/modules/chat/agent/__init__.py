"""
Agent 模块
支持多领域通用Agent架构
"""

from src.modules.chat.agent.executor import GeneralAgentExecutor, HospitalAgentExecutor
from src.modules.chat.agent.schemas import (
    HospitalChatRequest,
    HospitalChatResponse,
    AgentStepResult,
    SafetyCheckResult,
    QuestionRewriteResult,
    RetrievalResult,
)
from src.modules.chat.agent.prompts import PromptTemplateManager

__all__ = [
    # 执行器
    "GeneralAgentExecutor",  # 通用执行器
    "HospitalAgentExecutor",  # 向后兼容，别名
    # Schema
    "HospitalChatRequest",
    "HospitalChatResponse",
    "AgentStepResult",
    "SafetyCheckResult",
    "QuestionRewriteResult",
    "RetrievalResult",
    # 模板管理器
    "PromptTemplateManager",
]

"""Agent 执行步骤基类与上下文。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from src.modules.chat.agent.schemas import AgentStepResult
from src.modules.chat.schemas import ChatRequest


@dataclass
class AgentContext:
    """Agent 执行上下文，封装所有步骤共享的依赖与状态。"""

    request: ChatRequest
    domain: str
    config: Any  # AgentConfig
    llm_service: Any  # LLMService
    embedding_service: Any = None
    milvus_service: Any = None
    redis_cache_service: Any = None
    langfuse_handler: Any = None
    conversation_id: str = ""
    user_id: str = ""
    question_embedding: Optional[List[float]] = None
    graph_context: str = ""
    memory_context: str = ""  # MRAG 记忆检索结果
    retrieval_query: Optional[str] = None  # 多轮融合后的检索 query；为空则回退 request.message
    experiment_overrides: Any = None  # GrowthBook 实验变体 pipeline_overrides（注入载体；守卫式 get_override 读取）

    def get_override(self, name: str, default: Any) -> Any:
        """实验 override 优先；无 override 或该字段为 None 时回退 default。

        用显式 None 判断（非 `or`），避免 False/0 被误覆盖导致实验无法关闭开关。
        """
        if self.experiment_overrides is None:
            return default
        v = getattr(self.experiment_overrides, name, None)
        return default if v is None else v


class BaseStep(ABC):
    """Agent 执行步骤抽象基类（模板方法模式）。"""

    step_name: str = ""
    step_order: int = 0

    @abstractmethod
    async def execute(self, ctx: AgentContext) -> AgentStepResult:
        """执行单步逻辑，返回步骤结果。"""
        ...

    def _skip_result(self, ctx: AgentContext) -> AgentStepResult:
        """步骤未启用时的默认跳过结果。"""
        return AgentStepResult(
            step_name=self.step_name,
            step_order=self.step_order,
            input_data={},
            output_data={},
            status="skipped",
        )

    def _error_result(self, ctx: AgentContext, error: Exception) -> AgentStepResult:
        """步骤执行异常时的错误结果。"""
        return AgentStepResult(
            step_name=self.step_name,
            step_order=self.step_order,
            input_data={},
            status="failed",
            error_message=str(error),
        )

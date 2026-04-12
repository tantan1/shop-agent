"""
医院客服 Agent 数据结构定义
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# 导入共享的 Schema
from src.modules.chat.schemas import (
    HospitalChatRequest,
    HospitalChatResponse,
    HospitalAgentConfig,
)

__all__ = [
    "HospitalChatRequest",
    "HospitalChatResponse", 
    "HospitalAgentConfig",
    "AgentStepResult",
    "SafetyCheckResult",
    "QuestionRewriteResult",
    "RetrievalResult",
    "AgentStreamEvent",
]


class AgentStepResult(BaseModel):
    """Agent 单步执行结果"""
    step_name: str = Field(..., description="步骤名称")
    step_order: int = Field(..., description="步骤顺序")
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="输入数据")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="输出数据")
    status: str = Field(..., description="执行状态: success, failed, skipped")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    duration_ms: Optional[int] = Field(default=None, description="执行耗时(毫秒)")
    timestamp: datetime = Field(default_factory=datetime.now, description="执行时间")


class SafetyCheckResult(BaseModel):
    """安全审查结果"""
    is_safe: bool = Field(..., description="是否安全")
    risk_level: str = Field(default="low", description="风险等级: low, medium, high")
    risk_categories: List[str] = Field(default_factory=list, description="风险类别列表")
    warning_message: Optional[str] = Field(default=None, description="警告信息")
    can_proceed: bool = Field(..., description="是否可以继续流程")


class QuestionRewriteResult(BaseModel):
    """问题重写结果"""
    original_question: str = Field(..., description="原始问题")
    rewritten_queries: List[str] = Field(..., description="重写后的查询列表")
    keywords: List[str] = Field(default_factory=list, description="提取的关键词")


class RetrievalResult(BaseModel):
    """检索结果"""
    query: str = Field(..., description="检索查询")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="检索到的文档")
    scores: List[float] = Field(default_factory=list, description="相似度分数")


class AgentStreamEvent(BaseModel):
    """流式事件"""
    event_type: str = Field(..., description="事件类型: step_start, step_complete, token, error, done")
    step_name: Optional[str] = Field(default=None, description="步骤名称")
    content: Optional[str] = Field(default=None, description="内容")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")

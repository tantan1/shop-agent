"""
医院客服 Agent 数据结构定义
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# 导入共享的 Schema
from src.modules.chat.schemas import (
    ChatRequest,
    ChatResponse,
    AgentConfig,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "AgentConfig",
    "AgentStepResult",
    "SafetyCheckResult",
    "QuestionRewriteResult",
    "RetrievalResult",
    "AgentStreamEvent",
    # 结构化输出 Schema
    "SafetyCheckSchema",
    "ComplianceCheckSchema",
    "QuestionClassifySchema",
    "QuestionRewriteSchema",
    "AnswerQualitySchema",
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


# =============================================================================
# 结构化输出 Schema 注册表
# =============================================================================

class SafetyCheckSchema(BaseModel):
    """安全审查结果 Schema"""
    is_safe: bool = Field(default=True, description="是否安全")
    risk_level: Literal["low", "medium", "high"] = Field(default="low", description="风险等级")
    risk_categories: List[str] = Field(default_factory=list, description="风险类别列表")
    warning_message: Optional[str] = Field(default=None, description="警告信息")


class ComplianceCheckSchema(BaseModel):
    """合规检查结果 Schema"""
    compliant: bool = Field(default=True, description="是否合规")
    issue: Optional[str] = Field(default="", description="不合规原因：违禁品/敏感内容/广告等")


class QuestionClassifySchema(BaseModel):
    """问题分类结果 Schema"""
    category: Literal["咨询", "投诉", "建议", "技术支持", "其他"] = Field(default="咨询", description="问题类别")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")


class QuestionRewriteSchema(BaseModel):
    """问题改写结果 Schema（用于关键词提取）"""
    rewritten_queries: List[str] = Field(default_factory=list, description="重写后的检索关键词列表")


class AnswerQualitySchema(BaseModel):
    """答案质量评估 Schema"""
    is_solved: bool = Field(..., description="问题是否已解决")
    quality_score: int = Field(..., ge=0, le=10, description="质量评分 0-10")
    reasons: List[str] = Field(default_factory=list, description="评估理由")
    improvement_suggestion: Optional[str] = Field(default=None, description="改进建议")


# Schema 注册表：用于 with_structured_output
STRUCTURED_OUTPUT_SCHEMAS: Dict[str, type[BaseModel]] = {
    "SafetyCheckSchema": SafetyCheckSchema,
    "ComplianceCheckSchema": ComplianceCheckSchema,
    "QuestionClassifySchema": QuestionClassifySchema,
    "QuestionRewriteSchema": QuestionRewriteSchema,
    "AnswerQualitySchema": AnswerQualitySchema,
}


def get_structured_schema(schema_name: str) -> Optional[type[BaseModel]]:
    """获取结构化输出 Schema"""
    return STRUCTURED_OUTPUT_SCHEMAS.get(schema_name)

"""API 请求/响应模型。"""
from typing import List, Optional
from pydantic import BaseModel


class SubmitGoldRequest(BaseModel):
    canonical_tid: str
    correct_tool: Optional[str] = None           # 单意图正例(将与 correct_tools 合并)
    correct_tools: Optional[List[str]] = None    # 多意图正例(对应多个工具,§5.3 招5)
    rejected_tools: Optional[List[str]] = None  # 负例(被否定工具)
    reason: Optional[str] = None
    changed_by: Optional[str] = "annotator"
    label_source: Optional[str] = "full"         # full / confirm(预填一致)


class SubmitCorrectionRequest(BaseModel):
    """来自 chat 纠正按钮 / 运营台(§2.5)。经 conversation_id 定位。"""
    conversation_id: Optional[str] = None
    trace_id: Optional[str] = None
    original_tool: Optional[str] = None
    correct_tool: Optional[str] = None
    rejected_tools: Optional[List[str]] = None
    content: Optional[str] = None
    changed_by: Optional[str] = "chat_correction"


class SubmitFeedbackRequest(BaseModel):
    """点赞 / 点踩(§2.5)。"""
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    signal: str                                  # like / dislike
    reason: Optional[str] = None
    context: Optional[dict] = None


class CandidateRow(BaseModel):
    canonical_tid: str
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    query_text: Optional[str] = None
    category: Optional[str] = None
    is_multi_intent: bool = False
    candidate_tools: List[str] = []
    available_tools: List[str] = []
    llm_suggested_tool: Optional[str] = None
    top1_tool: Optional[str] = None
    margin: Optional[float] = None
    freq: int = 1
    label_source: Optional[str] = None


class SyncResponse(BaseModel):
    pulled: int
    auto_labeled: int
    spotcheck: int
    watermark: Optional[str] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    total: int
    unlabeled_in_queue: int
    auto_labeled: int
    manual_labeled: int
    spotcheck: int
    feedback_signals: int
    relabel_count: int

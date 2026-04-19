from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    message: str = Field(..., description="聊天信息")


class ChatQueryResponse(BaseModel):
    message: str = Field(..., description="回复消息")
    relevant_documents: Optional[List[str]] = Field(default=None, description="相关文档内容")
    document_count: Optional[int] = Field(default=0, description="检索到的文档数量")


class InsertDocumentRequest(BaseModel):
    document: str = Field(..., description="要插入的文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")


class BatchInsertRequest(BaseModel):
    documents: List[InsertDocumentRequest] = Field(..., description="批量插入的文档列表")


class InsertDocumentResponse(BaseModel):
    status: str = Field(..., description="插入状态")
    chunks_inserted: int = Field(..., description="插入的块数量")
    total_characters: int = Field(..., description="总字符数")


class BatchInsertResponse(BaseModel):
    status: str = Field(..., description="批量插入状态")
    documents_processed: int = Field(..., description="处理的文档数量")
    estimated_chunks: int = Field(..., description="预估的块数量")


class FileUploadMetadata(BaseModel):
    """文件上传元数据"""
    source: Optional[str] = Field(default="file_upload", description="文档来源")
    file_name: Optional[str] = Field(default=None, description="原始文件名")
    batch_id: Optional[str] = Field(default=None, description="批次ID")


# =============================================================================
# 医院客服 Agent 相关 Schema
# =============================================================================

class HospitalChatRequest(BaseModel):
    """医院客服聊天请求"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户消息，限制 1-2000 字符"
    )
    conversation_id: Optional[str] = Field(default=None, description="对话ID，用于多轮对话")
    stream: bool = Field(default=True, description="是否启用流式输出")


class HospitalChatResponse(BaseModel):
    """医院客服聊天响应"""
    message: str = Field(..., description="回复消息")
    conversation_id: str = Field(..., description="对话ID")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="处理步骤详情")
    documents_used: List[str] = Field(default_factory=list, description="使用的参考文档")
    safety_passed: bool = Field(default=True, description="安全审查是否通过")
    stream_available: bool = Field(default=True, description="是否支持流式输出")
    cache_hit: bool = Field(default=False, description="是否命中缓存")


class HospitalAgentConfig(BaseModel):
    """医院客服 Agent 配置"""
    model_name: str = Field(default="doubao-pro-251215", description="模型名称")
    temperature: float = Field(default=0.3, description="温度参数")
    max_retries: int = Field(default=3, description="最大重试次数")
    timeout: int = Field(default=60, description="超时时间(秒)")
    top_k: int = Field(default=5, description="检索返回的文档数量")
    enable_history: bool = Field(default=True, description="是否启用对话历史")
    max_history_turns: int = Field(default=5, description="最大历史对话轮次")


__all__ = [
    "ChatQueryRequest",
    "ChatQueryResponse",
    "InsertDocumentRequest",
    "BatchInsertRequest",
    "InsertDocumentResponse",
    "BatchInsertResponse",
    "FileUploadMetadata",
    "HospitalChatRequest",
    "HospitalChatResponse",
    "HospitalAgentConfig",
]


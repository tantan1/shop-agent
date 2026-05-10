from typing import Optional, List, Dict, Any, Type
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

class ChatRequest(BaseModel):
    """通用Agent聊天请求"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户消息，限制 1-2000 字符"
    )
    conversation_id: Optional[str] = Field(default=None, description="对话ID，用于多轮对话")
    stream: bool = Field(default=True, description="是否启用流式输出")
    domain: str = Field(default="ecommerce", description="业务领域: medical/ecommerce/customer_service/general")


class ChatResponse(BaseModel):
    """通用Agent聊天响应"""
    message: str = Field(..., description="回复消息")
    conversation_id: str = Field(..., description="对话ID")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="处理步骤详情")
    documents_used: List[str] = Field(default_factory=list, description="使用的参考文档")
    safety_passed: bool = Field(default=True, description="安全审查是否通过")
    stream_available: bool = Field(default=True, description="是否支持流式输出")
    cache_hit: bool = Field(default=False, description="是否命中缓存")
    domain: str = Field(default="ecommerce", description="处理的业务领域")
    status: str = Field(default="completed", description="执行状态: completed | waiting_for_confirmation | cancelled")
    interrupt_data: Optional[Dict[str, Any]] = Field(default=None, description="中断数据（人在回路暂停信息）")


class RefundConfirmRequest(BaseModel):
    """退款确认请求（人在回路）"""
    conversation_id: str = Field(..., description="对话ID（与原始聊天请求相同）")
    confirm: bool = Field(default=True, description="是否确认退款: true=批准, false=拒绝")
    remark: Optional[str] = Field(default=None, description="审批备注")


class AgentConfig(BaseModel):
    """Agent通用配置"""
    model_name: str = Field(default="doubao-pro-251215", description="模型名称")
    temperature: float = Field(default=0.3, description="温度参数")
    max_retries: int = Field(default=3, description="最大重试次数")
    timeout: int = Field(default=60, description="超时时间(秒)")
    top_k: int = Field(default=5, description="检索返回的文档数量")
    enable_history: bool = Field(default=True, description="是否启用对话历史")
    max_history_turns: int = Field(default=5, description="最大历史对话轮次")
    domain: str = Field(default="medical", description="业务领域")


# =============================================================================
# 商品嵌入与搜索相关 Schema
# =============================================================================

class ItemEmbedRequest(BaseModel):
    """商品嵌入请求（单个）"""
    item_id: str = Field(..., description="商品ID")
    title: str = Field(..., description="商品标题/文本内容")


class BatchItemEmbedRequest(BaseModel):
    """批量商品嵌入请求"""
    items: List[ItemEmbedRequest] = Field(
        ...,
        description="商品列表",
        max_length=1000  # 限制最大批量大小
    )
    batch_id: Optional[str] = Field(default=None, description="批次ID")


class ItemEmbedResponse(BaseModel):
    """商品嵌入响应"""
    status: str = Field(..., description="嵌入状态")
    message: Optional[str] = Field(default=None, description="状态消息或错误信息")
    items_processed: int = Field(default=0, description="处理的商品数量")
    items_inserted: int = Field(default=0, description="成功插入的商品数量")
    failed_items: List[str] = Field(default_factory=list, description="失败的商品ID列表")


class ItemSearchRequest(BaseModel):
    """商品搜索请求"""
    query: str = Field(
        ...,
        description="搜索查询",
        min_length=1,
        max_length=500
    )
    top_k: int = Field(default=10, description="返回商品数量", ge=1, le=100)


class ItemSearchResult(BaseModel):
    """商品搜索结果项"""
    item_id: str = Field(..., description="商品ID")
    content: str = Field(..., description="商品内容")
    score: float = Field(..., description="相关性分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ItemSearchResponse(BaseModel):
    """商品搜索响应"""
    query: str = Field(..., description="原始查询")
    total: int = Field(..., description="结果总数")
    items: List[ItemSearchResult] = Field(..., description="商品列表")


# =============================================================================
# 意图识别相关 Schema
# =============================================================================

class IntentResult(BaseModel):
    """意图识别结果"""
    intent: str = Field(default="rag_answer", description="意图: rag_answer | call_remote_api")
    action: Optional[str] = Field(default=None, description="远程API操作类型: query-order | check-shipping | request-return")
    params: Optional[Dict[str, Any]] = Field(default=None, description="远程API调用参数")
    # ---- 复杂性门控字段 ----
    similarity_score: Optional[float] = Field(default=None, description="FAISS 匹配的余弦相似度分(0~1)")
    complexity: Optional[str] = Field(default=None, description="查询复杂性: simple | multi_step | needs_agent")
    # simple: 单次 tool 调用即可；multi_step: 需要 tool+RAG 组合；needs_agent: 需要 LLM 自主规划
    complexity_reason: Optional[str] = Field(default=None, description="复杂性判定的依据说明")


# =============================================================================
# 意图参数抽取 Schema（小模型 structured output 专用）
# =============================================================================

class QueryOrderParams(BaseModel):
    """查订单——从用户 query 中提取的参数"""
    order_id: Optional[str] = Field(default=None, description="订单号，如 WB202405270001")
    phone: Optional[str] = Field(default=None, description="手机号后四位")
    status_filter: Optional[str] = Field(default=None, description="筛选状态: 待付款/已发货/派送中/已签收")


class CheckShippingParams(BaseModel):
    """查物流——从用户 query 中提取的参数"""
    tracking_number: Optional[str] = Field(default=None, description="快递单号，如 SF1234567890")
    order_id: Optional[str] = Field(default=None, description="订单号")


class RequestReturnParams(BaseModel):
    """退货退款——从用户 query 中提取的参数"""
    order_id: Optional[str] = Field(default=None, description="要退货的订单号")
    reason: Optional[str] = Field(default=None, description="退货原因: 质量问题/不想要/发错货/其他")


class CheckBalanceParams(BaseModel):
    """查余额——从用户 query 中提取的参数"""
    # 目前查余额不需要参数
    pass


class CouponInquiryParams(BaseModel):
    """查优惠券——从用户 query 中提取的参数"""
    coupon_type: Optional[str] = Field(default=None, description="券类型: 满减券/折扣券/运费券")


# 意图 → 参数 Schema 注册表
INTENT_PARAM_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "query-order":      QueryOrderParams,
    "check-shipping":   CheckShippingParams,
    "request-return":   RequestReturnParams,
    "check-balance":    CheckBalanceParams,
    "coupon-inquiry":   CouponInquiryParams,
}


# ---- 参数抽取提示词 ----
# 每个意图对应一个精简的 prompt 模板，告诉小模型从 query 里提取什么

PARAM_EXTRACTION_PROMPTS: Dict[str, str] = {
    "query-order": (
        "从用户消息中提取查询订单的参数。\n"
        "- order_id: 订单号通常是数字组合(如 202405270001)\n"
        "- phone: 手机号后四位\n"
        "- status_filter: 用户想看的订单状态(待付款/已发货/派送中/已签收)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "check-shipping": (
        "从用户消息中提取查询物流的参数。\n"
        "- tracking_number: 快递单号(如 SF1234567890、YT123456、JD001234567)\n"
        "- order_id: 订单号\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "request-return": (
        "从用户消息中提取退货退款的参数。\n"
        "- order_id: 要退货的订单号\n"
        "- reason: 退货原因(质量问题/不想要/发错货/与描述不符/其他)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
    "check-balance": (
        "用户查询账户余额或积分，当前无需额外参数。"
    ),
    "coupon-inquiry": (
        "从用户消息中提取查询优惠券的参数。\n"
        "- coupon_type: 券类型(满减券/折扣券/运费券/通用)\n"
        "- 如果没有提到某个参数，留空即可"
    ),
}


__all__ = [
    "ChatQueryRequest",
    "ChatQueryResponse",
    "InsertDocumentRequest",
    "BatchInsertRequest",
    "InsertDocumentResponse",
    "BatchInsertResponse",
    "FileUploadMetadata",
    "ChatRequest",
    "ChatResponse",
    "RefundConfirmRequest",
    "AgentConfig",
    "ItemEmbedRequest",
    "BatchItemEmbedRequest",
    "ItemEmbedResponse",
    "ItemSearchRequest",
    "ItemSearchResult",
    "ItemSearchResponse",
    "IntentResult",
]


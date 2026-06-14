from typing import Optional, List, Dict, Any, Type, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="聊天信息（1-4000字符）"
    )


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
        max_length=5000,
        description="用户消息，限制 1-5000 字符。超过内置 token 预算会自动智能截断并提示用户"
    )
    conversation_id: Optional[str] = Field(default=None, description="对话ID，用于多轮对话")
    stream: bool = Field(default=True, description="是否启用流式输出")
    domain: str = Field(default="ecommerce", description="业务领域: medical/ecommerce/customer_service/general")
    # ── A/B 实验字段（可选，未传则自动分配）──
    experiment_group: Optional[str] = Field(default=None, description="A/B实验组标识（如 control / treatment_A），留空则由服务端自动分配")


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
    input_truncated: bool = Field(default=False, description="用户输入是否因过长被自动截断")
    input_original_tokens: Optional[int] = Field(default=None, description="原始输入的 token 数（截断前）")
    input_truncated_tokens: Optional[int] = Field(default=None, description="截断后的 token 数")
    # ── A/B 实验反馈 ──
    experiment_group: Optional[str] = Field(default=None, description="实际命中的实验组标识（供前端日志收集）")


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
    order_id: Optional[str] = Field(
        default=None,
        description="订单号，如 WB202405270001",
        json_schema_extra={"semantic": "order_id"},
    )
    phone: Optional[str] = Field(
        default=None,
        description="手机号后四位",
        json_schema_extra={"semantic": "phone"},
    )
    status_filter: Optional[str] = Field(
        default=None,
        description="筛选状态: 待付款/已发货/派送中/已签收",
        json_schema_extra={"semantic": "order_status"},
    )


class CheckShippingParams(BaseModel):
    """查物流——从用户 query 中提取的参数"""
    tracking_number: Optional[str] = Field(
        default=None,
        description="快递单号，如 SF1234567890",
        json_schema_extra={"semantic": "tracking_number"},
    )
    order_id: Optional[str] = Field(
        default=None,
        description="订单号",
        json_schema_extra={"semantic": "order_id"},
    )


class RequestReturnParams(BaseModel):
    """退货退款——从用户 query 中提取的参数"""
    order_id: Optional[str] = Field(
        default=None,
        description="要退货的订单号",
        json_schema_extra={"semantic": "order_id"},
    )
    reason: Optional[str] = Field(
        default=None,
        description="退货原因: 质量问题/不想要/发错货/其他",
        json_schema_extra={"semantic": "return_reason"},
    )


class CheckBalanceParams(BaseModel):
    """查余额——从用户 query 中提取的参数"""
    # 目前查余额不需要参数
    pass


class CouponInquiryParams(BaseModel):
    """查优惠券——从用户 query 中提取的参数"""
    coupon_type: Optional[str] = Field(
        default=None,
        description="券类型: 满减券/折扣券/运费券",
        json_schema_extra={"semantic": "coupon_type"},
    )


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
    "ExperimentCreateRequest",
    "ExperimentPauseRequest",
    "ExperimentValidateRequest",
    # ── A2A Schemas ──
    "AgentSkill",
    "AgentAuth",
    "AgentRateLimitInfo",
    "AgentEndpoint",
    "AgentCapabilities",
    "AgentCard",
    "A2ATaskRequest",
    "A2ATaskStatusResponse",
    "A2ATaskListResponse",
    "WebhookSubscriptionRequest",
    "WebhookSubscriptionResponse",
    "A2AConversationSummary",
    "A2AConversationListResponse",
    "A2AHealthResponse",
]


# =============================================================================
# A/B 实验管理 API Schema
# =============================================================================

class VariantSchema(BaseModel):
    """实验变体请求"""
    name: str = Field(..., description="变体名称: control / treatment_A")
    variant_type: str = Field(default="treatment", description="变体类型: control | treatment")
    traffic_percent: float = Field(..., ge=0, le=100, description="流量百分比（0-100）")
    pipeline_overrides: Dict[str, Any] = Field(default_factory=dict, description="管道覆盖配置")
    description: str = Field(default="", description="变体描述")


class SafetyGuardSchema(BaseModel):
    """安全护栏请求"""
    metric: str = Field(..., description="监控指标: escalation_rate | error_rate | p99_latency_ms 等")
    threshold: float = Field(..., description="阈值")
    comparison: str = Field(default="gt", description="比较方式: gt | lt | pct_change")
    window_seconds: int = Field(default=300, ge=60, description="监控窗口秒数")
    action: str = Field(default="pause", description="触发动作: pause | stop")


class ExperimentCreateRequest(BaseModel):
    """创建/更新实验请求"""
    id: str = Field(..., min_length=3, max_length=64, description="实验唯一ID，如 exp_reranker_001")
    name: str = Field(..., description="实验名称")
    description: str = Field(default="", description="实验描述")
    variants: List[VariantSchema] = Field(..., min_length=2, description="至少包含对照组+实验组")
    safety_guards: List[SafetyGuardSchema] = Field(default_factory=list, description="安全护栏配置")
    domains: List[str] = Field(default_factory=lambda: ["ecommerce"], description="生效领域")
    owner: str = Field(default="", description="实验负责人")


class ExperimentPauseRequest(BaseModel):
    """暂停/恢复实验请求"""
    id: str = Field(..., description="实验ID")
    status: str = Field(default="paused", description="目标状态: paused | running | stopped")


class ExperimentValidateRequest(BaseModel):
    """分流均匀性验证请求"""
    id: str = Field(..., description="实验ID")
    sample_user_count: int = Field(default=10000, ge=1000, le=100000, description="模拟用户数（1000-100000）")


# =============================================================================
# Agent Card (A2A protocol) Schema — 增强版
# =============================================================================

class AgentSkill(BaseModel):
    """Agent Card 中的单个技能声明。
    与 A2A 协议兼容：id + name + description + tags + examples
    """
    id: str = Field(..., description="技能唯一标识（如 query-order）")
    name: str = Field(..., description="技能展示名（如 订单查询）")
    description: str = Field(..., description="技能描述")
    tags: List[str] = Field(default_factory=list, description="标签")
    examples: List[str] = Field(default_factory=list, description="触发示例")


class AgentAuth(BaseModel):
    """Agent 认证方式声明（A2A 协议增强）。"""
    type: Literal["api_key_header", "bearer_token", "none"] = Field(
        default="api_key_header", description="认证类型"
    )
    header_name: str = Field(default="X-API-Key", description="认证 Header 名称")
    scopes: List[str] = Field(default_factory=list, description="权限范围")

    class Config:
        populate_by_name = True


class AgentRateLimitInfo(BaseModel):
    """Agent 速率限制信息（A2A 协议增强）。"""
    requests_per_minute: int = Field(default=60, description="每分钟请求上限")
    burst: int = Field(default=10, description="突发并发容量")


class AgentEndpoint(BaseModel):
    """Agent 可用端点声明（A2A 协议增强）。"""
    method: str = Field(..., description="HTTP 方法: GET/POST/PATCH/DELETE")
    path: str = Field(..., description="端点路径（如 /a2a/tasks/send）")
    identifier: str = Field(default="", description="端点标识（用于唯一引用）")
    description: str = Field(default="", description="端点描述")
    content_type: str = Field(default="application/json", description="请求/响应 Content-Type")
    requires_auth: bool = Field(default=True, description="是否需要认证")


class AgentCapabilities(BaseModel):
    """Agent 能力声明（A2A 协议标准字段）。"""
    streaming: bool = Field(default=True, description="是否支持流式输出")
    push_notifications: bool = Field(
        default=False, alias="pushNotifications", description="是否支持推送通知"
    )
    async_tasks: bool = Field(
        default=True, alias="asyncTasks", description="是否支持异步任务"
    )

    class Config:
        populate_by_name = True


class AgentCard(BaseModel):
    """Agent Card —— A2A 协议的能力发现入口。

    标准端点：GET /.well-known/agent-card.json  或  GET /api/v1/chatagent/agent/card
    协议参考：A2A (Agent-to-Agent) specification
    """
    name: str = Field(..., description="Agent 名称")
    description: str = Field(..., description="Agent 描述")
    url: str = Field(..., description="Agent 访问地址")
    provider: str = Field(default="shop-agent", description="提供方标识")
    version: str = Field(default="1.0.0", description="Agent 版本")
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = Field(default_factory=list, description="技能列表")
    authentication: Optional[AgentAuth] = Field(default=None, description="认证方式声明")
    rate_limit: Optional[AgentRateLimitInfo] = Field(default=None, description="速率限制信息")
    endpoints: List[AgentEndpoint] = Field(default_factory=list, description="可用端点列表")
    documentation_url: Optional[str] = Field(default=None, description="文档地址")

    class Config:
        populate_by_name = True


# =============================================================================
# A2A 异步任务 API Schema
# =============================================================================

class A2ATaskRequest(BaseModel):
    """A2A 异步任务提交请求。"""
    skill_id: Optional[str] = Field(
        default=None, description="指定 skill（不传则自动路由意图）"
    )
    message: str = Field(
        ..., min_length=1, max_length=5000, description="用户自然语言消息"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="上下文透传（键值对）"
    )
    callback_url: Optional[str] = Field(
        default=None, description="任务完成后回调的 Webhook URL"
    )
    conversation_id: Optional[str] = Field(default=None, description="对话 ID（多轮）")
    domain: str = Field(default="ecommerce", description="业务领域")


class A2ATaskStatusResponse(BaseModel):
    """A2A 任务状态查询响应。"""
    task_id: str = Field(..., description="任务唯一 ID")
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = Field(
        ..., description="任务状态"
    )
    result: Optional[str] = Field(default=None, description="任务结果（completed 时填充）")
    error: Optional[str] = Field(default=None, description="错误信息（failed 时填充）")
    created_at: str = Field(..., description="创建时间 (ISO 8601)")
    started_at: Optional[str] = Field(default=None, description="开始时间 (ISO 8601)")
    completed_at: Optional[str] = Field(default=None, description="完成时间 (ISO 8601)")
    conversation_id: Optional[str] = Field(default=None, description="关联对话 ID")
    domain: str = Field(default="ecommerce", description="业务领域")


class A2ATaskListResponse(BaseModel):
    """A2A 任务列表响应。"""
    total: int = Field(..., description="任务总数")
    tasks: List[A2ATaskStatusResponse] = Field(default_factory=list, description="任务列表")


# =============================================================================
# A2A Webhook Schema
# =============================================================================

class WebhookSubscriptionRequest(BaseModel):
    """Webhook 订阅请求。"""
    url: str = Field(..., description="回调 URL")
    events: List[str] = Field(
        default_factory=lambda: ["task.completed", "task.failed"],
        description="订阅事件列表: task.completed | task.failed | skill.error",
    )
    secret: Optional[str] = Field(
        default=None, description="HMAC 签名密钥（用于验证回调来源）"
    )
    ttl_seconds: Optional[int] = Field(
        default=86400, ge=60, le=604800, description="有效期秒数（默认 24h，最大 7 天）"
    )


class WebhookSubscriptionResponse(BaseModel):
    """Webhook 订阅响应。"""
    subscription_id: str = Field(..., description="订阅唯一 ID")
    url: str = Field(..., description="回调 URL")
    events: List[str] = Field(default_factory=list, description="订阅事件")
    created_at: str = Field(..., description="创建时间 (ISO 8601)")
    expires_at: Optional[str] = Field(default=None, description="过期时间 (ISO 8601)")


# =============================================================================
# A2A Conversation Schema
# =============================================================================

class A2AConversationSummary(BaseModel):
    """A2A 对话摘要。"""
    conversation_id: str = Field(..., description="对话 ID")
    message_count: int = Field(default=0, description="消息数量")
    created_at: str = Field(..., description="创建时间 (ISO 8601)")
    last_active_at: str = Field(..., description="最近活跃时间 (ISO 8601)")
    domain: str = Field(default="ecommerce", description="业务领域")
    status: str = Field(default="active", description="状态: active | archived")


class A2AConversationListResponse(BaseModel):
    """A2A 对话列表响应。"""
    total: int = Field(..., description="对话总数")
    conversations: List[A2AConversationSummary] = Field(default_factory=list)


class A2AHealthResponse(BaseModel):
    """A2A 专用健康检查响应。"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        default="healthy", description="整体健康状态"
    )
    agent_name: str = Field(default="shop-agent", description="Agent 名称")
    version: str = Field(default="1.0.0", description="版本")
    uptime_seconds: float = Field(default=0, description="启动秒数")
    dependencies: Dict[str, str] = Field(
        default_factory=dict, description="依赖状态映射"
    )
    skills_count: int = Field(default=0, description="已加载 Skill 数量")
    mcp_enabled: bool = Field(default=False, description="MCP Server 是否启用")


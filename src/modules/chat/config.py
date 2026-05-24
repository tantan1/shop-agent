from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Type

from src.core.config import config


class ChatConfig:
    """聊天服务配置 - 使用全局配置"""
    
    # 从全局配置获取值
    tongyi_api_key: str = config.TONGYI_API_KEY
    chat_model: str = config.CHAT_MODEL
    tool_selector_model: str = config.TOOL_SELECTOR_MODEL  # P1 工具选择器专用轻量模型（更快更便宜）
    # P1 本地模型（设置后优先用本地模型替代云端 API）
    tool_selector_local_model: str = config.TOOL_SELECTOR_LOCAL_MODEL
    tool_selector_local_device: str = config.TOOL_SELECTOR_LOCAL_DEVICE
    tool_selector_local_load_in_4bit: bool = config.TOOL_SELECTOR_LOCAL_LOAD_IN_4BIT
    temperature: float = 0.7
    
    # 本地小模型配置（参数抽取用）
    local_param_model: str = config.LOCAL_PARAM_MODEL
    local_param_device: str = config.LOCAL_PARAM_DEVICE
    local_param_max_tokens: int = config.LOCAL_PARAM_MAX_TOKENS
    local_param_load_in_4bit: bool = config.LOCAL_PARAM_LOAD_IN_4BIT
    
    embedding_model: str = config.EMBEDDING_MODEL
    embedding_provider: str = config.EMBEDDING_PROVIDER  # local | volcengine
    # 向量数据库提供者: milvus | pgvector
    vector_store_provider: str = config.VECTOR_STORE_PROVIDER
    # Milvus 配置
    milvus_host: str = config.MILVUS_HOST
    milvus_port: int = config.MILVUS_PORT
    milvus_collection_name: str = "chat_embeddings"
    # PostgreSQL pgvector 配置
    pgvector_host: str = config.PGVECTOR_HOST
    pgvector_port: int = config.PGVECTOR_PORT
    pgvector_db: str = config.PGVECTOR_DB
    pgvector_user: str = config.PGVECTOR_USER
    pgvector_password: str = config.PGVECTOR_PASSWORD
    pgvector_table: str = config.PGVECTOR_TABLE
    
    @property
    def embedding_dimension(self) -> int:
        """根据 provider 返回对应维度"""
        _DIMS = {
            "BAAI/bge-small-zh-v1.5": 512,
            "BAAI/bge-large-zh-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "doubao-embedding-vision-251215": 2048,
            "doubao-embedding-text-240915": 1024,
        }
        return _DIMS.get(self.embedding_model, 1024)
    
    # Redis 缓存配置
    redis_vector_enabled: bool = True
    redis_vector_threshold: float = 0.85  # 相似度阈值
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""  # Redis 密码（留空则无密码连接）
    cache_expire_days: int = 7  # 缓存过期天数
    
    # 火山引擎配置
    volcengine_api_key: str = config.VOLCENGINE_API_KEY
    volcengine_embedding_endpoint: str = config.VOLCENGINE_EMBEDDING_ENDPOINT
    
    # 默认领域
    default_domain: str = "ecommerce"
    
    # 同义词归一化（L1+L2 静态匹配，零 LLM 成本）
    synonym_normalize_enabled: bool = True
    # L3 LLM 归一化（默认关闭，需 API 调用）
    synonym_normalize_llm_enabled: bool = config.SYNONYM_NORMALIZE_LLM_ENABLED


chat_config = ChatConfig()


# =============================================================================
# 通用Agent配置系统
# =============================================================================

class AgentStepConfig(BaseModel):
    """单步骤配置"""
    enabled: bool = True
    name: str = ""
    prompt_template_key: str = ""  # 提示词模板key
    output_format: Literal["text", "json"] = "text"  # 输出格式：text=普通文本，json=结构化JSON
    response_schema: Optional[str] = None  # Pydantic Schema 名称（用于结构化输出）
    timeout_ms: int = 30000
    model: Optional[str] = None  # 可指定使用特定模型

    class Config:
        extra = "allow"  # 允许额外字段


class AgentConfig(BaseModel):
    """通用Agent配置"""
    domain: str = "general"  # 领域标识
    name: str = "通用助手"
    description: str = ""
    
    # 步骤配置
    step1: AgentStepConfig = Field(default_factory=lambda: AgentStepConfig(
        name="问题理解",
        prompt_template_key="step1_understand"
    ))
    step2: AgentStepConfig = Field(default_factory=lambda: AgentStepConfig(
        name="内容审查",
        prompt_template_key="step2_review"
    ))
    step3: AgentStepConfig = Field(default_factory=lambda: AgentStepConfig(
        name="知识检索",
        prompt_template_key="step3_retrieve"
    ))
    step4: AgentStepConfig = Field(default_factory=lambda: AgentStepConfig(
        name="回答生成",
        prompt_template_key="step4_generate"
    ))
    
    # 检索配置
    top_k: int = 5
    max_history_turns: int = 10
    max_retrieval_queries: int = 3
    retrieval_score_threshold: float = 0.0  # Milvus RRF 融合分数最低阈值（0=不过滤，低于此阈值的文档被丢弃）
    rrf_k: int = 60  # RRFRanker k 参数（越小高分权重越大，推荐范围 10~100）
    relevance_filter_enabled: bool = False  # 是否启用 LLM 相关性过滤（过滤语义不相关的检索结果）
    rerank_enabled: bool = True  # 是否启用 BGE-Reranker 重排序
    rerank_threshold: float = 0.3  # Rerank 相关性分数最低阈值（0~1，低于此值的文档被丢弃）
    rerank_top_k: int = 10  # Rerank 后保留的文档数量
    rerank_initial_top_k: int = 10  # 从 Milvus 先多取几条给 Reranker 筛（降低到10以减少CrossEncoder推理开销）
    
    # 缓存配置
    cache_enabled: bool = True
    cache_threshold: float = 0.85
    
    # 低质量模式检测
    low_quality_patterns: List[str] = Field(default_factory=lambda: [
        "暂无相关检索结果",
        "抱歉，服务暂时繁忙",
        "我无法回答",
        "无法提供",
        "未查询到"
    ])
    
    # 安全审查敏感词（JSON解析失败时的兜底检测）
    sensitive_keywords: List[str] = Field(default_factory=lambda: ["诊断", "处方", "胸痛"])

    # 内容过滤服务开关（规则引擎，零 LLM 成本）
    content_filter_enabled: bool = Field(default=True, description="是否启用内容安全过滤服务")
    content_filter_output_block: bool = Field(default=True, description="输出过滤是否硬阻断（True=命中拦截）")

    # Step2 安全审查本地小模型（省 API 费，非合规才升级云端 LLM 复核）
    step2_safety_local_model_enabled: bool = Field(default=False, description="Step2 是否启用本地小模型优先")

    class Config:
        extra = "allow"  # 允许额外字段


# =============================================================================
# 预定义领域配置
# =============================================================================

def _create_medical_config() -> AgentConfig:
    """创建医疗领域配置"""
    return AgentConfig(
        domain="medical",
        name="医疗助手",
        description="医院智能客服助手，为患者提供准确、安全的就医咨询",
        step1=AgentStepConfig(
            enabled=True,
            name="问题改写",
            prompt_template_key="medical_step1_rewrite",
            output_format="json",
            response_schema="QuestionRewriteSchema"
        ),
        step2=AgentStepConfig(
            enabled=True,
            name="安全审查",
            prompt_template_key="medical_step2_safety",
            output_format="json",  # 启用结构化输出
            response_schema="SafetyCheckSchema"
        ),
        step3=AgentStepConfig(
            enabled=True,
            name="医学知识检索",
            prompt_template_key="medical_step3_retrieve"
        ),
        step4=AgentStepConfig(
            enabled=True,
            name="医疗回答生成",
            prompt_template_key="medical_step4_generate"
        ),
        top_k=5,
        max_history_turns=10,
        max_retrieval_queries=1,  # 只检索最优查询，避免多次 embedding API 调用
        retrieval_score_threshold=0.003,  # RRF 融合分数阈值
        rrf_k=40,  # RRF 融合 k：越小越精确，越大越全（40=偏精确）
        rerank_initial_top_k=15,  # 多拉几条候选给 Reranker 精选
        cache_enabled=True,
        cache_threshold=0.85,
        low_quality_patterns=[
            "暂无相关检索结果",
            "抱歉，服务暂时繁忙",
            "我无法回答",
            "无法提供",
            "未查询到"
        ],
        sensitive_keywords=[
            "诊断", "处方", "胸痛", "开药", "用药", "手术",
            "自杀", "自残", "安乐死",
        ],
    )


def _create_ecommerce_config() -> AgentConfig:
    """创建电商领域配置"""
    return AgentConfig(
        domain="ecommerce",
        name="电商助手",
        description="电商客服助手，为用户提供商品咨询、订单处理等服务",
        step1=AgentStepConfig(
            enabled=False,
            name="需求分析",
            prompt_template_key="ecommerce_step1_analyze"
        ),
        step2=AgentStepConfig(
            enabled=False,
            name="合规检查",
            prompt_template_key="ecommerce_step2_compliance",
            output_format="json",
            response_schema="ComplianceCheckSchema"
        ),
        step3=AgentStepConfig(
            enabled=True,
            name="商品检索",
            prompt_template_key="ecommerce_step3_query"
        ),
        step4=AgentStepConfig(
            enabled=True,
            name="商品推荐",
            prompt_template_key="ecommerce_step4_generate"
        ),
        top_k=10,
        max_history_turns=5,
        max_retrieval_queries=5,
        retrieval_score_threshold=0.005,  # RRF 融合分数阈值（过滤低相关性文档）
        rrf_k=40,  # 电商场景突出高分商品
        rerank_enabled=True,  # 启用 Rerank 重排序
        rerank_threshold=0.3,  # 低于 0.3 的文档丢弃
        rerank_top_k=5,
        rerank_initial_top_k=20,  # Milvus 先召回 20 条让 Reranker 筛
        cache_enabled=True,
        cache_threshold=0.80,
        low_quality_patterns=[
            "抱歉，暂无相关商品",
            "服务暂时繁忙",
            "我无法为您推荐"
        ],
        sensitive_keywords=[
            "毒品", "枪支", "管制刀具", "色情", "赌博", "诈骗",
            "翻墙", "VPN", "个人信息", "身份证号", "银行卡号",
            "刷单", "刷好评", "假货", "假币",
        ],
    )


def _create_customer_service_config() -> AgentConfig:
    """创建客服领域配置"""
    return AgentConfig(
        domain="customer_service",
        name="客服助手",
        description="通用客服助手，处理用户咨询、投诉、建议等",
        step1=AgentStepConfig(
            enabled=True,
            name="问题分类",
            prompt_template_key="service_step1_classify",
            output_format="json",
            response_schema="QuestionClassifySchema"
        ),
        step2=AgentStepConfig(
            enabled=True,
            name="敏感检测",
            prompt_template_key="service_step2_sensitive",
            output_format="json",
            response_schema="SafetyCheckSchema"
        ),
        step3=AgentStepConfig(
            enabled=True,
            name="知识库检索",
            prompt_template_key="service_step3_knowledge"
        ),
        step4=AgentStepConfig(
            enabled=True,
            name="回复生成",
            prompt_template_key="service_step4_reply"
        ),
        top_k=5,
        max_history_turns=20,
        max_retrieval_queries=3,
        cache_enabled=True,
        cache_threshold=0.85,
        sensitive_keywords=[
            "色情", "暴力", "恐怖", "政治敏感", "侮辱", "攻击",
            "个人信息", "银行卡", "密码", "非法集会",
        ],
    )


def _create_general_config() -> AgentConfig:
    """创建通用配置"""
    return AgentConfig(
        domain="general",
        name="通用助手",
        description="通用AI助手，提供各类咨询和帮助",
        step1=AgentStepConfig(
            enabled=True,
            name="问题理解",
            prompt_template_key="general_step1_understand"
        ),
        step2=AgentStepConfig(
            enabled=True,
            name="内容审查",
            prompt_template_key="general_step2_review"
        ),
        step3=AgentStepConfig(
            enabled=True,
            name="信息检索",
            prompt_template_key="general_step3_retrieve"
        ),
        step4=AgentStepConfig(
            enabled=True,
            name="回答生成",
            prompt_template_key="general_step4_generate"
        ),
        top_k=5,
        max_history_turns=10,
        max_retrieval_queries=1,  # 只检索最优查询，避免多次 embedding API 调用
        retrieval_score_threshold=0.003,  # RRF 融合分数阈值
        rrf_k=40,  # RRF 融合 k：越小越精确，越大越全（40=偏精确）
        rerank_initial_top_k=15,  # 多拉几条候选给 Reranker 精选
        cache_enabled=True,
        cache_threshold=0.85,
        sensitive_keywords=[
            "色情", "暴力", "政治敏感", "非法",
            "诈骗", "赌博", "毒品",
        ],
    )


# 领域配置注册表
DOMAIN_CONFIGS: Dict[str, AgentConfig] = {
    "medical": _create_medical_config(),
    "ecommerce": _create_ecommerce_config(),
    "customer_service": _create_customer_service_config(),
    "general": _create_general_config(),
}


def get_agent_config(domain: str) -> AgentConfig:
    """获取指定领域的Agent配置"""
    return DOMAIN_CONFIGS.get(domain, _create_general_config())


def get_available_domains() -> List[Dict[str, str]]:
    """获取所有可用领域列表"""
    return [
        {"domain": config.domain, "name": config.name, "description": config.description}
        for config in DOMAIN_CONFIGS.values()
    ]


def register_domain_config(domain: str, config: AgentConfig) -> None:
    """注册新的领域配置"""
    DOMAIN_CONFIGS[domain] = config

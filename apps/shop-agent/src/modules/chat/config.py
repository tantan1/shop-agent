from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import config


class ChatConfig:
    """聊天服务配置 - 使用全局配置"""

    # 从全局配置获取值
    tongyi_api_key: str = config.TONGYI_API_KEY
    chat_model: str = config.CHAT_MODEL
    tool_selector_model: str = config.TOOL_SELECTOR_MODEL  # P2 工具选择器专用轻量模型（更快更便宜）
    # P2 本地模型（设置后优先用本地模型替代云端 API）
    tool_selector_local_model: str = config.TOOL_SELECTOR_LOCAL_MODEL
    tool_selector_local_device: str = config.TOOL_SELECTOR_LOCAL_DEVICE
    tool_selector_local_load_in_4bit: bool = config.TOOL_SELECTOR_LOCAL_LOAD_IN_4BIT
    temperature: float = 0.7

    # 本地小模型配置（参数抽取用）
    local_param_model: str = config.LOCAL_PARAM_MODEL
    local_param_device: str = config.LOCAL_PARAM_DEVICE
    local_param_max_tokens: int = config.LOCAL_PARAM_MAX_TOKENS
    local_param_load_in_4bit: bool = config.LOCAL_PARAM_LOAD_IN_4BIT

    # 本地小模型后端（transformers | ollama）+ Ollama 端点/模型名
    local_model_backend: str = config.LOCAL_MODEL_BACKEND
    ollama_base_url: str = config.OLLAMA_BASE_URL
    ollama_param_model: str = config.OLLAMA_PARAM_MODEL
    ollama_tool_selector_model: str = config.OLLAMA_TOOL_SELECTOR_MODEL
    ollama_timeout: int = config.OLLAMA_TIMEOUT
    # vLLM 小模型端点（LOCAL_MODEL_BACKEND=vllm 时生效）
    vllm_base_url: str = config.VLLM_BASE_URL
    vllm_param_model: str = config.VLLM_PARAM_MODEL
    vllm_tool_selector_model: str = config.VLLM_TOOL_SELECTOR_MODEL
    vllm_timeout: int = config.VLLM_TIMEOUT

    # P2 本地工具选择（小模型专项辅助层）。
    # 设计决策（承接架构评审）：本地 1.7B 只作为「意图加权软过滤」的补充确认，
    # 而非通用兜底。因此：
    #  - enable_p2_local_classify=False 时完全跳过本地模型，直接用 P1 结果；
    #  - 即便开启，也只在候选工具数 > p2_local_classify_min_candidates（默认 4）
    #    时才介入，避免对少量候选做无意义的二次推理。
    enable_p2_local_classify: bool = config.ENABLE_P2_LOCAL_CLASSIFY
    p2_local_classify_min_candidates: int = config.P2_LOCAL_CLASSIFY_MIN_CANDIDATES

    embedding_model: str = config.EMBEDDING_MODEL
    # Embedding 后端: local=进程内 sentence-transformers | ollama=进程外 Ollama API | vllm=vLLM bge-m3 直连
    embedding_provider: str = config.EMBEDDING_PROVIDER
    # Ollama embedding 模型名（EMBEDDING_PROVIDER=ollama 时生效）
    ollama_embedding_model: str = config.OLLAMA_EMBEDDING_MODEL
    # vLLM bge-m3 嵌入（EMBEDDING_PROVIDER=vllm 时生效，直连）
    vllm_embedding_base_url: str = config.VLLM_EMBEDDING_BASE_URL
    vllm_embedding_model: str = config.VLLM_EMBEDDING_MODEL

    # 重排后端: local=进程内 | vllm=远程调 vLLM bge-reranker（直连）
    reranker_provider: str = config.RERANKER_PROVIDER
    vllm_rerank_base_url: str = config.VLLM_RERANK_BASE_URL
    vllm_rerank_model: str = config.VLLM_RERANK_MODEL

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
        """根据 provider 返回对应维度（支持完整路径匹配）"""
        _DIMS = {
            "BAAI/bge-m3": 1024,
        }
        model = self.embedding_model
        # 处理完整路径（如 ./models/BAAI/bge-m3）
        for key, dim in _DIMS.items():
            if model.endswith(key) or model == key:
                return dim
        return 1024

    # Redis 缓存配置（优先读环境变量，回退默认值，对齐 k8s 注入 REDIS_HOST/REDIS_AUTH）
    redis_vector_enabled: bool = True
    redis_vector_threshold: float = 0.85  # 相似度阈值
    redis_host: str = config.REDIS_HOST
    redis_port: int = config.REDIS_PORT_NUM
    redis_password: str = config.REDIS_PASSWORD  # Redis 密码（留空则无密码连接）
    cache_expire_days: int = 7  # 缓存过期天数
    embedding_cache_version: str = (
        "v1"  # 语义缓存版本：embedding 模型/维度变更时递增，使旧向量缓存失效
    )

    # 火山引擎配置
    volcengine_api_key: str = config.VOLCENGINE_API_KEY

    # 默认领域
    default_domain: str = "ecommerce"

    # 同义词归一化（L1+L2 静态匹配，零 LLM 成本）
    synonym_normalize_enabled: bool = True
    # L3 LLM 归一化（默认关闭，需 API 调用）
    synonym_normalize_llm_enabled: bool = config.SYNONYM_NORMALIZE_LLM_ENABLED

    # TTS 配置
    tts_provider: str = config.TTS_PROVIDER
    baidu_tts_api_key: str = config.BAIDU_TTS_API_KEY
    baidu_tts_secret_key: str = config.BAIDU_TTS_SECRET_KEY

    # 数字人配置
    avatar_provider: str = config.AVATAR_PROVIDER
    baidu_avatar_api_key: str = config.BAIDU_AVATAR_API_KEY
    baidu_avatar_secret_key: str = config.BAIDU_AVATAR_SECRET_KEY

    # 速率限制（压测可调）
    global_rate_limit: int = config.GLOBAL_RATE_LIMIT
    chat_rate_limit: int = config.CHAT_RATE_LIMIT

    # LLM 适配器类型（langchain | mock）；mock 用于压测 0 Token 消耗
    LLM_ADAPTER_TYPE: str = config.LLM_ADAPTER_TYPE

    # Mock LLM 仿真（LLM_ADAPTER_TYPE=mock 时生效；压测 0 Token 消耗）
    mock_llm_latency_min: int = config.MOCK_LLM_LATENCY_MIN
    mock_llm_latency_max: int = config.MOCK_LLM_LATENCY_MAX
    mock_llm_error_rate: float = config.MOCK_LLM_ERROR_RATE
    mock_llm_output_tokens: int = config.MOCK_LLM_OUTPUT_TOKENS

    # 遗忘机制配置
    forgetting_enabled: bool = True
    forgetting_archive_days: int = 90
    forgetting_delete_days: int = 365
    forgetting_check_interval_hours: int = 24


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

    model_config = ConfigDict(extra="allow")  # 允许额外字段


class AgentConfig(BaseModel):
    """通用Agent配置"""

    domain: str = "general"  # 领域标识
    name: str = "通用助手"
    description: str = ""

    # 步骤配置
    step1: AgentStepConfig = Field(
        default_factory=lambda: AgentStepConfig(
            name="问题理解", prompt_template_key="step1_understand"
        )
    )
    step2: AgentStepConfig = Field(
        default_factory=lambda: AgentStepConfig(name="内容审查", prompt_template_key="step2_review")
    )
    step3: AgentStepConfig = Field(
        default_factory=lambda: AgentStepConfig(
            name="知识检索", prompt_template_key="step3_retrieve"
        )
    )
    step4: AgentStepConfig = Field(
        default_factory=lambda: AgentStepConfig(
            name="回答生成", prompt_template_key="step4_generate"
        )
    )

    # 检索配置
    top_k: int = 5
    max_history_turns: int = 10
    long_conversation_threshold: int = 10  # 长对话阈值：轮次超过该值才启动 L3 每 5 轮兜底提取（电商对话多在 5-10 轮内解决）
    max_history_chars: int = 300  # 回放进窗口时单条历史消息的最大字符数（截断以控制 token 消耗）
    max_retrieval_queries: int = 3
    retrieval_score_threshold: float = (
        0.0  # Milvus RRF 融合分数最低阈值（0=不过滤，低于此阈值的文档被丢弃）
    )
    rrf_k: int = 60  # RRFRanker k 参数（越小高分权重越大，推荐范围 10~100）
    relevance_filter_enabled: bool = False  # 是否启用 LLM 相关性过滤（过滤语义不相关的检索结果）
    rerank_enabled: bool = True  # 是否启用 BGE-Reranker 重排序
    rerank_threshold: float = 0.3  # Rerank 相关性分数最低阈值（0~1，低于此值的文档被丢弃）
    rerank_top_k: int = 10  # Rerank 后保留的文档数量
    rerank_initial_top_k: int = (
        10  # 从 Milvus 先多取几条给 Reranker 筛（降低到10以减少CrossEncoder推理开销）
    )

    # 缓存配置
    cache_enabled: bool = True
    cache_threshold: float = 0.85

    # 低质量模式检测
    low_quality_patterns: List[str] = Field(
        default_factory=lambda: [
            "暂无相关检索结果",
            "抱歉，服务暂时繁忙",
            "我无法回答",
            "无法提供",
            "未查询到",
        ]
    )

    # 安全审查敏感词（JSON解析失败时的兜底检测）
    sensitive_keywords: List[str] = Field(default_factory=lambda: ["诊断", "处方", "胸痛"])

    # 内容过滤服务开关（规则引擎，零 LLM 成本）
    content_filter_enabled: bool = Field(default=True, description="是否启用内容安全过滤服务")
    content_filter_output_block: bool = Field(
        default=True, description="输出过滤是否硬阻断（True=命中拦截）"
    )

    # Step2 安全审查本地小模型（省 API 费，非合规才升级云端 LLM 复核）
    step2_safety_local_model_enabled: bool = Field(
        default=False, description="Step2 是否启用本地小模型优先"
    )

    model_config = ConfigDict(extra="allow")  # 允许额外字段


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
            response_schema="QuestionRewriteSchema",
        ),
        step2=AgentStepConfig(
            enabled=True,
            name="安全审查",
            prompt_template_key="medical_step2_safety",
            output_format="json",  # 启用结构化输出
            response_schema="SafetyCheckSchema",
        ),
        step3=AgentStepConfig(
            enabled=True, name="医学知识检索", prompt_template_key="medical_step3_retrieve"
        ),
        step4=AgentStepConfig(
            enabled=True, name="医疗回答生成", prompt_template_key="medical_step4_generate"
        ),
        top_k=5,
        max_history_turns=10,
        max_history_chars=300,
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
            "未查询到",
        ],
        sensitive_keywords=[
            "诊断",
            "处方",
            "胸痛",
            "开药",
            "用药",
            "手术",
            "自杀",
            "自残",
            "安乐死",
        ],
    )


def _create_ecommerce_config() -> AgentConfig:
    """创建电商领域配置"""
    return AgentConfig(
        domain="ecommerce",
        name="电商助手",
        description="电商客服助手，为用户提供商品咨询、订单处理等服务",
        step1=AgentStepConfig(
            enabled=False, name="需求分析", prompt_template_key="ecommerce_step1_analyze"
        ),
        step2=AgentStepConfig(
            enabled=False,
            name="合规检查",
            prompt_template_key="ecommerce_step2_compliance",
            output_format="json",
            response_schema="ComplianceCheckSchema",
        ),
        step3=AgentStepConfig(
            enabled=True, name="商品检索", prompt_template_key="ecommerce_step3_query"
        ),
        step4=AgentStepConfig(
            enabled=True, name="商品推荐", prompt_template_key="ecommerce_step4_generate"
        ),
        top_k=10,
        max_history_turns=5,
        max_history_chars=300,
        max_retrieval_queries=5,
        retrieval_score_threshold=0.005,  # RRF 融合分数阈值（过滤低相关性文档）
        rrf_k=40,  # 电商场景突出高分商品
        rerank_enabled=True,  # 启用 Rerank 重排序
        rerank_threshold=0.3,  # 低于 0.3 的文档丢弃
        rerank_top_k=5,
        rerank_initial_top_k=20,  # Milvus 先召回 20 条让 Reranker 筛
        cache_enabled=True,
        cache_threshold=0.80,
        low_quality_patterns=["抱歉，暂无相关商品", "服务暂时繁忙", "我无法为您推荐"],
        sensitive_keywords=[
            "毒品",
            "枪支",
            "管制刀具",
            "色情",
            "赌博",
            "诈骗",
            "翻墙",
            "VPN",
            "个人信息",
            "身份证号",
            "银行卡号",
            "刷单",
            "刷好评",
            "假货",
            "假币",
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
            response_schema="QuestionClassifySchema",
        ),
        step2=AgentStepConfig(
            enabled=True,
            name="敏感检测",
            prompt_template_key="service_step2_sensitive",
            output_format="json",
            response_schema="SafetyCheckSchema",
        ),
        step3=AgentStepConfig(
            enabled=True, name="知识库检索", prompt_template_key="service_step3_knowledge"
        ),
        step4=AgentStepConfig(
            enabled=True, name="回复生成", prompt_template_key="service_step4_reply"
        ),
        top_k=5,
        max_history_turns=20,
        max_history_chars=300,
        max_retrieval_queries=3,
        cache_enabled=True,
        cache_threshold=0.85,
        sensitive_keywords=[
            "色情",
            "暴力",
            "恐怖",
            "政治敏感",
            "侮辱",
            "攻击",
            "个人信息",
            "银行卡",
            "密码",
            "非法集会",
        ],
    )


def _create_general_config() -> AgentConfig:
    """创建通用配置"""
    return AgentConfig(
        domain="general",
        name="通用助手",
        description="通用AI助手，提供各类咨询和帮助",
        step1=AgentStepConfig(
            enabled=True, name="问题理解", prompt_template_key="general_step1_understand"
        ),
        step2=AgentStepConfig(
            enabled=True, name="内容审查", prompt_template_key="general_step2_review"
        ),
        step3=AgentStepConfig(
            enabled=True, name="信息检索", prompt_template_key="general_step3_retrieve"
        ),
        step4=AgentStepConfig(
            enabled=True, name="回答生成", prompt_template_key="general_step4_generate"
        ),
        top_k=5,
        max_history_turns=10,
        max_history_chars=300,
        max_retrieval_queries=1,  # 只检索最优查询，避免多次 embedding API 调用
        retrieval_score_threshold=0.003,  # RRF 融合分数阈值
        rrf_k=40,  # RRF 融合 k：越小越精确，越大越全（40=偏精确）
        rerank_initial_top_k=15,  # 多拉几条候选给 Reranker 精选
        cache_enabled=True,
        cache_threshold=0.85,
        sensitive_keywords=[
            "色情",
            "暴力",
            "政治敏感",
            "非法",
            "诈骗",
            "赌博",
            "毒品",
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

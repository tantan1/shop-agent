from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from src.core.config import config


class ChatConfig:
    """聊天服务配置 - 使用全局配置"""
    
    # 从全局配置获取值
    tongyi_api_key: str = config.TONGYI_API_KEY
    chat_model: str = config.CHAT_MODEL
    temperature: float = 0.7
    embedding_model: str = config.EMBEDDING_MODEL
    milvus_host: str = config.MILVUS_HOST
    milvus_port: int = config.MILVUS_PORT
    milvus_collection_name: str = "chat_embeddings"
    embedding_dimension: int = 2048  # Doubao-embedding 维度
    
    # Redis 缓存配置
    redis_vector_enabled: bool = True
    redis_vector_threshold: float = 0.85  # 相似度阈值
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_expire_days: int = 7  # 缓存过期天数
    
    # 火山引擎配置
    volcengine_api_key: str = config.VOLCENGINE_API_KEY
    volcengine_embedding_endpoint: str = config.VOLCENGINE_EMBEDDING_ENDPOINT
    
    # 默认领域
    default_domain: str = "medical"


chat_config = ChatConfig()


# =============================================================================
# 通用Agent配置系统
# =============================================================================

class AgentStepConfig(BaseModel):
    """单步骤配置"""
    enabled: bool = True
    name: str = ""
    prompt_template_key: str = ""  # 提示词模板key
    timeout_ms: int = 30000
    model: Optional[str] = None  # 可指定使用特定模型


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
            prompt_template_key="medical_step1_rewrite"
        ),
        step2=AgentStepConfig(
            enabled=True,
            name="安全审查",
            prompt_template_key="medical_step2_safety"
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
        max_retrieval_queries=3,
        cache_enabled=True,
        cache_threshold=0.85,
        low_quality_patterns=[
            "暂无相关检索结果",
            "抱歉，服务暂时繁忙",
            "我无法回答",
            "无法提供",
            "未查询到"
        ]
    )


def _create_ecommerce_config() -> AgentConfig:
    """创建电商领域配置"""
    return AgentConfig(
        domain="ecommerce",
        name="电商助手",
        description="电商客服助手，为用户提供商品咨询、订单处理等服务",
        step1=AgentStepConfig(
            enabled=True,
            name="需求分析",
            prompt_template_key="ecommerce_step1_analyze"
        ),
        step2=AgentStepConfig(
            enabled=False,  # 电商场景可能不需要安全审查
            name="合规检查",
            prompt_template_key="ecommerce_step2_compliance"
        ),
        step3=AgentStepConfig(
            enabled=True,
            name="商品检索",
            prompt_template_key="ecommerce_step3_search"
        ),
        step4=AgentStepConfig(
            enabled=True,
            name="商品推荐",
            prompt_template_key="ecommerce_step4_recommend"
        ),
        top_k=10,
        max_history_turns=5,
        max_retrieval_queries=5,
        cache_enabled=True,
        cache_threshold=0.80,
        low_quality_patterns=[
            "抱歉，暂无相关商品",
            "服务暂时繁忙",
            "我无法为您推荐"
        ]
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
            prompt_template_key="service_step1_classify"
        ),
        step2=AgentStepConfig(
            enabled=True,
            name="敏感检测",
            prompt_template_key="service_step2_sensitive"
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
        cache_threshold=0.85
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
            enabled=False,
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
        max_retrieval_queries=3,
        cache_enabled=True,
        cache_threshold=0.85
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

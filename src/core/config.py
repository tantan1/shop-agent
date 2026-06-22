from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # 应用配置
    DEBUG_MODE: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # 数据库配置
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "123456"
    DB_NAME: str = "fastapi_dev"

    # 日志配置
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "json"

    # 固定API密钥配置
    FIXED_API_KEY: str

    # 通义千问API配置
    TONGYI_API_KEY: str = ""
    
    # 火山引擎 Doubao API配置
    VOLCENGINE_API_KEY: str = ""
    
    # 聊天模型配置（云端模型，用于 Agent 回答生成等复杂任务）
    CHAT_MODEL: str = Field(default="")

    # P2 工具选择器专用模型（更轻量更快，qwen-turbo 延迟约为主模型 40%）
    TOOL_SELECTOR_MODEL: str = Field(default="")
    
    # P2 工具选择器本地模型路径（设置后优先用本地模型替代云端 API）
    # 推荐: Qwen2.5-1.5B-Instruct（速度和准确度的最佳平衡点）
    TOOL_SELECTOR_LOCAL_MODEL: str = ""  # 如 ./models/Qwen2.5-1.5B-Instruct
    TOOL_SELECTOR_LOCAL_DEVICE: str = "cpu"  # cpu | auto
    TOOL_SELECTOR_LOCAL_LOAD_IN_4BIT: bool = False
    
    # 本地小模型配置（用于参数抽取，transformers 直接加载，无需部署）
    LOCAL_PARAM_MODEL: str = Field(default="./models/Qwen2.5-0.5B-Instruct")  # 轻量级中文模型，~1GB，CPU 可跑
    LOCAL_PARAM_DEVICE: str = "auto"  # cpu | cuda | auto（auto 优先 GPU）
    LOCAL_PARAM_MAX_TOKENS: int = 256  # 参数抽取很短，256 足够
    LOCAL_PARAM_LOAD_IN_4BIT: bool = True  # 4bit 量化，节省内存（需 bitsandbytes）
    
    # Embedding 模型（本地 BGE/Sentence-Transformers）
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-zh-v1.5")

    # BGE-Reranker 本地模型路径（用于 RAG 检索结果重排序）
    # 优先从 ModelScope 本地缓存加载（国内秒下），不存在则回退 HuggingFace 自动下载
    RERANKER_LOCAL_MODEL_PATH: str = ""  # 如 C:/Users/.../modelscope/BAAI/bge-reranker-base
    
    # 向量数据库提供者: milvus | pgvector
    VECTOR_STORE_PROVIDER: str = "milvus"

    # Milvus向量数据库配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # PostgreSQL pgvector 配置（VECTOR_STORE_PROVIDER=pgvector 时生效）
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DB: str = "shop_agent"
    PGVECTOR_USER: str = "postgres"
    PGVECTOR_PASSWORD: str = "postgres"
    PGVECTOR_TABLE: str = "documents"
    # 远程业务API配置（意图识别触发远程调用时使用）
    REMOTE_API_BASE_URL: str = ""
    REMOTE_API_TIMEOUT: int = 10
    
    # 意图识别模式: local=关键词+向量(免费), llm=通义千问(精准)
    INTENT_RECOGNITION_MODE: str = "local"
    
    # 参数抽取模式: local=正则+关键词(免费,毫秒级), local_model=transformers本地小模型(免费,智能), llm=通义千问structured output(精准)
    PARAM_EXTRACTION_MODE: str = "local"
    
    # FAISS 意图向量匹配参数
    INTENT_VECTOR_SIMILARITY_THRESHOLD: float = 0.65  # 余弦相似度阈值（BGE归一化向量用内积）

    # 同义词归一化配置
    # L1+L2: 静态同义词表 + 文本标准化（默认开启，零LLM成本，零延迟）
    SYNONYM_NORMALIZE_ENABLED: bool = True
    # L3: LLM归一化（默认关闭，需API调用，约500-1500ms延迟，覆盖长尾表达）
    SYNONYM_NORMALIZE_LLM_ENABLED: bool = False

    # NebulaGraph 图数据库配置（商品关系图谱，增强 RAG 的结构化知识）
    NEBULA_GRAPH_ADDRS: str = "127.0.0.1:9669"  # graphd 地址，逗号分隔多地址
    NEBULA_USER: str = "root"
    NEBULA_PASSWORD: str = "nebula"
    NEBULA_SPACE: str = "shop_graph"  # 图空间名
    NEBULA_TIMEOUT: int = 3000  # 连接超时 ms
    NEBULA_POOL_SIZE: int = 4  # 连接池大小
    NEBULA_GRAPH_ENABLED: bool = True  # 是否启用图查询增强

    # Step2 输入安全审查本地小模型配置
    # 开启后 Step2 优先用本地小模型做合规分类（省 API 费），非合规才升级云端 LLM 复核
    STEP1_SAFETY_LOCAL_MODEL_ENABLED: bool = False

    # Token 预估器配置（用于 Token 消耗限流）
    # Qwen3 全系列共用 tokenizer，指向本地 tokenizer.json 即可
    TOKENIZER_PATH: str = "./models/Qwen3-1.7B/tokenizer.json"
    # Token 消耗限流默认值（每窗口 max_tokens）
    TOKEN_LIMIT_MAX_TOKENS: int = 100000  # 每分钟最大 token 消耗
    TOKEN_LIMIT_WINDOW_SECONDS: int = 60  # 窗口 60 秒
    TOKEN_LIMIT_ENABLED: bool = True  # 是否启用 token 消耗限流
    # 用户输入长度管控（基于 token 而非字符数，与 LLM 实际消耗一致）
    MAX_USER_MESSAGE_TOKENS: int = 2000  # 单条用户消息的最大 token 数（~1300 中文字/4000 英文字）
    # 截断策略: keep_both_ends | keep_start_only | keep_end_only
    # keep_both_ends: 保留首 40% + 尾 20%，中间插入省略标记（推荐，核心意图在首部，关键细节在尾部）
    # keep_start_only: 仅保留开头（适合客服场景）
    TRUNCATION_STRATEGY: str = "keep_both_ends"
    # 截断提示语（{original_tokens}/{truncated_tokens}/{max_tokens} 会被替换）
    TRUNCATION_WARNING_TEMPLATE: str = (
        "⚠️ 您的输入较长（原始 {original_tokens} token，已自动保留核心 {truncated_tokens} token）。"
        "如需更精准的回答，建议精简描述后重新提问。\n\n"
    )

    # MCP Server 配置
    MCP_SERVER_NAME: str = "shop-agent"
    MCP_ENABLED: bool = False  # 是否启用 MCP Server
    MCP_TRANSPORT: str = "stdio"  # stdio | sse | streamable-http

    # MCP Client 配置 —— Agent 作为 Client 消费远程 MCP Server 的工具
    # JSON 数组，每个元素包含 name、url、headers（可选）
    # 示例: '[{"name":"order-system","url":"http://localhost:3002/mcp"}]'
    MCP_CLIENT_SERVERS: str = ""
    MCP_CLIENT_ENABLED: bool = False  # 是否启用 MCP Client 模式

    # 基于角色的工具权限控制（默认关闭，生产环境按需开启）
    PERMISSION_ENABLED: bool = False

    # TTS 配置
    TTS_PROVIDER: str = "edge"  # edge | baidu
    BAIDU_TTS_API_KEY: str = ""
    BAIDU_TTS_SECRET_KEY: str = ""

    # 数字人配置
    AVATAR_PROVIDER: str = "static"  # static | baidu
    BAIDU_AVATAR_API_KEY: str = ""
    BAIDU_AVATAR_SECRET_KEY: str = ""

    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = (".env", ".env.prod")  # 多个环境文件，后者优先
        extra = "ignore"  # 忽略未知的环境变量


config = Settings()

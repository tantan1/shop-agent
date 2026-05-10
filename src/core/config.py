from pydantic_settings import BaseSettings


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
    VOLCENGINE_EMBEDDING_ENDPOINT: str = "https://ark.cn-beijing.volces.com/api/v3/projects/default/embeddings"
    
    # 聊天模型配置（云端模型，用于 Agent 回答生成等复杂任务）
    CHAT_MODEL: str = "qwen3.6-flash-2026-04-16"

    # P1 工具选择器专用模型（更轻量更快，qwen-turbo 延迟约为主模型 40%）
    TOOL_SELECTOR_MODEL: str = "qwen3.6-flash-2026-04-16"
    
    # P1 工具选择器本地模型路径（设置后优先用本地模型替代云端 API）
    # 推荐: Qwen2.5-1.5B-Instruct（速度和准确度的最佳平衡点）
    TOOL_SELECTOR_LOCAL_MODEL: str = ""  # 如 ./models/Qwen2.5-1.5B-Instruct
    TOOL_SELECTOR_LOCAL_DEVICE: str = "cpu"  # cpu | auto
    TOOL_SELECTOR_LOCAL_LOAD_IN_4BIT: bool = False
    
    # 本地小模型配置（用于参数抽取，transformers 直接加载，无需部署）
    LOCAL_PARAM_MODEL: str = "./models/Qwen2.5-0.5B-Instruct"  # 轻量级中文模型，~1GB，CPU 可跑
    LOCAL_PARAM_DEVICE: str = "auto"  # cpu | cuda | auto（auto 优先 GPU）
    LOCAL_PARAM_MAX_TOKENS: int = 256  # 参数抽取很短，256 足够
    LOCAL_PARAM_LOAD_IN_4BIT: bool = True  # 4bit 量化，节省内存（需 bitsandbytes）
    
    # Embedding 提供者: local | volcengine
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    
    # Milvus向量数据库配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # 远程业务API配置（意图识别触发远程调用时使用）
    REMOTE_API_BASE_URL: str = ""
    REMOTE_API_TIMEOUT: int = 10
    
    # 意图识别模式: local=关键词+向量(免费), llm=通义千问(精准)
    INTENT_RECOGNITION_MODE: str = "local"
    
    # 参数抽取模式: local=正则+关键词(免费,毫秒级), local_model=transformers本地小模型(免费,智能), llm=通义千问structured output(精准)
    PARAM_EXTRACTION_MODE: str = "local"
    
    # FAISS 意图向量匹配参数
    INTENT_VECTOR_SIMILARITY_THRESHOLD: float = 0.65  # 余弦相似度阈值（BGE归一化向量用内积）

    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = (".env", ".env.prod")  # 多个环境文件，后者优先
        extra = "ignore"  # 忽略未知的环境变量


config = Settings()

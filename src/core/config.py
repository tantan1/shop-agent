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
    
    # 聊天模型配置
    CHAT_MODEL: str = "qwen3.5-plus-2026-02-15"
    EMBEDDING_MODEL: str = "doubao-embedding-vision-251215"  # Doubao embedding 模型
    
    # Milvus向量数据库配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = (".env", ".env.prod")  # 多个环境文件，后者优先
        extra = "ignore"  # 忽略未知的环境变量


config = Settings()

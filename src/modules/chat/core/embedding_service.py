"""
嵌入服务模块
使用火山引擎 Ark SDK 实现文本嵌入
"""

from typing import List
import asyncio
from volcenginesdkarkruntime import Ark
from langchain_core.embeddings.embeddings import Embeddings

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("embedding_service")


class ArkEmbeddings(Embeddings):
    """火山引擎 Ark SDK 多模态嵌入封装类"""

    def __init__(self, client: Ark = None, model: str = None):
        """
        初始化嵌入服务
        
        Args:
            client: Ark 客户端实例，如果为 None 则自动创建
            model: 嵌入模型名称，如果为 None 则使用配置中的模型
        """
        if client is None:
            if not chat_config.volcengine_api_key:
                raise ValueError("VOLCENGINE_API_KEY 未配置")
            client = Ark(api_key=chat_config.volcengine_api_key)
        
        self.client = client
        self.model = model or chat_config.embedding_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        embeddings = []
        for text in texts:
            input_data = [{"type": "text", "text": text}]
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=input_data
            )
            embeddings.append(response.data.embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        input_data = [{"type": "text", "text": text}]
        response = self.client.multimodal_embeddings.create(
            model=self.model,
            input=input_data
        )
        return response.data.embedding

    def embed_with_image(self, text: str, image_url: str) -> List[float]:
        """嵌入文本和图片"""
        input_data = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        response = self.client.multimodal_embeddings.create(
            model=self.model,
            input=input_data
        )
        return response.data.embedding

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入文档"""
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入单个查询"""
        return await asyncio.to_thread(self.embed_query, text)


class EmbeddingService:
    """嵌入服务管理类"""
    
    _instance = None
    _embeddings: ArkEmbeddings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_embeddings(self) -> ArkEmbeddings:
        """获取嵌入实例"""
        if self._embeddings is None:
            self._embeddings = ArkEmbeddings()
        return self._embeddings
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        embeddings = self.get_embeddings()
        return await embeddings.aembed_documents(texts)
    
    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        embeddings = self.get_embeddings()
        return await embeddings.aembed_query(text)

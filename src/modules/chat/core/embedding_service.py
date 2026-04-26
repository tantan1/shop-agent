"""
嵌入服务模块
使用火山引擎 Ark SDK 实现文本嵌入
"""

from typing import List, Optional
import asyncio
import aiohttp
from volcenginesdkarkruntime import Ark
from langchain_core.embeddings.embeddings import Embeddings

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("embedding_service")

# API 请求批次大小限制
BATCH_SIZE = 25


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
        if not texts:
            return []
        
        # 检查 API 是否支持批量输入
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            input_data = [{"type": "text", "text": text} for text in batch]
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=input_data
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

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
        """真正的异步批量嵌入文档"""
        if not texts:
            return []
        
        # 使用 asyncio.gather 并行执行批量嵌入
        tasks = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            task = self._embed_batch_async(batch)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # 扁平化结果
        embeddings = []
        for batch_embeddings in results:
            embeddings.extend(batch_embeddings)
        return embeddings
    
    async def _embed_batch_async(self, batch: List[str]) -> List[List[float]]:
        """异步嵌入单个批次"""
        def _sync_call():
            input_data = [{"type": "text", "text": text} for text in batch]
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=input_data
            )
            return [item.embedding for item in response.data]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入单个查询"""
        def _sync_call():
            input_data = [{"type": "text", "text": text}]
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=input_data
            )
            return response.data.embedding
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)


class EmbeddingService:
    """嵌入服务管理类"""
    
    _instance: Optional["EmbeddingService"] = None
    _embeddings: Optional[ArkEmbeddings] = None
    _initialized: bool = False
    
    def __init__(self):
        """初始化嵌入服务"""
        if EmbeddingService._initialized:
            raise RuntimeError("请使用 get_instance() 获取 EmbeddingService 实例")
        EmbeddingService._initialized = True
    
    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
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

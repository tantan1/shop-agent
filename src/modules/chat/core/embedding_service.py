"""
嵌入服务模块
使用火山引擎 Ark SDK 实现文本嵌入
支持 LangChain 标准回调机制
"""

from typing import List, Optional, Any, Dict
import asyncio
from volcenginesdkarkruntime import Ark
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.callbacks import CallbackManager

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config
from src.modules.monitoring.metrics import (
    embedding_request_counter,
    embedding_request_duration,
    embedding_token_counter,
)

logger = APILogger("embedding_service")


class ArkEmbeddings(Embeddings):
    """火山引擎 Ark SDK 多模态嵌入封装类 - 支持 LangChain 回调"""

    def __init__(
        self,
        client: Ark = None,
        model: str = None,
        callback_manager: Optional[CallbackManager] = None
    ):
        """
        初始化嵌入服务
        
        Args:
            client: Ark 客户端实例，如果为 None 则自动创建
            model: 嵌入模型名称，如果为 None 则使用配置中的模型
            callback_manager: LangChain 回调管理器
        """
        if client is None:
            if not chat_config.volcengine_api_key:
                raise ValueError("VOLCENGINE_API_KEY 未配置")
            client = Ark(api_key=chat_config.volcengine_api_key)
        
        self.client = client
        self.model = model or chat_config.embedding_model
        self.callback_manager = callback_manager
        self._is_tracking = callback_manager is not None

    def _get_tokens_from_response(self, response: Any) -> tuple:
        """从响应中提取 token 使用量"""
        tokens = 0
        if hasattr(response, 'usage') and response.usage:
            tokens = response.usage.get('text_tokens', 0) or response.usage.get('total_tokens', 0)
        return tokens

    def _emit_embedding_metrics(self, tokens: int, count: int = 1):
        """通过 LangChain 回调发射指标"""
        if tokens > 0:
            embedding_token_counter.labels(provider="volcengine", type="text").inc(tokens)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档（同步版本，逐条调用）"""
        if not texts:
            return []
        
        import time
        start_time = time.time()
        status = "success"
        total_tokens = 0
        
        try:
            all_embeddings = []
            for text in texts:
                input_data = [{"type": "text", "text": text}]
                response = self.client.multimodal_embeddings.create(
                    model=self.model,
                    input=input_data
                )
                all_embeddings.append(response.data.embedding)
                # 提取 token
                tokens = self._get_tokens_from_response(response)
                total_tokens += tokens
            return all_embeddings
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            embedding_request_counter.labels(provider="volcengine", status=status).inc(len(texts))
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            if total_tokens > 0:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        import time
        start_time = time.time()
        status = "success"
        total_tokens = 0
        
        try:
            input_data = [{"type": "text", "text": text}]
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=input_data
            )
            # 提取 token
            total_tokens = self._get_tokens_from_response(response)
            return response.data.embedding
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            embedding_request_counter.labels(provider="volcengine", status=status).inc(1)
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            if total_tokens > 0:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)

    def embed_with_image(self, text: str, image_url: str) -> List[float]:
        """嵌入文本和图片"""
        import time
        start_time = time.time()
        status = "success"
        text_tokens = 0
        image_tokens = 0
        
        try:
            input_data = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
            response = self.client.multimodal_embeddings.create(
                model=self.model,
                input=input_data
            )
            # 提取 token
            if hasattr(response, 'usage') and response.usage:
                text_tokens = response.usage.get('text_tokens', 0)
                image_tokens = response.usage.get('image_tokens', 0)
            return response.data.embedding
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            embedding_request_counter.labels(provider="volcengine", status=status).inc(1)
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            if text_tokens > 0:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(text_tokens)
            if image_tokens > 0:
                embedding_token_counter.labels(provider="volcengine", type="image").inc(image_tokens)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入文档（API只支持单条，需逐条调用并并行化）"""
        if not texts:
            return []
        
        import time
        start_time = time.time()
        status = "success"
        total_tokens = 0
        
        try:
            # 使用信号量限制并发数（避免API限流）
            semaphore = asyncio.Semaphore(10)
            
            async def embed_one(text: str) -> tuple:
                async with semaphore:
                    def _sync_call():
                        input_data = [{"type": "text", "text": text}]
                        response = self.client.multimodal_embeddings.create(
                            model=self.model,
                            input=input_data
                        )
                        tokens = self._get_tokens_from_response(response)
                        return response.data.embedding, tokens
                    
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, _sync_call)
            
            # 并行执行所有嵌入
            tasks = [embed_one(text) for text in texts]
            results = await asyncio.gather(*tasks)
            
            # 汇总 token 使用量
            for embedding, tokens in results:
                total_tokens += tokens
            
            return [emb for emb, _ in results]
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            embedding_request_counter.labels(provider="volcengine", status=status).inc(len(texts))
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            if total_tokens > 0:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入单个查询"""
        import time
        start_time = time.time()
        status = "success"
        total_tokens = 0
        
        try:
            def _sync_call():
                input_data = [{"type": "text", "text": text}]
                response = self.client.multimodal_embeddings.create(
                    model=self.model,
                    input=input_data
                )
                tokens = self._get_tokens_from_response(response)
                return response.data.embedding, tokens
            
            loop = asyncio.get_event_loop()
            result, tokens = await loop.run_in_executor(None, _sync_call)
            total_tokens = tokens
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            embedding_request_counter.labels(provider="volcengine", status=status).inc(1)
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            if total_tokens > 0:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)


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
        import time
        start_time = time.time()
        status = "success"
        
        try:
            embeddings = self.get_embeddings()
            return await embeddings.aembed_documents(texts)
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            embedding_request_counter.labels(provider="volcengine", status=status).inc(len(texts))
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            logger.info(f"批量嵌入完成", count=len(texts), duration_ms=int(duration * 1000))
    
    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        import time
        start_time = time.time()
        status = "success"
        
        try:
            embeddings = self.get_embeddings()
            return await embeddings.aembed_query(text)
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            embedding_request_counter.labels(provider="volcengine", status=status).inc(1)
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            logger.info(f"单条嵌入完成", text_length=len(text), duration_ms=int(duration * 1000))

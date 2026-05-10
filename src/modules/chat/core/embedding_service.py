"""
嵌入服务模块
支持本地模型（sentence-transformers）和火山引擎 Ark SDK 两种提供者
兼容 LangChain 标准回调机制
"""

import os
from typing import List, Optional, Any, Dict
import asyncio
import threading

# 国内访问 HuggingFace 自动走镜像（必须在 sentence-transformers 导入前设置）
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "") or "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.callbacks import CallbackManager
from langfuse import observe

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config
from src.modules.monitoring.metrics import (
    embedding_request_counter,
    embedding_request_duration,
    embedding_token_counter,
)

logger = APILogger("embedding_service")

# 延迟导入，避免未安装的 SDK 报错
_Ark = None


def _get_ark():
    global _Ark
    if _Ark is None:
        from volcenginesdkarkruntime import Ark as _ArkCls
        _Ark = _ArkCls
    return _Ark


# =============================================================================
# 本地模型（sentence-transformers）
# =============================================================================

class LocalEmbeddings(Embeddings):
    """本地 BGE/Sentence-Transformers 嵌入（免费，毫秒级）"""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        normalize: bool = True,
        batch_size: int = 32,
        callback_manager: Optional[CallbackManager] = None
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or chat_config.embedding_model
        self.normalize = normalize
        self.batch_size = batch_size
        self.callback_manager = callback_manager

        # 线程安全：模型加载一次，推理可复用
        self._lock = threading.Lock()
        self._model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            logger.info(f"加载本地 embedding 模型: {self.model_name}")
            self._model = __import__("sentence_transformers").SentenceTransformer(
                self.model_name
            )
            logger.info(
                f"本地 embedding 模型加载完成, 维度={self._model.get_embedding_dimension()}"
            )

    @property
    def model(self):
        self._load_model()
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import time
        start = time.time()

        try:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            return vectors.tolist()
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="local", status="success").inc(len(texts))
            embedding_request_duration.labels(provider="local").observe(dur)

    def embed_query(self, text: str) -> List[float]:
        import time
        start = time.time()

        try:
            vectors = self.model.encode(
                [text],
                normalize_embeddings=self.normalize,
                batch_size=1,
                show_progress_bar=False,
            )
            return vectors[0].tolist()
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="local", status="success").inc(1)
            embedding_request_duration.labels(provider="local").observe(dur)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)


# =============================================================================
# 火山引擎 Ark（保留，provider=volcengine 时使用）
# =============================================================================

class ArkEmbeddings(Embeddings):
    """火山引擎 Ark SDK 多模态嵌入封装"""

    def __init__(
        self,
        client=None,
        model: str = None,
        callback_manager: Optional[CallbackManager] = None
    ):
        Ark = _get_ark()
        if client is None:
            if not chat_config.volcengine_api_key:
                raise ValueError("VOLCENGINE_API_KEY 未配置")
            client = Ark(api_key=chat_config.volcengine_api_key)

        self.client = client
        self.model = model or chat_config.embedding_model
        self.callback_manager = callback_manager

    def _tokens(self, response) -> int:
        if hasattr(response, "usage") and response.usage:
            return response.usage.get("text_tokens", 0) or response.usage.get("total_tokens", 0)
        return 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import time
        start = time.time()
        total_tokens = 0
        try:
            result = []
            for text in texts:
                r = self.client.multimodal_embeddings.create(
                    model=self.model,
                    input=[{"type": "text", "text": text}]
                )
                result.append(r.data.embedding)
                total_tokens += self._tokens(r)
            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="volcengine", status="success").inc(len(texts))
            embedding_request_duration.labels(provider="volcengine").observe(dur)
            if total_tokens:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)

    def embed_query(self, text: str) -> List[float]:
        import time
        start = time.time()
        total_tokens = 0
        try:
            r = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[{"type": "text", "text": text}]
            )
            total_tokens = self._tokens(r)
            return r.data.embedding
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="volcengine", status="success").inc(1)
            embedding_request_duration.labels(provider="volcengine").observe(dur)
            if total_tokens:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)

    def embed_with_image(self, text: str, image_url: str) -> List[float]:
        import time
        start = time.time()
        text_tokens = 0
        image_tokens = 0
        try:
            r = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )
            if hasattr(r, "usage") and r.usage:
                text_tokens = r.usage.get("text_tokens", 0)
                image_tokens = r.usage.get("image_tokens", 0)
            return r.data.embedding
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="volcengine", status="success").inc(1)
            embedding_request_duration.labels(provider="volcengine").observe(dur)
            if text_tokens:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(text_tokens)
            if image_tokens:
                embedding_token_counter.labels(provider="volcengine", type="image").inc(image_tokens)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import time
        start = time.time()
        total_tokens = 0
        try:
            sem = asyncio.Semaphore(10)
            async def _one(text: str):
                async with sem:
                    loop = asyncio.get_event_loop()
                    def _sync():
                        r = self.client.multimodal_embeddings.create(
                            model=self.model,
                            input=[{"type": "text", "text": text}]
                        )
                        return r.data.embedding, self._tokens(r)
                    return await loop.run_in_executor(None, _sync)

            results = await asyncio.gather(*[_one(t) for t in texts])
            total_tokens = sum(t for _, t in results)
            return [e for e, _ in results]
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="volcengine", status="success").inc(len(texts))
            embedding_request_duration.labels(provider="volcengine").observe(dur)
            if total_tokens:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)

    async def aembed_query(self, text: str) -> List[float]:
        import time
        start = time.time()
        total_tokens = 0
        try:
            loop = asyncio.get_event_loop()
            def _sync():
                r = self.client.multimodal_embeddings.create(
                    model=self.model,
                    input=[{"type": "text", "text": text}]
                )
                return r.data.embedding, self._tokens(r)
            result, total_tokens = await loop.run_in_executor(None, _sync)
            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="volcengine", status="success").inc(1)
            embedding_request_duration.labels(provider="volcengine").observe(dur)
            if total_tokens:
                embedding_token_counter.labels(provider="volcengine", type="text").inc(total_tokens)


# =============================================================================
# 嵌入服务管理类（提供者路由）
# =============================================================================

class EmbeddingService:
    """嵌入服务管理类"""

    _instance: Optional["EmbeddingService"] = None
    _embeddings: Optional[Embeddings] = None
    _initialized: bool = False

    def __init__(self):
        if EmbeddingService._initialized:
            raise RuntimeError("请使用 get_instance() 获取 EmbeddingService 实例")
        EmbeddingService._initialized = True

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
        return cls._instance

    @property
    def _provider(self) -> str:
        return getattr(chat_config, "embedding_provider", "volcengine")

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            if self._provider == "local":
                logger.info(f"使用本地 embedding: {chat_config.embedding_model}")
                self._embeddings = LocalEmbeddings()
            else:
                logger.info(f"使用火山引擎 embedding: {chat_config.embedding_model}")
                self._embeddings = ArkEmbeddings()
        return self._embeddings

    @observe(name="embedding.batch")
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        import time
        start = time.time()
        try:
            return await self.get_embeddings().aembed_documents(texts)
        finally:
            dur = (time.time() - start) * 1000
            logger.info(f"批量嵌入完成", count=len(texts), duration_ms=int(dur))

    @observe(name="embedding.query")
    async def embed_query(self, text: str) -> List[float]:
        import time
        start = time.time()
        try:
            return await self.get_embeddings().aembed_query(text)
        finally:
            dur = (time.time() - start) * 1000
            logger.info(f"单条嵌入完成", text_length=len(text), duration_ms=int(dur))

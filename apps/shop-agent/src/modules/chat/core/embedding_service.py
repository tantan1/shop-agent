"""
嵌入服务模块
基于本地模型（sentence-transformers），兼容 LangChain 标准回调机制
"""

import asyncio  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
from typing import List, Optional  # noqa: E402

# 国内访问 HuggingFace 自动走镜像（必须在 sentence-transformers 导入前设置）
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "") or "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)

from langchain_core.callbacks import CallbackManager  # noqa: E402
from langchain_core.embeddings.embeddings import Embeddings  # noqa: E402

from src.modules.monitoring.langfuse_callback import observe  # noqa: E402


def _get_current_span():
    """获取当前 OpenTelemetry span（Langfuse @observe 底层使用的 span）。"""
    try:
        from opentelemetry import trace  # noqa: E402

        return trace.get_current_span()
    except ImportError:
        return None


from src.modules.chat.config import chat_config  # noqa: E402
from src.modules.monitoring.metrics import (  # noqa: E402
    embedding_request_counter,
    embedding_request_duration,
)
from src.shared.logger import APILogger  # noqa: E402

logger = APILogger("embedding_service")


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
        callback_manager: Optional[CallbackManager] = None,
    ):
        from sentence_transformers import SentenceTransformer  # noqa: E402

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
            self._model = __import__("sentence_transformers").SentenceTransformer(self.model_name)
            logger.info(
                f"本地 embedding 模型加载完成, 维度={self._model.get_embedding_dimension()}"
            )

    @property
    def model(self):
        self._load_model()
        return self._model

    @observe(name="embedding.documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import time  # noqa: E402

        start = time.time()

        try:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            result = vectors.tolist()

            # Langfuse: 记录批量嵌入的模型和用量（OTel span attribute）
            _span = _get_current_span()
            if _span is not None and not isinstance(_span, type(None)):
                try:
                    _span.set_attribute("embedding.model", self.model_name)
                    _span.set_attribute("embedding.provider", "local")
                    _span.set_attribute("embedding.batch_count", len(texts))
                    _span.set_attribute("embedding.input_chars_total", sum(len(t) for t in texts))
                    _span.set_attribute(
                        "embedding.output_dim",
                        vectors.shape[1] if vectors.ndim > 1 else len(result[0]),
                    )
                    _span.set_attribute(
                        "embedding.input_tokens_est", max(1, sum(len(t) for t in texts) // 2)
                    )
                except Exception:
                    pass

            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="local", status="success").inc(len(texts))
            embedding_request_duration.labels(provider="local").observe(dur)

    @observe(name="embedding.query")
    def embed_query(self, text: str, instruction: str = None) -> List[float]:
        """对单条查询文本做 embedding。

        Args:
            text: 查询文本
            instruction: BGE-M3 指令前缀。BGE-M3 是 instruction-tuned 模型，
                         加入任务前缀可激活模型在检索任务上的最佳编码路径（5-10% 精度提升）。
                         示例: "为这个句子生成表示以用于检索相关文章："
        """
        import time  # noqa: E402

        start = time.time()

        try:
            if instruction:
                text = f"{instruction}{text}"
            vectors = self.model.encode(
                [text],
                normalize_embeddings=self.normalize,
                batch_size=1,
                show_progress_bar=False,
            )
            result = vectors[0].tolist()

            # Langfuse: 记录本地嵌入模型的名称和用量估计（OTel span attribute）
            _span = _get_current_span()
            if _span is not None and not isinstance(_span, type(None)):
                try:
                    _span.set_attribute("embedding.model", self.model_name)
                    _span.set_attribute("embedding.provider", "local")
                    _span.set_attribute("embedding.input_chars", len(text))
                    _span.set_attribute("embedding.output_dim", len(result))
                    _span.set_attribute("embedding.input_tokens_est", max(1, len(text) // 2))
                except Exception:
                    pass

            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="local", status="success").inc(1)
            embedding_request_duration.labels(provider="local").observe(dur)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str, instruction: str = None) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text, instruction)


# =============================================================================
# Ollama 嵌入（进程外，k8s 部署用；Ollama 0.1.47+ 提供 /api/embed）
# =============================================================================


class OllamaEmbeddings(Embeddings):
    """Ollama 进程外 embedding（零 ML 依赖，httpx 调 /api/embed）

    用于 k8s 部署形态：模型权重由 Ollama 进程管理（本地 GGUF），
    应用侧无需 sentence-transformers / torch，避免镜像体积膨胀。
    """

    def __init__(
        self,
        model_name: str = "bge-m3",
        base_url: str = None,
        timeout: int = None,
        normalize: bool = True,
        batch_size: int = 32,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self.model_name = model_name or chat_config.ollama_embedding_model
        self.base_url = (base_url or chat_config.ollama_base_url).rstrip("/")
        self.timeout = timeout or chat_config.ollama_timeout
        self.normalize = normalize
        self.batch_size = batch_size
        self.callback_manager = callback_manager

    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        import httpx  # noqa: E402

        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model_name, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["embeddings"]

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._embed_async(texts))
        finally:
            loop.close()

    @observe(name="embedding.ollama.documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import time  # noqa: E402

        start = time.time()
        try:
            result = self._embed_sync(texts)
            if self.normalize:
                result = self._normalize(result)
            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="ollama", status="success").inc(len(texts))
            embedding_request_duration.labels(provider="ollama").observe(dur)

    @observe(name="embedding.ollama.query")
    def embed_query(self, text: str, instruction: str = None) -> List[float]:
        import time  # noqa: E402

        start = time.time()
        try:
            if instruction:
                text = f"{instruction}{text}"
            result = self._embed_sync([text])[0]
            if self.normalize:
                result = self._normalize([result])[0]
            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="ollama", status="success").inc(1)
            embedding_request_duration.labels(provider="ollama").observe(dur)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = await self._embed_async(texts)
        return self._normalize(result)

    async def aembed_query(self, text: str, instruction: str = None) -> List[float]:
        if instruction:
            text = f"{instruction}{text}"
        result = await self._embed_async([text])
        return self._normalize(result)[0]

    @staticmethod
    def _normalize(vectors: List[List[float]]) -> List[List[float]]:
        """L2 归一化（与 sentence-transformers normalize_embeddings=True 一致）"""
        import math  # noqa: E402

        out = []
        for v in vectors:
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


# =============================================================================
# vLLM bge-m3 嵌入（进程外，OpenAI 兼容 /v1/embeddings；本地 GPU 直连）
# =============================================================================


class VLLMEmbeddings(Embeddings):
    """vLLM bge-m3 embedding（直连容器名，OpenAI 兼容 /v1/embeddings）

    用于本地 GPU 部署形态：模型权重由 vLLM 管理，应用侧零 ML 依赖。
    """

    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        timeout: int = None,
        normalize: bool = True,
        batch_size: int = 32,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self.model_name = model_name or chat_config.vllm_embedding_model
        self.base_url = (base_url or chat_config.vllm_embedding_base_url).rstrip("/")
        self.timeout = timeout or 60
        self.normalize = normalize
        self.batch_size = batch_size
        self.callback_manager = callback_manager

    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        import httpx  # noqa: E402

        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.model_name, "input": texts}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        items = sorted(
            data.get("data", []),
            key=lambda it: it.get("index", 0),
        )
        return [it["embedding"] for it in items]

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._embed_async(texts))
        finally:
            loop.close()

    @observe(name="embedding.vllm.documents")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import time  # noqa: E402

        start = time.time()
        try:
            result = self._embed_sync(texts)
            if self.normalize:
                result = OllamaEmbeddings._normalize(result)
            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="vllm", status="success").inc(len(texts))
            embedding_request_duration.labels(provider="vllm").observe(dur)

    @observe(name="embedding.vllm.query")
    def embed_query(self, text: str, instruction: str = None) -> List[float]:
        import time  # noqa: E402

        start = time.time()
        try:
            if instruction:
                text = f"{instruction}{text}"
            result = self._embed_sync([text])[0]
            if self.normalize:
                result = OllamaEmbeddings._normalize([result])[0]
            return result
        finally:
            dur = time.time() - start
            embedding_request_counter.labels(provider="vllm", status="success").inc(1)
            embedding_request_duration.labels(provider="vllm").observe(dur)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = await self._embed_async(texts)
        if self.normalize:
            result = OllamaEmbeddings._normalize(result)
        return result

    async def aembed_query(self, text: str, instruction: str = None) -> List[float]:
        if instruction:
            text = f"{instruction}{text}"
        result = await self._embed_async([text])
        if self.normalize:
            result = OllamaEmbeddings._normalize(result)
        return result[0]


# =============================================================================
# 嵌入服务管理类
# =============================================================================


class EmbeddingService:
    """嵌入服务管理类（provider: local=sentence-transformers | ollama=Ollama API）"""

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

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            provider = chat_config.embedding_provider
            if provider == "ollama":
                logger.info(
                    f"使用 Ollama embedding: {chat_config.ollama_embedding_model} "
                    f"@ {chat_config.ollama_base_url}"
                )
                self._embeddings = OllamaEmbeddings()
            elif provider == "vllm":
                logger.info(
                    f"使用 vLLM embedding: {chat_config.vllm_embedding_model} "
                    f"@ {chat_config.vllm_embedding_base_url}"
                )
                self._embeddings = VLLMEmbeddings()
            else:
                logger.info(f"使用本地 embedding: {chat_config.embedding_model}")
                self._embeddings = LocalEmbeddings()
        return self._embeddings

    @observe(name="embedding.batch")
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        import time  # noqa: E402

        start = time.time()
        try:
            return await self.get_embeddings().aembed_documents(texts)
        finally:
            dur = (time.time() - start) * 1000
            logger.info("批量嵌入完成", count=len(texts), duration_ms=int(dur))

    @observe(name="embedding.query")
    async def embed_query(self, text: str, instruction: str = None) -> List[float]:
        import time  # noqa: E402

        start = time.time()
        try:
            return await self.get_embeddings().aembed_query(text, instruction)
        finally:
            dur = (time.time() - start) * 1000
            logger.info("单条嵌入完成", text_length=len(text), duration_ms=int(dur))

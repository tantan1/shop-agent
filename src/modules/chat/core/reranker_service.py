"""
Reranker 服务模块
基于 BGE-Reranker-base (sentence-transformers CrossEncoder) 对检索结果进行重排序 + 低相关性截断
"""

from typing import List, Tuple, Optional
import threading

from langfuse import observe


def _get_current_span():
    """获取当前 OpenTelemetry span（Langfuse @observe 底层使用的 span）。"""
    try:
        from opentelemetry import trace
        return trace.get_current_span()
    except ImportError:
        return None

from src.shared.logger import APILogger
from src.core.config import config

logger = APILogger("reranker_service")


class RerankerService:
    """
    BGE-Reranker 封装（懒加载单例），使用 sentence-transformers CrossEncoder
    
    用法:
        reranker = RerankerService.get_instance()
        scores = reranker.compute_scores(
            query="用户问题",
            documents=["文档1", "文档2", ...]
        )
    """

    _instance: Optional["RerankerService"] = None
    _lock = threading.Lock()
    _model: object = None  # CrossEncoder 实例

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: str = None, use_fp16: bool = False):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._use_fp16 = use_fp16
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "RerankerService":
        """获取单例实例（线程安全）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试或重新加载）"""
        with cls._lock:
            if cls._instance is not None and cls._instance._model is not None:
                del cls._instance._model
                cls._instance._model = None
            cls._instance = None

    def _ensure_initialized(self):
        """确保模型已加载（懒加载），优先从本地 ModelScope 缓存加载"""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                from sentence_transformers import CrossEncoder
                import os

                # 优先使用配置指定的本地模型路径（支持 .env 配置）
                local_path = config.RERANKER_LOCAL_MODEL_PATH
                if local_path and os.path.isdir(local_path):
                    model_path = local_path
                    logger.info(f"使用本地模型 (config): {model_path}")
                else:
                    model_path = self._model_name
                    logger.info(f"本地模型不存在，从 HuggingFace 加载: {model_path}")

                logger.info(f"正在加载 Reranker 模型: {model_path}")
                self._model = CrossEncoder(model_path)
                self._initialized = True
                logger.info(f"Reranker 模型加载完成: {model_path}")
            except ImportError:
                logger.error(
                    "sentence-transformers 未安装，无法使用 Reranker。请执行: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Reranker 模型加载失败: {str(e)}")
                raise

    @observe(name="reranker.scores")
    def compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        计算 query 与每篇文档的相关性分数
        
        Args:
            query: 用户问题
            documents: 文档内容列表
            
        Returns:
            相关性分数列表（0~1，越高越相关）
        """
        self._ensure_initialized()

        if not documents:
            return []

        # CrossEncoder 接受 [(query, doc), ...] 格式
        pairs = [(query, doc) for doc in documents]

        try:
            scores = self._model.predict(pairs)
            result = [round(float(s), 4) for s in scores]

            # Langfuse: 记录 Reranker 模型名称和输入量（OTel span attribute）
            _span = _get_current_span()
            if _span is not None and not isinstance(_span, type(None)):
                try:
                    _span.set_attribute("reranker.model", self._model_name)
                    _span.set_attribute("reranker.provider", "local")
                    _span.set_attribute("reranker.num_documents", len(documents))
                    _span.set_attribute("reranker.input_chars_query", len(query))
                    _span.set_attribute("reranker.input_chars_total", sum(len(d) for d in documents))
                    _span.set_attribute("reranker.input_tokens_est", max(1, (len(query) + sum(len(d) for d in documents)) // 2))
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.error(f"Reranker 计算分数失败: {str(e)[:200]}")
            return [0.0] * len(documents)

    @observe(name="reranker.rerank")
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[int, float, str]]:
        """
        重排序 + 阈值截断
        
        Args:
            query: 用户问题
            documents: 文档内容列表
            top_k: 返回前K条
            threshold: 最低相关性分数阈值（低于此分的文档被丢弃）
            
        Returns:
            [(原始索引, 相关性分数, 文档内容), ...]  按分数降序排列
        """
        scores = self.compute_scores(query, documents)

        # 按分数降序排列，保留指定排名
        ranked = sorted(
            [(i, score, documents[i]) for i, score in enumerate(scores)],
            key=lambda x: x[1],
            reverse=True
        )

        # 阈值截断 + top_k
        filtered = [
            (idx, score, doc)
            for idx, score, doc in ranked
            if score >= threshold
        ][:top_k]

        # Langfuse: 记录 Reranker 输出信息（OTel span attribute）
        _span = _get_current_span()
        if _span is not None and not isinstance(_span, type(None)):
            try:
                _span.set_attribute("reranker.model", self._model_name)
                _span.set_attribute("reranker.provider", "local")
                _span.set_attribute("reranker.input_docs", len(documents))
                _span.set_attribute("reranker.output_docs", len(filtered))
                _span.set_attribute("reranker.top_k", top_k)
                _span.set_attribute("reranker.threshold", float(threshold))
            except Exception:
                pass

        logger.debug(
            f"Rerank 完成",
            input_count=len(documents),
            output_count=len(filtered),
            threshold=threshold,
            top_k=top_k
        )

        return filtered

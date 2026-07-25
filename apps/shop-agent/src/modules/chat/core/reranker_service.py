"""
Reranker 服务模块
基于 BGE-Reranker-base (sentence-transformers CrossEncoder) 对检索结果进行重排序 + 低相关性截断
"""

import threading  # noqa: E402
from typing import List, Optional, Tuple  # noqa: E402

from src.modules.monitoring.langfuse_callback import observe  # noqa: E402


def _get_current_span():
    """获取当前 OpenTelemetry span（Langfuse @observe 底层使用的 span）。"""
    try:
        from opentelemetry import trace  # noqa: E402

        return trace.get_current_span()
    except ImportError:
        return None


from src.core.config import config  # noqa: E402
from src.modules.chat.config import chat_config  # noqa: E402
from src.shared.logger import APILogger  # noqa: E402

logger = APILogger("reranker_service")


class RerankerService:
    """
    BGE-Reranker 封装（懒加载单例）
    后端（RERANKER_PROVIDER）:
      - local: sentence-transformers CrossEncoder 进程内推理
      - vllm:  远程调用 vLLM bge-reranker 的 /v1/rerank（直连容器名）

    用法:
        reranker = RerankerService.get_instance()
        scores = reranker.compute_scores(
            query="用户问题",
            documents=["文档1", "文档2", ...]
        )
    """

    _instance: Optional["RerankerService"] = None
    _lock = threading.Lock()
    _model: object = None  # CrossEncoder 实例（仅 local 后端）

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(self, model_name: str = None, use_fp16: bool = False):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._use_fp16 = use_fp16
        self._initialized = False
        self._provider = (chat_config.reranker_provider or "local").strip().lower()

    @property
    def provider(self) -> str:
        return self._provider

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
        """确保本地模型已加载（仅 local 后端；vllm 后端无进程内模型）。"""
        if self._provider == "vllm":
            return
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                import os  # noqa: E402

                from sentence_transformers import CrossEncoder  # noqa: E402

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
    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        计算 query 与每篇文档的相关分数

        Args:
            query: 用户问题
            documents: 文档内容列表

        Returns:
            相关性分数列表（0~1，越高越相关，与 documents 顺序对齐）
        """
        if not documents:
            return []

        if self._provider == "vllm":
            return self._compute_scores_vllm(query, documents)
        return self._compute_scores_local(query, documents)

    def _compute_scores_vllm(self, query: str, documents: List[str]) -> List[float]:
        """远程调用 vLLM bge-reranker 的 /v1/rerank（Jina/Cohere 兼容接口）。

        响应 results[].index 为原始文档下标、relevance_score 为 sigmoid 归一化分数。
        """
        import httpx  # noqa: E402

        base = (chat_config.vllm_rerank_base_url or "").rstrip("/")
        model = chat_config.vllm_rerank_model or self._model_name
        if not base:
            logger.error("vLLM rerank 未配置 VLLM_RERANK_BASE_URL，跳过重排（保留 dense 结果）")
            raise RuntimeError("VLLM_RERANK_BASE_URL 未配置，reranker unavailable")

        # bge-reranker 容器 --max-model-len=512，过长的 chunk 会触发 vLLM 400。
        # 按字符保守截断到 480 字（中文约 240 token，远低于 512 上限）。
        max_doc_chars = 480
        safe_documents = [d[:max_doc_chars] for d in documents]

        payload = {
            "model": model,
            "query": query,
            "documents": safe_documents,
            "top_n": len(safe_documents),
            "return_documents": False,
        }
        try:
            resp = httpx.post(
                f"{base}/v1/rerank",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            # 不能吞异常返回 0.0 — 0.0 会被 rerank() 的阈值过滤全部丢掉，
            # 反而比直接失败更致命。记录响应体以便定位 400 原因，
            # 交给 executor 的 try/except 保留原始 dense 结果。
            body = ""
            try:
                body = resp.text[:300]
            except Exception:
                pass
            logger.error(
                f"vLLM Rerank 调用失败: {str(e)[:200]} | status={resp.status_code} body={body}"
            )
            raise
        except Exception as e:
            logger.error(f"vLLM Rerank 请求异常: {str(e)[:200]}")
            raise

        result = [0.0] * len(documents)
        for item in data.get("results", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(documents):
                result[idx] = round(float(item.get("relevance_score", 0.0)), 4)

        # Langfuse: 记录 Reranker 模型名称和输入量（OTel span attribute）
        _span = _get_current_span()
        if _span is not None and not isinstance(_span, type(None)):
            try:
                _span.set_attribute("reranker.model", model)
                _span.set_attribute("reranker.provider", "vllm")
                _span.set_attribute("reranker.num_documents", len(documents))
                _span.set_attribute("reranker.input_chars_query", len(query))
                _span.set_attribute("reranker.input_chars_total", sum(len(d) for d in documents))
                _span.set_attribute(
                    "reranker.input_tokens_est",
                    max(1, (len(query) + sum(len(d) for d in documents)) // 2),
                )
            except Exception:
                pass

        return result

    def _compute_scores_local(self, query: str, documents: List[str]) -> List[float]:
        self._ensure_initialized()

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
                    _span.set_attribute(
                        "reranker.input_chars_total", sum(len(d) for d in documents)
                    )
                    _span.set_attribute(
                        "reranker.input_tokens_est",
                        max(1, (len(query) + sum(len(d) for d in documents)) // 2),
                    )
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.error(f"Reranker 计算分数失败: {str(e)[:200]}")
            return [0.0] * len(documents)

    @observe(name="reranker.rerank")
    def rerank(
        self, query: str, documents: List[str], top_k: int = 10, threshold: float = 0.0
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
            reverse=True,
        )

        # 阈值截断 + top_k
        filtered = [(idx, score, doc) for idx, score, doc in ranked if score >= threshold][:top_k]

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
            "Rerank 完成",
            input_count=len(documents),
            output_count=len(filtered),
            threshold=threshold,
            top_k=top_k,
        )

        return filtered

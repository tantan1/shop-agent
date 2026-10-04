"""P2 线性头分类器（ONNX Runtime 推理，非生成式 LLM）。

复用 ``scripts/eval/tool_select/eval_embedding_baseline.py::ToolHead`` 训练产出的权重，
对候选工具做 "query embedding -> 线性头" 打分。

- 权重: 与本品打包的 ONNX 文件（默认 ``src/modules/chat/agent/assets/p2_linear_head.onnx``）
- 输入: 用户 query 经统一 EmbeddingService（vLLM ``/models/bge-small-zh-v1.5``，512 维）
        归一化后的 embedding；**训练与推理共用同一 embedding 端点**，保证向量完全一致。
- 推理: ONNX Runtime 执行线性前向（W·x + b），进程内不加载任何 PyTorch / 本地 embedding 模型。
- 打分: 仅对 scope 中"落在 head 词表内"的候选取 logits 做 top-k；
        词表外候选无法评分，原样透传（保证不误删可能正确的工具）。

与旧实现的差异:
    - 旧: 进程内 sentence-transformers 本地加载 bge-small-zh-v1.5 + torch 前向（重、且训练/推理易错位）。
    - 新: 统一走 vLLM embedding 端点 + ONNX Runtime 推理（轻量，且与 P1 FAISS 共用同一 embedding）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


# 与本品打包的权重/元信息目录（位于 src/modules/chat/agent/assets/，随镜像构建 COPY 进容器）。
# 本模块在 src/modules/chat/core/ 下，故取 parents[1]=chat 再进 agent/assets。
_ASSET_DIR = Path(__file__).resolve().parents[1] / "agent" / "assets"


class ToolHeadClassifier:
    """惰性加载的线性头分类器单例（ONNX Runtime 推理）。"""

    _instance = None

    def __init__(
        self,
        onnx_path: str,
        meta_path: str,
        embed_service,
        top_k: int = 3,
    ) -> None:
        self._onnx_path = Path(onnx_path)
        self._meta_path = Path(meta_path)
        # embed_service 为 EmbeddingService.get_embeddings() 返回的同步 embedding 后端
        # （生产为 VLLMEmbeddings，其 embed_query 为同步方法，返回已归一化向量）。
        self._emb = embed_service
        self._top_k = top_k
        self._session = None
        self._classes: List[str] = []
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._load_failed = False

    @classmethod
    def get_instance(cls) -> "ToolHeadClassifier":
        if cls._instance is None:
            from src.modules.chat.config import chat_config
            from src.modules.chat.core.embedding_service import EmbeddingService

            cls._instance = cls(
                onnx_path=chat_config.p2_head_model_path,
                meta_path=chat_config.p2_head_meta_path,
                embed_service=EmbeddingService.get_instance().get_embeddings(),
                top_k=chat_config.p2_head_top_k,
            )
        return cls._instance

    def _resolve_asset(self, path: Path) -> Path:
        """给定路径不存在时，回退到本品打包 assets 目录下的同名文件。

        保证 Docker 构建后（COPY ./ /code）即使工作目录不同也能定位到镜像内打包的权重。
        """
        if path.is_absolute() and path.exists():
            return path
        cand = Path(path)
        if not cand.exists():
            cand = _ASSET_DIR / path.name
        return cand

    # ── 加载（仅首次 score / warmup 时触发）──
    def _load(self) -> bool:
        if ort is None:
            self._load_failed = True
            logger.warning("P2 线性头不可用：onnxruntime 未安装，P2 退化为透传")
            return False
        if self._session is not None:
            return True
        if self._load_failed:
            return False
        try:
            meta_path = self._resolve_asset(self._meta_path)
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._classes = list(meta["classes"])

            onnx_path = self._resolve_asset(self._onnx_path)
            self._session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

            logger.info(
                "P2 线性头加载完成(ONNX): onnx=%s n_classes=%s embed=%s",
                str(onnx_path),
                len(self._classes),
                meta.get("embed_model"),
            )
            return True
        except Exception as e:
            self._load_failed = True
            logger.warning(f"P2 线性头加载失败，P2 退化为透传: {e}")
            return False

    def warmup(self) -> bool:
        """预热：加载 ONNX 会话，避免首次推理延迟。返回 True 表示预热成功。"""
        return self._load()

    def score(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
        query_embedding: Any | None = None,
    ) -> Optional[List[str]]:
        """对候选子集打分，返回 top-k 工具名（均为 ``candidates`` 子集）。见 ``score_with_probs``。"""
        pairs = self.score_with_probs(query, candidates, top_k, query_embedding)
        return [n for n, _ in pairs] if pairs is not None else None

    def score_with_probs(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
        query_embedding: Any | None = None,
    ) -> Optional[List[Tuple[str, float]]]:
        """同 ``score``，但返回 ``[(工具名, softmax 概率), ...]``（真实置信度，设计 1）。

        ``query_embedding``：流水线预计算好的 query 向量（已归一化，来自 vLLM 端点）。
        传入时直接复用，避免 P2 再打一次 vLLM（同时绕开同步 embed 的跨事件循环 httpx 问题）。
        未传入时回退到 ``self._emb.embed_query``（独立调用场景）。

        返回 ``None`` 表示无法评分（加载失败 / 候选全在词表外），调用方应透传候选。
        """
        if not self._load():
            return None
        k = top_k or self._top_k
        try:
            if query_embedding is not None:
                # 复用流水线一次性算好的 query embedding
                q_emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            else:
                # 复用统一 EmbeddingService（vLLM bge-small-zh-v1.5）取 query embedding
                q_emb = np.array(self._emb.embed_query(query), dtype=np.float32).reshape(1, -1)

            logits = self._session.run(
                [self._output_name], {self._input_name: q_emb}
            )[0][0]

            idx_of = {n: i for i, n in enumerate(self._classes)}
            in_vocab = [c for c in candidates if c in idx_of]
            if not in_vocab:
                # 候选全部不在 head 词表内 -> 无法评分，透传
                return None

            cand_logits = np.array([logits[idx_of[c]] for c in in_vocab])
            # softmax 归一化（线性头输出已可比较）
            exp = np.exp(cand_logits - cand_logits.max())
            probs = exp / exp.sum()
            order = np.argsort(-probs)
            top = order[: min(k, len(in_vocab))]
            return [(in_vocab[i], float(probs[i])) for i in top]
        except Exception as e:
            logger.warning(f"P2 线性头推理失败，P2 退化为透传: {e}")
            return None

"""
意图识别器 —— 本地 FAISS 向量匹配

负责：
1. FAISS 意图向量索引构建（复用已有 BGE 模型）
2. 否定词过滤 → 直接回退 RAG
3. 意图命中后的复杂性检测（simple vs multi_step）
"""

import time as _perf_time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.core.config import config
from src.modules.chat.schemas import IntentResult
from src.shared.logger import APILogger

logger = APILogger("intent_recognizer")

# ═══════════════════════════════════════════════════════════════════════════════
# 常量 & 模式定义
# ═══════════════════════════════════════════════════════════════════════════════

# 否定词：问的是"政策/流程/规则"，走 RAG 而非远程 API
NEGATION_PATTERNS: List[str] = [
    "退货政策",
    "退款流程",
    "退货流程",
    "怎么退",
    "如何退",
    "退货条件",
    "退款规则",
    "退换政策",
    "什么是",
    "说明",
]

# 信号1：问题里包含推理/多步关键词 → 不是一次 tool 调用能搞定的
REACT_TRIGGER_PATTERNS: List[str] = [
    "为什么",
    "怎么办",
    "怎么处理",
    "怎么操作",
    "怎么解决",
    "帮我处理",
    "帮我操作",
    "帮我解决",
    "同时",
    "并且",
    "还要",
    "另外",
    "然后再",
    "能不能",
    "可以吗",
    "行不行",  # 需要判断 → 可能需要 RAG 查政策
    "哪里有问题",
    "什么原因",
    "怎么回事",  # 需要诊断
]

# 信号2：某些意图本质上是多步骤流程，永远不该 direct tool dispatch
ALWAYS_AGENT_ACTIONS: set = {"request-return"}
# 原因：退货需要 (1)查退货政策(RAG) → (2)查订单是否符合条件(tool) → (3)创建退货单(tool)

# 信号3：FAISS 分在阈值边缘（疑似歧义查询），交给 Agent 处理更稳妥
COMPLEXITY_SCORE_THRESHOLD: float = 0.85

# 意图示例（用于构建 FAISS 索引）
INTENT_EXAMPLES: Dict[str, List[str]] = {
    "query-order": [
        "帮我查一下我的订单到哪了",
        "我的订单什么时候发货",
        "看看我最近买了什么东西",
        "查询订单状态",
    ],
    "check-shipping": [
        "快递现在在什么地方，物流信息",
        "我的包裹到哪了，查物流",
        "什么时候能送到",
        "配送进度怎么查",
    ],
    "request-return": [
        "我要退货退款，这个商品不满意",
        "申请七天无理由退货",
        "想把这个东西退掉，怎么操作",
        "退款什么时候到账",
    ],
    "check-balance": [
        "我账户里还有多少余额",
        "查一下我的积分有多少",
        "我的钱包余额",
        "账户资产查询",
    ],
    "coupon-inquiry": [
        "我有什么优惠券可以用",
        "领取优惠券在哪里领",
        "看看有没有满减券",
        "红包怎么用",
    ],
}


class IntentRecognizer:
    """意图识别器 —— 本地 FAISS 向量匹配 + LLM 兜底"""

    # FAISS 意图向量索引（类级别共享，所有实例复用同一份索引）
    _faiss_index: Optional[faiss.IndexFlatIP] = None
    _intent_actions: List[str] = []  # 意图标签列表，与 FAISS 索引行对齐
    _intent_dim: int = 0  # BGE 向量维度，首次构建时推断

    def __init__(self, embedding_service, llm_service=None):
        """
        Args:
            embedding_service: EmbeddingService 实例（必需，用于向量化）
            llm_service: LLMService 实例（可选，用于 LLM 模式意图识别/参数抽取）
        """
        self._embedding_service = embedding_service
        self._llm_service = llm_service

        self._intent_examples: Dict[str, List[str]] = INTENT_EXAMPLES

    # ════════════════════════════════════════════════════════════════════════
    # FAISS 意图索引
    # ════════════════════════════════════════════════════════════════════════

    async def warmup(self):
        """预热 FAISS 意图索引（在启动时调用，避免首次请求等待）"""
        await self._init_faiss_intent_index()

    @classmethod
    def warmup_sync(cls, embedding_service):
        """同步预热 FAISS 意图索引（用于 lifespan 中，不依赖事件循环）"""
        if cls._faiss_index is not None:
            return
        try:
            emb = embedding_service.get_embeddings()
            actions_flat: List[str] = []
            examples_flat: List[str] = []
            for action, phrases in INTENT_EXAMPLES.items():
                for phrase in phrases:
                    actions_flat.append(action)
                    examples_flat.append(phrase)

            vecs = emb.embed_documents(examples_flat)  # 同步版本
            vecs_np = np.array(vecs, dtype=np.float32)
            dim = vecs_np.shape[1]
            cls._intent_dim = dim
            index = faiss.IndexFlatIP(dim)
            index.add(vecs_np)
            cls._faiss_index = index
            cls._intent_actions = actions_flat
            logger.info(
                f"FAISS意图索引构建完成, dim={dim}, "
                f"intents={len(INTENT_EXAMPLES)}, total_vectors={len(actions_flat)}"
            )
        except Exception as e:
            logger.warning(f"FAISS意图索引构建失败: {e}")

    async def _init_faiss_intent_index(self):
        """初始化 FAISS 意图向量索引（懒加载，复用已有BGE模型）"""
        if IntentRecognizer._faiss_index is not None:
            return
        if not self._embedding_service:
            logger.warning("Embedding服务未初始化，跳过FAISS意图索引构建")
            return
        try:
            emb = self._embedding_service.get_embeddings()

            # 展开：每个意图有 N 种提问方式 → 每个方式一条 FAISS 行
            actions_flat: List[str] = []
            examples_flat: List[str] = []
            for action, phrases in self._intent_examples.items():
                for phrase in phrases:
                    actions_flat.append(action)
                    examples_flat.append(phrase)

            # 批量编码所有意图示例 → numpy 矩阵
            vecs = await emb.aembed_documents(examples_flat)
            vecs_np = np.array(vecs, dtype=np.float32)

            # BGE 输出已 L2 归一化，用 IndexFlatIP（内积 = 余弦相似度）
            dim = vecs_np.shape[1]
            IntentRecognizer._intent_dim = dim
            index = faiss.IndexFlatIP(dim)  # Inner Product on normalized vectors = Cosine
            index.add(vecs_np)  # FAISS 要求 contiguous float32

            IntentRecognizer._faiss_index = index
            IntentRecognizer._intent_actions = actions_flat
            logger.info(
                f"FAISS意图索引构建完成, dim={dim}, "
                f"intents={len(self._intent_examples)}, total_vectors={len(actions_flat)}"
            )
        except Exception as e:
            logger.warning(f"FAISS意图索引构建失败: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # 复杂性检测
    # ════════════════════════════════════════════════════════════════════════

    def assess_complexity(
        self, message: str, action: str, similarity_score: float
    ) -> Tuple[str, str]:
        """
        意图命中后，判断是否需要走 ReAct Agent 而非直接 tool 调用。

        三个信号综合判断：
        1. 意图类型：某些 action 本质多步 (如 request_return)
        2. 关键词模式：含"为什么/怎么办/帮我"等推理/多步信号
        3. FAISS 相似分：低分(0.75~0.85)意味着表达模糊 → 可能需要 RAG 补全

        Returns:
            (complexity_label, reason_str)
            - "simple": 直接 tool dispatch
            - "multi_step": 需要 tool+RAG，交给 Agent
        """
        reasons = []
        complexity_score = 0  # 每命中一个信号 +1

        # 信号1：该意图永远需要 Agent
        if action in ALWAYS_AGENT_ACTIONS:
            reasons.append(f"{action} 是多步骤意图（查政策→验条件→执行）")
            complexity_score += 3  # 强信号

        # 信号2：含推理/多步关键词
        matched_triggers = [p for p in REACT_TRIGGER_PATTERNS if p in message]
        if matched_triggers:
            reasons.append(f"含推理/多步关键词: {matched_triggers}")
            complexity_score += 1

        # 信号3：FAISS 相似分在阈值边缘（不太确定用户到底要什么）
        if similarity_score < COMPLEXITY_SCORE_THRESHOLD:
            reasons.append(
                f"相似分 {similarity_score:.3f} < {COMPLEXITY_SCORE_THRESHOLD}，可能存在歧义"
            )
            complexity_score += 1

        complexity = "multi_step" if complexity_score >= 1 else "simple"
        reason = "; ".join(reasons) if reasons else "表达清晰，直接调用工具即可"

        logger.info(
            "复杂性检测",
            action=action,
            score=round(similarity_score, 3),
            complexity=complexity,
            triggers=complexity_score,
            reason=reason,
        )
        return complexity, reason

    # ════════════════════════════════════════════════════════════════════════
    # 统一入口
    # ════════════════════════════════════════════════════════════════════════
    # 统一入口
    # ════════════════════════════════════════════════════════════════════════

    async def recognize(self, message: str, langfuse_handler=None) -> IntentResult:
        """本地意图识别（否定过滤 + FAISS 向量匹配）。无 LLM 调用，延迟 < 5ms"""
        t_total_start = _perf_time.perf_counter()

        # ---- 第一层：否定模式过滤（咨询类问题走 RAG） ----
        for neg in NEGATION_PATTERNS:
            if neg in message:
                dt = (_perf_time.perf_counter() - t_total_start) * 1000
                print(f"[⏱] 意图识别 [否定命中] {dt:.0f}ms → rag_answer")
                return IntentResult(intent="rag_answer")

        # ---- 第二层：FAISS 向量语义匹配 ----
        t0 = _perf_time.perf_counter()
        await self._init_faiss_intent_index()
        t_init_faiss = (_perf_time.perf_counter() - t0) * 1000

        t_get_emb = 0.0
        t_embed = 0.0

        if IntentRecognizer._faiss_index is not None and self._embedding_service:
            try:
                t0 = _perf_time.perf_counter()
                emb = self._embedding_service.get_embeddings()
                t_get_emb = (_perf_time.perf_counter() - t0) * 1000

                t0 = _perf_time.perf_counter()
                query_vec = await emb.aembed_query(message)
                t_embed = (_perf_time.perf_counter() - t0) * 1000

                query_vec_np = np.array([query_vec], dtype=np.float32)

                k = min(2, IntentRecognizer._faiss_index.ntotal)
                scores, indices = IntentRecognizer._faiss_index.search(query_vec_np, k)

                best_score = float(scores[0][0])
                best_idx = int(indices[0][0])
                best_action = IntentRecognizer._intent_actions[best_idx]

                if k >= 2:
                    second_score = float(scores[0][1])
                    second_action = IntentRecognizer._intent_actions[int(indices[0][1])]
                    logger.debug(
                        f"FAISS向量匹配 top-2: ({best_action},{best_score:.3f}) "
                        f"({second_action},{second_score:.3f})"
                    )

                threshold = getattr(config, "INTENT_VECTOR_SIMILARITY_THRESHOLD", 0.65)
                if best_score > threshold:
                    complexity, reason = self.assess_complexity(message, best_action, best_score)
                    t_total = (_perf_time.perf_counter() - t_total_start) * 1000
                    print(
                        f"[⏱] 意图识别 [FAISS命中] total={t_total:.0f}ms "
                        f"init_faiss={t_init_faiss:.0f}ms get_emb={t_get_emb:.0f}ms "
                        f"embed={t_embed:.0f}ms action={best_action}"
                    )
                    logger.info(
                        "本地意图识别(FAISS向量命中)",
                        action=best_action,
                        score=round(best_score, 3),
                        complexity=complexity,
                    )
                    return IntentResult(
                        intent="call_remote_api",
                        action=best_action,
                        similarity_score=round(best_score, 4),
                        complexity=complexity,
                        complexity_reason=reason,
                    )
            except Exception as e:
                logger.warning(f"FAISS向量意图识别失败: {e}")

        # ---- 默认：走RAG回答 ----
        t_total = (_perf_time.perf_counter() - t_total_start) * 1000
        logger.info(
            f"意图识别 [默认rag] total={t_total:.0f}ms "
            f"init_faiss={t_init_faiss:.0f}ms get_emb={t_get_emb:.0f}ms "
            f"embed={t_embed:.0f}ms"
        )
        return IntentResult(intent="rag_answer")

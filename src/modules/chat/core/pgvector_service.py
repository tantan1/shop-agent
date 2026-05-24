"""
PostgreSQL pgvector 向量数据库服务模块
提供与 MilvusService 相同接口的向量存储和混合检索功能

Dense + BM25(ts_rank) → RRF 融合检索

依赖: pip install pgvector psycopg2
"""
import math
import re
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool
from langchain_core.documents import Document

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("pgvector_service")

# 搜索参数常量
DEFAULT_TOP_K = 3
MAX_TOP_K = 1000
RRF_K_DEFAULT = 60


class PgVectorService:
    """PostgreSQL pgvector 向量数据库服务

    接口与 MilvusService 完全一致：
    - search_similar(query_embedding, top_k, output_fields) → List[Document]
    - hybrid_search(query_embedding, query_text, top_k, rrf_k) → List[Document]
    - insert_documents(texts, embeddings, metadata_list)
    - flush()
    - close()

    Dense:   pgvector HNSW 索引 (vector_cosine_ops)
    Sparse:  PostgreSQL ts_rank (GIN 索引) 作为 BM25 替代
    RRF 融合: Python 侧计算
    """

    _instance: Optional["PgVectorService"] = None
    _initialized: bool = False
    _service_initialized: bool = False

    def __init__(self):
        if PgVectorService._service_initialized:
            raise RuntimeError("请使用 get_instance() 获取 PgVectorService 实例")
        PgVectorService._service_initialized = True
        self._pool: Optional[SimpleConnectionPool] = None
        self._table_name = chat_config.pgvector_table

    @classmethod
    def get_instance(cls) -> "PgVectorService":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
        return cls._instance

    # ════════════════════════════════════════════════════════════════════
    # 连接与初始化
    # ════════════════════════════════════════════════════════════════════

    def initialize(self):
        """初始化 PostgreSQL 连接、pgvector 扩展及数据表"""
        if self._initialized:
            return

        try:
            # 连接池
            self._pool = SimpleConnectionPool(
                minconn=2,
                maxconn=10,
                host=chat_config.pgvector_host,
                port=chat_config.pgvector_port,
                dbname=chat_config.pgvector_db,
                user=chat_config.pgvector_user,
                password=chat_config.pgvector_password,
            )

            # 初始化 schema
            conn = self._pool.getconn()
            try:
                self._init_schema(conn)
            finally:
                self._pool.putconn(conn)

            self._initialized = True
            dim = chat_config.embedding_dimension
            logger.info(
                f"PgVectorService 初始化完成: "
                f"host={chat_config.pgvector_host}:{chat_config.pgvector_port}, "
                f"db={chat_config.pgvector_db}, table={self._table_name}, dim={dim}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize PgVectorService: {str(e)}")
            raise

    def _init_schema(self, conn):
        """创建 pgvector 扩展、数据表及索引"""
        dim = chat_config.embedding_dimension
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector({dim}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            # HNSW 向量索引
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding
                ON {self._table_name}
                USING hnsw (embedding vector_cosine_ops)
            """)
            # GIN 全文搜索索引（BM25 替代）
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_text
                ON {self._table_name}
                USING gin (to_tsvector('simple', text))
            """)
        conn.commit()

    def _get_conn(self):
        if self._pool is None:
            self.initialize()
        return self._pool.getconn()

    def _put_conn(self, conn):
        if self._pool:
            self._pool.putconn(conn)

    # ════════════════════════════════════════════════════════════════════
    # 参数校验
    # ════════════════════════════════════════════════════════════════════

    def _validate_search_params(self, query_embedding: List[float], top_k: int) -> None:
        if not query_embedding:
            raise ValueError("query_embedding 不能为空")
        if len(query_embedding) != chat_config.embedding_dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {chat_config.embedding_dimension}, "
                f"实际 {len(query_embedding)}"
            )
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if top_k > MAX_TOP_K:
            raise ValueError(f"top_k 不能超过 {MAX_TOP_K}")

    def _validate_insert_params(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ) -> None:
        if not texts:
            logger.warning("尝试插入空文档列表")
            return
        if len(texts) != len(embeddings):
            raise ValueError(f"输入列表长度不一致: texts={len(texts)}, embeddings={len(embeddings)}")
        if len(texts) != len(metadata_list):
            raise ValueError(f"输入列表长度不一致: texts={len(texts)}, metadata_list={len(metadata_list)}")

    # ════════════════════════════════════════════════════════════════════
    # Dense 向量检索（纯 HNSW 余弦相似度）
    # ════════════════════════════════════════════════════════════════════

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = DEFAULT_TOP_K,
        output_fields: List[str] = None
    ) -> List[Document]:
        """纯向量检索（pgvector HNSW 余弦相似度）"""
        try:
            self._validate_search_params(query_embedding, top_k)

            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # pgvector <=> 操作符 = 余弦距离；1 - 距离 = 余弦相似度
                    cur.execute(
                        f"""
                        SELECT text, metadata,
                               1 - (embedding <=> %s::vector) AS score
                        FROM {self._table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, query_embedding, top_k)
                    )
                    rows = cur.fetchall()
            finally:
                self._put_conn(conn)

            documents = []
            for row in rows:
                metadata = row.get("metadata") or {}
                metadata["distance"] = row["score"]
                documents.append(Document(
                    page_content=row["text"] or "",
                    metadata=metadata
                ))
            return documents

        except Exception as e:
            logger.error(f"Pgvector search_similar failed: {str(e)}")
            raise

    # ════════════════════════════════════════════════════════════════════
    # Sparse 检索（PostgreSQL ts_rank 模拟 BM25）
    # ════════════════════════════════════════════════════════════════════

    def _sparse_search(
        self,
        query_text: str,
        top_k: int,
        conn
    ) -> Dict[int, float]:
        """使用 PostgreSQL ts_rank 做关键词稀疏检索

        返回 {doc_id: ts_rank_score} 映射，score 已归一化
        """
        # 将中文查询转为 PostgreSQL 可识别的 token 形式
        ts_query = self._build_tsquery(query_text)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id,
                       ts_rank(to_tsvector('simple', text), to_tsquery('simple', %s)) AS score
                FROM {self._table_name}
                WHERE to_tsvector('simple', text) @@ to_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (ts_query, ts_query, top_k)
            )
            # 归一化到 [0, 1]
            results = {}
            rows = cur.fetchall()
            if not rows:
                return results
            max_score = max(r[1] for r in rows) or 1.0
            for doc_id, score in rows:
                results[doc_id] = score / max_score
            return results

    def _build_tsquery(self, text: str) -> str:
        """将查询文本转为 PostgreSQL tsquery 格式

        PostgreSQL tsquery 使用 & (AND) 和 | (OR) 连接词
        对于中文，简单按字拆分并用 & 连接
        """
        # 去除特殊字符，按字/词拆分
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text)
        if not tokens:
            return "'0'"
        # 中文按单字拆分，英文保持单词
        parts = []
        for token in tokens:
            if re.match(r'[\u4e00-\u9fff]', token):
                # 中文：按连续字保持、用 & 连接（AND 语义）
                if len(token) <= 4:
                    parts.append(token)
                else:
                    # 长中文短语，取前2字和后2字
                    parts.append(token[:2])
                    parts.append(token[-2:])
            else:
                parts.append(token)
        return " & ".join(parts)

    # ════════════════════════════════════════════════════════════════════
    # RRF 融合（Python 侧）
    # ════════════════════════════════════════════════════════════════════

    def _rrf_fusion(
        self,
        dense_results: List[Dict],
        sparse_scores: Dict[int, float],
        top_k: int,
        rrf_k: int = RRF_K_DEFAULT
    ) -> List[Document]:
        """RRF (Reciprocal Rank Fusion) 融合 Dense + Sparse 结果

        RRF(d) = sum_{r in rankings} 1 / (k + rank(d, r))
        """
        # 构建稠密排名（1-based）
        dense_ranks = {}
        for rank, doc in enumerate(dense_results, start=1):
            dense_ranks[doc["id"]] = rank

        # 构建稀疏排名（1-based，按 score 降序）
        sorted_sparse = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)
        sparse_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(sorted_sparse, start=1)}

        # RRF 计算
        rrf_scores = {}
        all_doc_ids = set(list(dense_ranks.keys()) + list(sparse_ranks.keys()))

        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score

        # 按 RRF 分数降序取 top_k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 创建 id → Document 映射
        doc_map = {d["id"]: d for d in dense_results}

        # 对于 sparse 独有的文档，从 DB 取回
        missing_ids = [did for did, _ in sorted_docs if did not in doc_map]
        if missing_ids:
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT id, text, metadata FROM {self._table_name} WHERE id = ANY(%s)",
                        (missing_ids,)
                    )
                    for row in cur.fetchall():
                        doc_map[row["id"]] = row
            finally:
                self._put_conn(conn)

        documents = []
        for doc_id, rrf_score in sorted_docs:
            row = doc_map.get(doc_id)
            if row is None:
                continue
            metadata = row.get("metadata") or {}
            metadata["distance"] = rrf_score
            documents.append(Document(
                page_content=row["text"] or "",
                metadata=metadata
            ))

        return documents

    # ════════════════════════════════════════════════════════════════════
    # 混合检索（Dense + Sparse → RRF 融合）
    # ════════════════════════════════════════════════════════════════════

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = RRF_K_DEFAULT,
    ) -> List[Document]:
        """混合检索：Dense (HNSW余弦) + Sparse (ts_rank) → RRF 融合

        Args:
            query_embedding: 稠密查询向量
            query_text: 查询原始文本（用于稀疏检索）
            top_k: 返回数量
            rrf_k: RRF k 参数（越小高分权重越大，默认 60）

        Returns:
            已按 RRF 融合排序的文档列表
        """
        try:
            if not query_embedding:
                raise ValueError("query_embedding 不能为空")
            if not query_text:
                raise ValueError("query_text 不能为空")
            if top_k <= 0:
                raise ValueError("top_k 必须大于 0")

            conn = self._get_conn()
            try:
                # 1. Dense 检索（取 top_k * 2 候选）
                dense_results = []
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT id, text, metadata,
                               1 - (embedding <=> %s::vector) AS score
                        FROM {self._table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, query_embedding, top_k * 2)
                    )
                    dense_results = cur.fetchall()

                # 2. Sparse 检索（ts_rank，取 top_k * 2 候选）
                sparse_scores = self._sparse_search(query_text, top_k * 2, conn)

                # 3. RRF 融合
                documents = self._rrf_fusion(dense_results, sparse_scores, top_k, rrf_k)

                logger.debug(f"混合检索返回 {len(documents)} 条结果")
                return documents

            finally:
                self._put_conn(conn)

        except Exception as e:
            logger.error(f"Pgvector hybrid_search failed: {str(e)}")
            raise

    # ════════════════════════════════════════════════════════════════════
    # 文档插入
    # ════════════════════════════════════════════════════════════════════

    def insert_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ):
        """批量插入文档

        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表（list of list[float]）
            metadata_list: 元数据列表
        """
        try:
            self._validate_insert_params(texts, embeddings, metadata_list)
            if not texts:
                return

            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    rows = [
                        (text, embedding, metadata)
                        for text, embedding, metadata in zip(texts, embeddings, metadata_list)
                    ]
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {self._table_name} (text, embedding, metadata)
                        VALUES %s
                        """,
                        rows,
                        template="(%s, %s::vector, %s::jsonb)"
                    )
                conn.commit()
                logger.debug(f"插入 {len(texts)} 条文档到 pgvector")
            finally:
                self._put_conn(conn)

        except Exception as e:
            logger.error(f"Pgvector insert_documents failed: {str(e)}")
            raise

    # ════════════════════════════════════════════════════════════════════
    # 管理方法
    # ════════════════════════════════════════════════════════════════════

    def flush(self):
        """pgvector 每次 insert 已自动 commit，此方法为接口兼容保留"""
        pass

    def close(self):
        """关闭连接池"""
        if self._pool:
            self._pool.closeall()
            self._pool = None
        self._initialized = False

    @property
    def collection(self):
        """接口兼容 MilvusService 的 collection 属性，返回自身"""
        return self

"""
Milvus 向量数据库服务模块
提供文档存储和相似性搜索功能
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility,
    Function, FunctionType, AnnSearchRequest, RRFRanker
)

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("milvus_service")


class MetricType(Enum):
    """度量类型枚举"""
    COSINE = "COSINE"
    IP = "IP"
    L2 = "L2"


# HNSW 索引参数常量
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
SEARCH_EF_MIN = 50
DEFAULT_TOP_K = 3
MAX_TOP_K = 1000


class MilvusService:
    """Milvus 向量数据库服务"""
    
    _instance: Optional["MilvusService"] = None
    _collection: Optional[Collection] = None
    _initialized: bool = False
    _service_initialized: bool = False  # 区分服务初始化 vs 单例初始化
    
    def __init__(self):
        """初始化 Milvus 服务"""
        if MilvusService._service_initialized:
            raise RuntimeError("请使用 get_instance() 获取 MilvusService 实例")
        MilvusService._service_initialized = True
    
    @classmethod
    def get_instance(cls) -> "MilvusService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
        return cls._instance
    
    def initialize(self):
        """初始化 Milvus 连接和集合"""
        if self._initialized:
            return
        
        try:
            # 连接 Milvus
            connections.connect(
                "default",
                host=chat_config.milvus_host,
                port=chat_config.milvus_port
            )
            
            # 初始化集合
            self._initialize_collection()
            self._initialized = True
            logger.info(
                f"MilvusService 初始化完成: "
                f"host={chat_config.milvus_host}:{chat_config.milvus_port}, "
                f"collection={chat_config.milvus_collection_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize MilvusService: {str(e)}")
            raise
    
    def _ensure_indexes(self, collection: Collection) -> None:
        """确保集合有必要的索引（用于已有集合）"""
        indexed_fields = {idx.field_name for idx in collection.indexes}

        # 检查稠密向量索引
        if "embedding" not in indexed_fields:
            dense_index_params = {
                "metric_type": MetricType.COSINE.value,
                "index_type": "HNSW",
                "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION}
            }
            collection.create_index("embedding", dense_index_params)
            logger.info("已为 embedding 字段创建稠密向量索引")

        # 检查稀疏向量索引（BM25 检索必需）
        if "sparse_bm25" not in indexed_fields:
            sparse_index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25"
            }
            collection.create_index("sparse_bm25", sparse_index_params)
            logger.info("已为 sparse_bm25 字段创建稀疏向量索引")
    
    def _validate_collection_schema(self, collection: Collection) -> bool:
        """验证集合 schema 是否匹配"""
        schema = collection.schema
        fields = {f.name: f for f in schema.fields}

        # 检查 embedding 字段维度
        embedding_field = fields.get("embedding")
        if embedding_field is None:
            return False
        if embedding_field.params.get("dim") != chat_config.embedding_dimension:
            return False

        # 检查 sparse_bm25 字段是否存在（混合检索必需）
        if "sparse_bm25" not in fields:
            return False

        return True
    
    def _create_collection(self, collection_name: str) -> Collection:
        """创建新的集合（支持混合检索：Dense + Sparse BM25）"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=chat_config.embedding_dimension),
            FieldSchema(name="sparse_bm25", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        # 创建 BM25 函数（Milvus 2.5+ 原生支持）
        bm25_function = Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_bm25"]
        )

        schema = CollectionSchema(
            fields,
            "文档嵌入存储（支持混合检索）",
            functions=[bm25_function]  # PyMilvus 2.5+ 推荐方式
        )
        collection = Collection(collection_name, schema)

        # 创建稠密向量索引（稀疏向量索引由 BM25 Function 自动创建，无需手动创建）
        dense_index_params = {
            "metric_type": MetricType.COSINE.value,
            "index_type": "HNSW",
            "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION}
        }
        collection.create_index("embedding", dense_index_params)

        collection.load()
        logger.info(f"集合 {collection_name} 创建成功，已启用混合检索（Dense + Sparse BM25）")
        return collection
    
    def _initialize_collection(self):
        """初始化 Milvus 集合"""
        collection_name = chat_config.milvus_collection_name
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            
            # 检查现有集合的维度是否匹配
            if not self._validate_collection_schema(collection):
                logger.warning(
                    f"集合 {collection_name} schema 不匹配，将重新创建"
                )
                utility.drop_collection(collection_name)
            else:
                self._collection = collection
        
        # 创建新的集合（如果不存在或被删除）
        if self._collection is None:
            self._collection = self._create_collection(collection_name)
        else:
            # 确保已存在的集合有必要的索引
            self._ensure_indexes(self._collection)
            # 确保已存在的集合已加载到内存
            self._collection.load()
    
    @property
    def collection(self) -> Collection:
        """获取集合实例"""
        if not self._initialized:
            self.initialize()
        return self._collection
    
    def _create_search_params(self, top_k: int) -> Dict[str, Any]:
        """创建搜索参数"""
        return {
            "metric_type": MetricType.COSINE.value,
            "params": {"ef": max(SEARCH_EF_MIN, top_k * 2)}
        }
    
    def _validate_search_params(self, query_embedding: List[float], top_k: int) -> None:
        """验证搜索参数"""
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
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = DEFAULT_TOP_K,
        output_fields: List[str] = None
    ) -> List[Document]:
        """
        搜索相似文档（纯向量检索）
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            output_fields: 输出字段列表
            
        Returns:
            相似文档列表
        """
        try:
            self._validate_search_params(query_embedding, top_k)
            
            if output_fields is None:
                output_fields = ["text", "metadata"]
            
            if not output_fields:
                raise ValueError("output_fields 不能为空列表")
            
            search_params = self._create_search_params(top_k)
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )

            documents = []
            for hits in results:
                for hit in hits:
                    metadata = hit.entity.get("metadata") or {}
                    metadata["distance"] = hit.distance
                    documents.append(Document(
                        page_content=hit.entity.get("text", ""),
                        metadata=metadata
                    ))

            return documents
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            raise

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        混合检索（Dense向量 + Sparse BM25）
        
        Milvus 2.6+ 原生支持，无需手动 RRF 融合。
        
        Args:
            query_embedding: 查询向量（稠密）
            query_text: 查询文本（用于稀疏BM25检索）
            top_k: 返回数量
            rrf_k: RRFRanker k 参数（越小高分权重越大，推荐 10~100，默认 60）
            
        Returns:
            相似文档列表（已按 RRF 融合排序）
        """
        try:
            if not query_embedding:
                raise ValueError("query_embedding 不能为空")
            if not query_text:
                raise ValueError("query_text 不能为空")
            if top_k <= 0:
                raise ValueError("top_k 必须大于 0")
            
            # 构建稠密向量检索请求
            dense_request = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": MetricType.COSINE.value,
                       "params": {"ef": max(SEARCH_EF_MIN, top_k * 2)}},
                limit=top_k * 2
            )
            
            # 构建稀疏向量检索请求（BM25）
            sparse_request = AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_bm25",
                param={"metric_type": "BM25"},
                limit=top_k * 2
            )
            
            # 执行混合搜索（内置 RRF 融合）
            reranker = RRFRanker(k=rrf_k)
            results = self.collection.hybrid_search(
                reqs=[dense_request, sparse_request],
                rerank=reranker,
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            documents = []
            for hits in results:
                for hit in hits:
                    metadata = hit.entity.get("metadata") or {}
                    metadata["distance"] = hit.distance
                    documents.append(Document(
                        page_content=hit.entity.get("text", ""),
                        metadata=metadata
                    ))
            
            logger.debug(f"混合检索返回 {len(documents)} 条结果")
            return documents
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise
    
    def _validate_insert_params(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ) -> None:
        """验证插入参数"""
        if not texts:
            logger.warning("尝试插入空文档列表")
            return
        
        if len(texts) != len(embeddings):
            raise ValueError(
                f"输入列表长度不一致: texts={len(texts)}, "
                f"embeddings={len(embeddings)}"
            )
        if len(texts) != len(metadata_list):
            raise ValueError(
                f"输入列表长度不一致: texts={len(texts)}, "
                f"metadata_list={len(metadata_list)}"
            )
        
        # 验证嵌入向量维度，并转换 tuple 为 list
        for i, embedding in enumerate(embeddings):
            # 转换为 list（处理可能的 tuple 类型）
            embedding = list(embedding) if isinstance(embedding, tuple) else embedding
            if len(embedding) != chat_config.embedding_dimension:
                raise ValueError(
                    f"第 {i} 个向量维度不匹配: 期望 {chat_config.embedding_dimension}, "
                    f"实际 {len(embedding)}"
                )
    
    def insert_documents(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        metadata_list: List[Dict[str, Any]]
    ):
        """
        批量插入文档
        
        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
            metadata_list: 元数据列表
        """
        try:
            self._validate_insert_params(texts, embeddings, metadata_list)
            
            if not texts:
                return
            
            data = [
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata
                }
                for text, embedding, metadata in zip(texts, embeddings, metadata_list)
            ]
            
            self.collection.insert(data)
                
        except Exception as e:
            logger.error(f"Document insert failed: {str(e)}")
            raise
    
    def flush(self):
        """刷新数据到存储"""
        self.collection.flush()
    
    def close(self):
        """关闭连接"""
        if self._collection:
            self._collection.release()
            self._collection = None
        connections.disconnect("default")
        self._initialized = False
        # 不再重置单例，允许连接复用

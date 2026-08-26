"""
记忆 Milvus 服务：独立管理 memory_blocks Collection
与 RAG 文档 Collection（chat_embeddings）隔离
"""
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src.modules.chat.config import chat_config
from src.shared.logger import APILogger

logger = APILogger("memory_milvus_service")


class MemoryBlockType:
    """记忆块类型常量"""
    PREFERENCE = "preference"
    ORDER = "order"
    COMPLAINT = "complaint"
    RESOLUTION = "resolution"
    INTERACTION = "interaction"
    PENDING = "pending"
    NEXT_ACTION = "next_action"


@dataclass
class SearchFilter:
    """记忆检索过滤条件"""
    user_id: str
    block_type: Optional[str] = None
    importance_threshold: int = 1
    filter_expr: Optional[str] = None


class MemoryBlockService:
    """记忆块 Milvus 服务（memory_blocks Collection）"""

    _instance: Optional["MemoryBlockService"] = None
    _collection: Optional[Collection] = None
    _initialized: bool = False

    COLLECTION_NAME = "memory_blocks"

    def __init__(self):
        if MemoryBlockService._instance is not None:
            raise RuntimeError("请使用 get_instance() 获取 MemoryBlockService 实例")
        MemoryBlockService._instance = self

    @classmethod
    def get_instance(cls) -> "MemoryBlockService":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def initialize(self) -> None:
        """初始化 memory_blocks Collection"""
        if self._initialized:
            return

        try:
            # 确保 Milvus 连接已建立
            if not connections.has_connection("default"):
                connections.connect(
                    "default",
                    host=chat_config.milvus_host,
                    port=chat_config.milvus_port,
                )

            collection_name = self.COLLECTION_NAME
            if not utility.has_collection(collection_name):
                self._create_collection(collection_name)
            else:
                self._collection = Collection(collection_name)
                self._ensure_indexes(self._collection)

            self._collection.load()
            self._initialized = True
            logger.info(f"MemoryBlockService 初始化完成: collection={collection_name}")
        except Exception as e:
            logger.error(f"MemoryBlockService 初始化失败: {str(e)}")
            raise

    def _create_collection(self, collection_name: str) -> Collection:
        """创建 memory_blocks Collection"""
        fields = [
            FieldSchema(
                name="block_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True
            ),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="block_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="value", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=chat_config.embedding_dimension,
            ),
            FieldSchema(name="importance", dtype=DataType.INT64),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="last_accessed_at", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="access_count", dtype=DataType.INT64),
            FieldSchema(name="expires_at", dtype=DataType.VARCHAR, max_length=64, nullable=True),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(
            fields,
            "分层记忆架构：L2 短期记忆 + L3 长期记忆",
        )
        collection = Collection(collection_name, schema)

        # 向量索引
        collection.create_index(
            "embedding",
            {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )

        # 标量索引
        for field_name in ["user_id", "block_type", "expires_at"]:
            collection.create_index(field_name, {"index_type": "INVERTED"})

        logger.info(f"Collection {collection_name} 创建成功")
        return collection

    def _ensure_indexes(self, collection: Collection) -> None:
        """确保索引存在"""
        indexed_fields = {idx.field_name for idx in collection.indexes}
        if "embedding" not in indexed_fields:
            self._collection.create_index(
                "embedding",
                {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        for field_name in ["user_id", "block_type", "expires_at"]:
            if field_name not in indexed_fields:
                collection.create_index(field_name, {"index_type": "INVERTED"})

    @property
    def collection(self) -> Collection:
        if not self._initialized:
            self.initialize()
        return self._collection

    def insert_block(self, block: Dict[str, Any]) -> str:
        """插入单个记忆块"""
        block_id = block.get("block_id") or str(uuid.uuid4())
        now = datetime.now().isoformat()
        data = {
            "block_id": block_id,
            "user_id": block["user_id"],
            "block_type": block["block_type"],
            "label": block.get("label", ""),
            "value": block.get("value", ""),
            "embedding": block["embedding"],
            "importance": block.get("importance", 3),
            "source": block.get("source", "conversation"),
            "created_at": block.get("created_at", now),
            "updated_at": now,
            "last_accessed_at": now,
            "access_count": block.get("access_count", 0),
            "expires_at": block.get("expires_at"),
            "metadata": block.get("metadata", {}),
        }
        self.collection.insert([data])
        self.collection.flush()
        return block_id

    def search(
        self,
        query_embedding: List[float],
        search_filter: SearchFilter,
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """向量检索记忆块"""
        if output_fields is None:
            output_fields = [
                "block_id", "user_id", "block_type", "label", "value",
                "importance", "source", "created_at", "expires_at",
                "last_accessed_at", "access_count", "metadata",
            ]

        filters = [f'user_id == "{search_filter.user_id}"', f"importance >= {search_filter.importance_threshold}"]

        if search_filter.block_type:
            filters.append(f'block_type == "{search_filter.block_type}"')

        if search_filter.filter_expr:
            filters.append(search_filter.filter_expr)

        expr = " and ".join(filters)

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": max(50, top_k * 2)}},
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )

        blocks = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                block = {field: entity.get(field) for field in output_fields}
                block["score"] = hit.distance
                blocks.append(block)

        return blocks

    def delete(self, filter_expr: str) -> int:
        """按条件删除记忆块，返回删除数量"""
        results = self.collection.query(
            expr=filter_expr,
            output_fields=["block_id"],
        )
        if not results:
            return 0
        ids = [r["block_id"] for r in results]
        self.collection.delete(f"block_id in {ids}")
        self.collection.flush()
        return len(ids)

    def update_access(self, block_id: str) -> None:
        """更新访问计数和最后访问时间"""
        now = datetime.now().isoformat()
        self.collection.update(
            expr=f'block_id == "{block_id}"',
            field_values={
                "last_accessed_at": now,
                "access_count": {"$inc": 1},
            },
        )

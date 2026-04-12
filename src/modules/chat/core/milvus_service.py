"""
Milvus 向量数据库服务模块
提供文档存储和相似性搜索功能
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("milvus_service")


class MilvusService:
    """Milvus 向量数据库服务"""
    
    _instance = None
    _collection: Optional[Collection] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "MilvusService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
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
            logger.info("MilvusService initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize MilvusService: {str(e)}")
            raise
    
    def _initialize_collection(self):
        """初始化 Milvus 集合"""
        collection_name = chat_config.milvus_collection_name
        
        if utility.has_collection(collection_name):
            self._collection = Collection(collection_name)
            
            # 检查现有集合的维度是否匹配
            schema = self._collection.schema
            embedding_field = next((f for f in schema.fields if f.name == "embedding"), None)
            if embedding_field and embedding_field.params.get("dim") != chat_config.embedding_dimension:
                logger.warning(f"集合维度不匹配 ({embedding_field.params.get('dim')} vs {chat_config.embedding_dimension})，删除旧集合...")
                utility.drop_collection(collection_name)
                self._collection = None
        
        # 创建新的集合（如果不存在或被删除）
        if self._collection is None:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=chat_config.embedding_dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]

            schema = CollectionSchema(fields, "文档嵌入存储")
            self._collection = Collection(collection_name, schema)

            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            self._collection.create_index("embedding", index_params)
        
        self._collection.load()
    
    @property
    def collection(self) -> Collection:
        """获取集合实例"""
        if not self._initialized:
            self.initialize()
        return self._collection
    
    def _create_search_params(self, top_k: int = 3) -> Dict[str, Any]:
        """创建搜索参数"""
        return {
            "metric_type": "COSINE",
            "params": {"ef": max(50, top_k * 2)}
        }
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 3,
        output_fields: List[str] = None
    ) -> List[Document]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            output_fields: 输出字段列表
            
        Returns:
            相似文档列表
        """
        try:
            if output_fields is None:
                output_fields = ["text", "metadata"]
            
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
                    documents.append(Document(
                        page_content=hit.entity.get("text", ""),
                        metadata=metadata
                    ))

            return documents
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            raise
    
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
            data = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadata_list)):
                data.append({
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata
                })
            
            if data:
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
        connections.disconnect("default")
        self._initialized = False
        MilvusService._instance = None
        MilvusService._collection = None

from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.chat.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    InsertDocumentRequest,
    ChatRequest,
    ChatResponse,
    RefundConfirmRequest,
    ItemSearchResponse,
)
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.pgvector_service import PgVectorService
from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.tool_registry import ToolService
from src.modules.chat.core.item_service import ItemService
from src.modules.chat.core.intent_recognizer import IntentRecognizer
from src.modules.chat.core.document_service import DocumentService
from src.modules.chat.core.redis_cache_service import get_redis_cache_service
from src.modules.chat.agent.orchestrator import AgentOrchestrator
from src.shared.exceptions import ValidationException
from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("chatagent_service")


def _create_vector_service():
    """工厂方法：根据 VECTOR_STORE_PROVIDER 创建对应的向量数据库服务"""
    provider = chat_config.vector_store_provider
    if provider == "pgvector":
        logger.info("使用 PostgreSQL pgvector 作为向量存储")
        return PgVectorService.get_instance()
    else:
        logger.info("使用 Milvus 作为向量存储")
        return MilvusService.get_instance()


class ChatAgentService:
    """智能客服服务 —— 服务初始化、依赖注入、对外委托"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self._embedding_service = None
        self._vector_service = None  # MilvusService | PgVectorService
        self._llm_service = None
        self._redis_cache_service = None
        self._tool_service = ToolService()
        self._item_service = None
        self._intent_recognizer = None
        self._document_service = None
        self._orchestrator = None
        self._initialized = False

    # =========================================================================
    # 属性（懒加载单例服务）
    # =========================================================================

    @property
    def embedding(self):
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService.get_instance()
        return self._embedding_service.get_embeddings()

    @property
    def embeddings(self):
        return self.embedding

    @property
    def vector_service(self):
        """统一向量服务属性（Milvus 或 PgVector）"""
        if self._vector_service is None:
            self._vector_service = _create_vector_service()
        return self._vector_service

    # 向后兼容的 milvus 别名
    @property
    def milvus(self):
        return self.vector_service

    @property
    def llm(self) -> LLMService:
        if self._llm_service is None:
            self._llm_service = LLMService.get_instance()
        return self._llm_service

    # =========================================================================
    # 生命周期
    # =========================================================================

    async def close(self):
        if self._vector_service:
            self._vector_service.close()
        if self._llm_service:
            self._llm_service.close()
        if self._redis_cache_service:
            self._redis_cache_service.close()
        self._initialized = False

    async def _initialize(self):
        if self._initialized:
            return

        try:
            if not chat_config.volcengine_api_key:
                raise ValidationException("火山引擎API密钥未配置", "请设置VOLCENGINE_API_KEY环境变量")

            # 初始化 LLM 服务（单例）
            self._llm_service = LLMService.get_instance()
            self._llm_service.initialize()

            # 初始化向量数据库服务（Milvus 或 PgVector）
            self._vector_service = _create_vector_service()
            self._vector_service.initialize()

            # 初始化嵌入服务（单例）
            self._embedding_service = EmbeddingService.get_instance()

            # 初始化商品服务（注入 vector_service，兼容 Milvus/PgVector）
            self._item_service = ItemService(vector_service=self._vector_service)

            # 初始化 Redis 缓存服务（单例）
            if chat_config.redis_vector_enabled:
                self._redis_cache_service = get_redis_cache_service()
                if self._redis_cache_service.is_available:
                    logger.info("Redis 缓存服务初始化成功")
                else:
                    logger.warning("Redis 缓存服务不可用，将禁用问题去重功能")

            # 初始化意图识别器
            self._intent_recognizer = IntentRecognizer(
                embedding_service=self._embedding_service,
                llm_service=self._llm_service,
            )

            # 初始化文档服务
            self._document_service = DocumentService(
                embedding_service=self._embedding_service,
                milvus_service=self._vector_service,
                item_service=self._item_service,
            )

            # 初始化 Agent 编排器（依赖已就绪的所有子服务）
            self._orchestrator = AgentOrchestrator(
                llm_service=self._llm_service,
                embedding_service=self._embedding_service,
                milvus_service=self._vector_service,
                intent_recognizer=self._intent_recognizer,
                tool_service=self._tool_service,
                redis_cache_service=self._redis_cache_service,
            )

            self._initialized = True
            logger.info("ChatAgentService initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize ChatAgentService: {str(e)}")
            raise ValidationException("初始化聊天服务失败", str(e))

    # =========================================================================
    # 对话入口（委托给 AgentOrchestrator）
    # =========================================================================

    async def chat(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """旧版 RAG 聊天接口"""
        await self._initialize()
        return await self._orchestrator.chat_rag(request)

    async def chat_with_agent(self, request: ChatRequest,
                            experiment_assignment=None) -> ChatResponse:
        """通用 Agent 多步骤对话（含意图识别 + 路由分发）"""
        try:
            await self._initialize()
            return await self._orchestrator.chat_with_agent(
                request, experiment_assignment=experiment_assignment
            )
        except ValidationException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"电商 Agent 执行失败: {str(e)}\n{traceback.format_exc()}")
            logger.log_business_event(
                "电商Agent对话",
                success=False,
                error=str(e),
                message_length=len(request.message)
            )
            raise ValidationException("电商客服对话失败", str(e))

    # =========================================================================
    # 文档插入 / 文件上传（委托给 DocumentService）
    # =========================================================================

    async def insert_documents(self, request: InsertDocumentRequest) -> dict:
        await self._initialize()
        return await self._document_service.insert_documents(request)

    async def batch_insert_documents(self, documents: List[InsertDocumentRequest]) -> dict:
        await self._initialize()
        return await self._document_service.batch_insert_documents(documents)

    async def insert_file(self, file_content: bytes, file_name: str, metadata: dict = None) -> dict:
        await self._initialize()
        return await self._document_service.insert_file(file_content, file_name, metadata)

    # =========================================================================
    # 商品嵌入相关方法（委托给 ItemService）
    # =========================================================================

    async def embed_items(self, items: List[Dict[str, str]], batch_id: str = None) -> Dict[str, Any]:
        return await self._item_service.embed_items(items, batch_id)

    async def embed_items_from_file(self, file_content: bytes, file_name: str, batch_id: str = None) -> Dict[str, Any]:
        return await self._item_service.embed_items_from_file(file_content, file_name, batch_id)

    async def search_items_api(self, query: str, top_k: int = 10) -> ItemSearchResponse:
        return await self._item_service.search_items_api(query, top_k)

    # =========================================================================
    # 人在回路：退款确认
    # =========================================================================

    async def confirm_refund(self, request: RefundConfirmRequest) -> ChatResponse:
        """审批退款申请 —— 恢复人在回路中断的 graph 执行。

        Args:
            request: 包含 conversation_id、confirm、remark

        Returns:
            ChatResponse（含最终执行结果）

        Raises:
            ValidationException: 未找到对应的中断记录
        """
        await self._initialize()

        from src.modules.chat.agent.react_agent import ReActAgent

        response = await ReActAgent.resume_execution(
            thread_id=request.conversation_id,
            confirm=request.confirm,
            tool_service=self._tool_service,
        )

        if response is None:
            raise ValidationException(
                "未找到待确认的退款申请",
                f"conversation_id={request.conversation_id} 可能已过期或已处理",
            )

        logger.log_business_event(
            "退款审批确认",
            success=True,
            conversation_id=request.conversation_id,
            confirm=request.confirm,
            remark=request.remark or "",
        )

        return response

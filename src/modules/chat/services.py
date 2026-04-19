from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.modules.chat.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    InsertDocumentRequest,
    HospitalChatRequest,
    HospitalChatResponse
)
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.redis_cache_service import get_redis_cache_service
from src.shared.exceptions import NotFoundException, ValidationException
from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("chatagent_service")


class ChatAgentService:
    """智能客服服务"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self._embedding_service = None
        self._milvus_service = None
        self._llm_service = None
        self._redis_cache_service = None
        self._initialized = False

    @property
    def embedding(self):
        """获取嵌入服务实例"""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService.get_instance()
        return self._embedding_service.get_embeddings()
    
    @property
    def embeddings(self):
        """获取嵌入服务（兼容旧代码）"""
        return self.embedding
    
    @property
    def milvus(self) -> MilvusService:
        """获取 Milvus 服务"""
        if self._milvus_service is None:
            self._milvus_service = MilvusService.get_instance()
        return self._milvus_service
    
    @property
    def llm(self) -> LLMService:
        """获取 LLM 服务"""
        if self._llm_service is None:
            self._llm_service = LLMService.get_instance()
        return self._llm_service

    async def close(self):
        """关闭资源连接"""
        if self._milvus_service:
            self._milvus_service.close()
        if self._llm_service:
            self._llm_service.close()
        if self._redis_cache_service:
            self._redis_cache_service.close()
        self._initialized = False

    async def _initialize(self):
        """初始化服务"""
        if self._initialized:
            return

        try:
            if not chat_config.volcengine_api_key:
                raise ValidationException("火山引擎API密钥未配置", "请设置VOLCENGINE_API_KEY环境变量")

            # 初始化 LLM 服务（单例）
            self._llm_service = LLMService.get_instance()
            self._llm_service.initialize()

            # 初始化嵌入服务（单例）
            self._embedding_service = EmbeddingService.get_instance()

            # 初始化 Milvus 服务（单例）
            self._milvus_service = MilvusService.get_instance()
            self._milvus_service.initialize()

            # 初始化 Redis 缓存服务（单例）
            if chat_config.redis_vector_enabled:
                self._redis_cache_service = get_redis_cache_service()
                if self._redis_cache_service.is_available:
                    logger.info("Redis 缓存服务初始化成功")
                else:
                    logger.warning("Redis 缓存服务不可用，将禁用问题去重功能")

            self._initialized = True
            logger.info("ChatAgentService initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize ChatAgentService: {str(e)}")
            raise ValidationException("初始化聊天服务失败", str(e))

    async def _search_similar_documents(self, query: str, top_k: int = 3) -> List[Document]:
        """搜索相似的文档"""
        try:
            await self._initialize()
            
            # 生成查询嵌入
            query_embedding = await self.embeddings.aembed_query(query)
            
            # 在 Milvus 中搜索
            documents = self.milvus.search_similar(query_embedding, top_k)
            return documents

        except Exception as e:
            import traceback
            logger.error(f"Document search failed: {str(e)}\n{traceback.format_exc()}")
            raise ValidationException("文档搜索失败", str(e))

    async def _generate_response(self, query: str, documents: List[Document]) -> str:
        """基于检索到的文档生成回答"""
        try:
            await self._initialize()
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in documents[:3]]) or "暂无相关信息"
            
            # 获取当前时间
            from datetime import datetime
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
            
            # 从 prompts 模块获取默认模板
            from src.modules.chat.agent.prompts import PromptTemplateManager
            template = PromptTemplateManager.get("medical", "medical_step4_generate")
            
            # 构建 prompt
            if template:
                prompt_content = template.format(
                    rag_context=context,
                    user_question=query,
                    current_time=current_time,
                    safety_check_result="风险等级: low",
                    safety_reminder="",
                    chat_history="",
                    product_info=context,
                    knowledge_base=context,
                    context=context,
                    category=""
                )
            else:
                prompt_content = f"基于以下信息回答用户问题。\n\n知识库：\n{context}\n\n问题：{query}\n\n回答："

            # 使用 LLM 服务调用通义千问
            response = await self.llm.chat_qwen_with_prompt(
                prompt=prompt_content,
                system_prompt="你是一个医疗客服助手"
            )
            return response

        except Exception as e:
            import traceback
            logger.error(f"Response generation failed: {str(e)}\n{traceback.format_exc()}")
            return "抱歉，我暂时无法回答这个问题。"

    async def chat(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """RAG 聊天接口"""
        try:
            await self._initialize()

            # 1. 搜索相似文档
            similar_docs = await self._search_similar_documents(request.message)
            
            # 2. 生成回答
            response_text = await self._generate_response(request.message, similar_docs)
            
            # 3. 构建响应
            response = ChatQueryResponse(
                message=response_text,
                relevant_documents=[doc.page_content for doc in similar_docs],
                document_count=len(similar_docs)
            )

            # 记录业务事件
            logger.log_business_event(
                "RAG聊天查询",
                success=True,
                query=request.message,
                document_count=len(similar_docs),
                response_length=len(response_text)
            )

            return response

        except (NotFoundException, ValidationException):
            raise
        except Exception as e:
            logger.log_business_event(
                "RAG聊天查询",
                success=False,
                error=str(e),
                query=request.message
            )
            raise ValidationException("聊天查询失败", str(e))

    async def insert_documents(self, request: InsertDocumentRequest) -> dict:
        """向 Milvus 数据库插入文档数据"""
        try:
            await self._initialize()

            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # 分割文本
            chunks = text_splitter.split_text(request.document)
            
            # 生成嵌入
            embeddings = await self.embeddings.aembed_documents(chunks)
            
            # 准备插入数据
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata_list.append({
                    "source": request.metadata.get("source", "unknown"),
                    "batch_id": request.metadata.get("batch_id"),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            # 插入到 Milvus
            self.milvus.insert_documents(chunks, embeddings, metadata_list)
            self.milvus.flush()
                
            # 记录业务事件
            logger.log_business_event(
                "文档插入",
                success=True,
                document_length=len(request.document),
                chunks_count=len(chunks),
                source=request.metadata.get("source", "unknown")
            )

            return {
                "status": "success",
                "chunks_inserted": len(chunks),
                "total_characters": len(request.document)
            }

        except Exception as e:
            logger.log_business_event(
                "文档插入",
                success=False,
                error=str(e),
                document_length=len(request.document),
                source=request.metadata.get("source", "unknown")
            )
            raise ValidationException("文档插入失败", str(e))

    async def batch_insert_documents(self, documents: List[InsertDocumentRequest]) -> dict:
        """批量插入文档（优化：利用 Milvus 批量插入）"""
        try:
            await self._initialize()

            all_chunks = []
            all_embeddings = []
            all_metadata = []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            for doc_request in documents:
                chunks = text_splitter.split_text(doc_request.document)
                embeddings = await self.embeddings.aembed_documents(chunks)
                
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_embeddings.append(embeddings[i])
                    all_metadata.append({
                        "source": doc_request.metadata.get("source", "unknown"),
                        "batch_id": doc_request.metadata.get("batch_id"),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
            
            # 一次性批量插入
            if all_chunks:
                self.milvus.insert_documents(all_chunks, all_embeddings, all_metadata)
                self.milvus.flush()

            logger.log_business_event(
                "批量文档插入",
                success=True,
                documents_count=len(documents),
                total_chunks=len(all_chunks)
            )

            return {
                "status": "success",
                "documents_processed": len(documents),
                "chunks_inserted": len(all_chunks)
            }

        except Exception as e:
            logger.log_business_event(
                "批量文档插入",
                success=False,
                error=str(e),
                documents_count=len(documents)
            )
            raise ValidationException("批量文档插入失败", str(e))

    @staticmethod
    def parse_text_file(file_content: bytes, file_name: str) -> str:
        """解析纯文本文件内容"""
        allowed_extensions = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', '.css', '.js', '.py', '.java', '.c', '.cpp', '.h', '.yml', '.yaml', '.ini', '.cfg', '.log', '.conf'}
        
        import os
        _, ext = os.path.splitext(file_name.lower())
        
        if ext not in allowed_extensions:
            raise ValidationException(
                "不支持的文件类型",
                f"仅支持纯文本文件: {', '.join(allowed_extensions)}"
            )
        
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        
        for encoding in encodings:
            try:
                return file_content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        
        return file_content.decode('utf-8', errors='ignore')

    async def insert_file(self, file_content: bytes, file_name: str, metadata: dict = None) -> dict:
        """上传并插入纯文本文件"""
        try:
            await self._initialize()
            
            # 解析文件内容
            text_content = self.parse_text_file(file_content, file_name)
            
            if not text_content.strip():
                raise ValidationException("文件内容为空", "请上传非空的文件")
            
            # 构建元数据
            file_metadata = metadata or {}
            file_metadata.update({
                "source": "file_upload",
                "file_name": file_name
            })
            
            # 创建插入请求
            request = InsertDocumentRequest(
                document=text_content,
                metadata=file_metadata
            )
            
            # 插入文档
            result = await self.insert_documents(request)
            
            # 记录业务事件
            logger.log_business_event(
                "文件上传嵌入",
                success=True,
                file_name=file_name,
                chunks_inserted=result["chunks_inserted"],
                total_characters=result["total_characters"]
            )
            
            return {
                "status": "success",
                "file_name": file_name,
                "chunks_inserted": result["chunks_inserted"],
                "total_characters": result["total_characters"]
            }
            
        except ValidationException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"文件上传失败: {str(e)}\n{traceback.format_exc()}")
            logger.log_business_event(
                "文件上传嵌入",
                success=False,
                file_name=file_name,
                error=str(e)
            )
            raise ValidationException("文件上传失败", str(e))

    async def chat_with_hospital_agent(
        self, 
        request: HospitalChatRequest
    ) -> HospitalChatResponse:
        """使用通用 Agent 进行多步骤对话"""
        try:
            await self._initialize()
            
            # 延迟导入避免循环依赖
            from src.modules.chat.agent.executor import GeneralAgentExecutor
            from src.modules.chat.agent.schemas import HospitalAgentConfig
            
            # 获取领域配置
            domain = getattr(request, 'domain', 'medical')
            
            # 创建 Agent 执行器（传入domain自动加载对应配置）
            executor = GeneralAgentExecutor(
                domain=domain,
                llm_service=self._llm_service,
                embedding_service=self._embedding_service,
                milvus_service=self._milvus_service,
                redis_cache_service=self._redis_cache_service,
            )
            
            # 执行 Agent
            response = await executor.execute(request)
            response.domain = domain  # 设置响应领域
            
            # 记录业务事件
            logger.log_business_event(
                f"{executor.agent_name}对话",
                success=True,
                domain=domain,
                conversation_id=response.conversation_id,
                message_length=len(request.message),
                response_length=len(response.message),
                safety_passed=response.safety_passed
            )
            
            return response
            
        except Exception as e:
            import traceback
            logger.error(f"医院客服 Agent 执行失败: {str(e)}\n{traceback.format_exc()}")
            logger.log_business_event(
                "医院客服Agent对话",
                success=False,
                error=str(e),
                message_length=len(request.message)
            )
            raise ValidationException("医院客服对话失败", str(e))

from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.modules.chat.schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    InsertDocumentRequest,
    HospitalChatRequest,
    HospitalChatResponse,
    BatchItemEmbedRequest,
    ItemEmbedResponse,
    ItemSearchRequest,
    ItemSearchResponse,
    ItemSearchResult,
)
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.redis_cache_service import get_redis_cache_service
from src.shared.exceptions import NotFoundException, ValidationException
from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("chatagent_service")

# 嵌入 API 批次大小（Ark API 限制）
API_BATCH_SIZE = 25
# Milvus 每次插入的最大条数
MILVUS_BATCH_SIZE = 500


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

    async def search_items(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        混合检索商品（Dense + Sparse BM25），按 item_id 去重
        
        检索流程：
        1. 对查询进行向量嵌入
        2. 调用 Milvus 原生混合检索（稠密向量 + 稀疏BM25）
        3. 按 metadata.item_id 去重，保留分数最高的分片分数
        4. 按分数降序排序，返回商品列表
        
        Args:
            query: 用户查询
            top_k: 返回商品数量
            
        Returns:
            商品列表，每项包含 item_id、content、score、metadata
        """
        try:
            await self._initialize()
            
            # 1. 生成查询嵌入
            query_embedding = await self.embeddings.aembed_query(query)
            
            # 2. Milvus 混合检索（多拿一些，给去重留空间）
            docs = self.milvus.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=max(top_k * 3, 30)
            )
            
            # 3. 按 item_id 去重，保留最高分
            item_map: Dict[str, Dict[str, Any]] = {}
            
            for doc in docs:
                item_id = doc.metadata.get("item_id", "")
                score = doc.metadata.get("distance", 0.0)
                
                if not item_id:
                    continue
                
                if item_id not in item_map or score > item_map[item_id]["score"]:
                    item_map[item_id] = {
                        "item_id": item_id,
                        "content": doc.page_content,
                        "score": score,
                        "metadata": doc.metadata
                    }
            
            # 4. 按分数排序，返回 top_k
            results = sorted(item_map.values(), key=lambda x: x["score"], reverse=True)
            
            logger.info(f"商品检索完成: query={query}, 命中日杂数={len(item_map)}, 返回={min(top_k, len(results))}")
            return results[:top_k]
        
        except Exception as e:
            logger.error(f"商品检索失败: {str(e)}")
            raise

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

    # =========================================================================
    # 商品嵌入相关方法
    # =========================================================================

    async def embed_items(
        self,
        items: List[Dict[str, str]],
        batch_id: str = None
    ) -> Dict[str, Any]:
        """
        批量嵌入商品数据并存入 Milvus
        
        每个商品会作为一个独立的文档插入，使用 Milvus 2.6+ 原生混合检索支持。
        
        Args:
            items: 商品列表，每项包含 'item_id' 和 'title'
            batch_id: 批次ID（可选）
            
        Returns:
            嵌入结果统计
        """
        try:
            await self._initialize()
            
            if not items:
                return {
                    "status": "success",
                    "message": "没有商品数据需要处理",
                    "items_processed": 0,
                    "items_inserted": 0,
                    "failed_items": []
                }
            
            # 准备数据
            texts = []
            item_ids = []
            failed_items = []
            
            for item in items:
                item_id = item.get("item_id", "").strip()
                title = item.get("title", "").strip()
                
                if not item_id or not title:
                    failed_items.append(item_id or "unknown")
                    logger.warning(f"跳过无效商品: item_id={item_id}, title={title[:50]}")
                    continue
                
                texts.append(title)
                item_ids.append(item_id)
            
            if not texts:
                return {
                    "status": "error",
                    "message": "没有有效的商品数据",
                    "items_processed": 0,
                    "items_inserted": 0,
                    "failed_items": failed_items
                }
            
            # 批量嵌入
            all_embeddings = []
            total = len(texts)
            
            for batch_start in range(0, total, API_BATCH_SIZE):
                batch_end = min(batch_start + API_BATCH_SIZE, total)
                batch_texts = texts[batch_start:batch_end]
                
                try:
                    batch_embeddings = await self.embeddings.aembed_documents(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"嵌入进度: {batch_end}/{total}")
                except Exception as e:
                    logger.error(f"嵌入失败（批次 {batch_start}-{batch_end}）: {str(e)}")
                    failed_items.extend(item_ids[batch_start:batch_end])
                    continue
            
            if not all_embeddings:
                return {
                    "status": "error",
                    "message": "所有商品嵌入失败",
                    "items_processed": total,
                    "items_inserted": 0,
                    "failed_items": failed_items
                }
            
            # 分批插入 Milvus
            inserted_count = 0
            embeddings_count = len(all_embeddings)
            
            for batch_start in range(0, embeddings_count, MILVUS_BATCH_SIZE):
                batch_end = min(batch_start + MILVUS_BATCH_SIZE, embeddings_count)
                
                batch_texts = texts[batch_start:batch_end]
                batch_embeddings = all_embeddings[batch_start:batch_end]
                batch_metadata = [
                    {
                        "item_id": item_ids[batch_start + i],
                        "source": "item_title",
                        "batch_id": batch_id
                    }
                    for i in range(batch_end - batch_start)
                ]
                
                try:
                    self.milvus.insert_documents(
                        texts=batch_texts,
                        embeddings=batch_embeddings,
                        metadata_list=batch_metadata
                    )
                    inserted_count += (batch_end - batch_start)
                    logger.info(f"插入进度: {batch_end}/{embeddings_count}")
                except Exception as e:
                    logger.error(f"插入失败（批次 {batch_start}-{batch_end}）: {str(e)}")
                    failed_items.extend(item_ids[batch_start:batch_end])
            
            # 刷新 Milvus
            self.milvus.flush()
            
            logger.log_business_event(
                "商品嵌入",
                success=True,
                items_count=total,
                inserted_count=inserted_count,
                failed_count=len(failed_items),
                batch_id=batch_id
            )
            
            return {
                "status": "success",
                "message": f"成功嵌入 {inserted_count} 个商品",
                "items_processed": total,
                "items_inserted": inserted_count,
                "failed_items": failed_items
            }
            
        except Exception as e:
            logger.error(f"商品嵌入失败: {str(e)}")
            logger.log_business_event(
                "商品嵌入",
                success=False,
                error=str(e)
            )
            raise ValidationException("商品嵌入失败", str(e))

    async def embed_items_from_file(
        self,
        file_content: bytes,
        file_name: str,
        batch_id: str = None
    ) -> Dict[str, Any]:
        """
        从文件解析并嵌入商品数据
        
        支持的文件格式：
        - .txt / .tsv: 每行格式为 "ID\\t商品标题"
        - .csv: CSV 格式，需包含 item_id 和 title 列
        
        Args:
            file_content: 文件内容（字节）
            file_name: 文件名
            batch_id: 批次ID（可选）
            
        Returns:
            嵌入结果统计
        """
        try:
            await self._initialize()
            
            # 解析文件内容
            text_content = self.parse_text_file(file_content, file_name)
            
            if not text_content.strip():
                raise ValidationException("文件内容为空", "请上传非空的文件")
            
            # 解析商品数据
            items = []
            lines = text_content.strip().split("\n")
            
            for line_no, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                
                # 支持 TSV 格式（ID\t标题）
                if "\t" in line:
                    parts = line.split("\t", 1)
                    if len(parts) >= 2:
                        items.append({
                            "item_id": parts[0].strip(),
                            "title": parts[1].strip()
                        })
                    else:
                        logger.warning(f"跳过格式错误行 {line_no}: {line[:50]}")
                else:
                    # 如果没有制表符，整行作为标题，使用行号作为 ID
                    items.append({
                        "item_id": f"line_{line_no}",
                        "title": line
                    })
            
            if not items:
                raise ValidationException("文件中没有有效的商品数据", "请检查文件格式")
            
            logger.info(f"从文件 {file_name} 解析到 {len(items)} 条商品数据")
            
            # 调用批量嵌入
            result = await self.embed_items(items, batch_id)
            result["file_name"] = file_name
            result["items_parsed"] = len(items)
            
            return result
            
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"从文件嵌入商品失败: {str(e)}")
            raise ValidationException("从文件嵌入商品失败", str(e))

    async def search_items_api(
        self,
        query: str,
        top_k: int = 10
    ) -> ItemSearchResponse:
        """
        商品搜索 API 方法（返回结构化响应）
        
        Args:
            query: 搜索查询
            top_k: 返回商品数量
            
        Returns:
            ItemSearchResponse 对象
        """
        try:
            results = await self.search_items(query, top_k)
            
            items = [
                ItemSearchResult(
                    item_id=result["item_id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result.get("metadata", {})
                )
                for result in results
            ]
            
            return ItemSearchResponse(
                query=query,
                total=len(items),
                items=items
            )
            
        except Exception as e:
            logger.error(f"商品搜索 API 失败: {str(e)}")
            raise ValidationException("商品搜索失败", str(e))

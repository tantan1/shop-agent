"""
文档服务 —— Token 安全兜底切分 + 文档插入/批插入/文件上传

负责：
1. Token 估算与安全切分（保守估算，不依赖 tiktoken）
2. 语义切分 + Token 安全兜底两层切分
3. 单文档插入、批量插入、文件上传嵌入
"""

from typing import List, Dict, Any
import logging

from src.modules.chat.schemas import InsertDocumentRequest
from src.shared.exceptions import ValidationException
from src.shared.logger import APILogger

logger = APILogger("document_service")

# ═══════════════════════════════════════════════════════════════════════════════
# Token 限制安全兜底配置
# ═══════════════════════════════════════════════════════════════════════════════
# Doubao-embedding 最大输入 tokens（通常 4096，取 80% 留安全余量）
EMBEDDING_MAX_TOKENS = 4096
TOKEN_SAFETY_MARGIN = 0.8
# 安全 chunk 大小：中文 1 字符约 1.2~2 tokens，这里按最坏 1:1 估算
# 即 embed(3276 chars) <= 4096 * 0.8 = 3276 tokens，确保不超限
MAX_CHUNK_CHARS = int(EMBEDDING_MAX_TOKENS * TOKEN_SAFETY_MARGIN)
# 安全分割的默认参数（用于兜底对超限 chunk 再切分）
FALLBACK_CHUNK_SIZE = MAX_CHUNK_CHARS
FALLBACK_CHUNK_OVERLAP = 200


class DocumentService:
    """文档服务 —— 语义切分 + Token 安全兜底 + 向量化 + 写入 Milvus"""

    def __init__(self, embedding_service, milvus_service, item_service=None):
        """
        Args:
            embedding_service: EmbeddingService 实例
            milvus_service: MilvusService 实例
            item_service: ItemService 实例（用于 parse_text_file）
        """
        self._embedding_service = embedding_service
        self._milvus_service = milvus_service
        self._item_service = item_service

    # ════════════════════════════════════════════════════════════════════════
    # Token 估算 & 安全切分
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的 token 数量。

        优先使用 TokenEstimator（HF tokenizers，精度 100% + LRU 缓存），
        加载失败时降级为字符估算（保守，确保不超限）。
        """
        if not text:
            return 0

        # 优先复用 TokenEstimator（含 LRU 缓存，不再重复编码）
        try:
            from src.core.token_estimator import get_token_estimator
            estimator = get_token_estimator()
            if estimator.is_loaded:
                return estimator.estimate(text)
        except Exception:
            pass

        # 降级：保守字符估算（中文 /1.2，英文 /4.0，不超过 len(text)）
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        ascii_len = len(text) - non_ascii
        estimated = int(ascii_len / 4.0 + non_ascii / 1.2)
        return min(estimated, len(text))

    @staticmethod
    def ensure_token_limit(chunks: List[str]) -> List[str]:
        """
        确保所有 chunk 都在 embedding 模型的 token 限制内。

        对于超限的 chunk，使用 RecursiveCharacterTextSplitter 二次切分。
        未超限的 chunk 原样保留（保持语义完整性）。

        Args:
            chunks: 原始切分列表

        Returns:
            安全的切分列表（所有 chunk 均在 token 限制内）
        """
        safe_chunks = []
        overflow_count = 0

        for chunk in chunks:
            estimated_tokens = DocumentService.estimate_tokens(chunk)

            if estimated_tokens <= MAX_CHUNK_CHARS:
                # 未超限，原样保留
                safe_chunks.append(chunk)
            else:
                # 超限了，用 RecursiveCharacterTextSplitter 做机械切分兜底
                overflow_count += 1
                # 注意：langchain_text_splitters 的顶层 import 会在 Windows 下触发
                # pyarrow/sentence-transformers/sklearn/pandas 等 C 扩展的 DLL 加载冲突，
                # 因此改为懒加载。
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                fallback_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", "。", ".", " ", ""],
                    chunk_size=FALLBACK_CHUNK_SIZE,
                    chunk_overlap=FALLBACK_CHUNK_OVERLAP,
                    length_function=len,
                )
                sub_chunks = fallback_splitter.split_text(chunk)
                safe_chunks.extend(sub_chunks)

        if overflow_count > 0:
            logger.warning(
                f"Token 安全兜底触发：{overflow_count}/{len(chunks)} 个 chunk 超限（>{MAX_CHUNK_CHARS}字符），"
                f"已拆分为 {len(safe_chunks)} 个安全 chunk"
            )

        return safe_chunks

    # ════════════════════════════════════════════════════════════════════════
    # 文档切分 & 向量化
    # ════════════════════════════════════════════════════════════════════════

    async def _chunk_and_embed(self, text: str, source_meta: Dict[str, Any]) -> dict:
        """
        两层切分 + 向量化 + 写入 Milvus，返回统计信息。

        内部流程：
        1. 第一层：SemanticChunker 语义切分
        2. 第二层：Token 安全兜底切分
        3. 向量化
        4. 写入 Milvus
        """
        # 第一层：语义切分（基于向量相似度检测话题边界）
        from langchain_experimental.text_splitter import SemanticChunker
        text_splitter = SemanticChunker(
            embeddings=self._embedding_service.get_embeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,  # 相似度低于第 85 百分位就切分
        )

        raw_chunks = text_splitter.split_text(text)

        # 第二层：token 安全兜底
        chunks = self.ensure_token_limit(raw_chunks)

        # 向量化
        embeddings = await self._embedding_service.get_embeddings().aembed_documents(chunks)

        # 准备元数据
        metadata_list = []
        for i in range(len(chunks)):
            metadata_list.append({
                **source_meta,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        # 写入 Milvus
        self._milvus_service.insert_documents(chunks, embeddings, metadata_list)
        self._milvus_service.flush()

        return {
            "raw_chunks": len(raw_chunks),
            "safe_chunks": len(chunks),
        }

    # ════════════════════════════════════════════════════════════════════════
    # 对外 API
    # ════════════════════════════════════════════════════════════════════════

    async def insert_documents(self, request: InsertDocumentRequest) -> dict:
        """向 Milvus 数据库插入文档数据"""
        try:
            source_meta = {
                "source": request.metadata.get("source", "unknown"),
                "batch_id": request.metadata.get("batch_id"),
            }

            stats = await self._chunk_and_embed(request.document, source_meta)

            logger.log_business_event(
                "文档插入",
                success=True,
                document_length=len(request.document),
                raw_chunks=stats["raw_chunks"],
                safe_chunks=stats["safe_chunks"],
                source=request.metadata.get("source", "unknown")
            )

            return {
                "status": "success",
                "chunks_inserted": stats["safe_chunks"],
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
        """批量插入文档（先全部切分+向量化，再一次性写入 Milvus）"""
        try:
            all_chunks = []
            all_embeddings = []
            all_metadata = []

            for doc_request in documents:
                source_meta = {
                    "source": doc_request.metadata.get("source", "unknown"),
                    "batch_id": doc_request.metadata.get("batch_id"),
                }

                # 两层切分 + 向量化
                from langchain_experimental.text_splitter import SemanticChunker
                text_splitter = SemanticChunker(
                    embeddings=self._embedding_service.get_embeddings(),
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=85,
                )

                raw_chunks = text_splitter.split_text(doc_request.document)
                chunks = self.ensure_token_limit(raw_chunks)
                embeddings = await self._embedding_service.get_embeddings().aembed_documents(chunks)

                for i in range(len(chunks)):
                    all_chunks.append(chunks[i])
                    all_embeddings.append(embeddings[i])
                    all_metadata.append({
                        **source_meta,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    })

            # 一次性批量插入
            if all_chunks:
                self._milvus_service.insert_documents(all_chunks, all_embeddings, all_metadata)
                self._milvus_service.flush()

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

    async def insert_file(
        self,
        file_content: bytes,
        file_name: str,
        metadata: dict = None
    ) -> dict:
        """
        上传并插入纯文本文件。

        委托 ItemService.parse_text_file 解析文件内容，再走标准文档插入流程。
        """
        try:
            # 解析文件内容（委托给 ItemService）
            if self._item_service is None:
                raise ValidationException("ItemService 未初始化", "无法解析文件")
            text_content = self._item_service.parse_text_file(file_content, file_name)

            if not text_content.strip():
                raise ValidationException("文件内容为空", "请上传非空的文件")

            # 构建元数据
            file_metadata = metadata or {}
            file_metadata.update({
                "source": "file_upload",
                "file_name": file_name
            })

            # 走标准插入流程
            request = InsertDocumentRequest(
                document=text_content,
                metadata=file_metadata
            )
            result = await self.insert_documents(request)

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

"""
商品嵌入 & 搜索服务
"""
from typing import List, Dict, Any, Set
import importlib

from src.modules.chat.schemas import (
    ItemSearchResponse,
    ItemSearchResult,
)
from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.shared.exceptions import ValidationException
from src.shared.logger import APILogger

logger = APILogger("item_service")

# 嵌入 API 批次大小
API_BATCH_SIZE = 25
# Milvus 每次插入的最大条数
MILVUS_BATCH_SIZE = 500

# 允许解析的纯文本文件扩展名
ALLOWED_TEXT_EXTENSIONS: Set[str] = {
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
    '.css', '.js', '.py', '.java', '.c', '.cpp', '.h',
    '.yml', '.yaml', '.ini', '.cfg', '.log', '.conf',
}


class ItemService:
    """商品嵌入 & 搜索服务"""

    def __init__(self):
        self._embedding_svc = EmbeddingService.get_instance()
        self._milvus_svc = MilvusService.get_instance()

    @property
    def _embeddings(self):
        """获取嵌入函数"""
        return self._embedding_svc.get_embeddings()

    @property
    def _milvus(self) -> MilvusService:
        """获取 Milvus 服务"""
        return self._milvus_svc

    @staticmethod
    def parse_text_file(file_content: bytes, file_name: str) -> str:
        """解析纯文本文件内容"""
        import os
        _, ext = os.path.splitext(file_name.lower())

        if ext not in ALLOWED_TEXT_EXTENSIONS:
            raise ValidationException(
                "不支持的文件类型",
                f"仅支持纯文本文件: {', '.join(sorted(ALLOWED_TEXT_EXTENSIONS))}"
            )

        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']

        for encoding in encodings:
            try:
                return file_content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        return file_content.decode('utf-8', errors='ignore')

    async def embed_items(
        self,
        items: List[Dict[str, str]],
        batch_id: str = None
    ) -> Dict[str, Any]:
        """
        批量嵌入商品数据并存入 Milvus

        Args:
            items: 商品列表，每项包含 'item_id' 和 'title'
            batch_id: 批次ID（可选）

        Returns:
            嵌入结果统计
        """
        try:
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
                    batch_embeddings = await self._embeddings.aembed_documents(batch_texts)
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
                    self._milvus.insert_documents(
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
            self._milvus.flush()

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

        支持：.txt / .tsv（每行 "ID\\t标题"）

        Args:
            file_content: 文件内容（字节）
            file_name: 文件名
            batch_id: 批次ID（可选）

        Returns:
            嵌入结果统计
        """
        try:
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
                    items.append({
                        "item_id": f"line_{line_no}",
                        "title": line
                    })

            if not items:
                raise ValidationException("文件中没有有效的商品数据", "请检查文件格式")

            logger.info(f"从文件 {file_name} 解析到 {len(items)} 条商品数据")

            result = await self.embed_items(items, batch_id)
            result["file_name"] = file_name
            result["items_parsed"] = len(items)

            return result

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"从文件嵌入商品失败: {str(e)}")
            raise ValidationException("从文件嵌入商品失败", str(e))

    async def search_items(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        混合检索商品（Dense + Sparse BM25），按 item_id 去重

        Args:
            query: 用户查询
            top_k: 返回商品数量

        Returns:
            商品列表，每项包含 item_id、content、score、metadata
        """
        try:
            # 1. 生成查询嵌入
            query_embedding = await self._embeddings.aembed_query(query)

            # 2. Milvus 混合检索
            docs = self._milvus.hybrid_search(
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

            logger.info(f"商品检索完成: query={query}, 命中={len(item_map)}, 返回={min(top_k, len(results))}")
            return results[:top_k]

        except Exception as e:
            logger.error(f"商品检索失败: {str(e)}")
            raise

    async def search_items_api(
        self,
        query: str,
        top_k: int = 10
    ) -> ItemSearchResponse:
        """
        商品搜索 API（返回结构化响应）

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

"""
商品数据嵌入脚本

读取商品数据文件（每行格式：ID\t商品标题），逐行嵌入后存入 Milvus。
使用 Milvus 2.6+ 原生混合检索（Dense + Sparse BM25），无需手动分片。

用法:
    python scripts/embed_items.py --file data/items.txt [--batch 25] [--milvus-batch 500]
"""

import argparse
import asyncio
import os
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.modules.chat.core.pgvector_service import PgVectorService
from src.modules.chat.config import chat_config
from src.shared.logger import APILogger

logger = APILogger("embed_items")


def _get_vector_service():
    """工厂方法：根据 VECTOR_STORE_PROVIDER 创建对应的向量数据库服务"""
    provider = chat_config.vector_store_provider
    if provider == "pgvector":
        logger.info("使用 PostgreSQL pgvector 作为向量存储")
        return PgVectorService.get_instance()
    else:
        logger.info("使用 Milvus 作为向量存储")
        return MilvusService.get_instance()

# Ark API 批次大小（与 embedding_service.py 中的 BATCH_SIZE 一致）
API_BATCH_SIZE = 25
# Milvus 每次插入的最大条数
MILVUS_BATCH_SIZE = 500
# 断点续传记录文件
CHECKPOINT_FILE = "data/.embed_checkpoint.txt"


def load_checkpoint() -> int:
    """加载断点（已处理的行数）"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(line_no: int):
    """保存断点"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(str(line_no))


def parse_lines(file_path: str, skip_lines: int) -> Tuple[List[str], List[str]]:
    """
    解析数据文件，跳过已处理的行
    
    Returns:
        (texts, item_ids): 商品标题列表, 商品ID列表
    """
    texts = []
    item_ids = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i <= skip_lines:
                continue
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                logger.warning(f"跳过格式错误行 {i}: {line[:50]}")
                continue
            item_id, title = parts[0].strip(), parts[1].strip()
            if not title:
                continue
            item_ids.append(item_id)
            texts.append(title)
    
    return texts, item_ids


async def embed_and_insert(
    texts: List[str],
    item_ids: List[str],
    embedding_service: EmbeddingService,
    milvus_service: MilvusService,
    api_batch_size: int,
    milvus_batch_size: int,
    start_line_no: int,
    total_lines: int,
):
    """
    嵌入并插入 Milvus
    
    流程：
    1. 按 api_batch_size 分批调用嵌入 API
    2. 嵌入完成后按 milvus_batch_size 分批插入 Milvus
    3. 每插入一批保存一次断点
    """
    total = len(texts)
    logger.info(f"开始处理 {total} 条商品数据（从第 {start_line_no + 1} 行开始）")
    
    # ========== 1. 批量嵌入 ==========
    all_embeddings: List[List[float]] = []
    
    for batch_start in range(0, total, api_batch_size):
        batch_end = min(batch_start + api_batch_size, total)
        batch_texts = texts[batch_start:batch_end]
        
        try:
            batch_embeddings = await embedding_service.embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            processed = batch_end
            overall_line = start_line_no + processed
            logger.info(f"嵌入进度: {processed}/{total}（文件行号: {overall_line}）")
        
        except Exception as e:
            # 保存断点，下次从这批开始
            save_checkpoint(start_line_no + batch_start)
            logger.error(f"嵌入失败，断点已保存（行号: {start_line_no + batch_start}）: {str(e)}")
            raise
    
    logger.info(f"嵌入完成，共 {len(all_embeddings)} 条")
    
    # ========== 2. 分批插入 Milvus ==========
    for batch_start in range(0, total, milvus_batch_size):
        batch_end = min(batch_start + milvus_batch_size, total)
        
        batch_texts = texts[batch_start:batch_end]
        batch_embeddings = all_embeddings[batch_start:batch_end]
        batch_metadata = [
            {"item_id": item_ids[batch_start + i], "source": "item_title"}
            for i in range(batch_end - batch_start)
        ]
        
        try:
            milvus_service.insert_documents(
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadata_list=batch_metadata
            )
            save_checkpoint(start_line_no + batch_end)
            
            processed = batch_end
            overall_line = start_line_no + processed
            logger.info(f"插入进度: {processed}/{total}（文件行号: {overall_line}）")
        
        except Exception as e:
            save_checkpoint(start_line_no + batch_start)
            logger.error(f"插入失败，断点已保存（行号: {start_line_no + batch_start}）: {str(e)}")
            raise
    
    # ========== 3. 刷新 + 清除断点 ==========
    milvus_service.flush()
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    logger.info(f"全部完成！共处理 {total} 条商品数据")


async def main():
    parser = argparse.ArgumentParser(description="商品数据嵌入脚本")
    parser.add_argument("--file", type=str, required=True, help="数据文件路径（每行：ID\\t商品标题）")
    parser.add_argument("--batch", type=int, default=API_BATCH_SIZE, help=f"嵌入 API 批次大小（默认 {API_BATCH_SIZE}）")
    parser.add_argument("--milvus-batch", type=int, default=MILVUS_BATCH_SIZE, help=f"Milvus 插入批次大小（默认 {MILVUS_BATCH_SIZE}）")
    parser.add_argument("--reset-checkpoint", action="store_true", help="忽略断点，从头开始")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"文件不存在: {args.file}")
        return
    
    # 加载断点
    if args.reset_checkpoint and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("已重置断点")
    
    skip_lines = load_checkpoint()
    if skip_lines > 0:
        logger.info(f"从断点恢复：跳过前 {skip_lines} 行")
    
    # 解析文件（只解析未处理的行）
    texts, item_ids = parse_lines(args.file, skip_lines)
    if not texts:
        logger.info("所有数据已处理完成！")
        return
    
    # 初始化服务
    embedding_svc = EmbeddingService.get_instance()
    vector_svc = _get_vector_service()
    vector_svc.initialize()
    
    try:
        await embed_and_insert(
            texts=texts,
            item_ids=item_ids,
            embedding_service=embedding_svc,
            milvus_service=vector_svc,
            api_batch_size=args.batch,
            milvus_batch_size=args.milvus_batch,
            start_line_no=skip_lines,
            total_lines=skip_lines + len(texts),
        )
    finally:
        vector_svc.close()


if __name__ == "__main__":
    asyncio.run(main())

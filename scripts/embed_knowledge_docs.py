"""
知识库文档嵌入脚本

读取 knowledge/ 目录下的 .md / .txt / .docx / .pdf 文件，
分块后嵌入并存入 Milvus。
"""

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from src.modules.chat.core.embedding_service import EmbeddingService
from src.modules.chat.core.milvus_service import MilvusService
from src.shared.logger import APILogger

logger = APILogger("embed_knowledge")

# 分块参数
CHUNK_SIZE = 500     # 每块字符数
CHUNK_OVERLAP = 100  # 重叠字符数
BATCH_SIZE = 25      # 嵌入批次大小

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def read_file_content(filepath: Path) -> str:
    """读取各类文档内容"""
    suffix = filepath.suffix.lower()

    if suffix == ".md":
        return filepath.read_text(encoding="utf-8")
    elif suffix == ".txt":
        return filepath.read_text(encoding="utf-8")
    elif suffix == ".docx":
        import docx
        doc = docx.Document(str(filepath))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif suffix == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(str(filepath))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n".join(text_parts)
    else:
        return ""


def clean_text(text: str) -> str:
    """清洗文本：去掉过多空白、图片引用"""
    # 去掉图片引用
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 合并空白行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去掉纯表格行（HTML table 残留）
    text = re.sub(r'<table>.*?</table>', '', text, flags=re.DOTALL)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    按段落分块，尽量保持段落完整性。
    如果单个段落过长，按句子切分。
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 跳过过短的段落（标题等），合并到下一个段落
        if len(para) < 30 and not para.endswith((".", "。", "!", "！", "?")):
            if current:
                current += "\n" + para
            else:
                current = para
            continue
        
        if len(current) + len(para) + 2 <= chunk_size:
            if current:
                current += "\n\n" + para
            else:
                current = para
        else:
            if current:
                chunks.append(current)
            current = para
    
    if current:
        chunks.append(current)
    
    # 带重叠的滑动窗口分块（针对超长 chunks）
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # 按句子切分长段落
            sentences = re.split(r'(?<=[。！？.!?])\s*', chunk)
            sub_chunk = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sub_chunk) + len(sent) <= chunk_size:
                    sub_chunk += sent
                else:
                    if sub_chunk:
                        final_chunks.append(sub_chunk)
                    sub_chunk = sent
            if sub_chunk:
                final_chunks.append(sub_chunk)
    
    return final_chunks


async def main():
    # 找到所有可嵌入文件
    files = []
    for pattern in ["*.md", "*.txt", "*.docx", "*.pdf"]:
        files.extend(KNOWLEDGE_DIR.glob(pattern))
    # 排除 temp 目录
    files = [f for f in files if "temp_magicpdf" not in str(f)]
    
    if not files:
        logger.error("knowledge/ 目录下没有可嵌入的文件")
        return
    
    logger.info(f"找到 {len(files)} 个文档: {[f.name for f in files]}")
    
    # 读取并分块
    all_chunks = []
    all_metadata = []
    
    for fpath in files:
        logger.info(f"读取: {fpath.name}")
        content = read_file_content(fpath)
        if not content:
            logger.warning(f"  文件为空或无法解析: {fpath.name}")
            continue
        
        content = clean_text(content)
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk) < 20:  # 跳过太短的块
                continue
            all_chunks.append(chunk)
            all_metadata.append({
                "source": str(fpath.relative_to(PROJECT_ROOT)),
                "filename": fpath.name,
                "chunk_index": i,
                "chunk_count": len(chunks),
            })
    
    logger.info(f"\n共产生 {len(all_chunks)} 个文本块")
    if not all_chunks:
        return
    
    # 初始化服务
    logger.info("初始化 Embedding 服务...")
    emb_svc = EmbeddingService.get_instance()
    
    logger.info("初始化 Milvus 服务...")
    milvus_svc = MilvusService.get_instance()
    milvus_svc.initialize()
    
    # 批量嵌入 + 插入
    total = len(all_chunks)
    inserted = 0
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = all_chunks[batch_start:batch_end]
        batch_meta = all_metadata[batch_start:batch_end]
        
        # 嵌入
        try:
            batch_embeddings = await emb_svc.embed_texts(batch_texts)
        except Exception as e:
            logger.error(f"嵌入失败 (块 {batch_start}-{batch_end}): {e}")
            break
        
        # 插入 Milvus
        try:
            milvus_svc.insert_documents(
                texts=batch_texts,
                embeddings=batch_embeddings,
                metadata_list=batch_meta
            )
            inserted += len(batch_texts)
            logger.info(f"进度: {inserted}/{total}")
        except Exception as e:
            logger.error(f"插入失败 (块 {batch_start}-{batch_end}): {e}")
            break
    
    milvus_svc.flush()
    milvus_svc.close()
    logger.info(f"\n完成！成功嵌入 {inserted}/{total} 个文本块")


if __name__ == "__main__":
    asyncio.run(main())

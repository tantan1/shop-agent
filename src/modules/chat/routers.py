from fastapi import APIRouter, Depends, UploadFile, File, Form
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.shared.database import get_db
from src.modules.auth.dependencies import verify_api_key
from src.modules.chat.services import ChatAgentService
from src.modules.chat.schemas import (
    ChatQueryRequest,
    InsertDocumentRequest,
    BatchInsertRequest,
    HospitalChatRequest,
    ItemEmbedRequest,
    BatchItemEmbedRequest,
    ItemEmbedResponse,
    ItemSearchRequest,
    ItemSearchResponse,
)
from src.shared.responses import success_response

router = APIRouter(prefix="/chatagent", tags=["智能客服与文档管理"])


async def get_chatagent_service(db: AsyncSession = Depends(get_db)) -> ChatAgentService:
    """获取智能客服依赖"""
    return ChatAgentService(db)


@router.post("/chat", summary="通义千问RAG对话")
async def chat(
    request: ChatQueryRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """使用TongyiChat进行RAG对话，基于Milvus中的文档增强回答"""
    response = await chatagent_service.chat(request)
    return success_response(data=response.model_dump())


@router.post("/documents", summary="插入单个文档")
async def insert_document(
    request: InsertDocumentRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """向Milvus数据库插入单个文档，使用嵌入模型生成向量
    
    HTTP 请求示例:
    ```bash
    curl -X POST 'http://localhost:8000/api/chatagent/documents' \
      -H 'X-API-Key: your-api-key' \
      -H 'Content-Type: application/json' \
      -d '{
        "document": "这是一段要嵌入的文本内容，可以是产品说明、服务介绍等。",
        "metadata": {
          "source": "product_description",
          "batch_id": "batch_001"
        }
      }'
    ```
    
    响应示例:
    ```json
    {
      "code": 200,
      "message": "success",
      "data": {
        "status": "success",
        "chunks_inserted": 2,
        "total_characters": 28
      }
    }
    ```
    """
    result = await chatagent_service.insert_documents(request)
    return success_response(data=result)


@router.post("/documents/batch", summary="批量插入文档")
async def batch_insert_documents(
    request: BatchInsertRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """批量插入文档到Milvus数据库"""
    result = await chatagent_service.batch_insert_documents(request.documents)
    return success_response(data=result)


@router.post("/documents/upload", summary="上传文件并嵌入")
async def upload_document(
    file: UploadFile = File(..., description="要上传的纯文本文件"),
    source: Optional[str] = Form(default="file_upload", description="文档来源"),
    batch_id: Optional[str] = Form(default=None, description="批次ID"),
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """上传纯文本文件并自动解析内容进行嵌入
    
    支持的文件格式: .txt, .md, .csv, .json, .xml, .html, .css, .js, .py, .java, .c, .cpp, .yml, .yaml, .log 等
    
    HTTP 请求示例 (curl):
    ```bash
    curl -X POST 'http://localhost:8000/api/chatagent/documents/upload' \
      -H 'X-API-Key: your-api-key' \
      -F 'file=@/path/to/document.txt' \
      -F 'source=manual_upload' \
      -F 'batch_id=batch_001'
    ```
    
    Python 请求示例:
    ```python
    import requests
    
    with open('document.txt', 'rb') as f:
        files = {'file': ('document.txt', f, 'text/plain')}
        data = {'source': 'manual_upload', 'batch_id': 'batch_001'}
        headers = {'X-API-Key': 'your-api-key'}
        response = requests.post(
            'http://localhost:8000/api/chatagent/documents/upload',
            files=files,
            data=data,
            headers=headers
        )
    ```
    
    响应示例:
    ```json
    {
      "code": 200,
      "message": "success",
      "data": {
        "status": "success",
        "file_name": "document.txt",
        "chunks_inserted": 5,
        "total_characters": 1234
      }
    }
    ```
    """
    # 读取文件内容
    file_content = await file.read()
    
    # 构建元数据
    metadata = {
        "source": source,
        "batch_id": batch_id
    }
    
    # 上传并嵌入文件
    result = await chatagent_service.insert_file(
        file_content=file_content,
        file_name=file.filename,
        metadata=metadata
    )
    
    return success_response(data=result)


@router.get("/health", summary="服务健康检查")
async def health_check(
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """检查服务状态"""
    try:
        # 健康检查：测试初始化
        await chatagent_service._initialize()

        return success_response(data={
            "status": "healthy",
            "embeddings_service": "available",
            "llm": "available",
            "milvus": "connected"
        })
    except Exception as e:
        error_msg = str(e)
        return success_response(data={
            "status": "unhealthy",
            "error": error_msg[:200] if len(error_msg) > 200 else error_msg
        }, code=503)


@router.post("/hospital/chat", summary="通用Agent对话")
async def hospital_chat(
    request: HospitalChatRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    使用通用 Agent 进行多步骤对话

    Agent 流程：
    1. 问题理解/改写 - 将用户问题改写为更适合检索的查询
    2. 内容审查/安全检查 - 检查问题是否涉及敏感内容（取决于领域配置）
    3. 知识检索 - 使用 RAG 组件从知识库检索相关内容
    4. 答案生成 - 基于检索结果生成回答

    支持的领域 (domain 参数):
    - medical: 医疗客服（默认）
    - ecommerce: 电商客服
    - customer_service: 通用客服
    - general: 通用助手

    HTTP 请求示例:
    ```bash
    curl -X POST 'http://localhost:8000/api/chatagent/hospital/chat' \\
      -H 'X-API-Key: your-api-key' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "message": "我想咨询一下，心脏病患者可以做胃镜检查吗？",
        "stream": false,
        "domain": "medical"
      }'
    ```

    响应示例:
    ```json
    {
      "code": 200,
      "message": "success",
      "data": {
        "message": "关于您的问题...",
        "conversation_id": "conv_1234567890",
        "steps": [...],
        "documents_used": [...],
        "safety_passed": true,
        "stream_available": true,
        "domain": "medical"
      }
    }
    ```
    """
    response = await chatagent_service.chat_with_hospital_agent(request)
    return success_response(data=response.model_dump())


# =============================================================================
# 商品嵌入与搜索 API
# =============================================================================

@router.post("/items/embed", summary="嵌入单个商品", response_model=ItemEmbedResponse)
async def embed_item(
    item: ItemEmbedRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    嵌入单个商品并存入 Milvus
    
    请求示例:
    ```json
    {
        "item_id": "12345",
        "title": "Apple iPhone 15 Pro Max 256GB 原色钛金属"
    }
    ```
    
    响应示例:
    ```json
    {
        "code": 200,
        "message": "success",
        "data": {
            "status": "success",
            "items_processed": 1,
            "items_inserted": 1,
            "failed_items": []
        }
    }
    ```
    """
    result = await chatagent_service.embed_items(
        items=[{"item_id": item.item_id, "title": item.title}]
    )
    # 将 dict 转换为 Pydantic 模型，确保响应格式一致
    response_data = ItemEmbedResponse(**result)
    return success_response(data=response_data.model_dump())


@router.post("/items/embed/batch", summary="批量嵌入商品", response_model=ItemEmbedResponse)
async def batch_embed_items(
    request: BatchItemEmbedRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    批量嵌入商品并存入 Milvus
    
    请求示例:
    ```json
    {
        "items": [
            {"item_id": "12345", "title": "Apple iPhone 15 Pro Max"},
            {"item_id": "12346", "title": "Samsung Galaxy S24 Ultra"}
        ],
        "batch_id": "batch_001"
    }
    ```
    """
    items = [{"item_id": item.item_id, "title": item.title} for item in request.items]
    result = await chatagent_service.embed_items(
        items=items,
        batch_id=request.batch_id
    )
    # 将 dict 转换为 Pydantic 模型，确保响应格式一致
    response_data = ItemEmbedResponse(**result)
    return success_response(data=response_data.model_dump())


@router.post("/items/embed/file", summary="上传文件并嵌入商品")
async def upload_items_file(
    file: UploadFile = File(..., description="商品数据文件（支持 .txt/.tsv/.csv）"),
    batch_id: Optional[str] = Form(default=None, description="批次ID"),
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    上传商品数据文件并自动嵌入
    
    支持的文件格式：
    - .txt / .tsv: 每行格式为 "ID\\t商品标题"
    - .csv: CSV 格式，需包含 item_id 和 title 列
    
    HTTP 请求示例 (curl):
    ```bash
    curl -X POST 'http://localhost:8000/api/chatagent/items/embed/file' \\
      -H 'X-API-Key: your-api-key' \\
      -F 'file=@/path/to/items.tsv' \\
      -F 'batch_id=batch_001'
    ```
    """
    # 读取文件内容
    file_content = await file.read()
    
    # 嵌入商品
    result = await chatagent_service.embed_items_from_file(
        file_content=file_content,
        file_name=file.filename,
        batch_id=batch_id
    )
    
    return success_response(data=result)


@router.post("/items/search", summary="搜索商品", response_model=ItemSearchResponse)
async def search_items(
    request: ItemSearchRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    混合检索商品（Dense + Sparse BM25）
    
    使用 Milvus 2.6+ 原生混合检索，按 item_id 去重。
    
    请求示例:
    ```json
    {
        "query": "苹果手机",
        "top_k": 10
    }
    ```
    
    响应示例:
    ```json
    {
        "code": 200,
        "message": "success",
        "data": {
            "query": "苹果手机",
            "total": 5,
            "items": [
                {
                    "item_id": "12345",
                    "content": "Apple iPhone 15 Pro Max 256GB",
                    "score": 0.95,
                    "metadata": {...}
                }
            ]
        }
    }
    ```
    """
    result = await chatagent_service.search_items_api(
        query=request.query,
        top_k=request.top_k
    )
    # result 已经是 ItemSearchResponse 对象，直接转换为 dict
    return success_response(data=result.model_dump())
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
        return success_response(data={
            "status": "unhealthy",
            "error": str(e[:200]) if len(str(e)) > 200 else str(e)
        }, code=503)


@router.post("/hospital/chat", summary="医院客服Agent对话")
async def hospital_chat(
    request: HospitalChatRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    使用医院客服 Agent 进行多步骤对话

    Agent 流程：
    1. 问题重写 - 将用户问题改写为更适合检索的查询
    2. 安全审查 - 检查问题是否涉及医疗建议、处方、诊断等敏感内容
    3. 知识检索 - 使用现有的 RAG 组件从医疗知识库检索相关内容
    4. 答案生成 - 基于检索结果生成安全、准确的回复

    HTTP 请求示例:
    ```bash
    curl -X POST 'http://localhost:8000/api/chatagent/hospital/chat' \\
      -H 'X-API-Key: your-api-key' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "message": "我想咨询一下，心脏病患者可以做胃镜检查吗？",
        "stream": false
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
        "stream_available": true
      }
    }
    ```
    """
    response = await chatagent_service.chat_with_hospital_agent(request)
    return success_response(data=response.model_dump())


@router.post("/hospital/chat/stream", summary="医院客服Agent流式对话")
async def hospital_chat_stream(
    request: HospitalChatRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    使用医院客服 Agent 进行流式对话

    返回 SSE (Server-Sent Events) 流式响应

    HTTP 请求示例:
    ```bash
    curl -X POST 'http://localhost:8000/api/chatagent/hospital/chat/stream' \\
      -H 'X-API-Key: your-api-key' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "message": "我想咨询一下，心脏病患者可以做胃镜检查吗？",
        "stream": true
      }'
    ```

    SSE 事件格式:
    - step_start: 步骤开始
    - step_complete: 步骤完成
    - content: 内容块
    - done: 完成
    - error: 错误
    """
    from fastapi.responses import StreamingResponse

    async def event_generator():
        async for chunk in chatagent_service.chat_with_hospital_agent_stream(request):
            yield chunk
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
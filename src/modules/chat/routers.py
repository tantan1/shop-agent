from fastapi import APIRouter, Depends, UploadFile, File, Form, Request
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.shared.database import get_db
from src.modules.auth.dependencies import verify_api_key, get_current_client
from src.core.permissions import ClientInfo
from src.core.rate_limiter import get_rate_limiter
from src.modules.chat.services import ChatAgentService
from src.modules.chat.schemas import (
    ChatQueryRequest,
    InsertDocumentRequest,
    ExperimentCreateRequest,
    ExperimentPauseRequest,
    ExperimentValidateRequest,
    BatchInsertRequest,
    ChatRequest,
    ItemEmbedRequest,
    BatchItemEmbedRequest,
    ItemEmbedResponse,
    ItemSearchRequest,
    ItemSearchResponse,
    RefundConfirmRequest,
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


@router.post("/agent/chat", summary="通用Agent对话")
async def agent_chat(
    request: ChatRequest,
    req: Request,
    _: None = Depends(verify_api_key),
    current_client: ClientInfo = Depends(get_current_client),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service),
    _rl: None = Depends(get_rate_limiter().dependency(max_requests=15, window_seconds=60)),
):
    
    # ── 设置 Token 限流上下文 ──
    try:
        from src.core.config import config as core_config
        from src.modules.chat.core.llm_service import set_rate_limit_context

        client_ip = req.client.host if req.client else "unknown"
        conv_id = getattr(request, "conversation_id", "") or "new"
        rate_limit_key = f"{client_ip}:agent_chat:{conv_id}"
        enabled = getattr(core_config, "TOKEN_LIMIT_ENABLED", True)
        set_rate_limit_context(rate_limit_key, enabled=enabled)
    except Exception:
        pass  # 上下文设置失败不影响主流程

    # ── A/B 实验分配 ──
    experiment_assignment = None
    try:
        from src.modules.chat.core.experiment_service import ExperimentService
        exp_service = ExperimentService.get_instance()
        if exp_service.is_initialized:
            user_id = request.conversation_id or client_ip
            experiment_assignment = exp_service.assign(user_id, request.domain)
    except Exception:
        pass  # 实验分配失败不影响主流程

    try:
        response = await chatagent_service.chat_with_agent(
            request, experiment_assignment=experiment_assignment
        )
        # ── A2A 对话追踪：记录每条对话到内存存储 ──
        try:
            from src.modules.chat.a2a_routers import register_conversation_event
            register_conversation_event(
                conversation_id=response.conversation_id,
                user_message=request.message,
                assistant_message=response.message,
                domain=request.domain,
            )
        except Exception:
            pass
        return success_response(data=response.model_dump())
    except Exception as e:
        # 处理工具权限不足
        if "ToolPermissionError" in type(e).__name__:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={
                    "code": 403,
                    "message": "error",
                    "data": {
                        "error": str(e),
                        "permission_denied": True,
                    },
                },
            )
        # 处理 TokenLimitExceeded
        if "TokenLimitExceeded" in type(e).__name__:
            from fastapi.responses import JSONResponse
            remain = getattr(e, "remaining", 0)
            reset = getattr(e, "reset_seconds", 60)
            return JSONResponse(
                status_code=429,
                content={
                    "code": 429,
                    "message": "error",
                    "data": {
                        "error": "Token 消耗超限，请稍后再试",
                        "retry_after": reset,
                        "limit_type": "token",
                    },
                },
                headers={
                    "Retry-After": str(reset),
                    "X-Token-Limit-Remaining": str(remain),
                    "X-Token-Limit-Reset": str(reset),
                },
            )
        raise


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


@router.post("/agent/refund/confirm", summary="退款人工确认（人在回路）")
async def refund_confirm(
    request: RefundConfirmRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    """
    人对回路退款审批 —— 对 pending 的退款申请进行批准或拒绝。

    使用场景：
    1. 用户发起退款请求 → Agent 自动暂停，返回 status="waiting_for_confirmation"
    2. 管理员通过本接口审批 → 批准则继续执行退款，拒绝则取消退款

    请求示例:
    ```json
    {
        "conversation_id": "conv_1234567890",
        "confirm": true,
        "remark": "已核实，同意退款"
    }
    ```

    响应示例:
    ```json
    {
        "code": 200,
        "message": "success",
        "data": {
            "message": "已为您提交退货申请...",
            "conversation_id": "conv_1234567890",
            "status": "completed",
            ...
        }
    }
    ```
    """
    response = await chatagent_service.confirm_refund(request)
    return success_response(data=response.model_dump())




@router.post("/agent/test", summary="test")
async def agent_test(
    request: ChatRequest,
    _: None = Depends(verify_api_key),
    chatagent_service: ChatAgentService = Depends(get_chatagent_service)
):
    
    response = await chatagent_service.test_agent(request)
    return success_response(data=response.model_dump())


# =============================================================================
# A/B 实验管理 API
# =============================================================================

@router.post("/experiments", summary="创建/更新 A/B 实验")
async def create_experiment(
    exp_request: ExperimentCreateRequest,
    _: None = Depends(verify_api_key),
):
    """创建或更新 A/B 实验配置（写入 Redis，热加载生效）

    实验变量支持：
    - Reranker: threshold, top_k, enabled
    - Retrieval: strategy (hybrid/dense_only/bm25_only), top_k
    - LLM: model, temperature
    - Prompt: template_version_key
    - 以及 content_filter, synonym_normalize, nebula_graph 等 feature toggle

    请求示例:
    ```json
    {
      "id": "exp_reranker_001",
      "name": "Reranker 阈值消融实验",
      "variants": [
        {
          "name": "control",
          "variant_type": "control",
          "traffic_percent": 50,
          "pipeline_overrides": {"rerank_threshold": 0.3}
        },
        {
          "name": "treatment",
          "variant_type": "treatment",
          "traffic_percent": 50,
          "pipeline_overrides": {"rerank_threshold": 0.1}
        }
      ],
      "safety_guards": [
        {"metric": "escalation_rate", "threshold": 0.10, "comparison": "pct_change"}
      ],
      "domains": ["ecommerce"]
    }
    ```
    """
    from src.modules.chat.core.experiment_service import (
        ExperimentService, ExperimentDef, VariantDef, VariantType,
        PipelineOverrides, SafetyGuard, ExperimentStatus, SafetyMetricType,
    )
    from datetime import datetime

    exp_service = ExperimentService.get_instance()
    if not exp_service.is_initialized:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={
            "code": 503, "message": "ExperimentService 未初始化（Redis 不可用？）"})

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp = ExperimentDef(
        id=exp_request.id,
        name=exp_request.name,
        description=exp_request.description,
        status=ExperimentStatus.RUNNING,
        variants=[
            VariantDef(
                name=v.name,
                variant_type=VariantType(v.variant_type),
                traffic_percent=v.traffic_percent,
                pipeline_overrides=PipelineOverrides.from_dict(v.pipeline_overrides),
                description=v.description,
            )
            for v in exp_request.variants
        ],
        safety_guards=[
            SafetyGuard.from_dict(dict(
                metric=g.metric,
                threshold=g.threshold,
                comparison=g.comparison,
                window_seconds=g.window_seconds,
                action=g.action,
            ))
            for g in exp_request.safety_guards
        ],
        domains=exp_request.domains,
        owner=exp_request.owner,
        created_at=now_str,
        updated_at=now_str,
    )

    ok = exp_service.create_experiment(exp)
    return success_response(data={"status": "created" if ok else "failed", "experiment_id": exp_request.id})


@router.patch("/experiments", summary="暂停/恢复/停止 A/B 实验")
async def update_experiment_status(
    pause_request: ExperimentPauseRequest,
    _: None = Depends(verify_api_key),
):
    """修改实验状态
    - paused: 暂停实验（保留配置，所有用户退出实验）
    - running: 恢复运行
    - stopped: 永久停止（建议保留数据后 archive）

    请求示例:
    ```json
    {"id": "exp_reranker_001", "status": "paused"}
    ```
    """
    from src.modules.chat.core.experiment_service import ExperimentService, ExperimentStatus

    exp_service = ExperimentService.get_instance()
    exp = exp_service.get_experiment(pause_request.id)
    if not exp:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"code": 404, "message": f"实验 {pause_request.id} 不存在"})

    new_status = ExperimentStatus(pause_request.status)
    exp.status = new_status
    exp_service.create_experiment(exp)
    return success_response(data={"status": new_status.value, "experiment_id": pause_request.id})


@router.post("/experiments/validate", summary="验证实验流量分流均匀性")
async def validate_experiment_distribution(
    validate_request: ExperimentValidateRequest,
    _: None = Depends(verify_api_key),
):
    """模拟 N 个用户验证流量分配均匀性（回答'你测过分流均匀性吗？'的追问）

    请求示例:
    ```json
    {"id": "exp_reranker_001", "sample_user_count": 10000}
    ```

    响应:
    ```json
    {
      "experiment_id": "exp_reranker_001",
      "total_users": 10000,
      "variant_counts": {"control": 5023, "treatment": 4977},
      "chi_square": 0.21,
      "is_uniform": true
    }
    ```
    """
    from src.modules.chat.core.experiment_service import ExperimentService

    exp_service = ExperimentService.get_instance()
    sample_users = [f"user_{i:06d}" for i in range(validate_request.sample_user_count)]
    result = exp_service.validate_distribution(validate_request.id, sample_users)
    result["experiment_id"] = validate_request.id
    return success_response(data=result)


@router.get("/experiments", summary="列出所有 A/B 实验")
async def list_experiments(
    _: None = Depends(verify_api_key),
):
    """列出当前所有运行中的 A/B 实验"""
    from src.modules.chat.core.experiment_service import ExperimentService

    exp_service = ExperimentService.get_instance()
    experiments = exp_service.list_experiments()
    return success_response(data={
        "count": len(experiments),
        "experiments": [e.to_dict() for e in experiments],
    })


@router.get("/experiments/{experiment_id}", summary="获取单个 A/B 实验详情")
async def get_experiment_detail(
    experiment_id: str,
    _: None = Depends(verify_api_key),
):
    """获取单个实验的详细配置"""
    from src.modules.chat.core.experiment_service import ExperimentService

    exp_service = ExperimentService.get_instance()
    exp = exp_service.get_experiment(experiment_id)
    if not exp:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"code": 404, "message": f"实验 {experiment_id} 不存在"})

    return success_response(data=exp.to_dict())


@router.delete("/experiments/{experiment_id}", summary="删除 A/B 实验")
async def delete_experiment(
    experiment_id: str,
    _: None = Depends(verify_api_key),
):
    """删除实验配置（不可恢复）"""
    from src.modules.chat.core.experiment_service import ExperimentService

    exp_service = ExperimentService.get_instance()
    ok = exp_service.delete_experiment(experiment_id)
    return success_response(data={"deleted": ok, "experiment_id": experiment_id})


@router.post("/experiments/refresh", summary="强制刷新实验配置缓存")
async def refresh_experiments(
    _: None = Depends(verify_api_key),
):
    """强制从 Redis 重新加载实验配置（正常情况下每 30 秒自动刷新）"""
    from src.modules.chat.core.experiment_service import ExperimentService

    exp_service = ExperimentService.get_instance()
    exp_service.force_refresh()
    return success_response(data={"refreshed": True})


# =============================================================================
# Agent Card —— A2A 能力发现端点
# =============================================================================

from src.modules.chat.core.agent_card import build_agent_card  # 模块级导入，利用 lifespan 预热


@router.get("/agent/card", summary="Agent Card（A2A 能力发现）", include_in_schema=True)
async def agent_card():
    """返回 A2A 标准 Agent Card（缓存命中，<1ms）。

    外部系统通过此端点发现 Shop-Agent 的能力：
    - skills 列表来自 skills/ 目录的 SKILL.md（与 MCP tools/list 同源）
    - capabilities 声明流式/推送等协议支持

    无需认证（等同于 /.well-known/agent-card.json）。
    """
    card = build_agent_card()
    return card.model_dump(by_alias=True)
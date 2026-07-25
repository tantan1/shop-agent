"""旧版 RAG 聊天流程模块。"""
from __future__ import annotations

from langchain_core.documents import Document

from src.modules.chat.agent.prompts import PromptTemplateManager
from src.modules.chat.core.content_filter import ContentFilterService
from src.modules.chat.schemas import ChatQueryRequest, ChatQueryResponse
from src.shared.exceptions import ValidationException
from src.shared.logger import APILogger

logger = APILogger("orchestrator_rag")


async def search_similar_documents(embeddings, milvus, query: str, top_k: int = 3) -> list[Document]:
    """搜索相似的文档"""
    try:
        query_embedding = await embeddings.aembed_query(query)
        documents = milvus.search_similar(query_embedding, top_k)
        return documents
    except Exception as e:
        import traceback

        logger.error(f"Document search failed: {str(e)}\n{traceback.format_exc()}")
        raise ValidationException("文档搜索失败", str(e)) from e


async def generate_response(llm, query: str, documents: list[Document], langfuse_handler=None) -> str:
    """基于检索到的文档生成回答"""
    try:
        context = "\n\n".join([doc.page_content for doc in documents[:3]]) or "暂无相关信息"

        from datetime import datetime

        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")

        template = PromptTemplateManager.get("ecommerce", "ecommerce_step4_generate")

        if template:
            prompt_content = template.format(
                graph_context="",
                rag_context=context,
                user_question=query,
                current_time=current_time,
                safety_check_result="风险等级: low",
                safety_reminder="",
                chat_history="",
                product_info=context,
                knowledge_base=context,
                context=context,
                category="",
            )
        else:
            prompt_content = (
                f"基于以下信息回答用户问题。\n\n知识库：\n{context}\n\n问题：{query}\n\n回答："
            )

        response = await llm.chat_qwen_with_prompt(
            prompt=prompt_content,
            system_prompt="你是一个智能助手，基于知识库提供准确回答。不要虚构公司名称或品牌信息。",
            langfuse_handler=langfuse_handler,
        )
        return response
    except Exception as e:
        import traceback

        logger.error(f"Response generation failed: {str(e)}\n{traceback.format_exc()}")
        return "抱歉，我暂时无法回答这个问题。"


async def chat_rag(llm, embeddings, milvus, request: ChatQueryRequest, langfuse_handler=None) -> ChatQueryResponse:
    """RAG 聊天接口（旧版，简单检索→生成）"""
    getattr(request, "conversation_id", None) or "rag-default"
    domain = getattr(request, "domain", "ecommerce")

    try:
        search_query = await request.message
        similar_docs = await search_similar_documents(embeddings, milvus, search_query)

        response_text = await generate_response(
            llm, request.message, similar_docs, langfuse_handler=langfuse_handler
        )

        cf = ContentFilterService.get_instance()
        output_check = cf.filter_output(response_text, domain)
        if not output_check.is_safe:
            logger.warning(
                "RAG 输出安全检查未通过",
                domain=domain,
                risk_categories=output_check.risk_categories,
            )
            if output_check.filtered_text:
                response_text = output_check.filtered_text
            else:
                response_text = "抱歉，当前无法处理您的请求，请稍后重试。"

        response = ChatQueryResponse(
            message=response_text,
            relevant_documents=[doc.page_content for doc in similar_docs],
            document_count=len(similar_docs),
        )

        logger.log_business_event(
            "RAG聊天查询",
            success=True,
            query=request.message,
            document_count=len(similar_docs),
            response_length=len(response_text),
        )
        return response
    except Exception as e:
        logger.log_business_event(
            "RAG聊天查询",
            success=False,
            error=str(e),
            query=request.message,
        )
        raise ValidationException("聊天查询失败", str(e)) from e

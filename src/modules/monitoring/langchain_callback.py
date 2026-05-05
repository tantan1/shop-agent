"""
LangChain Prometheus 回调处理器
统一追踪 LLM 调用、Token 使用、工具执行等
使用 LangChain 标准回调机制
"""
from typing import Any, Dict, List, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult, Generation
from langchain_core.agents import AgentAction, AgentFinish

from src.modules.monitoring.metrics import (
    api_call_counter,
    api_duration_histogram,
    agent_conversation_counter,
    agent_token_counter,
    embedding_request_counter,
    embedding_request_duration,
    embedding_token_counter,
    exception_counter,
)


class PrometheusCallbackHandler(BaseCallbackHandler):
    """
    LangChain Prometheus 回调处理器
    
    自动追踪以下事件：
    - LLM 调用次数和耗时
    - Token 使用量 (prompt/completion)
    - Embedding 请求和 token
    - Agent 执行成功/失败
    - 工具调用
    - 异常
    """

    def __init__(self, module: str = "langchain"):
        super().__init__()
        self.module = module
        self._current_chain = ""
        self._start_time: Optional[float] = None

    # ==================== LLM 回调 ====================
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """LLM 开始调用"""
        import time
        self._start_time = time.time()
        model_name = serialized.get("name", "unknown") if serialized else "unknown"
        self._current_model = model_name

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """LLM 调用结束 - 统计 Token"""
        import time
        if self._start_time:
            duration = time.time() - self._start_time
            # 记录 LLM 调用耗时
            api_call_counter.labels(
                module=self.module,
                method="llm",
                status="success"
            ).inc()
            api_duration_histogram.labels(module=self.module).observe(duration)

        # 统计 Token 使用量 (LangChain 标准方式)
        if response and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            
            if prompt_tokens > 0:
                agent_token_counter.labels(type="prompt").inc(prompt_tokens)
            if completion_tokens > 0:
                agent_token_counter.labels(type="completion").inc(completion_tokens)
            if total_tokens > 0:
                agent_token_counter.labels(type="total").inc(total_tokens)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """LLM 调用错误"""
        exception_counter.labels(
            type=type(error).__name__,
            module=self.module
        ).inc()
        api_call_counter.labels(
            module=self.module,
            method="llm",
            status="error"
        ).inc()

    # ==================== Embedding 回调 ====================
    
    def on_embedding_start(
        self,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Embedding 开始"""
        import time
        self._embedding_start_time = time.time()

    def on_embedding_end(
        self,
        *,
        run_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Embedding 结束 - 统计 token 和耗时"""
        import time
        if hasattr(self, '_embedding_start_time'):
            duration = time.time() - self._embedding_start_time
            # 统计请求
            embedding_request_counter.labels(
                provider="volcengine",
                status="success"
            ).inc()
            embedding_request_duration.labels(provider="volcengine").observe(duration)
            
            # 统计 token
            if outputs and isinstance(outputs, dict):
                tokens = outputs.get("tokens", 0) or outputs.get("token_usage", 0)
                if tokens > 0:
                    embedding_token_counter.labels(provider="volcengine", type="text").inc(tokens)

    # ==================== Agent 回调 ====================
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Agent 执行动作（工具调用）"""
        api_call_counter.labels(
            module=self.module,
            method=f"tool:{action.tool}",
            status="success"
        ).inc()

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Agent 完成"""
        agent_conversation_counter.labels(status="success").inc()

    # ==================== Tool 回调 ====================
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """工具开始执行"""
        import time
        tool_name = serialized.get("name", "unknown") if serialized else "unknown"
        self._tool_start_time = time.time()
        self._current_tool = tool_name

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """工具执行结束"""
        import time
        if hasattr(self, "_tool_start_time"):
            duration = time.time() - self._tool_start_time
            tool_name = getattr(self, "_current_tool", "unknown")
            api_duration_histogram.labels(module=f"tool:{tool_name}").observe(duration)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """工具执行错误"""
        exception_counter.labels(
            type=type(error).__name__,
            module=getattr(self, "_current_tool", "tool")
        ).inc()

    # ==================== Chain 回调 ====================
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Chain 开始执行"""
        import time
        self._chain_start_time = time.time()
        chain_name = serialized.get("name", "unknown") if serialized else "unknown"
        self._current_chain = chain_name

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Chain 执行结束"""
        import time
        if hasattr(self, "_chain_start_time"):
            duration = time.time() - self._chain_start_time
            api_duration_histogram.labels(module=f"chain:{self._current_chain}").observe(duration)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Chain 执行错误"""
        exception_counter.labels(
            type=type(error).__name__,
            module=f"chain:{self._current_chain}"
        ).inc()

    # ==================== Retriever 回调 ====================
    
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """检索器开始"""
        import time
        self._retriever_start_time = time.time()

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """检索器结束"""
        import time
        if hasattr(self, "_retriever_start_time"):
            duration = time.time() - self._retriever_start_time
            api_duration_histogram.labels(module="retriever").observe(duration)


# 全局单例
_prometheus_callback: Optional[PrometheusCallbackHandler] = None


def get_prometheus_callback() -> PrometheusCallbackHandler:
    """获取 Prometheus 回调处理器单例"""
    global _prometheus_callback
    if _prometheus_callback is None:
        _prometheus_callback = PrometheusCallbackHandler()
    return _prometheus_callback

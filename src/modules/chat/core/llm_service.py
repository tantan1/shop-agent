"""
LLM 服务模块
支持通义千问和火山引擎 Doubao 模型
"""

from typing import List, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

from langchain_openai import ChatOpenAI

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config
from src.modules.monitoring.langchain_callback import get_prometheus_callback

logger = APILogger("llm_service")

# 类型变量用于泛型返回
T = TypeVar("T", bound=BaseModel)


class LLMService:
    """大语言模型服务"""
    
    _instance = None
    _qwen_llm: Optional[ChatOpenAI] = None
    _tool_selector_llm: Optional[ChatOpenAI] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "LLMService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self):
        """初始化 LLM 服务"""
        try:
            # 初始化通义千问模型
            if chat_config.tongyi_api_key:
                self._qwen_llm = ChatOpenAI(
                    model=chat_config.chat_model,
                    api_key=chat_config.tongyi_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0.7,
                    extra_body={"enable_thinking": False}
                )
                logger.info(f"通义千问模型初始化成功: {chat_config.chat_model}")
            
        except Exception as e:
            logger.error(f"LLM 服务初始化失败: {str(e)}")
            raise
    
    @property
    def qwen_llm(self) -> ChatOpenAI:
        """获取通义千问 LLM（懒加载）"""
        if self._qwen_llm is None:
            if not chat_config.tongyi_api_key:
                raise ValueError("TONGYI_API_KEY 未配置")
            self._qwen_llm = ChatOpenAI(
                model=chat_config.chat_model,
                api_key=chat_config.tongyi_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.7,
                extra_body={"enable_thinking": False}
            )
        return self._qwen_llm

    @property
    def tool_selector_llm(self) -> ChatOpenAI:
        """获取工具选择器专用轻量 LLM（更快、更便宜）。

        P1 工具选择任务极其简单（从 3-5 个工具名选一个），
        不需要主 Agent 的大模型，用小模型可降低延迟 50%+ 且不牺牲准确率。
        """
        if self._tool_selector_llm is None:
            if not chat_config.tongyi_api_key:
                return self.qwen_llm  # 降级到主模型
            model = getattr(chat_config, "tool_selector_model", None)
            if model is None:
                return self.qwen_llm  # 未配置则回退主模型
            self._tool_selector_llm = ChatOpenAI(
                model=model,
                api_key=chat_config.tongyi_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.0,  # 工具选择不需要创造性
                extra_body={"enable_thinking": False}
            )
            logger.info(f"工具选择器模型初始化成功: {model}")
        return self._tool_selector_llm
    
    async def chat_qwen(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        track_metrics: bool = True,
        langfuse_handler = None,
        **kwargs
    ) -> str:
        """
        使用通义千问模型聊天
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            track_metrics: 是否追踪指标（默认开启，LangChain回调会自动统计Token）
            langfuse_handler: 可选的 Langfuse CallbackHandler（由调用方传入，携带 session_id/user_id/tags）
            **kwargs: 其他参数
            
        Returns:
            模型回复内容
        """
        try:
            # 构建配置
            config = {}
            if track_metrics:
                callbacks = [get_prometheus_callback()]
                if langfuse_handler:
                    callbacks.append(langfuse_handler)
                config["callbacks"] = callbacks
            
            response = await self.qwen_llm.ainvoke(messages, config=config)
            return response.content
        except Exception as e:
            logger.error(f"通义千问调用失败: {str(e)}")
            raise

    async def chat_qwen_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[BaseModel],
        temperature: float = 0.0,
        track_metrics: bool = True,
        max_retries: int = 2,
        langfuse_handler = None,
        **kwargs
    ) -> BaseModel:
        """
        使用通义千问模型进行结构化输出（带重试和降级）
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            output_schema: Pydantic Schema，用于结构化输出
            temperature: 温度参数（结构化输出通常用0）
            track_metrics: 是否追踪指标
            max_retries: 最大重试次数（ValidationError 时）
            langfuse_handler: 可选的 Langfuse CallbackHandler
            **kwargs: 其他参数
            
        Returns:
            结构化对象（Pydantic Model 实例）
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # 构建配置
                config = {}
                if track_metrics:
                    callbacks = [get_prometheus_callback()]
                    if langfuse_handler:
                        callbacks.append(langfuse_handler)
                    config["callbacks"] = callbacks
                
                # 使用 with_structured_output 获取支持结构化输出的 LLM
                structured_llm = self.qwen_llm.with_structured_output(output_schema)
                
                response = await structured_llm.ainvoke(messages, config=config)
                return response
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # ValidationError：模型输出不符合 Schema
                if "ValidationError" in error_type or "validation" in str(e).lower():
                    if attempt < max_retries:
                        logger.warning(
                            f"结构化输出验证失败（尝试 {attempt + 1}/{max_retries}）: {str(e)[:200]}"
                        )
                        continue
                    else:
                        logger.error(f"结构化输出重试耗尽: {str(e)[:200]}")
                
                logger.error(f"通义千问结构化调用失败: {error_type} - {str(e)[:200]}")
                raise
        
        # 不应该到达这里，但以防万一
        raise last_error or Exception("结构化输出未知错误")

    def create_structured_llm(self, output_schema: Type[BaseModel]) -> ChatOpenAI:
        """
        创建支持结构化输出的 LLM 实例
        
        Args:
            output_schema: Pydantic Schema
            
        Returns:
            配置好的 ChatOpenAI 实例（已绑定 with_structured_output）
        """
        return self.qwen_llm.with_structured_output(output_schema)
    
    async def chat_qwen_with_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        langfuse_handler=None,
    ) -> str:
        """使用 prompt 字符串调用通义千问"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.chat_qwen(messages, temperature, langfuse_handler=langfuse_handler)
    
    
    def close(self):
        """关闭服务"""
        self._qwen_llm = None
        self._tool_selector_llm = None
        LLMService._instance = None

"""
LLM 服务模块
支持通义千问和火山引擎 Doubao 模型
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
from volcenginesdkarkruntime import Ark
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from src.shared.logger import APILogger
from src.modules.chat.config import chat_config

logger = APILogger("llm_service")


class LLMService:
    """大语言模型服务"""
    
    _instance = None
    _ark_client: Optional[Ark] = None
    _qwen_llm: Optional[ChatOpenAI] = None
    
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
            
            # 初始化火山引擎 Ark 客户端
            if chat_config.volcengine_api_key:
                self._ark_client = Ark(api_key=chat_config.volcengine_api_key)
                logger.info("火山引擎 Ark 客户端初始化成功")
                
        except Exception as e:
            logger.error(f"LLM 服务初始化失败: {str(e)}")
            raise
    
    @property
    def ark_client(self) -> Ark:
        """获取 Ark 客户端"""
        if self._ark_client is None:
            if not chat_config.volcengine_api_key:
                raise ValueError("VOLCENGINE_API_KEY 未配置")
            self._ark_client = Ark(api_key=chat_config.volcengine_api_key)
        return self._ark_client
    
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
    
    async def chat_qwen(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        使用通义千问模型聊天
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            模型回复内容
        """
        try:
            response = await self.qwen_llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"通义千问调用失败: {str(e)}")
            raise
    
    async def chat_qwen_with_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7
    ) -> str:
        """使用 prompt 字符串调用通义千问"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.chat_qwen(messages, temperature)
    
    
    def close(self):
        """关闭服务"""
        self._ark_client = None
        self._qwen_llm = None
        LLMService._instance = None

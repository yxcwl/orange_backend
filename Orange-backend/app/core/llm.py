"""
LLM 大模型客户端模块
封装与 LLM 的交互逻辑，支持多供应商切换
"""
"""
以下仅为测试修改基础参考，具体方法后续修改
"""

from typing import AsyncIterator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.utils.logger import logger


class LLMClient:
    """LLM 大模型客户端"""

    def __init__(self):
        settings = get_settings()
        self._chat_model = self._create_chat_model(settings)

    @staticmethod
    def _create_chat_model(settings) -> BaseChatModel:
        """
        根据配置创建对应的 ChatModel 实例

        Args:
            settings: 全局配置

        Returns:
            LangChain BaseChatModel 实例
        """
        provider = settings.LLM_PROVIDER.lower()

        if provider == "openai":
            return ChatOpenAI(
                model=settings.LLM_MODEL_NAME,
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
        # 后续可扩展其他供应商
        # elif provider == "zhipu":
        #     ...
        # elif provider == "deepseek":
        #     ...
        else:
            raise ValueError(f"不支持的 LLM 供应商: {provider}")

    @property
    def chat_model(self) -> BaseChatModel:
        """获取 ChatModel 实例"""
        return self._chat_model

    async def ainvoke(self, messages: list, **kwargs) -> str:
        """
        异步调用大模型

        Args:
            messages: LangChain 消息列表
            **kwargs: 额外参数

        Returns:
            模型回复文本
        """
        response = await self._chat_model.ainvoke(messages, **kwargs)
        return response.content

    async def astream(self, messages: list, **kwargs) -> AsyncIterator[str]:
        """
        异步流式调用大模型

        Args:
            messages: LangChain 消息列表
            **kwargs: 额外参数

        Yields:
            模型回复的文本片段
        """
        async for chunk in self._chat_model.astream(messages, **kwargs):
            if chunk.content:
                yield chunk.content


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取 LLMClient 单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
        logger.info("LLM 客户端初始化完成")
    return _llm_client

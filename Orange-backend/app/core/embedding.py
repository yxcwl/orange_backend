"""
Embedding 向量化模型模块
负责将文本转换为向量表示，支持多供应商切换
"""
"""
以下仅为测试修改基础参考，具体方法后续修改
"""
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config.settings import get_settings
from app.utils.logger import logger


class EmbeddingClient:
    """Embedding 向量化客户端"""

    def __init__(self):
        settings = get_settings()
        self._embeddings = self._create_embeddings(settings)

    @staticmethod
    def _create_embeddings(settings) -> Embeddings:
        """
        根据配置创建对应的 Embeddings 实例

        Args:
            settings: 全局配置

        Returns:
            LangChain Embeddings 实例
        """
        provider = settings.EMBEDDING_PROVIDER.lower()

        if provider == "openai":
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL_NAME,
                api_key=settings.EMBEDDING_API_KEY or settings.LLM_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL,
            )
        # 后续可扩展
        # elif provider == "local":
        #     from langchain_community.embeddings import HuggingFaceEmbeddings
        #     return HuggingFaceEmbeddings(model_name=...)
        else:
            raise ValueError(f"不支持的 Embedding 供应商: {provider}")

    @property
    def embeddings(self) -> Embeddings:
        """获取 Embeddings 实例"""
        return self._embeddings

    async def aembed_query(self, text: str) -> list[float]:
        """
        异步将查询文本向量化

        Args:
            text: 查询文本

        Returns:
            向量列表
        """
        return await self._embeddings.aembed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        异步将文档列表向量化

        Args:
            texts: 文档文本列表

        Returns:
            向量列表的列表
        """
        return await self._embeddings.aembed_documents(texts)


# 全局单例
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """获取 EmbeddingClient 单例"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
        logger.info("Embedding 客户端初始化完成")
    return _embedding_client

"""
Qdrant 向量数据库客户端模块
负责与 Qdrant 的连接管理、集合创建与操作
"""
"""
以下仅为测试修改基础参考，具体方法后续修改，需讨论是否现在进行启动
"""
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config.settings import get_settings
from app.utils.logger import logger


class QdrantManager:
    """Qdrant 向量数据库管理器"""

    def __init__(self):
        settings = get_settings()
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

    def ensure_collection(self) -> None:
        """确保集合存在，不存在则创建"""
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"已创建 Qdrant 集合: {self.collection_name}")
        else:
            logger.info(f"Qdrant 集合已存在: {self.collection_name}")

    def upsert_points(self, points: list[PointStruct]) -> None:
        """
        插入或更新向量点

        Args:
            points: PointStruct 列表
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"已写入 {len(points)} 条向量数据")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量相似度检索

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的 k 条结果
            score_threshold: 相似度阈值
            filters: 元数据过滤条件

        Returns:
            检索结果列表，每项包含 payload 和 score
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=filters,
        )
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def delete_points(self, point_ids: list[str]) -> None:
        """
        根据ID删除向量点

        Args:
            point_ids: 要删除的点ID列表
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=point_ids,
        )
        logger.info(f"已删除 {len(point_ids)} 条向量数据")

    def get_collection_info(self) -> dict:
        """获取集合信息"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }


# 全局单例
_qdrant_manager: Optional[QdrantManager] = None


def get_qdrant_manager() -> QdrantManager:
    """获取 QdrantManager 单例"""
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager()
        _qdrant_manager.ensure_collection()
    return _qdrant_manager

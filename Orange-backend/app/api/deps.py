"""
依赖注入模块
提供 FastAPI 路由中可复用的依赖项
"""

from app.services.chat_service import ChatService
from app.services.knowledge_service import KnowledgeService
from app.services.tool_service import ToolService
from app.config.settings import get_settings, Settings


def get_settings_dep() -> Settings:
    """获取全局配置"""
    return get_settings()


# 服务单例
_chat_service: ChatService | None = None
_knowledge_service: KnowledgeService | None = None
_tool_service: ToolService | None = None


def get_chat_service() -> ChatService:
    """获取对话服务实例"""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


def get_knowledge_service() -> KnowledgeService:
    """获取知识库服务实例"""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service


def get_tool_service() -> ToolService:
    """获取工具服务实例"""
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service

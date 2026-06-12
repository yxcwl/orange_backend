"""
依赖注入模块
提供 FastAPI 路由中可复用的依赖项
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings, Settings
from app.core.database import get_db
from app.models.db_models import User
from app.services.auth_service import get_auth_service, AuthService
from app.services.chat_service import ChatService
from app.services.knowledge_service import KnowledgeService
from app.services.tool_service import ToolService

# Bearer Token 提取器
_bearer_scheme = HTTPBearer(auto_error=False)


def get_settings_dep() -> Settings:
    """获取全局配置"""
    return get_settings()


# ==================== 认证依赖 ====================


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """
    从请求头中提取并验证 JWT token，返回当前用户

    用法:
        @router.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            ...

    未登录或 token 无效时返回 401
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await auth_service.get_current_user(db, credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或过期的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用",
        )

    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> User | None:
    """
    可选认证：有 token 则返回用户，无 token 返回 None

    适用于既允许匿名访问、又需要识别登录用户的接口
    """
    if credentials is None:
        return None

    user = await auth_service.get_current_user(db, credentials.credentials)
    if user and not user.is_active:
        return None
    return user


def require_role(required_role: str):
    """
    角色权限依赖工厂

    用法:
        @router.get("/admin-only", dependencies=[Depends(require_role("admin"))])
        async def admin_route(user: User = Depends(get_current_user)):
            ...
    """

    async def _check_role(user: User = Depends(get_current_user)) -> User:
        auth_service = get_auth_service()
        if not auth_service.check_permission(user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要 {required_role} 角色权限",
            )
        return user

    return _check_role


# ==================== 服务单例 ====================

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

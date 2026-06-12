"""
Orange-RAG 后端服务主入口
基于 FastAPI 框架，提供 LangChain + RAG 本地知识库服务
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.config.settings import get_settings
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} 启动中...")

    # 初始化 MySQL 数据库（建表）
    try:
        from app.core.database import init_db
        await init_db()
        logger.info("MySQL 数据库初始化完成")
    except Exception as e:
        logger.warning(f"MySQL 数据库初始化失败: {e}")

    # 初始化默认管理员账户
    try:
        from app.core.database import get_db
        from app.models.db_models import User
        from app.services.auth_service import get_auth_service
        from sqlalchemy import select

        auth_service = get_auth_service()
        async for db in get_db():
            result = await db.execute(select(User).where(User.username == "admin"))
            if result.scalar_one_or_none() is None:
                await auth_service.create_user(
                    db=db,
                    username="admin",
                    password="admin123",
                    nickname="系统管理员",
                    role="admin",
                )
                logger.info("默认管理员账户已创建: admin / admin123（请及时修改密码）")
            break
    except Exception as e:
        logger.warning(f"默认管理员账户初始化失败: {e}")

    # 初始化 Qdrant
    try:
        from app.core.qdrant import get_qdrant_manager
        get_qdrant_manager()
        logger.info("Qdrant 连接初始化完成")
    except Exception as e:
        logger.warning(f"Qdrant 连接失败（服务仍可启动）: {e}")

    # 注册工具
    try:
        import app.tools.fertilizer_calculator  # noqa: F401
        logger.info("工具注册完成")
    except Exception as e:
        logger.warning(f"工具注册失败: {e}")

    # 初始化 RAG 桥接
    try:
        from app.services.rag_bridge import get_rag_api
        get_rag_api()
        logger.info("RAG 桥接初始化完成")
    except Exception as e:
        logger.warning(f"RAG 桥接初始化失败: {e}")

    logger.info(f"{settings.APP_NAME} 启动完成")
    yield

    # 关闭清理
    try:
        from app.core.database import close_db
        await close_db()
    except Exception:
        pass
    logger.info(f"{settings.APP_NAME} 正在关闭...")


# 创建 FastAPI 应用
settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="广西橙子/柑橘产业智能助手 - LangChain + RAG 本地知识库后端服务",
    lifespan=lifespan,
)

# CORS 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 V1 路由，后续增添修改改这里进行注册
app.include_router(v1_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["健康检查"])
async def root():
    """服务根路径，返回基本信息"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

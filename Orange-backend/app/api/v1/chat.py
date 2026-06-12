"""
对话相关 API 路由
提供对话问答、流式输出、历史记录、快捷卡片等接口
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_chat_service
from app.core.database import get_db
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatHistoryItem,
    ChatHistoryResponse,
    QuickCard,
)
from app.schemas.common import ResponseBase
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["对话问答"])


@router.post("/completions", response_model=ResponseBase, summary="对话问答")
async def chat_completions(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    db: AsyncSession = Depends(get_db),
) -> ResponseBase:
    """
    对话问答接口

    - 支持多轮对话（通过 conversation_id 关联）
    - 支持流式/非流式输出
    - 自动检测并调用工具（如肥料计算器）
    - 严格基于知识库回答，标注来源
    """
    if request.stream:
        # 流式响应
        return StreamingResponse(
            _stream_generator(request, chat_service, db),
            media_type="text/event-stream",
        )

    # 非流式响应
    result = await chat_service.chat(request)

    # 记录问答日志到 MySQL
    try:
        from app.api.v1.admin import add_chat_log
        await add_chat_log(
            db=db,
            conversation_id=result.conversation_id,
            question=request.question,
            answer=result.answer,
            sources=result.sources if result.sources else None,
            tool_used=result.tool_used,
        )
    except Exception:
        pass  # 日志记录失败不影响主流程

    return ResponseBase(data=result.model_dump())

# 流式生成器数据来源的对接
# 后续看是否还需添加会话id
async def _stream_generator(
    request: ChatRequest,
    chat_service: ChatService,
    db: AsyncSession,
):
    """SSE 流式生成器"""
    full_answer = ""
    sources = None
    tool_used = None
    conversation_id = request.conversation_id

    async for chunk in chat_service.chat_stream(request):
        data = chunk.model_dump(exclude_none=True)
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 收集流式数据用于日志
        if chunk.type == "content" and chunk.content:
            full_answer += chunk.content
        if chunk.type == "source" and not sources:
            sources = []
        if chunk.type == "tool" and chunk.tool_name:
            tool_used = chunk.tool_name
        if hasattr(chunk, "conversation_id") and chunk.conversation_id:
            conversation_id = chunk.conversation_id

    # 流式结束后记录日志
    try:
        from app.api.v1.admin import add_chat_log
        await add_chat_log(
            db=db,
            conversation_id=conversation_id or "",
            question=request.question,
            answer=full_answer,
            sources=sources,
            tool_used=tool_used,
        )
    except Exception:
        pass


@router.get(
    "/history/{conversation_id}",
    response_model=ResponseBase,
    summary="获取对话历史",
)
async def get_chat_history(
    conversation_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> ResponseBase:
    """获取指定会话的对话历史"""
    messages = chat_service.get_chat_history(conversation_id)
    history_items = [
        ChatHistoryItem(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
        )
        for msg in messages
    ]
    return ResponseBase(
        data=ChatHistoryResponse(
            conversation_id=conversation_id,
            messages=history_items,
        ).model_dump()
    )


@router.get(
    "/quick-cards",
    response_model=ResponseBase,
    summary="获取快捷推荐卡片",
)
async def get_quick_cards(
    chat_service: ChatService = Depends(get_chat_service),
) -> ResponseBase:
    """
    获取预设快捷推荐卡片

    用于首页展示，帮助用户快速开始对话
    """
    cards = chat_service.get_quick_cards()
    return ResponseBase(data=[card.model_dump() for card in cards])

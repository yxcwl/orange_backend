"""
数据模型定义
用于内部业务逻辑的数据结构（非API Schema）
"""
"""
以下仅为测试修改基础参考，具体方法后续修改
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    """知识库文档模型"""
    id: str
    title: str
    source_type: str  # general / literature / policy / structured / external
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    chunk_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None


@dataclass
class Conversation:
    """会话模型"""
    id: str
    messages: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ChatLog:
    """问答日志模型"""
    id: str
    conversation_id: str
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    tool_used: Optional[str] = None
    is_corrected: bool = False  # 是否被人工纠错
    correction: Optional[str] = None  # 纠错内容
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

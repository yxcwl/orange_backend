"""
文档切片策略模块
支持多种切片方式：按字符数切片、按语义切片（文献友好）
"""

"""
以下仅为测试修改基础参考，具体方法后续修改
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from app.config.settings import get_settings
from app.utils.logger import logger


@dataclass
class ChunkResult:
    """切片结果"""
    content: str  # 切片文本内容
    metadata: dict = field(default_factory=dict)  # 元数据（来源、页码等）


class DocumentChunker:
    """文档切片器"""

    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.chunk_separator = settings.CHUNK_SEPARATOR

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
        source_type: str = "general",
    ) -> list[ChunkResult]:
        """
        对纯文本进行切片

        Args:
            text: 原始文本
            metadata: 附加元数据
            source_type: 来源类型（general / literature / policy）

        Returns:
            切片结果列表
        """
        metadata = metadata or {}

        # 根据来源类型选择切片策略
        if source_type == "literature":
            return self._chunk_literature(text, metadata)
        elif source_type == "markdown":
            return self._chunk_markdown(text, metadata)
        else:
            return self._chunk_recursive(text, metadata)

    def _chunk_recursive(
        self, text: str, metadata: dict
    ) -> list[ChunkResult]:
        """
        递归字符切片（通用策略）
        适用于：政策文件、普通文档
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

        chunks = splitter.split_text(text)
        logger.info(f"递归切片完成，共 {len(chunks)} 片")

        return [
            ChunkResult(
                content=chunk,
                metadata={**metadata, "chunk_index": i, "chunk_method": "recursive"},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _chunk_literature(
        self, text: str, metadata: dict
    ) -> list[ChunkResult]:
        """
        文献专用切片策略
        适用于：专业书籍、手册、文献
        特点：更小的切片尺寸，保留章节结构信息
        """
        # 文献使用更小的切片以保持语义完整性
        lit_chunk_size = min(self.chunk_size, 400)
        lit_overlap = min(self.chunk_overlap, 80)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=lit_chunk_size,
            chunk_overlap=lit_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

        chunks = splitter.split_text(text)
        logger.info(f"文献切片完成，共 {len(chunks)} 片")

        return [
            ChunkResult(
                content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "chunk_method": "literature",
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    def _chunk_markdown(
        self, text: str, metadata: dict
    ) -> list[ChunkResult]:
        """
        Markdown 按标题切片
        适用于：结构化的 Markdown 文档
        """
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
        )

        md_chunks = md_splitter.split_text(text)

        # 对每个 Markdown 块再进行子切片
        results = []
        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        for i, md_chunk in enumerate(md_chunks):
            sub_texts = sub_splitter.split_text(md_chunk.page_content)
            for j, sub_text in enumerate(sub_texts):
                results.append(
                    ChunkResult(
                        content=sub_text,
                        metadata={
                            **metadata,
                            **md_chunk.metadata,
                            "chunk_index": f"{i}-{j}",
                            "chunk_method": "markdown",
                        },
                    )
                )

        logger.info(f"Markdown 切片完成，共 {len(results)} 片")
        return results

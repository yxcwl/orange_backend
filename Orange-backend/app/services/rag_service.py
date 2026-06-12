"""
RAG 检索增强生成服务
核心流程：查询向量化 → 向量检索 → 构建Prompt → LLM生成回答
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from app.config.settings import get_settings
from app.core.qdrant import get_qdrant_manager
from app.core.llm import get_llm_client
from app.core.embedding import get_embedding_client
from app.schemas.chat import ChatStreamChunk
from app.utils.logger import logger


# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """你是一个专业的广西橙子/柑橘产业智能助手，专门为农户、客商和相关从业者提供信息服务。

## 核心原则
1. **严格基于知识库回答**：你必须且只能根据检索到的参考资料来回答问题。
2. **拒绝超范围问题**：如果知识库中没有相关内容，你必须明确告知用户"当前知识库中未找到相关信息，无法回答该问题"，不得自行编造或推测。
3. **标注来源**：每个回答必须标注参考资料的出处，格式为 [来源: 文档标题]。
4. **多轮对话**：结合上下文理解用户意图，必要时进行追问或引导。

## 回答格式
- 回答问题时，先给出直接答案，再补充详细说明
- 涉及数据时，必须引用具体来源
- 如果用户问题模糊，主动引导用户补充信息

## 工具调用
- 当用户需要计算类操作时（如施肥量计算），使用对应工具完成
- 工具调用结果也需要结合知识库内容进行解释
"""


class RAGService:
    """RAG 检索增强生成服务"""

    def __init__(self):
        self.settings = get_settings()
        self.qdrant = get_qdrant_manager()
        self.llm = get_llm_client()
        self.embedding = get_embedding_client()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量
            score_threshold: 相似度阈值
            filters: 元数据过滤

        Returns:
            检索结果列表
        """
        top_k = top_k or self.settings.RAG_TOP_K
        score_threshold = score_threshold or self.settings.RAG_SCORE_THRESHOLD

        # 查询向量化
        query_vector = await self.embedding.aembed_query(query)

        # 向量检索
        results = self.qdrant.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )

        logger.info(f"RAG 检索完成，查询: '{query[:30]}...'，命中 {len(results)} 条")
        return results

    def _build_context(self, search_results: list[dict]) -> str:
        """
        将检索结果构建为上下文文本

        Args:
            search_results: 检索结果列表

        Returns:
            格式化的上下文文本
        """
        if not search_results:
            return "（知识库中未找到相关内容）"

        context_parts = []
        for i, result in enumerate(search_results, 1):
            payload = result.get("payload", {})
            source_title = payload.get("document_title", "未知来源")
            content = payload.get("content", "")
            score = result.get("score", 0)

            context_parts.append(
                f"[参考资料 {i}] 来源: {source_title} (相似度: {score:.2f})\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        history: Optional[list[dict]] = None,
    ) -> list:
        """
        构建完整的消息列表

        Args:
            question: 用户问题
            context: 检索到的上下文
            history: 对话历史

        Returns:
            LangChain 消息列表
        """
        # 构建包含上下文的用户消息
        user_message = f"""## 检索到的参考资料
{context}

## 用户问题
{question}

## 回答要求
请严格基于上述参考资料回答用户问题。如果参考资料中没有相关信息，请明确告知。回答中请标注参考资料的来源。"""

        messages = [HumanMessage(content=user_message)]

        # 如果有历史对话，插入到前面
        if history:
            history_messages = []
            for msg in history[-self.settings.CHAT_HISTORY_MAX_TURNS * 2:]:
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))

            messages = history_messages + messages

        return messages

    async def generate(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        完整的 RAG 生成流程

        Args:
            question: 用户问题
            history: 对话历史
            filters: 元数据过滤

        Returns:
            包含 answer 和 sources 的字典
        """
        # 1. 检索
        search_results = await self.retrieve(question, filters=filters)

        # 2. 构建上下文
        context = self._build_context(search_results)

        # 3. 构建消息
        messages = self._build_messages(question, context, history)

        # 4. 使用 ChatPromptTemplate 构建完整 prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # 简化：直接使用消息列表调用
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if history:
            for msg in history[-self.settings.CHAT_HISTORY_MAX_TURNS * 2:]:
                full_messages.append(msg)
        full_messages.append({"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"})

        # 转换为 LangChain 消息
        lc_messages = []
        for msg in full_messages:
            if msg["role"] == "system":
                from langchain_core.messages import SystemMessage
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        # 5. 调用 LLM
        answer = await self.llm.ainvoke(lc_messages)

        # 6. 整理来源信息
        sources = [
            {
                "document_title": r.get("payload", {}).get("document_title", ""),
                "content": r.get("payload", {}).get("content", "")[:200],
                "score": r.get("score", 0),
                "chunk_id": r.get("id", ""),
            }
            for r in search_results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }

    async def generate_stream(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        filters: Optional[dict] = None,
    ):
        """
        流式 RAG 生成

        Yields:
            ChatStreamChunk
        """
        # 1. 检索
        search_results = await self.retrieve(question, filters=filters)
        context = self._build_context(search_results)

        # 2. 构建消息
        from langchain_core.messages import SystemMessage
        lc_messages = [SystemMessage(content=SYSTEM_PROMPT)]

        if history:
            for msg in history[-self.settings.CHAT_HISTORY_MAX_TURNS * 2:]:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

        lc_messages.append(
            HumanMessage(content=f"参考资料：\n{context}\n\n问题：{question}")
        )

        # 3. 先返回来源信息
        sources = [
            {
                "document_title": r.get("payload", {}).get("document_title", ""),
                "content": r.get("payload", {}).get("content", "")[:200],
                "score": r.get("score", 0),
                "chunk_id": r.get("id", ""),
            }
            for r in search_results
        ]

        yield ChatStreamChunk(type="source", sources=sources)

        # 4. 流式输出回答
        async for chunk in self.llm.astream(lc_messages):
            yield ChatStreamChunk(type="content", content=chunk)

        # 5. 结束标记
        yield ChatStreamChunk(type="done")

from __future__ import annotations

from dataclasses import dataclass

from rag_qa.config import Settings, get_settings
from rag_qa.document_loader import load_source
from rag_qa.embeddings import create_embedding_model
from rag_qa.llm_client import LLMClient, create_llm_client
from rag_qa.text_splitter import Chunk, split_document
from rag_qa.vector_store import InMemoryVectorStore, SearchResult


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    rag_answer: str
    plain_answer: str
    sources: list[dict[str, str | float]]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, llm_client: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = create_embedding_model(
            self.settings.embedding_provider,
            self.settings.embedding_model_name,
        )
        self.vector_store = InMemoryVectorStore(self.embedding_model)
        self.llm_client = llm_client or create_llm_client(
            provider=self.settings.llm_provider,
            base_url=self.settings.llm_base_url,
            model_name=self.settings.llm_model_name,
            api_key=self.settings.llm_api_key,
            timeout=self.settings.request_timeout,
        )
        self.chunks: list[Chunk] = []

    def build_knowledge_base(self, sources: list[str]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for source in sources:
            document = load_source(source, timeout=self.settings.request_timeout)
            chunks = split_document(
                document,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            all_chunks.extend(chunks)
        self.vector_store.add_chunks(all_chunks)
        self.chunks.extend(all_chunks)
        return all_chunks

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        results = self.vector_store.similarity_search(question, top_k=top_k or self.settings.top_k)
        context = format_context(results)
        rag_prompt = build_rag_prompt(question, context)
        plain_prompt = f"请直接回答问题，不要编造不确定信息：\n{question}"
        return RAGAnswer(
            question=question,
            rag_answer=self.llm_client.generate(rag_prompt),
            plain_answer=self.llm_client.generate(plain_prompt),
            sources=[
                {
                    "source": result.chunk.source,
                    "chunk_index": str(result.chunk.index),
                    "score": round(result.score, 4),
                    "content": result.chunk.content[:500],
                }
                for result in results
            ],
        )


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "你是一个严谨的知识库问答助手。请只基于参考资料回答问题；"
        "如果参考资料不足，请说明缺少依据。\n\n"
        f"参考资料：\n{context}\n\n"
        f"用户问题：{question}\n\n"
        "请给出结构清晰、可追溯的中文回答。"
    )


def format_context(results: list[SearchResult]) -> str:
    if not results:
        return "未检索到相关资料。"
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"[{index}] 来源：{result.chunk.source}；分数：{result.score:.4f}\n{result.chunk.content}"
        )
    return "\n\n".join(lines)

from pathlib import Path

from rag_qa.llm_client import MockLLMClient
from rag_qa.rag_chain import RAGPipeline


def test_rag_pipeline_answers_with_sources(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text(
        "RAG 会先检索外部资料，再把相关上下文提供给大模型生成回答。",
        encoding="utf-8",
    )
    pipeline = RAGPipeline(llm_client=MockLLMClient())
    chunks = pipeline.build_knowledge_base([str(source)])
    answer = pipeline.answer("RAG 的流程是什么？")
    assert chunks
    assert "RAG" in answer.rag_answer
    assert answer.sources

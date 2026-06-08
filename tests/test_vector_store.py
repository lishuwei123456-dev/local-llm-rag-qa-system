from rag_qa.document_loader import Document
from rag_qa.embeddings import HashingEmbeddingModel
from rag_qa.text_splitter import split_document
from rag_qa.vector_store import InMemoryVectorStore


def test_similarity_search_returns_relevant_chunk() -> None:
    document = Document(
        content="RAG 通过检索外部知识减少幻觉。\n\n天气预报用于查询气温。",
        source="demo.md",
        metadata={},
    )
    chunks = split_document(document, chunk_size=30, chunk_overlap=0)
    store = InMemoryVectorStore(HashingEmbeddingModel())
    store.add_chunks(chunks)
    results = store.similarity_search("RAG 如何减少幻觉", top_k=1)
    assert results
    assert "RAG" in results[0].chunk.content

from rag_qa.document_loader import Document
from rag_qa.text_splitter import split_document


def test_split_document_keeps_source_metadata() -> None:
    document = Document(
        content="RAG 是检索增强生成。" * 80,
        source="demo.md",
        metadata={"type": "file"},
    )
    chunks = split_document(document, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0].source == "demo.md"
    assert chunks[0].metadata["chunk_index"] == "0"

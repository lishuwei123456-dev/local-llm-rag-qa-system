from __future__ import annotations

from dataclasses import dataclass

from rag_qa.document_loader import Document


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    index: int
    metadata: dict[str, str]


def split_document(document: Document, chunk_size: int = 500, chunk_overlap: int = 80) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    text = document.content.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        content = text[start:end].strip()
        if content:
            chunks.append(
                Chunk(
                    content=content,
                    source=document.source,
                    index=len(chunks),
                    metadata={**document.metadata, "chunk_index": str(len(chunks))},
                )
            )
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks

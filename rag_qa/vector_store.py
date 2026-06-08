from __future__ import annotations

import math
from dataclasses import dataclass

from rag_qa.embeddings import EmbeddingModel
from rag_qa.text_splitter import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._vectors.extend(self.embedding_model.embed([chunk.content for chunk in chunks]))

    def similarity_search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if not self._chunks:
            return []
        query_vector = self.embedding_model.embed([query])[0]
        scored = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)

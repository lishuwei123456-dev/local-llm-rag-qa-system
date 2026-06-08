from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


class EmbeddingModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class HashingEmbeddingModel(EmbeddingModel):
    dimensions: int = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class HuggingFaceEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers to use HuggingFace embeddings.") from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors]


def create_embedding_model(provider: str, model_name: str) -> EmbeddingModel:
    if provider.lower() == "huggingface":
        try:
            return HuggingFaceEmbeddingModel(model_name)
        except Exception:
            return HashingEmbeddingModel()
    return HashingEmbeddingModel()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return words or [text[:32]]

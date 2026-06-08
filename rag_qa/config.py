from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "ollama"
    llm_model_name: str = "qwen2.5:7b"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "ollama"
    embedding_provider: str = "hash"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    request_timeout: int = 30


def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "qwen2.5:7b"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "ollama"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "4")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
    )

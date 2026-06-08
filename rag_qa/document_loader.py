from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Document:
    content: str
    source: str
    metadata: dict[str, str]


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_source(source: str, timeout: int = 30) -> Document:
    if is_url(source):
        return load_url(source, timeout=timeout)
    return load_file(source)


def load_file(path: str) -> Document:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")
    if file_path.suffix.lower() not in {".txt", ".md", ".markdown"}:
        raise ValueError("Only .txt and .md files are supported in the lightweight loader.")
    text = file_path.read_text(encoding="utf-8")
    return Document(
        content=clean_text(text),
        source=str(file_path),
        metadata={"type": "file", "name": file_path.name},
    )


def load_url(url: str, timeout: int = 30) -> Document:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("URL loading requires requests and beautifulsoup4.") from exc

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "local-rag-demo/0.1"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    title = soup.title.get_text(strip=True) if soup.title else url
    return Document(content=clean_text(text), source=url, metadata={"type": "url", "title": title})

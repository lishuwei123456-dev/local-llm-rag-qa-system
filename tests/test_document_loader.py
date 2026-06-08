from pathlib import Path

from rag_qa.document_loader import clean_text, load_file


def test_clean_text_compacts_spaces() -> None:
    assert clean_text("A   B\n\n\nC") == "A B\n\nC"


def test_load_file_reads_markdown(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("# RAG\n\n检索增强生成。", encoding="utf-8")
    document = load_file(str(source))
    assert "检索增强生成" in document.content
    assert document.metadata["type"] == "file"

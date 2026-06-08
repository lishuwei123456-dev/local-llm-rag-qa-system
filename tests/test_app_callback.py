from rag_qa.app import run_query


def test_run_query_returns_five_outputs_for_missing_source() -> None:
    outputs = run_query("", "RAG 是什么？", 4)
    assert len(outputs) == 5
    assert outputs[0] == "请至少输入一个知识来源。"


def test_run_query_returns_demo_answer() -> None:
    outputs = run_query("sample_data/demo_knowledge.md", "RAG 的核心流程是什么？", 4)
    assert len(outputs) == 5
    assert "知识库构建完成" in outputs[0]
    assert "RAG" in outputs[1]

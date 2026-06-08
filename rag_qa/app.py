from __future__ import annotations

from rag_qa.rag_chain import RAGPipeline


def run_query(sources_text: str, question: str) -> tuple[str, str, str]:
    sources = [line.strip() for line in sources_text.splitlines() if line.strip()]
    if not sources:
        return "请至少输入一个知识来源。", "", ""
    if not question.strip():
        return "请输入问题。", "", ""

    pipeline = RAGPipeline()
    chunks = pipeline.build_knowledge_base(sources)
    answer = pipeline.answer(question.strip())
    source_text = "\n\n".join(
        f"[{idx}] {item['source']} | score={item['score']}\n{item['content']}"
        for idx, item in enumerate(answer.sources, start=1)
    )
    summary = f"已构建 {len(chunks)} 个知识片段。\n\n{answer.rag_answer}"
    return summary, answer.plain_answer, source_text


def create_demo():
    import gradio as gr

    with gr.Blocks(title="Local LLM RAG QA") as demo:
        gr.Markdown("# 基于本地大模型与 RAG 的智能问答系统")
        sources = gr.Textbox(
            label="知识来源",
            lines=4,
            value="sample_data/demo_knowledge.md",
            placeholder="每行一个本地 .txt/.md 文件路径或网页 URL",
        )
        question = gr.Textbox(label="问题", value="RAG 为什么能减少模型幻觉？")
        submit = gr.Button("开始问答", variant="primary")
        rag_answer = gr.Textbox(label="RAG 增强回答", lines=8)
        plain_answer = gr.Textbox(label="普通回答对比", lines=5)
        retrieved = gr.Textbox(label="检索到的参考片段", lines=10)
        submit.click(run_query, inputs=[sources, question], outputs=[rag_answer, plain_answer, retrieved])
    return demo


def main() -> None:
    create_demo().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

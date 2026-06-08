from __future__ import annotations

import json

from rag_qa.rag_chain import RAGPipeline


def run_query(sources_text: str, question: str, top_k: int) -> tuple[str, str, str, str, str]:
    sources = [line.strip() for line in sources_text.splitlines() if line.strip()]
    if not sources:
        return "请至少输入一个知识来源。", "", "", "", "{}"
    if not question.strip():
        return "请输入问题。", "", "", "", "{}"

    pipeline = RAGPipeline()
    chunks = pipeline.build_knowledge_base(sources)
    answer = pipeline.answer(question.strip(), top_k=int(top_k))
    source_text = "\n\n".join(
        f"[{idx}] {item['source']} | score={item['score']}\n{item['content']}"
        for idx, item in enumerate(answer.sources, start=1)
    )
    summary = f"知识库构建完成：{len(chunks)} 个片段，命中 {len(answer.sources)} 条参考资料。"
    payload = {
        "question": answer.question,
        "chunk_count": len(chunks),
        "source_count": len(answer.sources),
        "sources": answer.sources,
    }
    return summary, answer.rag_answer, answer.plain_answer, source_text or "未检索到相关片段。", json.dumps(payload, ensure_ascii=False, indent=2)


def create_demo():
    import gradio as gr

    css = """
    .app-title {font-size: 28px; font-weight: 700; margin-bottom: 4px}
    .app-subtitle {color: #526071; margin-bottom: 18px}
    .metric-card textarea {font-family: ui-monospace, SFMono-Regular, Consolas, monospace}
    """

    with gr.Blocks(title="Local LLM RAG QA", css=css) as demo:
        gr.HTML(
            """
            <div class="app-title">基于本地大模型与 RAG 的智能问答系统</div>
            <div class="app-subtitle">加载本地文档或网页资料，构建向量检索库，并对比普通回答与 RAG 增强回答。</div>
            """
        )
        with gr.Row():
            with gr.Column(scale=5):
                sources = gr.Textbox(
                    label="知识来源",
                    lines=5,
                    value="sample_data/demo_knowledge.md",
                    placeholder="每行一个本地 .txt/.md 文件路径或网页 URL",
                )
                question = gr.Textbox(label="问题", value="RAG 为什么能减少模型幻觉？")
            with gr.Column(scale=2):
                top_k = gr.Slider(label="检索片段 Top-K", minimum=1, maximum=8, value=4, step=1)
                submit = gr.Button("开始问答", variant="primary")
                status = gr.Textbox(label="运行状态", lines=3)

        with gr.Tabs():
            with gr.Tab("RAG 增强回答"):
                rag_answer = gr.Textbox(label="基于检索资料生成的回答", lines=9)
            with gr.Tab("普通回答对比"):
                plain_answer = gr.Textbox(label="不注入知识片段的回答", lines=7)
            with gr.Tab("检索片段"):
                retrieved = gr.Textbox(label="命中的参考资料", lines=12)
            with gr.Tab("结构化结果"):
                json_output = gr.Code(label="JSON", language="json", lines=14)

        gr.Examples(
            examples=[
                ["sample_data/demo_knowledge.md", "RAG 的核心流程是什么？", 4],
                ["sample_data/demo_knowledge.md", "为什么 RAG 可以减少模型幻觉？", 3],
            ],
            inputs=[sources, question, top_k],
        )
        submit.click(
            run_query,
            inputs=[sources, question, top_k],
            outputs=[status, rag_answer, plain_answer, retrieved, json_output],
        )
    return demo


def main() -> None:
    create_demo().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

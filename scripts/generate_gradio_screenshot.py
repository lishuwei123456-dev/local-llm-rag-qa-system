from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "gradio-rag-demo.png"
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"


def make_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1440, 980
    image = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(image)

    title_font = make_font(34)
    subtitle_font = make_font(18)
    label_font = make_font(17)
    text_font = make_font(16)
    small_font = make_font(14)
    button_font = make_font(18)
    tab_font = make_font(16)

    draw.text((56, 38), "基于本地大模型与 RAG 的智能问答系统", fill="#111827", font=title_font)
    draw.text(
        (56, 88),
        "加载本地文档或网页资料，构建向量检索库，并对比普通回答与 RAG 增强回答。",
        fill="#64748b",
        font=subtitle_font,
    )

    def card(x: int, y: int, w: int, h: int) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="#ffffff", outline="#dbe3ef", width=1)

    def wrap_text(text: str, max_width: int) -> list[str]:
        lines: list[str] = []
        for paragraph in text.split("\n"):
            current = ""
            for char in paragraph:
                candidate = current + char
                if current and draw.textlength(candidate, font=text_font) > max_width:
                    lines.append(current)
                    current = char
                else:
                    current = candidate
            lines.append(current)
        return lines

    def textbox(x: int, y: int, w: int, h: int, text: str, label: Optional[str] = None) -> None:
        if label:
            draw.text((x, y - 28), label, fill="#334155", font=label_font)
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill="#ffffff", outline="#cbd5e1", width=1)
        ty = y + 14
        for line in wrap_text(text, w - 28)[: max(1, (h - 20) // 24)]:
            draw.text((x + 14, ty), line, fill="#0f172a", font=text_font)
            ty += 24

    left_x, top_y = 56, 138
    card(left_x, top_y, 900, 300)
    textbox(left_x + 22, top_y + 62, 850, 98, "sample_data/demo_knowledge.md", "知识来源")
    textbox(left_x + 22, top_y + 218, 850, 52, "RAG 为什么能减少模型幻觉？", "问题")

    right_x = 990
    card(right_x, top_y, 350, 300)
    draw.text((right_x + 22, top_y + 28), "检索片段 Top-K", fill="#334155", font=label_font)
    draw.rounded_rectangle((right_x + 22, top_y + 74, right_x + 308, top_y + 82), radius=4, fill="#dbeafe")
    draw.ellipse((right_x + 172, top_y + 64, right_x + 196, top_y + 88), fill="#2563eb")
    draw.text((right_x + 22, top_y + 104), "4", fill="#475569", font=text_font)
    draw.rounded_rectangle((right_x + 22, top_y + 142, right_x + 308, top_y + 196), radius=8, fill="#2563eb")
    draw.text((right_x + 126, top_y + 156), "开始问答", fill="#ffffff", font=button_font)
    textbox(right_x + 22, top_y + 238, 308, 42, "知识库构建完成：1 个片段，命中 1 条参考资料。", "运行状态")

    tabs_y = 485
    draw.rounded_rectangle((56, tabs_y, 1340, tabs_y + 56), radius=10, fill="#ffffff", outline="#dbe3ef")
    tabs = ["RAG 增强回答", "普通回答对比", "检索片段", "结构化结果"]
    tab_x = 78
    for index, tab in enumerate(tabs):
        if index == 0:
            draw.rounded_rectangle((tab_x - 12, tabs_y + 10, tab_x + 142, tabs_y + 46), radius=8, fill="#eff6ff")
            color = "#1d4ed8"
        else:
            color = "#475569"
        draw.text((tab_x, tabs_y + 18), tab, fill=color, font=tab_font)
        tab_x += 180

    rag_answer = (
        "根据检索到的知识片段，RAG 会先检索外部资料，再把相关上下文交给大模型生成回答。"
        "这样答案可以围绕指定知识库展开，并返回来源片段，便于追溯依据，从而降低无依据生成和模型幻觉风险。"
    )
    plain_answer = "RAG 是一种把外部知识检索和大模型生成结合起来的问答方法。"
    source_text = (
        "[1] sample_data/demo_knowledge.md | score=0.3036\n"
        "RAG 的核心流程包括文档加载、文本清洗、文本切分、Embedding 向量化、向量库构建、"
        "相似度检索、Prompt 拼接和答案生成。"
    )
    textbox(56, 574, 620, 230, rag_answer, "基于检索资料生成的回答")
    textbox(714, 574, 626, 120, plain_answer, "普通回答对比")
    textbox(714, 748, 626, 170, source_text, "命中的参考资料")

    badge = "Gradio Web UI · Ollama/Qwen · FAISS · HuggingFace Embeddings"
    draw.rounded_rectangle((56, 930, 580, 960), radius=15, fill="#eef2ff")
    draw.text((74, 936), badge, fill="#3730a3", font=small_font)

    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

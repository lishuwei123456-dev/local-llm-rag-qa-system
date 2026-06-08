# 基于本地大模型与 RAG 的智能问答系统

面向本地知识库问答场景的 RAG 工程项目。系统支持从网页或本地文档加载知识资料，完成文本清洗、切分、Embedding 向量化、相似片段检索，并通过 Ollama/Qwen 生成带参考依据的回答。项目同时提供 Gradio 交互界面、CLI 示例和轻量测试。

## 项目背景

该项目根据个人简历中的“基于本地大模型与 RAG 的智能问答系统开发”经历重建，重点展示从资料接入到检索增强生成的完整闭环：

- 使用 Ollama 部署本地 Qwen 模型。
- 使用 LangChain 思路组织文档加载、切分、检索和生成流程。
- 使用 HuggingFace Embeddings 和 FAISS 构建本地向量库。
- 使用 Gradio 构建可交互的问答页面。
- 支持对比普通 LLM 回答与 RAG 增强回答。

## 功能

- 本地文本、Markdown 和网页 URL 加载。
- 文本清洗、固定窗口切分和重叠保留。
- HuggingFace Embeddings/FAISS 接口，内置哈希向量 fallback 便于离线测试。
- Ollama OpenAI-compatible API 调用，内置 Mock LLM fallback 便于演示。
- RAG 回答、普通回答和检索来源片段同时返回。
- Gradio Web 页面和命令行入口。
- Pytest 单元测试和 GitHub Actions。

## 技术栈

Python、Ollama、Qwen、LangChain、FAISS、HuggingFace Embeddings、Sentence Transformers、Gradio、Requests、BeautifulSoup、dotenv。

## 目录结构

```text
local-llm-rag-qa-system/
├── rag_qa/
│   ├── app.py              # Gradio 页面
│   ├── cli.py              # 命令行入口
│   ├── config.py           # 环境变量配置
│   ├── document_loader.py  # 文档/网页加载
│   ├── embeddings.py       # 向量模型
│   ├── llm_client.py       # Ollama/Mock LLM
│   ├── rag_chain.py        # RAG 编排
│   ├── text_splitter.py    # 文本切分
│   └── vector_store.py     # 向量检索
├── sample_data/
│   └── demo_knowledge.md
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## 快速运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m rag_qa.app
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m rag_qa.app
```

打开 Gradio 页面后，输入知识来源和问题：

```text
sample_data/demo_knowledge.md
RAG 为什么能减少模型幻觉？
```

## Ollama/Qwen 配置

安装并启动 Ollama 后拉取模型：

```bash
ollama pull qwen2.5:7b
ollama serve
```

`.env` 示例：

```text
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:7b
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
```

如果本机暂时没有 Ollama，系统会使用 Mock LLM 输出可读的流程演示结果，方便先验证 RAG 链路。

## CLI 示例

```bash
python -m rag_qa.cli --source sample_data/demo_knowledge.md --question "RAG 的核心流程是什么？"
```

输出包含：

- `rag_answer`：检索增强回答。
- `plain_answer`：不注入知识片段的普通回答。
- `sources`：被检索命中的参考片段。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

测试覆盖文档加载、文本切分、向量检索和 RAG 编排。测试不依赖外部网络和 Ollama 服务。

## 与简历项目对应关系

| 简历描述 | 仓库实现 |
| --- | --- |
| 文档解析与文本切分 | `rag_qa/document_loader.py`、`rag_qa/text_splitter.py` |
| HuggingFaceEmbeddings 向量化 | `rag_qa/embeddings.py` |
| FAISS 语义检索 | `rag_qa/vector_store.py` |
| Qwen/Ollama 本地模型 | `rag_qa/llm_client.py` |
| RAG 检索增强生成 | `rag_qa/rag_chain.py` |
| Gradio 交互页面 | `rag_qa/app.py` |

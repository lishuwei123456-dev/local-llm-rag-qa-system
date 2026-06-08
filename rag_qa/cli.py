from __future__ import annotations

import argparse
import json

from rag_qa.rag_chain import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RAG question answering.")
    parser.add_argument("--source", action="append", required=True, help="Local .txt/.md file or URL.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.build_knowledge_base(args.source)
    answer = pipeline.answer(args.question)
    print(
        json.dumps(
            {
                "question": answer.question,
                "rag_answer": answer.rag_answer,
                "plain_answer": answer.plain_answer,
                "sources": answer.sources,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OllamaOpenAIClient(LLMClient):
    base_url: str
    model_name: str
    api_key: str = "ollama"
    timeout: int = 30

    def generate(self, prompt: str) -> str:
        import requests

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class MockLLMClient(LLMClient):
    def generate(self, prompt: str) -> str:
        if "参考资料" in prompt:
            return "根据检索到的知识片段，RAG 会先检索外部资料，再把相关上下文交给大模型生成回答，从而提升依据性并减少幻觉。"
        return "RAG 是一种把外部知识检索和大模型生成结合起来的问答方法。"


def create_llm_client(provider: str, base_url: str, model_name: str, api_key: str, timeout: int) -> LLMClient:
    if provider.lower() == "ollama":
        client = OllamaOpenAIClient(base_url=base_url, model_name=model_name, api_key=api_key, timeout=timeout)
        try:
            client.generate("请回复 ok。")
            return client
        except Exception:
            return MockLLMClient()
    return MockLLMClient()

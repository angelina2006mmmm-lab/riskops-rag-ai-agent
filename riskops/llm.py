"""Optional real LLM backend using OpenAI-compatible Chat Completions tool calling."""

import os
import httpx


class LLMConfigurationError(RuntimeError):
    pass


class ChatCompletionClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key or self.api_key == "replace_me":
            raise LLMConfigurationError(
                "Режим llm требует OPENAI_API_KEY и модель с поддержкой tool calling. "
                "Режим offline работает без ключа."
            )

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": self.model, "messages": messages, "tools": tools,
                   "tool_choice": "auto", "temperature": 0}
        try:
            with httpx.Client(timeout=40) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions", json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                result = response.json()
            return result["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"Ошибка LLM API: {type(exc).__name__}: {exc}") from exc

"""OpenAI プロバイダ — Chat Completions API."""
from __future__ import annotations

from src.llm.base import LLMError, LLMResult
from src.llm.client import post_json
from src.llm.prompt import build_messages
from src.tokens.morse_tokens import DisplayMode


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._api_key = api_key

    def transform(
        self,
        raw_text: str,
        mode: DisplayMode,
        *,
        timeout: float,
        lead_text: str | None = None,
        compact: bool = False,
    ) -> LLMResult:
        body = post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": self.model,
                "messages": build_messages(raw_text, mode, lead_text, compact),
            },
            {"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout,
        )
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI 応答の解析に失敗しました") from exc
        return LLMResult(text=text, provider=self.name, model=self.model)


__all__ = ["OpenAIProvider"]

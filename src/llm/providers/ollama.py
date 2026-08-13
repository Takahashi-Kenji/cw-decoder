"""Ollama (ローカル) プロバイダ — /api/chat. API キー不要."""
from __future__ import annotations

from src.llm.base import LLMError, LLMResult
from src.llm.client import post_json
from src.llm.prompt import build_messages
from src.tokens.morse_tokens import DisplayMode


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, endpoint: str = "http://localhost:11434") -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")

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
            f"{self.endpoint}/api/chat",
            {
                "model": self.model,
                "messages": build_messages(raw_text, mode, lead_text, compact),
                "stream": False,
                # **thinking を切る。** 清書は文字の変換であって推論ではないので、
                # 考えさせても質は上がらず待ち時間だけ延びる (運用者の判断)。
                # thinking 非対応のモデルはこのキーを無視するので害はない。
                "think": False,
                "options": {
                    # 清書は入力より長くならない。上限を切って暴走を防ぐ。
                    # 実測で gemma3:12b が記号を延々と繰り返す暴走を起こしたので、
                    # 上限と繰り返し penalty の両方で止める
                    "num_predict": 512,
                    "repeat_penalty": 1.15,
                    # 清書は創作ではないので、ぶれない方を選ぶ
                    "temperature": 0.2,
                },
            },
            {},
            timeout=timeout,
        )
        try:
            text = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama 応答の解析に失敗しました") from exc
        return LLMResult(text=text, provider=self.name, model=self.model)


__all__ = ["OllamaProvider"]

"""Claude (Anthropic) プロバイダ — Messages API."""
from __future__ import annotations

from src.llm.base import LLMError, LLMResult
from src.llm.client import post_json
from src.llm.prompt import build_messages
from src.tokens.morse_tokens import DisplayMode

_ANTHROPIC_VERSION = "2023-06-01"
_MAX_TOKENS = 2048


class ClaudeProvider:
    name = "claude"

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
        msgs = build_messages(raw_text, mode, lead_text, compact)
        system = next(m["content"] for m in msgs if m["role"] == "system")
        user_msgs = [m for m in msgs if m["role"] != "system"]
        body = post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                # 清書は校正であって創作ではない。**ぶれると測定もできない**
                # (2026-08-08、同じプロンプトの 2 回の実行で CER が 16.32% と
                #  20.92% に割れた。差 4.6pt は効果より大きい)
                "temperature": 0.0,
                "max_tokens": _MAX_TOKENS,
                "system": system,
                "messages": user_msgs,
            },
            {
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
            },
            timeout=timeout,
        )
        content = body.get("content")
        if not isinstance(content, list):
            raise LLMError("Claude 応答の解析に失敗しました (content 不正)")
        # 先頭ブロックが thinking 等の非 text の場合があるため、全 text ブロックを連結する.
        text = "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text.strip():
            stop = body.get("stop_reason")
            if stop == "refusal":
                raise LLMError(
                    "Claude が安全上の理由で応答を拒否しました。別の入力でお試しください。"
                )
            raise LLMError(
                f"Claude から本文を取得できませんでした (stop_reason={stop})"
            )
        return LLMResult(text=text, provider=self.name, model=self.model)


__all__ = ["ClaudeProvider"]

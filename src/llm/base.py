"""LLM プロバイダ共通の基底型."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.tokens.morse_tokens import DisplayMode


@dataclass(frozen=True)
class LLMResult:
    """LLM 清書結果. ``text`` は推測箇所 ⟦…⟧ マーカー入り."""

    text: str
    provider: str
    model: str


class LLMError(Exception):
    """全プロバイダ共通エラー型 (キー未設定/オフライン/HTTP/タイムアウト等を正規化)."""


class LLMProvider(Protocol):
    """LLM プロバイダの抽象インターフェース."""

    name: str
    model: str

    def transform(
        self,
        raw_text: str,
        mode: DisplayMode,
        *,
        timeout: float,
        lead_text: str | None = ...,
        compact: bool = ...,
    ) -> LLMResult:
        """生デコードテキストを清書して返す. 失敗時 ``LLMError``.

        ``lead_text`` は直前のやり取り。自動清書は未清書の増分だけを送るため、
        話のつながりを見せる目的で添える (出力には含めさせない).

        ``compact`` は短いプロンプトを使う指示。ローカルの小さいモデル向け.
        """
        ...


__all__ = ["LLMResult", "LLMError", "LLMProvider"]

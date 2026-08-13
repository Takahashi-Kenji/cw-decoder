"""LLM 清書を別スレッドで実行する Qt ワーカー."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from src.llm.base import LLMError, LLMProvider


class LLMWorker(QObject):
    """確定テキストを LLM で清書するワーカー.

    UI → ワーカーは request_transform Slot (QueuedConnection 経由) で呼ぶ.
    """

    result_ready = Signal(str)      # 清書結果 (⟦…⟧ マーカー入り)
    error = Signal(str)             # 日本語エラーメッセージ
    busy_changed = Signal(bool)

    def __init__(self, timeout_s: float = 30.0) -> None:
        super().__init__()
        self._timeout_s = timeout_s
        self._provider: LLMProvider | None = None

    @Slot(object)
    def set_provider(self, provider: object) -> None:
        self._provider = provider  # type: ignore[assignment]

    def set_compact(self, compact: bool) -> None:
        """短いプロンプトを使うか (ローカルの小さいモデル向け)."""
        self._compact = compact

    @Slot(str, str, str)
    def request_transform(
        self, raw_text: str, mode: str, lead_text: str = ""
    ) -> None:
        if self._provider is None:
            self.error.emit("LLM プロバイダが設定されていません。")
            return
        self.busy_changed.emit(True)
        try:
            result = self._provider.transform(
                raw_text, mode,  # type: ignore[arg-type]
                timeout=self._timeout_s,
                lead_text=lead_text or None,
                compact=getattr(self, "_compact", False),
            )
            self.result_ready.emit(result.text)
        except LLMError as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - 予期せぬ失敗もUIに通知して継続
            self.error.emit(f"LLM 清書中に予期せぬエラー: {exc!r}")
        finally:
            self.busy_changed.emit(False)


__all__ = ["LLMWorker"]

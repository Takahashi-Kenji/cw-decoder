"""清書前の全体再デコードを別スレッドで行うワーカー.

**なぜ別スレッドなのか**
------------------------
300 秒の一括デコードは CPU 4 スレッドで約 1.3 秒かかる (実測)。音声スレッドで
走らせると hop (0.5 秒) を大きく超えて 2〜3 hop ぶん取りこぼす。交信しながら
読む側を止めないため、清書用の再デコードは独立したスレッドで行う。

**エンジンは共有しない。** 音声スレッドのエンジンを借りると、結局そちらを
止めることになる。モデルは 17 MB なので二重に持って構わない。

出てくるテキストは**画面の確定テキストを置き換えない**。清書 (LLM) に渡す
入力としてのみ使う。交信用と記録用を分けるという方針による
(運用者、2026-08-14)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal, Slot

from src.infer.engine import InferenceEngine
from src.infer.word_correct import correct_text
from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import Mode


class RedecodeWorker(QObject):
    """音声をまるごと 1 回デコードしてテキストにする (清書用)."""

    # (テキスト, この結果が覆う末尾の絶対サンプル位置)
    result_ready = Signal(str, int)
    error = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, checkpoint_path: Path | str, device: str = "cpu") -> None:
        super().__init__()
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._engine: InferenceEngine | None = None

    def _ensure_engine(self) -> InferenceEngine:
        """**最初に使うときに読み込む。** 清書を使わない運用で 17 MB を
        無駄に確保しないため。"""
        if self._engine is None:
            self._engine = InferenceEngine.from_checkpoint(
                self._checkpoint_path, device=torch.device(self._device)
            )
        return self._engine

    @Slot(object, int, str, float, bool, bool)
    def request_redecode(
        self,
        audio: object,
        end_sample: int,
        mode: str,
        confidence_threshold: float,
        word_correct_enabled: bool,
        word_correct_ja_enabled: bool,
    ) -> None:
        """``audio`` を 1 回でデコードしてテキストを返す.

        Args:
            audio: 清書用バッファの切り出し (``np.ndarray``)。
            end_sample: この音声が覆う末尾の絶対サンプル位置。呼び出し側が
                「どこまで清書したか」を覚えるために返す値であって、
                ここでは中身を見ない。
            mode: ``"european"`` / ``"japanese"`` / ``"auto"``。
            confidence_threshold: 読めなかった印にする閾値 (表示側と揃える)。
        """
        wave = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wave.size == 0:
            self.result_ready.emit("", end_sample)
            return
        self.busy_changed.emit(True)
        try:
            engine = self._ensure_engine()
            tokens = engine.decode_chunk(wave)
            converter = TokenConverter(
                mode=mode, confidence_threshold=confidence_threshold  # type: ignore[arg-type]
            )
            text = converter.convert_timed(tokens).text
            if word_correct_enabled:
                text = correct_text(
                    text, japanese_enabled=word_correct_ja_enabled
                ).text
            self.result_ready.emit(text, end_sample)
        except Exception as exc:                              # noqa: BLE001
            # **清書は補助機能なので落とさない。** 失敗しても受信は続く
            self.error.emit(f"再デコードに失敗しました: {exc!r}")
        finally:
            self.busy_changed.emit(False)


__all__ = ["RedecodeWorker"]

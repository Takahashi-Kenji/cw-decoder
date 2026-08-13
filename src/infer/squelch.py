"""スキッシュ: レベルが閾値を下回る間、音を無音に置き換える.

なぜ必要か
----------
メル特徴抽出はサンプル内で z-score 正規化する (``MelConfig.normalize``) ため、
**音量の情報が捨てられる**。そのため -80 dBFS のノイズでも増幅されて信号のように
見え、30 秒の入力で 38 個ものトークン (ほとんど ``[SN]``/``[SK]``) が出る。
しかも確信度は 0.6〜0.72 と高く、**確信度閾値では防げない**。
デコーダの手前で止めるしかない (2026-08-07 の実測、``docs/word_break_bias.md`` の付録)。

設計上の要点
------------
**ブロックを捨てずに同じ長さの無音へ置き換える。** 捨てると時間軸 (``total_consumed``)
が止まり、確定タイミングと改行の間隔判定が壊れる。

判定は **BPF 通過後**のレベルで行うこと。ノイズの大半は帯域外なので、生の入力より
分離が良い (実測で BPF はノイズ由来のトークンを 25 個 → 2 個に減らす)。
"""
from __future__ import annotations

import numpy as np

# この値以下の閾値はスキッシュ無効とみなす。
# -120 dBFS が実質的な下限なので、それより下なら「常に開いている」のと同じ。
DISABLED_BELOW_DB = -100.0


class Squelch:
    """レベルが閾値を下回る間、ブロックを無音に置き換える.

    Args:
        threshold_db: 開く閾値 (dBFS). ``DISABLED_BELOW_DB`` 以下なら無効.
        hold_sec: 閾値を下回ってから閉じるまでの保持時間 (秒).
            符号の頭やキーアップ中の短い谷で閉じないための余裕.
        sample_rate: サンプリングレート.
    """

    def __init__(
        self,
        threshold_db: float,
        hold_sec: float = 1.0,
        sample_rate: int = 8000,
    ) -> None:
        self.threshold_db = threshold_db
        self.hold_sec = hold_sec
        self.sample_rate = sample_rate
        self._hold_remaining_s = 0.0

    @property
    def enabled(self) -> bool:
        return self.threshold_db > DISABLED_BELOW_DB

    @property
    def is_open(self) -> bool:
        """現在開いているか (デバッグ・表示用)."""
        return not self.enabled or self._hold_remaining_s > 0.0

    def reset(self) -> None:
        self._hold_remaining_s = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        """開いていればそのまま、閉じていれば同じ長さの無音を返す."""
        if not self.enabled or block.size == 0:
            return block
        rms = float(np.sqrt(np.mean(block * block)))
        db = 20.0 * np.log10(rms) if rms > 1e-6 else -120.0
        if db >= self.threshold_db:
            # **そのブロック自体が閾値以上なら必ず通す。** hold は「切れた後も
            # しばらく開けておく」ための猶予であって、開く条件ではない
            # (hold=0 のとき閾値超えのブロックまで無音にする不具合があった)。
            self._hold_remaining_s = self.hold_sec
            return block
        self._hold_remaining_s = max(
            0.0, self._hold_remaining_s - block.size / self.sample_rate
        )
        if self._hold_remaining_s > 0.0:
            return block
        return np.zeros_like(block)


__all__ = ["DISABLED_BELOW_DB", "Squelch"]

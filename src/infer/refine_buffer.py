"""清書用の長い音声バッファ.

**デコード用リング (``SlidingWindowDecoder``、既定 30 秒) とは別に持つ。**

なぜ分けるのか
--------------
清書のために長い履歴が欲しいが、**デコード用リングを広げてはいけない**。
``refine_closed_turns`` (2 段階確定) はターンの音をリングから取り出して
**同期的に**デコードするので、リングが長いとその 1 回が長くなり、音声スレッドが
止まる。実測では 300 秒の一括デコードに **約 1.3 秒** かかり、hop (0.5 秒) を
大きく超える。せっかく直した凍結が別の形で戻ることになる。

そこで清書専用のバッファを別に持ち、**別スレッドから読むだけ**にする。
交信しながら読む側 (1・2 段目と 2 段階確定) には一切影響させない。

    音声入力
      ├→ デコード用リング 30 秒 …… 1・2 段目、2 段階確定 (音声スレッド)
      └→ 清書用バッファ 300 秒 …… 清書前の全体再デコード (別スレッド)

位置の数え方
------------
``snapshot`` が返す ``start`` は**総投入サンプル数を基準にした絶対位置**である。
清書の進み具合を「何文字目まで」ではなく**時間で**覚えるために使う。
再デコードするとテキストは作り直されるので、文字位置では管理できない。
"""
from __future__ import annotations

import threading

import numpy as np

# 保持する長さ (秒) の既定値。
#
# **5 分を超える交信はまずない** という運用者の判断 (2026-08-14)。
# メモリは 8 kHz float32 で 300 秒 = 約 9.6 MB。実測の一括デコードは
# CPU 4 スレッドで 1,268 ms、ピークメモリ 9.2 MB。
DEFAULT_REFINE_CAPACITY_S = 300.0


class RefineBuffer:
    """清書のために音声を貯めておくリングバッファ.

    **スレッド安全**。音声スレッドが ``push`` し、清書スレッドが ``snapshot``
    する。``snapshot`` は必ずコピーを返す (参照を渡すと書き込みと競合する)。
    """

    def __init__(
        self,
        capacity_s: float = DEFAULT_REFINE_CAPACITY_S,
        sample_rate: int = 8000,
    ) -> None:
        self.sample_rate = sample_rate
        self.capacity = max(1, int(capacity_s * sample_rate))
        self._lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._total_consumed = 0

    @property
    def total_consumed(self) -> int:
        """落ちた分も含めた総投入サンプル数 (絶対位置の基準)."""
        with self._lock:
            return self._total_consumed

    def push(self, audio: np.ndarray) -> None:
        """音声を追加する. 容量を超えた古い分は捨てる."""
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = audio.astype(np.float32, copy=False)
        with self._lock:
            self._total_consumed += audio.size
            self._audio = np.concatenate([self._audio, audio])
            if self._audio.size > self.capacity:
                self._audio = self._audio[-self.capacity:]

    def snapshot(self, from_sample: int | None = None) -> tuple[np.ndarray, int]:
        """保持している音声の**コピー**と、その先頭の絶対位置を返す.

        Args:
            from_sample: この絶対位置以降だけを返す。``None`` なら全部。
                バッファから落ちた位置を指された場合は、**残っている先頭まで
                切り上げる** (無いものは返しようがない)。

        Returns:
            ``(音声, 先頭の絶対位置)``。音声が無ければ長さ 0 の配列を返す。
        """
        with self._lock:
            start_abs = self._total_consumed - self._audio.size
            if from_sample is None:
                return self._audio.copy(), start_abs
            offset = max(0, from_sample - start_abs)
            if offset >= self._audio.size:
                return np.zeros(0, dtype=np.float32), self._total_consumed
            return self._audio[offset:].copy(), start_abs + offset

    def clear(self) -> None:
        with self._lock:
            self._audio = np.zeros(0, dtype=np.float32)
            self._total_consumed = 0


__all__ = ["DEFAULT_REFINE_CAPACITY_S", "RefineBuffer"]

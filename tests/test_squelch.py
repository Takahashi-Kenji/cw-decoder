"""スキッシュのテスト."""
from __future__ import annotations

import numpy as np
import pytest

from src.infer.squelch import DISABLED_BELOW_DB, Squelch

SR = 8000
BLOCK = 160   # 20 ms


def _block(db: float) -> np.ndarray:
    """指定 dBFS の一定振幅ブロック (RMS = 振幅)."""
    amp = 10.0 ** (db / 20.0)
    return np.full(BLOCK, amp, dtype=np.float32)


class TestDisabled:
    def test_threshold_below_limit_disables(self) -> None:
        sq = Squelch(threshold_db=DISABLED_BELOW_DB, hold_sec=1.0, sample_rate=SR)
        assert sq.enabled is False
        quiet = _block(-90.0)
        assert np.array_equal(sq.process(quiet), quiet)

    def test_default_desktop_value_minus_60_is_enabled(self) -> None:
        """-60 は「実質無効」だが仕組みとしては有効 (無効判定は -100 以下)."""
        assert Squelch(threshold_db=-60.0).enabled is True


class TestGating:
    def _sq(self, thr: float = -40.0, hold: float = 0.1) -> Squelch:
        return Squelch(threshold_db=thr, hold_sec=hold, sample_rate=SR)

    def test_loud_block_passes_through(self) -> None:
        sq = self._sq()
        loud = _block(-20.0)
        assert np.array_equal(sq.process(loud), loud)
        assert sq.is_open is True

    def test_quiet_block_is_silenced_after_hold(self) -> None:
        sq = self._sq(hold=0.0)
        out = sq.process(_block(-60.0))
        assert np.all(out == 0.0)
        assert out.shape == (BLOCK,)

    def test_length_is_preserved(self) -> None:
        """捨てずに置き換える。捨てると時間軸が止まり確定と改行が壊れる."""
        sq = self._sq(hold=0.0)
        for db in (-20.0, -60.0, -60.0, -20.0):
            assert sq.process(_block(db)).shape == (BLOCK,)

    def test_hold_keeps_open_briefly_after_signal_stops(self) -> None:
        """信号が切れても hold の間は開いたまま (符号の頭を削らない)."""
        sq = self._sq(hold=0.1)          # 100 ms = 5 ブロック
        sq.process(_block(-20.0))        # 開く
        for _ in range(4):               # 80 ms 経過。まだ開いている
            out = sq.process(_block(-60.0))
            assert np.any(out != 0.0)
        for _ in range(3):               # hold 切れ
            out = sq.process(_block(-60.0))
        assert np.all(out == 0.0)

    def test_reopens_when_signal_returns(self) -> None:
        sq = self._sq(hold=0.0)
        assert np.all(sq.process(_block(-60.0)) == 0.0)
        loud = _block(-20.0)
        assert np.array_equal(sq.process(loud), loud)

    def test_reset_closes(self) -> None:
        sq = self._sq(hold=1.0)
        sq.process(_block(-20.0))
        sq.reset()
        assert np.all(sq.process(_block(-60.0)) == 0.0)

    def test_empty_block_is_returned_as_is(self) -> None:
        sq = self._sq()
        empty = np.zeros(0, dtype=np.float32)
        assert sq.process(empty).size == 0


class TestAgainstRealNoise:
    """実際のノイズでゲートが閉じ、信号で開くこと."""

    def test_noise_closed_signal_open(self) -> None:
        rng = np.random.default_rng(0)
        sq = Squelch(threshold_db=-25.0, hold_sec=0.0, sample_rate=SR)
        noise = (rng.normal(0, 1, BLOCK) * 0.02).astype(np.float32)   # 約 -34 dBFS
        signal = (np.sin(np.arange(BLOCK) * 0.5) * 0.5).astype(np.float32)  # 約 -9 dBFS
        assert np.all(sq.process(noise) == 0.0)
        assert np.any(sq.process(signal) != 0.0)

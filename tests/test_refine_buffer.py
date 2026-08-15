"""清書用バッファのテスト.

**リアルタイム側とは別に持つ。** デコード用リング (30 秒) を広げると
``refine_closed_turns`` が同期的に長い区間をデコードして音声スレッドを
止めてしまう (300 秒で約 1.3 秒。hop 0.5 秒を大きく超える)。

そこで清書専用に長いバッファを別に持ち、**別スレッドから読むだけ**にする。
交信しながら読む側には一切影響させない。
"""
from __future__ import annotations

import numpy as np

from src.infer.refine_buffer import RefineBuffer


class TestCapacity:
    def test_keeps_recent_audio_up_to_capacity(self) -> None:
        buf = RefineBuffer(capacity_s=1.0, sample_rate=100)
        buf.push(np.arange(150, dtype=np.float32))
        audio, start = buf.snapshot()
        assert len(audio) == 100
        # 古い 50 サンプルは落ちている
        assert audio[0] == 50.0
        assert start == 50

    def test_shorter_than_capacity_is_kept_whole(self) -> None:
        buf = RefineBuffer(capacity_s=1.0, sample_rate=100)
        buf.push(np.arange(40, dtype=np.float32))
        audio, start = buf.snapshot()
        assert len(audio) == 40
        assert start == 0

    def test_snapshot_is_a_copy(self) -> None:
        """**別スレッドが読む。** 参照を渡すと書き込みと競合する."""
        buf = RefineBuffer(capacity_s=1.0, sample_rate=100)
        buf.push(np.ones(10, dtype=np.float32))
        audio, _ = buf.snapshot()
        buf.push(np.zeros(10, dtype=np.float32))
        assert audio.tolist() == [1.0] * 10


class TestSnapshotFrom:
    """自動清書は「まだ清書していない区間」だけを読む."""

    def test_returns_audio_after_the_given_position(self) -> None:
        buf = RefineBuffer(capacity_s=10.0, sample_rate=100)
        buf.push(np.arange(500, dtype=np.float32))
        audio, start = buf.snapshot(from_sample=200)
        assert start == 200
        assert audio[0] == 200.0
        assert len(audio) == 300

    def test_position_before_the_buffer_is_clamped(self) -> None:
        """バッファから落ちた位置を指されたら、残っている先頭から返す."""
        buf = RefineBuffer(capacity_s=1.0, sample_rate=100)
        buf.push(np.arange(300, dtype=np.float32))
        audio, start = buf.snapshot(from_sample=0)
        assert start == 200                     # 落ちずに残っている先頭
        assert len(audio) == 100

    def test_position_at_the_end_returns_empty(self) -> None:
        buf = RefineBuffer(capacity_s=10.0, sample_rate=100)
        buf.push(np.arange(100, dtype=np.float32))
        audio, start = buf.snapshot(from_sample=100)
        assert len(audio) == 0
        assert start == 100


class TestBookkeeping:
    def test_total_consumed_counts_everything(self) -> None:
        """落ちた分も含めた総投入数 (絶対位置の基準になる)."""
        buf = RefineBuffer(capacity_s=1.0, sample_rate=100)
        buf.push(np.zeros(250, dtype=np.float32))
        assert buf.total_consumed == 250

    def test_clear_resets(self) -> None:
        buf = RefineBuffer(capacity_s=1.0, sample_rate=100)
        buf.push(np.zeros(50, dtype=np.float32))
        buf.clear()
        audio, start = buf.snapshot()
        assert len(audio) == 0
        assert start == 0
        assert buf.total_consumed == 0

    def test_multi_dimensional_input_is_flattened(self) -> None:
        buf = RefineBuffer(capacity_s=1.0, sample_rate=100)
        buf.push(np.zeros((10, 1), dtype=np.float32))
        assert buf.total_consumed == 10

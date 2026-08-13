"""音声キャプチャのテスト (バックエンド非依存部分)."""
from __future__ import annotations

import numpy as np

from src.infer.audio import _resample_to_8k, list_input_devices


class TestResample:
    def test_passthrough_at_8k(self) -> None:
        sig = np.random.randn(1000).astype(np.float32)
        out = _resample_to_8k(sig, 8000)
        np.testing.assert_array_equal(sig, out)

    def test_downsample_44100_to_8000(self) -> None:
        sr_in = 44100
        sig = np.random.randn(sr_in).astype(np.float32)
        out = _resample_to_8k(sig, sr_in)
        # 1 秒の信号 → 8000 サンプル前後
        assert abs(len(out) - 8000) < 10

    def test_downsample_48000_to_8000(self) -> None:
        sig = np.random.randn(48000).astype(np.float32)
        out = _resample_to_8k(sig, 48000)
        assert abs(len(out) - 8000) < 10


class TestListDevices:
    def test_returns_list(self) -> None:
        # 環境によっては空リストもありうるが、クラッシュしないこと
        devices = list_input_devices()
        assert isinstance(devices, list)

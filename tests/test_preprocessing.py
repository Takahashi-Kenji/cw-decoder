"""メルスペクトログラム前処理のテスト."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.train.preprocessing import MelConfig, MelExtractor


class TestMelConfig:
    def test_default_values(self) -> None:
        cfg = MelConfig()
        assert cfg.sample_rate == 8000
        assert cfg.n_mels == 64
        assert cfg.win_length == 200
        assert cfg.hop_length == 80


class TestMelExtractor:
    def test_shape_single_waveform(self) -> None:
        ext = MelExtractor()
        wave = torch.randn(8000)
        spec = ext(wave)
        assert spec.shape[0] == 1
        assert spec.shape[1] == 64
        # フレーム数 = 8000 / 80 + 1 = 101
        assert spec.shape[2] == 101

    def test_shape_batched(self) -> None:
        ext = MelExtractor()
        wave = torch.randn(4, 8000)
        spec = ext(wave)
        assert spec.shape == (4, 64, 101)

    def test_output_dtype(self) -> None:
        ext = MelExtractor()
        wave = torch.randn(8000)
        assert ext(wave).dtype == torch.float32

    def test_frame_count_matches_actual(self) -> None:
        ext = MelExtractor()
        wave = torch.randn(4000)
        spec = ext(wave)
        assert ext.frame_count(4000) == spec.shape[-1]

    def test_normalization_zero_mean_unit_std(self) -> None:
        ext = MelExtractor(MelConfig(normalize=True))
        wave = torch.randn(2, 8000)
        spec = ext(wave)
        # サンプル単位の z-score
        for b in range(2):
            assert spec[b].mean().abs().item() < 1e-4
            assert abs(spec[b].std().item() - 1.0) < 1e-2

    def test_no_normalization_preserves_db_scale(self) -> None:
        ext = MelExtractor(MelConfig(normalize=False))
        wave = torch.randn(8000)
        spec = ext(wave)
        # 非正規化なら値は dB スケール ([-80, 0] 程度)
        assert spec.min().item() < 0.0

    def test_runs_on_cuda_if_available(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        ext = MelExtractor().cuda()
        wave = torch.randn(2, 8000, device="cuda")
        spec = ext(wave)
        assert spec.is_cuda
        assert spec.shape == (2, 64, 101)

    def test_signal_peak_appears_in_mel_band(self) -> None:
        """600 Hz トーンに対し、対応する mel ビンが他より大きい."""
        ext = MelExtractor(MelConfig(normalize=False))
        sr = 8000
        t = torch.arange(sr).float() / sr
        wave = torch.sin(2 * np.pi * 600.0 * t)
        spec = ext(wave).squeeze(0)  # (n_mels, T)
        mean_per_bin = spec.mean(dim=-1)  # (n_mels,)
        peak_bin = int(mean_per_bin.argmax().item())
        # 600 Hz はフィルタバンクの中央付近に来るはず (n_mels=64, f_max=4000)
        assert 5 < peak_bin < 35

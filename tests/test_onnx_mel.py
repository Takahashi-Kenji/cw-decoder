"""ConvMelExtractor が MelExtractor と数値的に一致することの検証."""
from __future__ import annotations

import numpy as np
import torch

from src.train.onnx_mel import ConvMelExtractor
from src.train.preprocessing import MelExtractor


def _reference_and_conv(wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref = MelExtractor().eval()
    conv = ConvMelExtractor().eval()
    with torch.no_grad():
        return ref(wave), conv(wave)


def test_matches_reference_on_noise() -> None:
    """ホワイトノイズで最大絶対誤差 1e-4 未満."""
    rng = np.random.default_rng(0)
    wave = torch.from_numpy(rng.standard_normal(8000 * 3).astype(np.float32) * 0.1)
    ref, got = _reference_and_conv(wave)
    assert got.shape == ref.shape
    assert torch.max(torch.abs(got - ref)).item() < 1e-4


def test_matches_reference_on_tone_bursts() -> None:
    """CW に近い 600 Hz のトーンバーストで一致すること."""
    sr = 8000
    t = np.arange(sr * 2) / sr
    envelope = ((np.sin(2 * np.pi * 3.0 * t) > 0).astype(np.float32))
    wave = torch.from_numpy((np.sin(2 * np.pi * 600.0 * t) * envelope).astype(np.float32))
    ref, got = _reference_and_conv(wave)
    assert torch.max(torch.abs(got - ref)).item() < 1e-4


def test_frame_count_matches() -> None:
    """フレーム数が MelExtractor.frame_count と一致すること."""
    conv = ConvMelExtractor().eval()
    for n in (800, 4001, 8000, 24000):
        wave = torch.zeros(1, n)
        with torch.no_grad():
            out = conv(wave)
        assert out.shape[2] == MelExtractor().frame_count(n)


def test_accepts_1d_and_2d_input() -> None:
    """(T,) と (1, T) のどちらでも同じ結果を返すこと."""
    rng = np.random.default_rng(1)
    wave1d = torch.from_numpy(rng.standard_normal(8000).astype(np.float32) * 0.1)
    conv = ConvMelExtractor().eval()
    with torch.no_grad():
        a = conv(wave1d)
        b = conv(wave1d.unsqueeze(0))
    assert torch.equal(a, b)

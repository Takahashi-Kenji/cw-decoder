"""ONNX エクスポート可能なメルスペクトログラム変換.

``torchaudio.transforms.MelSpectrogram`` は内部で ``torch.stft`` を呼ぶが、
ONNX の STFT op はランタイム側の対応が不安定なため、ここでは STFT を
**Conv1d (cos/sin カーネル)** として展開する。Conv1d と行列積と初等演算だけで
構成されるため、どの ONNX ランタイムでも確実に動く。

``MelExtractor`` と数値的に一致することを ``tests/test_onnx_mel.py`` で検証する。
一致が壊れると推論精度が静かに劣化するため、このテストは必ず維持すること。
"""
from __future__ import annotations

import math

import torch
import torchaudio
from torch import Tensor, nn

from src.train.preprocessing import MelConfig


class ConvMelExtractor(nn.Module):
    """波形 → 対数メルスペクトログラム (ONNX エクスポート可能).

    入力: ``(B, T_wave)`` または ``(T_wave,)`` (float32)
    出力: ``(B, n_mels, T_frames)`` (float32)
    """

    def __init__(self, config: MelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MelConfig()
        c = self.config

        self.register_buffer("dft_kernel", self._build_dft_kernel(c), persistent=False)
        self.register_buffer("mel_fb", self._build_mel_fb(c), persistent=False)

    @staticmethod
    def _build_dft_kernel(c: MelConfig) -> Tensor:
        """DFT 基底 × Hann 窓 の Conv1d カーネル ``(2 * n_bins, 1, n_fft)``.

        前半 ``n_bins`` チャネルが実部、後半 ``n_bins`` チャネルが虚部.
        """
        n_fft = c.n_fft
        # periodic=True が torchaudio (torch.stft) の既定
        win = torch.hann_window(c.win_length, periodic=True, dtype=torch.float64)
        # win_length < n_fft のとき torch.stft は窓を中央寄せでゼロ詰めする
        pad_left = (n_fft - c.win_length) // 2
        window = torch.zeros(n_fft, dtype=torch.float64)
        window[pad_left:pad_left + c.win_length] = win

        n_bins = n_fft // 2 + 1
        k = torch.arange(n_bins, dtype=torch.float64).unsqueeze(1)   # (n_bins, 1)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(0)    # (1, n_fft)
        angle = 2.0 * math.pi * k * n / n_fft
        real = torch.cos(angle) * window
        imag = -torch.sin(angle) * window
        kernel = torch.cat([real, imag], dim=0).unsqueeze(1)         # (2*n_bins, 1, n_fft)
        return kernel.to(torch.float32)

    @staticmethod
    def _build_mel_fb(c: MelConfig) -> Tensor:
        """メルフィルタバンク ``(n_mels, n_bins)``.

        自前で式を書かず torchaudio の関数から取る (ズレの余地を作らない).
        """
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=c.n_fft // 2 + 1,
            f_min=c.f_min,
            f_max=c.f_max,
            n_mels=c.n_mels,
            sample_rate=c.sample_rate,
            norm=None,
            mel_scale="htk",
        )                                   # (n_bins, n_mels)
        return fb.transpose(0, 1).contiguous()   # (n_mels, n_bins)

    def forward(self, waveform: Tensor) -> Tensor:
        c = self.config
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        x = waveform.unsqueeze(1)                       # (B, 1, T_wave)

        # center=True 相当の reflect padding
        pad = c.n_fft // 2
        x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")

        spec = torch.nn.functional.conv1d(x, self.dft_kernel, stride=c.hop_length)
        n_bins = c.n_fft // 2 + 1
        real = spec[:, :n_bins, :]
        imag = spec[:, n_bins:, :]
        power = real * real + imag * imag               # (B, n_bins, T_frames)

        mel = torch.matmul(self.mel_fb, power)          # (B, n_mels, T_frames)

        # AmplitudeToDB(stype="power", top_db=80)
        spec_db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        spec_db = torch.maximum(spec_db, spec_db.max() - c.top_db)

        if c.normalize:
            mean = spec_db.mean(dim=(-2, -1), keepdim=True)
            std = spec_db.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            spec_db = (spec_db - mean) / std
        return spec_db


__all__ = ["ConvMelExtractor"]

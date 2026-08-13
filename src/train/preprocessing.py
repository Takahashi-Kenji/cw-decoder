"""音声前処理 (メルスペクトログラム).

要件 §3.3.1:

- サンプリングレート: 8 kHz
- ``n_mels = 64``, win 25 ms, hop 10 ms
- torchaudio で GPU 上で計算
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from torch import Tensor, nn


@dataclass(frozen=True)
class MelConfig:
    """メルスペクトログラム設定."""

    sample_rate: int = 8000
    n_mels: int = 64
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 256          # 2^n で win_length (=200) を覆う最小値
    f_min: float = 50.0
    f_max: float = 4000.0     # Nyquist 直前
    top_db: float = 80.0
    normalize: bool = True

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.win_ms / 1000.0)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000.0)


class MelExtractor(nn.Module):
    """波形 → 対数メルスペクトログラム.

    入力: ``(B, T_wave)`` または ``(T_wave,)`` (float32)
    出力: ``(B, n_mels, T_frames)`` (float32)

    GPU 上で計算するため、``model.to(device)`` で device 遷移する.
    """

    def __init__(self, config: MelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MelConfig()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=self.config.top_db
        )

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.mel(waveform)               # (B, n_mels, T)
        spec_db = self.to_db(spec)              # (B, n_mels, T)
        if self.config.normalize:
            # サンプル内 z-score 正規化 (バッチ次元は触らない)
            mean = spec_db.mean(dim=(-2, -1), keepdim=True)
            std = spec_db.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            spec_db = (spec_db - mean) / std
        return spec_db

    def frame_count(self, wave_length: int) -> int:
        """波形長から出力フレーム数を計算 (center=True の場合)."""
        return wave_length // self.config.hop_length + 1


__all__ = ["MelConfig", "MelExtractor"]

"""CW デコーダモデル (CNN + BiLSTM + Linear + CTC).

要件 §3.3.2:

- CNN 2〜3 層 (周波数・時間方向の局所特徴)
- BiLSTM 2 層 (hidden 256)
- Linear → CTC 出力
- パラメータ数 数百万 (数 MB 〜 十数 MB)
- 入力時間次元は保存 (CTC の時間アラインメント用)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """モデル構成."""

    n_mels: int = 64
    vocab_size: int = 72              # Phase 1 統合トークン (blank 含む)
    cnn_channels: tuple[int, int, int] = (32, 32, 64)
    cnn_freq_stride: tuple[int, int, int] = (2, 2, 1)   # 周波数方向のみ縮約
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    classifier_dropout: float = 0.1

    @property
    def cnn_freq_factor(self) -> int:
        f = 1
        for s in self.cnn_freq_stride:
            f *= s
        return f

    @property
    def cnn_output_freq(self) -> int:
        # 切り上げ (Conv2d stride による floor 動作を補正)
        result = self.n_mels
        for s in self.cnn_freq_stride:
            result = (result + s - 1) // s
        return result

    @property
    def lstm_input_size(self) -> int:
        return self.cnn_channels[-1] * self.cnn_output_freq


class CWModel(nn.Module):
    """CW モールス信号デコーダ (CTC モデル).

    入力: ``(B, n_mels, T)`` (メルスペクトログラム)
    出力: ``(B, T, vocab_size)`` (logits)

    時間次元 T は入力から出力までほぼ保持される (CNN は周波数方向のみ stride>1).
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        # CNN ブロック (Conv2d → BatchNorm → ReLU)
        in_ch = 1
        cnn_layers: list[nn.Module] = []
        for ch, stride_f in zip(c.cnn_channels, c.cnn_freq_stride, strict=True):
            cnn_layers.extend([
                nn.Conv2d(
                    in_ch, ch,
                    kernel_size=(3, 3),
                    stride=(stride_f, 1),
                    padding=(1, 1),
                ),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = ch
        self.cnn = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(
            input_size=c.lstm_input_size,
            hidden_size=c.lstm_hidden,
            num_layers=c.lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=c.lstm_dropout if c.lstm_layers > 1 else 0.0,
        )

        self.classifier_dropout = nn.Dropout(c.classifier_dropout)
        self.classifier = nn.Linear(c.lstm_hidden * 2, c.vocab_size)

    def forward(self, mel: Tensor) -> Tensor:
        """``mel`` から logits を計算.

        Args:
            mel: ``(B, n_mels, T)`` または ``(B, 1, n_mels, T)``.

        Returns:
            logits ``(B, T, vocab_size)``.
        """
        if mel.dim() == 3:
            x = mel.unsqueeze(1)
        else:
            x = mel
        x = self.cnn(x)                       # (B, C, F', T)
        b, ch, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, ch * f)  # (B, T, C*F')
        x, _ = self.lstm(x)                   # (B, T, 2H)
        x = self.classifier_dropout(x)
        x = self.classifier(x)                # (B, T, V)
        return x

    def output_length(self, input_length: int) -> int:
        """入力 (mel time T) に対する出力 time. 現状は同じ."""
        return input_length

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["CWModel", "ModelConfig"]

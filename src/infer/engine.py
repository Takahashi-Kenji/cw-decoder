"""推論エンジン (モデル読込 + 単一チャンク推論).

CTC greedy decode を **フレーム位置付き** で行い、ストリーミングマージで
オーバーラップ重複を取り除けるようにする.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.infer.ctc import FrameToken, ctc_greedy_decode_frames
from src.tokens.morse_tokens import BLANK_TOKEN_ID, VOCAB_SIZE
from src.train.checkpoint import build_model_from_checkpoint
from src.train.model import CWModel, ModelConfig
from src.train.preprocessing import MelConfig, MelExtractor


def ctc_greedy_decode_with_frames(
    log_probs: torch.Tensor, blank_id: int = BLANK_TOKEN_ID
) -> list[list[FrameToken]]:
    """``(B, T, V)`` log-softmax から ``FrameToken`` 列を返す (torch 入力版).

    中身は ``src.infer.ctc`` の numpy 実装に委ねる。**二重に実装しない。**
    配布物から torch を外すために numpy 版が要るが、両方を持つと片方だけ
    直したときに静かにずれる。ここは torch → numpy の変換だけを行う。
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be 3D, got {log_probs.shape}")
    return ctc_greedy_decode_frames(
        log_probs.detach().cpu().numpy(), blank_id=blank_id
    )


class InferenceEngine:
    """モデル + メルスペクトログラム抽出器のラッパ.

    単一チャンクの音声波形を受け取り、``FrameToken`` 列を返す.
    """

    def __init__(
        self,
        model: CWModel,
        mel_extractor: MelExtractor,
        device: torch.device,
        blank_id: int = BLANK_TOKEN_ID,
    ) -> None:
        self.model = model
        self.mel_extractor = mel_extractor
        self.device = device
        self.blank_id = blank_id
        self.model.train(False)
        self.mel_extractor.train(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: torch.device | str = "cpu",
    ) -> InferenceEngine:
        """学習済みチェックポイントから推論エンジンを構築."""
        dev = torch.device(device)
        model = build_model_from_checkpoint(checkpoint_path, map_location=dev)
        model.to(dev)
        mel_extractor = MelExtractor().to(dev)
        return cls(model, mel_extractor, dev)

    @classmethod
    def untrained(
        cls,
        device: torch.device | str = "cpu",
        model_config: ModelConfig | None = None,
        mel_config: MelConfig | None = None,
    ) -> InferenceEngine:
        """未学習モデルのエンジン (UI 動作確認用)."""
        dev = torch.device(device)
        model = CWModel(model_config or ModelConfig(vocab_size=VOCAB_SIZE)).to(dev)
        mel_extractor = MelExtractor(mel_config).to(dev)
        return cls(model, mel_extractor, dev)

    @torch.no_grad()
    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]:
        """1 つの音声チャンクをデコード.

        Args:
            waveform: ``(T_wave,)`` の float32 配列 (8 kHz サンプル).

        Returns:
            ``FrameToken`` の列 (CTC greedy decode 結果).
        """
        if waveform.size == 0:
            return []
        wave = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(self.device)
        mel = self.mel_extractor(wave)
        logits = self.model(mel)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return ctc_greedy_decode_with_frames(log_probs, blank_id=self.blank_id)[0]

    @property
    def frame_hop_samples(self) -> int:
        """1 フレームあたりのサンプル数 (= mel hop_length)."""
        return self.mel_extractor.config.hop_length


__all__ = [
    "FrameToken",
    "InferenceEngine",
    "ctc_greedy_decode_with_frames",
]

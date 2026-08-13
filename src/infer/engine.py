"""推論エンジン (モデル読込 + 単一チャンク推論).

CTC greedy decode を **フレーム位置付き** で行い、ストリーミングマージで
オーバーラップ重複を取り除けるようにする.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.tokens.morse_tokens import BLANK_TOKEN_ID, VOCAB_SIZE
from src.train.checkpoint import build_model_from_checkpoint
from src.train.model import CWModel, ModelConfig
from src.train.preprocessing import MelConfig, MelExtractor


@dataclass(frozen=True)
class FrameToken:
    """1 つのデコード済みトークン (フレーム位置付き)."""

    token_id: int
    confidence: float
    frame_start: int    # この token を出力した最初のフレーム
    frame_end: int      # 最後のフレーム (inclusive)


def ctc_greedy_decode_with_frames(
    log_probs: torch.Tensor, blank_id: int = BLANK_TOKEN_ID
) -> list[list[FrameToken]]:
    """``(B, T, V)`` log-softmax から ``FrameToken`` 列を返す.

    各 ``FrameToken`` は ``[frame_start, frame_end]`` の連続フレームから
    出力された 1 つのトークン. blank と重複は collapse 済み.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be 3D, got {log_probs.shape}")
    b, t, _ = log_probs.shape
    probs = log_probs.exp()
    max_prob, argmax = probs.max(dim=-1)
    am = argmax.detach().cpu().numpy()
    mp = max_prob.detach().cpu().numpy()

    results: list[list[FrameToken]] = []
    for i in range(b):
        out: list[FrameToken] = []
        prev = -1
        run_start = 0
        run_max = 0.0
        for j in range(t):
            tok = int(am[i, j])
            conf = float(mp[i, j])
            if tok == prev:
                if tok != blank_id:
                    run_max = max(run_max, conf)
                continue
            # 切り替わり: 前のランを確定
            if prev != -1 and prev != blank_id:
                out.append(FrameToken(
                    token_id=prev, confidence=run_max,
                    frame_start=run_start, frame_end=j - 1,
                ))
            prev = tok
            run_start = j
            run_max = conf
        # 末尾
        if prev != -1 and prev != blank_id:
            out.append(FrameToken(
                token_id=prev, confidence=run_max,
                frame_start=run_start, frame_end=t - 1,
            ))
        results.append(out)
    return results


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

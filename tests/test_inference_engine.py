"""推論エンジンとストリーミングデコーダのテスト."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.infer.engine import (
    FrameToken,
    InferenceEngine,
    ctc_greedy_decode_with_frames,
)
from src.synth.keying import KeyingParams, codes_to_waveform
from src.synth.synthesizer import SynthConfig, synthesize_from_text
from src.tokens.morse_tokens import TOKEN_TO_ID
from src.train.checkpoint import save_checkpoint
from src.train.model import CWModel, ModelConfig


def _make_log_probs(seq_per_batch: list[list[int]], vocab: int = 5) -> torch.Tensor:
    b = len(seq_per_batch)
    t = max(len(s) for s in seq_per_batch)
    logits = torch.full((b, t, vocab), -10.0)
    for i, seq in enumerate(seq_per_batch):
        for j, tok in enumerate(seq):
            logits[i, j, tok] = 0.0
        for j in range(len(seq), t):
            logits[i, j, 0] = 0.0
    return torch.log_softmax(logits, dim=-1)


class TestCTCDecodeWithFrames:
    def test_frame_ranges_correct(self) -> None:
        # [1, 1, 0, 2] → token 1 at frames [0, 1], token 2 at frame [3, 3]
        lp = _make_log_probs([[1, 1, 0, 2]])
        result = ctc_greedy_decode_with_frames(lp, blank_id=0)[0]
        assert len(result) == 2
        assert result[0].token_id == 1
        assert result[0].frame_start == 0
        assert result[0].frame_end == 1
        assert result[1].token_id == 2
        assert result[1].frame_start == 3
        assert result[1].frame_end == 3

    def test_blank_only_returns_empty(self) -> None:
        lp = _make_log_probs([[0, 0, 0]])
        result = ctc_greedy_decode_with_frames(lp, blank_id=0)[0]
        assert result == []


class TestInferenceEngine:
    def test_untrained_engine_runs(self) -> None:
        engine = InferenceEngine.untrained(device="cpu")
        wave = np.random.randn(8000).astype(np.float32)
        result = engine.decode_chunk(wave)
        # 未学習でもクラッシュせず FrameToken のリストを返す
        assert isinstance(result, list)

    def test_empty_waveform(self) -> None:
        engine = InferenceEngine.untrained(device="cpu")
        assert engine.decode_chunk(np.zeros(0, dtype=np.float32)) == []

    def test_runs_on_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        engine = InferenceEngine.untrained(device="cuda")
        wave = np.random.randn(8000).astype(np.float32)
        result = engine.decode_chunk(wave)
        assert isinstance(result, list)

    def test_from_checkpoint(self, tmp_path) -> None:
        model = CWModel(ModelConfig(vocab_size=72))
        save_checkpoint(tmp_path / "ckpt.pt", model, None, None, step=0, epoch=0)
        engine = InferenceEngine.from_checkpoint(tmp_path / "ckpt.pt", device="cpu")
        wave = np.random.randn(4000).astype(np.float32)
        assert isinstance(engine.decode_chunk(wave), list)

    def test_frame_hop_samples(self) -> None:
        engine = InferenceEngine.untrained(device="cpu")
        assert engine.frame_hop_samples == 80  # 10 ms at 8 kHz


class _MockEngine:
    """予めセットしたフレームトークンを返すモックエンジン."""

    def __init__(self, tokens_per_chunk: list[list[FrameToken]], hop_samples: int = 80):
        self._calls = 0
        self._tokens = tokens_per_chunk
        self.frame_hop_samples = hop_samples

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]:
        toks = self._tokens[self._calls]
        self._calls += 1
        return toks


class TestSlidingWindowDecoder:
    """ライブ連続モードのスモーク (詳細は tests/test_sliding_window.py)."""

    def test_push_and_redecode_runs(self) -> None:
        from src.infer.sliding_window import SlidingWindowDecoder
        eng = InferenceEngine.untrained("cpu")
        d = SlidingWindowDecoder(eng, window_s=5.0, hop_s=1.0, commit_lag_s=1.0)
        d.push(np.zeros(8000, dtype=np.float32))
        view = d.redecode()
        assert hasattr(view, "committed")

    def test_window_truncates_to_window_samples(self) -> None:
        from src.infer.sliding_window import SlidingWindowDecoder
        eng = InferenceEngine.untrained("cpu")
        d = SlidingWindowDecoder(eng, window_s=2.0, sample_rate=8000)
        d.push(np.zeros(8000 * 5, dtype=np.float32))   # 5s 投入
        assert d._ring.size == 16000                    # 窓 2s = 16000 に切詰め

    def test_reset_runs(self) -> None:
        from src.infer.sliding_window import SlidingWindowDecoder
        eng = InferenceEngine.untrained("cpu")
        d = SlidingWindowDecoder(eng, window_s=5.0)
        d.push(np.zeros(8000, dtype=np.float32))
        d.reset()
        assert d._ring.size == 0


class TestEndToEnd:
    """合成音声 → 推論 → スライディングウィンドウ のエンドツーエンド (未学習なので結果は問わず)."""

    def test_pipeline_runs_without_error(self) -> None:
        from src.infer.sliding_window import DecodeView, SlidingWindowDecoder
        engine = InferenceEngine.untrained(device="cpu")
        decoder = SlidingWindowDecoder(
            engine, window_s=5.0, hop_s=1.0, commit_lag_s=1.0, sample_rate=8000
        )
        # 合成: 欧文 CW
        rng = np.random.default_rng(0)
        config = SynthConfig(mode="european", snr_db=20.0,
                             keying=KeyingParams(wpm=20.0, rise_fall_ms=5.0))
        result = synthesize_from_text("HELLO WORLD", config, rng)
        # ストリームに小ブロックで投入
        block_size = 800  # 0.1s
        for start in range(0, len(result.samples), block_size):
            block = result.samples[start : start + block_size]
            decoder.push(block)
        view: DecodeView = decoder.finalize()
        # 未学習モデルでもクラッシュしないこと
        assert isinstance(view.committed, list)

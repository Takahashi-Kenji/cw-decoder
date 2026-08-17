"""numpy 版 CTC greedy デコードが torch 版と完全一致することを固定する.

**なぜ固定するか**: 配布物から torch を外すために CTC を numpy で書き直した。
書き直しは「同じ入力に同じ出力」でなければ意味がない。ここが崩れると、
デスクトップ版とブラウザ版・学習側の結果が静かにずれる。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.infer.ctc import FrameToken, ctc_greedy_decode_frames
from src.infer.engine import ctc_greedy_decode_with_frames
from src.tokens.morse_tokens import BLANK_TOKEN_ID, VOCAB_SIZE


def _random_log_probs(rng: np.random.Generator, batch: int, frames: int) -> np.ndarray:
    """log_softmax 済みに相当する ``(B, T, V)`` を作る."""
    logits = rng.normal(size=(batch, frames, VOCAB_SIZE)).astype(np.float32)
    m = logits.max(axis=-1, keepdims=True)
    return (logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).astype(np.float32)


class TestMatchesTorchVersion:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_same_tokens_and_frames(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        lp = _random_log_probs(rng, batch=2, frames=97)

        got = ctc_greedy_decode_frames(lp)
        expected = ctc_greedy_decode_with_frames(torch.from_numpy(lp))

        assert len(got) == len(expected)
        for g_seq, e_seq in zip(got, expected, strict=True):
            assert [t.token_id for t in g_seq] == [t.token_id for t in e_seq]
            assert [t.frame_start for t in g_seq] == [t.frame_start for t in e_seq]
            assert [t.frame_end for t in g_seq] == [t.frame_end for t in e_seq]
            for g, e in zip(g_seq, e_seq, strict=True):
                assert g.confidence == pytest.approx(e.confidence, abs=1e-6)

    def test_blank_only_gives_nothing(self) -> None:
        lp = np.full((1, 10, VOCAB_SIZE), -20.0, dtype=np.float32)
        lp[:, :, BLANK_TOKEN_ID] = 0.0
        assert ctc_greedy_decode_frames(lp) == [[]]

    def test_repeats_collapse_into_one_token(self) -> None:
        """同じトークンが続いたら 1 つにまとめ、区間はその全体になること."""
        lp = np.full((1, 6, VOCAB_SIZE), -20.0, dtype=np.float32)
        lp[:, :, BLANK_TOKEN_ID] = 0.0
        lp[0, 2:5, BLANK_TOKEN_ID] = -20.0
        lp[0, 2:5, 7] = 0.0
        (seq,) = ctc_greedy_decode_frames(lp)
        assert [t.token_id for t in seq] == [7]
        assert (seq[0].frame_start, seq[0].frame_end) == (2, 4)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            ctc_greedy_decode_frames(np.zeros((4, VOCAB_SIZE), dtype=np.float32))


class TestFrameTokenIsShared:
    def test_engine_reexports_the_same_class(self) -> None:
        """``engine`` 側の ``FrameToken`` が別クラスになっていないこと.

        別クラスになると ``isinstance`` が通らず、層をまたいだ受け渡しで
        静かに壊れる (この種の食い違いは過去に 3 度踏んでいる)。
        """
        from src.infer.engine import FrameToken as EngineFrameToken

        assert EngineFrameToken is FrameToken

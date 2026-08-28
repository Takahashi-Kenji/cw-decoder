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
from src.tokens.morse_tokens import VOCAB_SIZE, WORD_BREAK_TOKEN_ID
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


class TestWordBreakBias:
    """語間スペースの出しやすさを argmax の前に調整する.

    held-out の実測 (2026-08-25): モデルは**欧文ではぴったり** (50/50) だが
    **和文では 1.8 倍に膨らむ** (52/29)。和文の挿入誤り 34 個のうち 27 個が語間で、
    欧文と和文の TER 差 (13.06% 対 27.15%) はほぼこれで説明できる。

    掃引の結果、欧文 -1.0 / 和文 -5.0 で TER も CER も同時に改善した。
    """

    @staticmethod
    def _log_probs(wb_logit: float, other_logit: float = 0.0) -> np.ndarray:
        """1 フレームだけの (1, 1, V)。WORD_BREAK と別トークンを競わせる."""
        v = VOCAB_SIZE
        x = np.full((1, 1, v), -20.0, dtype=np.float32)
        x[0, 0, WORD_BREAK_TOKEN_ID] = wb_logit
        x[0, 0, 1] = other_logit
        return x

    def test_バイアス_0_は従来と同じ(self) -> None:
        x = self._log_probs(wb_logit=-0.1)
        assert (ctc_greedy_decode_frames(x)[0][0].token_id
                == ctc_greedy_decode_frames(x, word_break_bias=0.0)[0][0].token_id)

    def test_負のバイアスで語間が引っ込む(self) -> None:
        x = self._log_probs(wb_logit=-0.1, other_logit=-0.5)   # 素では語間が勝つ
        assert ctc_greedy_decode_frames(x)[0][0].token_id == WORD_BREAK_TOKEN_ID
        assert ctc_greedy_decode_frames(x, word_break_bias=-1.0)[0][0].token_id == 1

    def test_正のバイアスで語間が出る(self) -> None:
        """ブラウザ版は元々「語間を増やす」ために正の値で使っていた."""
        x = self._log_probs(wb_logit=-1.0, other_logit=-0.2)   # 素では別トークンが勝つ
        assert ctc_greedy_decode_frames(x)[0][0].token_id == 1
        assert (ctc_greedy_decode_frames(x, word_break_bias=+2.0)[0][0].token_id
                == WORD_BREAK_TOKEN_ID)

    def test_他のトークンの確信度は変わらない(self) -> None:
        x = self._log_probs(wb_logit=-9.0, other_logit=-0.2)
        a = ctc_greedy_decode_frames(x)[0][0]
        b = ctc_greedy_decode_frames(x, word_break_bias=-5.0)[0][0]
        assert a.token_id == b.token_id == 1
        assert a.confidence == pytest.approx(b.confidence)

    def test_語間の確信度はバイアス後の値(self) -> None:
        """ブラウザ版 (web/src/decode/ctc.ts) と同じ規約に揃える."""
        x = self._log_probs(wb_logit=0.0, other_logit=-20.0)
        got = ctc_greedy_decode_frames(x, word_break_bias=-1.0)[0][0]
        assert got.token_id == WORD_BREAK_TOKEN_ID
        assert got.confidence == pytest.approx(float(np.exp(-1.0)), rel=1e-5)

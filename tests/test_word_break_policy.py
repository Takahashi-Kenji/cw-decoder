"""WORD_BREAK 抑制ポリシーのテスト."""
from __future__ import annotations

import pytest
import torch

from src.infer.word_break_policy import (
    WordBreakPolicy,
    apply_logit_bias,
    filter_word_breaks,
)
from src.train.decode import CTCDecodeResult
from src.tokens.morse_tokens import VOCAB_SIZE, WORD_BREAK_TOKEN_ID


def _log_probs(t: int = 4, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    """``(1, t, vocab)`` の決定的な log-softmax を作る."""
    g = torch.Generator().manual_seed(20260731)
    return torch.log_softmax(torch.randn(1, t, vocab, generator=g), dim=-1)


class TestWordBreakPolicy:
    def test_default_is_identity(self) -> None:
        assert WordBreakPolicy().is_identity is True

    def test_nonzero_bias_is_not_identity(self) -> None:
        assert WordBreakPolicy(logit_bias=-1.0).is_identity is False

    def test_nonzero_threshold_is_not_identity(self) -> None:
        assert WordBreakPolicy(conf_threshold=0.5).is_identity is False

    def test_rejects_nan_bias(self) -> None:
        with pytest.raises(ValueError, match="logit_bias"):
            WordBreakPolicy(logit_bias=float("nan"))

    def test_rejects_inf_bias(self) -> None:
        with pytest.raises(ValueError, match="logit_bias"):
            WordBreakPolicy(logit_bias=float("-inf"))

    def test_rejects_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="conf_threshold"):
            WordBreakPolicy(conf_threshold=1.5)

    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="conf_threshold"):
            WordBreakPolicy(conf_threshold=-0.1)


class TestApplyLogitBias:
    def test_zero_bias_returns_input_unchanged(self) -> None:
        lp = _log_probs()
        assert torch.equal(apply_logit_bias(lp, 0.0), lp)

    def test_negative_bias_lowers_word_break_probability(self) -> None:
        lp = _log_probs()
        out = apply_logit_bias(lp, -3.0)
        assert (out[..., WORD_BREAK_TOKEN_ID] < lp[..., WORD_BREAK_TOKEN_ID]).all()

    def test_preserves_ratio_between_other_tokens(self) -> None:
        """WORD_BREAK 以外のトークン同士の確率比は歪まない."""
        lp = _log_probs()
        out = apply_logit_bias(lp, -3.0)
        # log 空間での差 = 確率比の log。id 1 と 2 は WORD_BREAK ではない
        before = lp[..., 1] - lp[..., 2]
        after = out[..., 1] - out[..., 2]
        assert torch.allclose(before, after, atol=1e-6)

    def test_output_is_normalized(self) -> None:
        out = apply_logit_bias(_log_probs(), -2.0)
        assert torch.allclose(out.exp().sum(dim=-1), torch.ones(1, 4), atol=1e-6)

    def test_positive_bias_is_allowed(self) -> None:
        """掃引の対称性確認に使うため β>0 も通す."""
        lp = _log_probs()
        out = apply_logit_bias(lp, 2.0)
        assert (out[..., WORD_BREAK_TOKEN_ID] > lp[..., WORD_BREAK_TOKEN_ID]).all()

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            apply_logit_bias(torch.zeros(4, VOCAB_SIZE), -1.0)

    def test_rejects_vocab_without_word_break(self) -> None:
        """WORD_BREAK 追加前の古い ckpt を黙って処理しない."""
        with pytest.raises(ValueError, match="WORD_BREAK"):
            apply_logit_bias(torch.log_softmax(torch.zeros(1, 4, 10), dim=-1), -1.0)


class TestFilterWordBreaks:
    @staticmethod
    def _result() -> CTCDecodeResult:
        """WORD_BREAK 2 個 (低確信度 0.2 / 高確信度 0.9) と通常トークン 2 個."""
        return CTCDecodeResult(
            token_ids=[1, WORD_BREAK_TOKEN_ID, 2, WORD_BREAK_TOKEN_ID],
            confidences=[0.15, 0.2, 0.3, 0.9],
        )

    def test_zero_threshold_returns_input(self) -> None:
        r = self._result()
        assert filter_word_breaks(r, 0.0) is r

    def test_removes_low_confidence_word_break(self) -> None:
        out = filter_word_breaks(self._result(), 0.5)
        assert out.token_ids == [1, 2, WORD_BREAK_TOKEN_ID]

    def test_keeps_high_confidence_word_break(self) -> None:
        out = filter_word_breaks(self._result(), 0.5)
        assert out.token_ids[-1] == WORD_BREAK_TOKEN_ID

    def test_keeps_low_confidence_non_word_break(self) -> None:
        """確信度が低くても WORD_BREAK 以外は残す."""
        out = filter_word_breaks(self._result(), 0.5)
        assert 1 in out.token_ids and 2 in out.token_ids

    def test_confidences_stay_aligned(self) -> None:
        out = filter_word_breaks(self._result(), 0.5)
        assert len(out.token_ids) == len(out.confidences)
        assert out.confidences == [0.15, 0.3, 0.9]

    def test_removes_all_when_threshold_is_one(self) -> None:
        out = filter_word_breaks(self._result(), 1.0)
        assert WORD_BREAK_TOKEN_ID not in out.token_ids

    def test_does_not_mutate_input(self) -> None:
        r = self._result()
        filter_word_breaks(r, 0.5)
        assert len(r.token_ids) == 4

    def test_rejects_out_of_range_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            filter_word_breaks(self._result(), 1.5)

    def test_rejects_length_mismatch(self) -> None:
        bad = CTCDecodeResult(token_ids=[1, 2], confidences=[0.5])
        with pytest.raises(ValueError):
            filter_word_breaks(bad, 0.5)

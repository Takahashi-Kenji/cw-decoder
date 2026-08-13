"""CTC greedy decode のテスト."""
from __future__ import annotations

import torch

from src.train.decode import CTCDecodeResult, ctc_greedy_decode


def _make_log_probs(seq_per_batch: list[list[int]], vocab: int = 5) -> torch.Tensor:
    """各バッチに対し、各時刻で one-hot に近い分布を持つ log_probs を生成."""
    b = len(seq_per_batch)
    t = max(len(s) for s in seq_per_batch)
    logits = torch.full((b, t, vocab), -10.0)
    for i, seq in enumerate(seq_per_batch):
        for j, tok in enumerate(seq):
            logits[i, j, tok] = 0.0
        for j in range(len(seq), t):
            logits[i, j, 0] = 0.0  # 末尾は blank
    return torch.log_softmax(logits, dim=-1)


class TestCTCGreedyDecode:
    def test_collapse_duplicates(self) -> None:
        lp = _make_log_probs([[1, 1, 1, 2, 2, 0, 3, 3]])
        results = ctc_greedy_decode(lp, blank_id=0)
        assert results[0].token_ids == [1, 2, 3]

    def test_remove_blanks(self) -> None:
        lp = _make_log_probs([[0, 1, 0, 2, 0]])
        results = ctc_greedy_decode(lp, blank_id=0)
        assert results[0].token_ids == [1, 2]

    def test_blank_separates_same_token(self) -> None:
        # CTC convention: blank 区切りなら同じ token が複数回出る
        lp = _make_log_probs([[1, 0, 1]])
        results = ctc_greedy_decode(lp, blank_id=0)
        assert results[0].token_ids == [1, 1]

    def test_batched(self) -> None:
        lp = _make_log_probs([[1, 2, 3], [4, 4, 4]])
        results = ctc_greedy_decode(lp, blank_id=0)
        assert results[0].token_ids == [1, 2, 3]
        assert results[1].token_ids == [4]

    def test_input_lengths_truncate(self) -> None:
        lp = _make_log_probs([[1, 2, 3, 4, 4]])
        lengths = torch.tensor([3])
        results = ctc_greedy_decode(lp, input_lengths=lengths, blank_id=0)
        assert results[0].token_ids == [1, 2, 3]

    def test_confidences_in_unit_range(self) -> None:
        lp = _make_log_probs([[1, 2, 3]])
        results = ctc_greedy_decode(lp, blank_id=0)
        for c in results[0].confidences:
            assert 0.0 <= c <= 1.0

    def test_returns_correct_type(self) -> None:
        lp = _make_log_probs([[1]])
        results = ctc_greedy_decode(lp, blank_id=0)
        assert isinstance(results[0], CTCDecodeResult)

    def test_all_blank_returns_empty(self) -> None:
        lp = _make_log_probs([[0, 0, 0]])
        results = ctc_greedy_decode(lp, blank_id=0)
        assert results[0].token_ids == []
        assert results[0].confidences == []

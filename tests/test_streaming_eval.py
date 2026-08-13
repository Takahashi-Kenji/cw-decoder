"""streaming vs offline 一致率ユーティリティのテスト."""
from __future__ import annotations

from scripts.eval_streaming_vs_offline import token_match_ratio


def test_identical_sequences_ratio_one() -> None:
    assert token_match_ratio([1, 2, 3], [1, 2, 3]) == 1.0


def test_one_diff_in_four() -> None:
    r = token_match_ratio([1, 2, 3, 4], [1, 2, 9, 4])
    assert 0.7 < r < 0.8


def test_empty_both_is_one() -> None:
    assert token_match_ratio([], []) == 1.0

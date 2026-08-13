"""ラベル検品の集計ロジック (torch 非依存の純関数)."""
from __future__ import annotations

from src.train.metrics import TokenErrorAnalysis, describe_token


def _is_code(code: str) -> bool:
    return bool(code) and all(ch in "・-" for ch in code)


def recall_by_code_length(analysis: TokenErrorAnalysis) -> dict[int, tuple[int, float]]:
    """符号長ごとの (ref 出現数, recall%) を返す.

    設計書 §2.3 の「6要素符号 (プロサイン・記号) だけ recall が崩壊する」現象を
    検出するための集計。実符号 (・ と - のみ) を対象にし、WORD_BREAK や
    未知トークンは除外する。
    """
    by_len_correct: dict[int, int] = {}
    by_len_ref: dict[int, int] = {}
    for tid, counts in analysis.counts.items():
        code = str(describe_token(tid)["code"])
        if not _is_code(code):
            continue
        n = len(code)
        by_len_correct[n] = by_len_correct.get(n, 0) + counts.correct
        by_len_ref[n] = by_len_ref.get(n, 0) + counts.ref_count
    result: dict[int, tuple[int, float]] = {}
    for n, ref in by_len_ref.items():
        recall = 100.0 * by_len_correct.get(n, 0) / ref if ref else 0.0
        result[n] = (ref, recall)
    return result


__all__ = ["recall_by_code_length"]

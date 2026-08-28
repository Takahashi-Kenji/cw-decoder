"""ラベル検品の集計ロジック (torch 非依存の純関数)."""
from __future__ import annotations

from dataclasses import dataclass

from src.tokens.morse_tokens import HORE_CODE, RATA_CODE
from src.train.metrics import TokenErrorAnalysis, describe_token

# 先頭/末尾のどこまでを「境界付近」と見るか (符号いくつぶん)。
# 包絡線から組み立てた符号列には雑音の尖りが孤立した ``・`` として混じるため、
# 境界ちょうどだけを見ると本物のホレ/ラタを見逃す。実測 (held-out 和文 10 件) では
# 先頭に最大 7 個の尖りが乗っていた。
BOUNDARY_WINDOW = 8


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


@dataclass(frozen=True)
class BoundaryExtras:
    """音の境界付近にあってラベルに無い運用記号の位置.

    値は「組み立てた符号列の何番目か」。``None`` は「無い」。
    """

    head: int | None = None      # 先頭付近のホレ
    tail: int | None = None      # 末尾付近のラタ


def boundary_extras(heard: list[str], labelled: list[str]) -> BoundaryExtras:
    """音の先頭/末尾に、ラベルに無いホレ・ラタが居ないかを返す.

    **境界ちょうどではなく境界付近を見る** (:data:`BOUNDARY_WINDOW`)。
    2026-08-03 に held-out 和文からホレ/ラタを「打鍵されていない」として
    除去したが、2026-08-12 に波形と独立デコーダ (別実装のデコーダ) で**除去の方が誤り**と
    判明した。境界一致だけで数えると 10 件中 2 件しか拾えず、実際には
    雑音の尖りを挟んで 7 件で鳴っている。**見逃す側に倒すと、正しく読めた分が
    誤りとして数えられ続ける。**

    この関数が答えられるのは「境界付近にラベルに無い運用記号があるか」まで。
    本文の逐一照合は守備範囲外 (そこは人間が波形を見る)。
    """
    head = _find_missing(heard[:BOUNDARY_WINDOW], labelled, HORE_CODE)
    tail_window = heard[-BOUNDARY_WINDOW:] if heard else []
    tail = _find_missing(tail_window, labelled, RATA_CODE)
    if tail is not None:
        tail += max(0, len(heard) - BOUNDARY_WINDOW)
    return BoundaryExtras(head=head, tail=tail)


def _find_missing(window: list[str], labelled: list[str], code: str) -> int | None:
    """``window`` の中の ``code`` の位置。ラベルに 1 つでもあれば ``None``."""
    if code in labelled or code not in window:
        return None
    return window.index(code)


__all__ = ["BOUNDARY_WINDOW", "BoundaryExtras", "boundary_extras", "recall_by_code_length"]

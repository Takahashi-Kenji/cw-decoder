"""欧文ストリームライン (2026-08-29 運用者要望) のテスト.

和文受信中も、同じ符号を**欧文表で読んだ 1 行の文字列**を流す。
欧文表に無い符号は ``_``。モード切替 (ホレ/ラタ) は**一切行わない**。
"""
from __future__ import annotations

from src.tokens.converter import FALLBACK_CHAR, render_european_stream
from src.tokens.morse_tokens import (
    HORE_CODE,
    TOKEN_TO_ID,
    WORD_BREAK_TOKEN_ID,
)


def _id(code: str) -> int:
    return TOKEN_TO_ID[code]


class TestRenderEuropeanStream:
    def test_european_codes_read_as_letters(self) -> None:
        # ・- = A (和文では イ)。CQ = -・-・ --・-
        ids = [_id("-・-・"), _id("--・-")]
        assert render_european_stream(ids) == "CQ"

    def test_japanese_only_code_becomes_underscore(self) -> None:
        # ロ ・-・- は欧文表に無い
        assert render_european_stream([_id("・-・-")]) == FALLBACK_CHAR

    def test_hore_is_not_a_mode_switch(self) -> None:
        """ホレを読んでも欧文表のまま。ホレ自体は欧文表に無いので ``_``."""
        ids = [_id(HORE_CODE), _id("・-")]        # ホレ, イ(=A)
        assert render_european_stream(ids) == "_A"

    def test_dakuten_reads_as_i(self) -> None:
        """濁点 ・・ は同符号の欧文 I として出る (同じ符号を欧文表で読む)."""
        assert render_european_stream([_id("・・")]) == "I"

    def test_word_break_becomes_space(self) -> None:
        ids = [_id("・-"), WORD_BREAK_TOKEN_ID, _id("-・・・")]
        assert render_european_stream(ids) == "A B"

    def test_consecutive_word_breaks_collapse(self) -> None:
        ids = [_id("・-"), WORD_BREAK_TOKEN_ID, WORD_BREAK_TOKEN_ID, _id("-・・・")]
        assert render_european_stream(ids) == "A B"

    def test_leading_word_break_is_dropped(self) -> None:
        assert render_european_stream([WORD_BREAK_TOKEN_ID, _id("・-")]) == "A"

    def test_prosign_keeps_bracket_form(self) -> None:
        assert render_european_stream([_id("・・・-・-")]) == "[SK]"

    def test_empty_is_empty(self) -> None:
        assert render_european_stream([]) == ""

    def test_output_is_append_only_for_growing_input(self) -> None:
        """確定列は追記型なので、入力が伸びたら出力は前回の前置きを保つ.

        (UI が差分追記できる = 0.5 秒ごとの作り直しで選択が壊れない前提)
        """
        ids = [_id("-・-・"), WORD_BREAK_TOKEN_ID, _id("--・-")]
        for cut in range(len(ids)):
            shorter = render_european_stream(ids[:cut])
            longer = render_european_stream(ids)
            assert longer.startswith(shorter)

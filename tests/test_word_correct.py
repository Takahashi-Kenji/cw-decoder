"""辞書による語の補正のテスト."""
from __future__ import annotations

import pytest

from src.infer.word_correct import (
    EUROPEAN_LEXICON,
    correct_text,
    is_protected,
    is_real_question,
    nearest_word,
    segment_word,
    substitution_cost,
    word_distance,
)


class TestSubstitutionCost:
    """置換費用は**符号の近さ**で決まること."""

    def test_same_char_is_free(self) -> None:
        assert substitution_cost("A", "A") == 0.0

    def test_question_mark_is_cheap(self) -> None:
        """`?` は「読めなかった」印なのでどの文字にも安く化ける."""
        assert substitution_cost("?", "Q") == pytest.approx(0.3)

    def test_one_dot_apart_is_cheaper_than_unrelated(self) -> None:
        """D(-・・) と B(-・・・) は点 1 個差。T(-) と Q(--・-) より近い."""
        assert substitution_cost("D", "B") < substitution_cost("D", "Q")

    def test_cost_is_symmetric_for_letters(self) -> None:
        assert substitution_cost("D", "B") == pytest.approx(substitution_cost("B", "D"))

    def test_unknown_char_costs_full(self) -> None:
        """符号表に無い文字 (和文カナ等) は 1.0."""
        assert substitution_cost("ア", "B") == 1.0


class TestWordDistance:
    def test_identical_is_zero(self) -> None:
        assert word_distance("CQ", "CQ") == 0.0

    def test_missing_letter_costs_one(self) -> None:
        assert word_distance("NAM", "NAME") == pytest.approx(1.0)


class TestProtected:
    """辞書で触ってはいけない語."""

    @pytest.mark.parametrize("word", ["JH0ILL", "599", "73", "0", "JA1ABC"])
    def test_words_with_digits_are_protected(self, word: str) -> None:
        """コールサインと RST。存在しない局を作るのは `?` より悪い."""
        assert is_protected(word) is True

    @pytest.mark.parametrize("word", ["[SK]", "[KN]", "[SN]"])
    def test_prosigns_are_protected(self, word: str) -> None:
        assert is_protected(word) is True

    @pytest.mark.parametrize("word", ["CQ", "NAM", "?RST"])
    def test_plain_words_are_not_protected(self, word: str) -> None:
        assert is_protected(word) is False

    def test_protected_word_passes_through_correct_text(self) -> None:
        assert correct_text("JH0ILL").text == "JH0ILL"


class TestRealQuestionMark:
    """`?` は「読めなかった」印であると同時に実在の文字 (・・--・・) でもある."""

    @pytest.mark.parametrize("word", ["QSL?", "QRZ?", "QTH?"])
    def test_lexicon_word_plus_question_is_real(self, word: str) -> None:
        assert is_real_question(word) is True

    @pytest.mark.parametrize("word", ["?RST", "?", "XYZ?"])
    def test_others_are_not_real(self, word: str) -> None:
        assert is_real_question(word) is False

    def test_qsl_question_is_kept(self) -> None:
        """`QSL?` の `?` を削ると意味が変わる (実測で 0.42pt 悪化した)."""
        assert correct_text("QSL?").text == "QSL?"


class TestNearestWord:
    def test_lexicon_word_returns_itself(self) -> None:
        assert nearest_word("CQ") == "CQ"

    def test_extra_leading_letter_snaps(self) -> None:
        assert nearest_word("CQRT") == "QRT"

    def test_one_element_apart_snaps(self) -> None:
        """TNZ→TNX。X(-・・-) と Z(--・・) は符号が近い."""
        assert nearest_word("TNZ") == "TNX"
        assert nearest_word("QTG") == "QTC"

    def test_leading_unknown_snaps(self) -> None:
        assert nearest_word("?RST") == "RST"

    def test_tied_candidates_are_refused(self) -> None:
        """QSB と QSY が同距離。どちらか選ぶより `?` のまま残す方が良い."""
        assert nearest_word("QSC") is None

    def test_nearest_wrong_but_ambiguous_is_refused(self) -> None:
        """?UR の最近傍は CUL (誤り) だが margin が無いので寄せない.

        margin の歯止めが無ければ誤った語を自信ありげに出していた。
        """
        assert nearest_word("?UR") is None

    def test_far_word_is_not_snapped(self) -> None:
        """語彙から遠い語は「分からない」として None."""
        assert nearest_word("ZZZZZZ") is None

    def test_empty_returns_none(self) -> None:
        assert nearest_word("") is None


class TestSegmentWord:
    def test_two_words_are_split(self) -> None:
        assert segment_word("CQDE") == ["CQ", "DE"]

    def test_three_words_are_split(self) -> None:
        assert segment_word("CQCQDE") == ["CQ", "CQ", "DE"]

    def test_lexicon_word_is_not_split(self) -> None:
        """語彙にある語はそれ自体が答え (CUAGN を CU+AGN に割らない)."""
        assert segment_word("CUAGN") == ["CUAGN"]

    def test_unsplittable_word_is_returned_as_is(self) -> None:
        assert segment_word("XYZZY") == ["XYZZY"]

    def test_short_word_is_not_split(self) -> None:
        assert segment_word("CQ") == ["CQ"]

    def test_prefers_fewest_parts(self) -> None:
        parts = segment_word("QRZDE")
        assert parts == ["QRZ", "DE"]


class TestCorrectText:
    def test_empty_text(self) -> None:
        result = correct_text("")
        assert result.text == ""
        assert result.changed is False

    def test_clean_text_is_untouched(self) -> None:
        result = correct_text("CQ CQ DE JA1ABC K")
        assert result.text == "CQ CQ DE JA1ABC K"
        assert result.changed is False

    def test_merged_words_are_split(self) -> None:
        result = correct_text("CQ CQCQDE JF1GL K")
        assert result.text == "CQ CQ CQ DE JF1GL K"
        assert result.changed is True

    def test_newlines_are_preserved(self) -> None:
        """改行は送信のターンの切れ目。潰すと改行機能が効かなくなる."""
        assert correct_text("CQDE\nTNX").text == "CQ DE\nTNX"

    def test_multiple_spaces_are_preserved(self) -> None:
        assert correct_text("CQ  DE").text == "CQ  DE"

    def test_trailing_space_is_preserved(self) -> None:
        """確定列の末尾スペースは暫定列との境界に効く (消すと語が繋がる)."""
        assert correct_text("CQ ").text == "CQ "

    def test_leading_space_is_preserved(self) -> None:
        assert correct_text(" CQ").text == " CQ"


class TestSpans:
    """UI が色を変えるための範囲情報."""

    def test_span_points_at_corrected_text(self) -> None:
        result = correct_text("CQRT")
        assert result.text == "QRT"
        assert len(result.spans) == 1
        span = result.spans[0]
        assert result.text[span.start:span.end] == "QRT"
        assert span.original == "CQRT"

    def test_span_offsets_survive_length_change(self) -> None:
        """切り直しで語が伸びても、後続の範囲がずれないこと."""
        result = correct_text("TNX CQDE TNZ")
        assert result.text == "TNX CQ DE TNX"
        assert [result.text[s.start:s.end] for s in result.spans] == ["CQ DE", "TNX"]

    def test_span_offsets_across_newline(self) -> None:
        result = correct_text("CQ\nCQRT")
        assert result.text == "CQ\nQRT"
        span = result.spans[0]
        assert result.text[span.start:span.end] == "QRT"

    def test_no_spans_when_unchanged(self) -> None:
        assert correct_text("CQ DE K").spans == ()


class TestLexicon:
    def test_contains_no_digits(self) -> None:
        """数字を含む語を語彙に入れると寄せ先の候補になってしまう."""
        for word in EUROPEAN_LEXICON:
            assert not any(ch.isdigit() for ch in word), word

    def test_has_no_duplicates(self) -> None:
        assert len(set(EUROPEAN_LEXICON)) == len(EUROPEAN_LEXICON)

    def test_common_abbreviations_present(self) -> None:
        for word in ("CQ", "DE", "RST", "QTH", "TNX", "QSL", "QSO"):
            assert word in EUROPEAN_LEXICON


class TestJapaneseIsUntouched:
    """和文は語彙が別。悪くしないこと (良くもならない)."""

    def test_kana_passes_through(self) -> None:
        text = "ナイトキキマネカラ　カラダニ"
        assert correct_text(text).text == text

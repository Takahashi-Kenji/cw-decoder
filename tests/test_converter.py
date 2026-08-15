"""トークン列 → 文字列変換器のテスト."""
from __future__ import annotations

import pytest

from src.tokens.converter import FALLBACK_CHAR, TokenConverter
from src.tokens.morse_tokens import BLANK_TOKEN_ID, TOKEN_TO_ID


def ids(codes: list[str]) -> list[int]:
    return [TOKEN_TO_ID[c] for c in codes]


# ============================================================
# 欧文基本
# ============================================================
class TestEuropeanBasic:
    def test_simple_letters(self) -> None:
        conv = TokenConverter(mode="european")
        result = conv.convert(ids(["・-", "-・・・", "-・-・"]))
        assert result.text == "ABC"
        assert result.fallback_log == []

    def test_with_blank_ignored(self) -> None:
        conv = TokenConverter(mode="european")
        token_ids = [TOKEN_TO_ID["・-"], BLANK_TOKEN_ID, TOKEN_TO_ID["-・・・"]]
        result = conv.convert(token_ids)
        assert result.text == "AB"

    def test_digits(self) -> None:
        conv = TokenConverter(mode="european")
        result = conv.convert(ids(["・----", "・・---", "・・・--"]))
        assert result.text == "123"

    def test_empty_input(self) -> None:
        conv = TokenConverter(mode="european")
        result = conv.convert([])
        assert result.text == ""
        assert result.fallback_log == []


# ============================================================
# 和文基本
# ============================================================
class TestJapaneseBasic:
    def test_iroha(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・-", "・-・-", "-・・・"]))
        assert result.text == "イロハ"
        assert result.fallback_log == []

    def test_long_vowel(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・-・・", "・--・-"]))  # カ + ー
        assert result.text == "カー"

    def test_kuten(self) -> None:
        # 区切点 、 = ・-・-・- (欧文 . と同符号)
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・-", "・-・-・-"]))
        assert result.text == "イ、"


# ============================================================
# 濁点・半濁点合成
# ============================================================
class TestDakutenComposition:
    def test_ka_plus_dakuten_becomes_ga(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・-・・", "・・"]))
        assert result.text == "ガ"
        assert result.fallback_log == []

    def test_ha_plus_handakuten_becomes_pa(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["-・・・", "・・--・"]))
        assert result.text == "パ"

    def test_ha_plus_dakuten_becomes_ba(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["-・・・", "・・"]))
        assert result.text == "バ"

    def test_dakuten_without_preceding_kana_is_fallback(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・・"]))
        assert result.text == FALLBACK_CHAR
        assert len(result.fallback_log) == 1
        assert result.fallback_log[0].kind == "TABLE_MISS"

    def test_dakuten_after_non_composable_kana_is_fallback(self) -> None:
        # イ は濁音化対象外 → イ + ゛ は「イ + 読めなかった印」になる
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・-", "・・"]))
        assert result.text == f"イ{FALLBACK_CHAR}"
        assert len(result.fallback_log) == 1
        assert result.fallback_log[0].kind == "TABLE_MISS"

    def test_two_kana_then_dakuten_composes_with_last(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・-・・", "-・・・", "・・"]))  # カ ハ ゛
        assert result.text == "カバ"


# ============================================================
# モード不一致
# ============================================================
class TestModeMismatch:
    def test_japanese_only_code_in_european_mode(self) -> None:
        # コ ---- は欧文に該当なし
        conv = TokenConverter(mode="european")
        result = conv.convert(ids(["----"]))
        assert result.text == FALLBACK_CHAR
        assert result.fallback_log[0].kind == "TABLE_MISS"

    def test_european_only_prosign_in_japanese_mode(self) -> None:
        # SK ・・・-・- は和文に該当なし
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・・・-・-"]))
        assert result.text == FALLBACK_CHAR
        assert result.fallback_log[0].kind == "TABLE_MISS"


# ============================================================
# 確信度閾値
# ============================================================
class TestConfidence:
    def test_low_confidence_replaces_with_fallback(self) -> None:
        conv = TokenConverter(mode="european", confidence_threshold=0.5)
        token_ids = ids(["・-", "-・・・", "-・-・"])
        result = conv.convert(token_ids, confidences=[0.3, 0.9, 0.9])
        assert result.text == f"{FALLBACK_CHAR}BC"
        assert len(result.fallback_log) == 1
        assert result.fallback_log[0].kind == "LOW_CONFIDENCE"
        assert result.fallback_log[0].confidence == pytest.approx(0.3)

    def test_confidence_at_threshold_passes(self) -> None:
        conv = TokenConverter(mode="european", confidence_threshold=0.5)
        result = conv.convert(ids(["・-"]), confidences=[0.5])
        assert result.text == "A"

    def test_multiple_low_confidence(self) -> None:
        conv = TokenConverter(mode="european", confidence_threshold=0.7)
        token_ids = ids(["・-", "-・・・", "-・-・"])
        result = conv.convert(token_ids, confidences=[0.3, 0.4, 0.9])
        assert result.text == f"{FALLBACK_CHAR}{FALLBACK_CHAR}C"
        assert len(result.fallback_log) == 2

    def test_mismatched_confidence_length_raises(self) -> None:
        conv = TokenConverter(mode="european")
        with pytest.raises(ValueError):
            conv.convert(ids(["・-", "-・・・"]), confidences=[0.9])


# ============================================================
# 特殊表示 (ホレ / ラタ / SK / SN)
# ============================================================
class TestSpecialDisplay:
    def test_horea_in_japanese_mode(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["-・・---"]))
        assert result.text == "[ホレ]"

    def test_rata_in_japanese_mode(self) -> None:
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・・・-・"]))
        assert result.text == "[ラタ]"

    def test_sn_in_european_mode_uses_brackets(self) -> None:
        # 同符号 ・・・-・ は EU では [SN]
        conv = TokenConverter(mode="european")
        result = conv.convert(ids(["・・・-・"]))
        assert result.text == "[SN]"

    def test_sk_in_european_mode(self) -> None:
        conv = TokenConverter(mode="european")
        result = conv.convert(ids(["・・・-・-"]))
        assert result.text == "[SK]"


# ============================================================
# 同形符号のモード別解釈
# ============================================================
class TestSharedCodeModeBranching:
    def test_dot_dash_is_a_in_european(self) -> None:
        conv = TokenConverter(mode="european")
        assert conv.convert(ids(["・-"])).text == "A"

    def test_dot_dash_is_i_in_japanese(self) -> None:
        conv = TokenConverter(mode="japanese")
        assert conv.convert(ids(["・-"])).text == "イ"

    def test_double_dot_is_i_in_european(self) -> None:
        conv = TokenConverter(mode="european")
        assert conv.convert(ids(["・・"])).text == "I"

    def test_double_dot_alone_in_japanese_is_fallback(self) -> None:
        # 和文では単独の ・・ は濁点 → 直前カナ無しで TABLE_MISS
        conv = TokenConverter(mode="japanese")
        result = conv.convert(ids(["・・"]))
        assert result.text == FALLBACK_CHAR


# ============================================================
# 入力バリデーション
# ============================================================
class TestValidation:
    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            TokenConverter(mode="chinese")  # type: ignore[arg-type]

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
    def test_out_of_range_threshold_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            TokenConverter(mode="european", confidence_threshold=threshold)


# --- 確定列と暫定列を分けて変換したときの語間スペース ---
#
# ライブ経路は確定列(黒)と暫定列(グレー)を別々に convert して連結する。
# 先頭の WORD_BREAK は「行頭に余計なスペースを出さない」ため捨てられるが、
# 境界が語間に落ちると連結時にスペースが消える。
# 実運用で "GL 73 CQ DE JH0ILL K" が "GL 73CQ DE JH0ILL" になる形で発覚
# (2026-08-04 報告・再現済み)。

def _ids(text: str, mode: str = "european") -> list[int]:
    from src.tokens.morse_tokens import TOKEN_TO_ID, text_to_codes
    return [TOKEN_TO_ID[c] for c in text_to_codes(text, mode)]


def test_leading_word_break_is_dropped_by_default():
    """既定では行頭のスペースを出さない (従来の挙動)."""
    conv = TokenConverter(mode="european", confidence_threshold=0.0)
    ids = _ids("GL 73 CQ")
    assert conv.convert(ids[5:]).text == "CQ"


def test_keep_leading_space_preserves_boundary_space():
    """暫定列の先頭が WORD_BREAK なら、連結のためにスペースを残せる."""
    conv = TokenConverter(mode="european", confidence_threshold=0.0)
    ids = _ids("GL 73 CQ")
    assert conv.convert(ids[5:], keep_leading_space=True).text == " CQ"


def test_split_at_word_break_concatenates_correctly():
    """確定/暫定を語間で割っても、連結すると一括変換と一致する."""
    conv = TokenConverter(mode="european", confidence_threshold=0.0)
    text = "GL 73 CQ DE JH0ILL K"
    ids = _ids(text)
    whole = conv.convert(ids).text
    for split in range(1, len(ids)):
        committed = conv.convert(ids[:split]).text
        keep = bool(committed) and not committed.endswith(" ")
        provisional = conv.convert(ids[split:], keep_leading_space=keep).text
        assert committed + provisional == whole, f"split={split} で不一致"


def test_keep_leading_space_does_not_double_space():
    """確定列が既にスペースで終わっている場合は足さない (呼び出し側の判断)."""
    conv = TokenConverter(mode="european", confidence_threshold=0.0)
    ids = _ids("GL 73 CQ")
    committed = conv.convert(ids[:6]).text          # 'GL 73 ' (WORD_BREAK まで含む)
    assert committed.endswith(" ")
    provisional = conv.convert(ids[6:], keep_leading_space=False).text
    assert committed + provisional == "GL 73 CQ"


def test_keep_leading_space_ignored_when_no_leading_break():
    """先頭が WORD_BREAK でなければ何も変わらない."""
    conv = TokenConverter(mode="european", confidence_threshold=0.0)
    ids = _ids("GL 73 CQ")
    assert conv.convert(ids[6:], keep_leading_space=True).text == "CQ"

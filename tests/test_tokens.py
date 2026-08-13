"""符号トークン定義と e-Gov 規則別表第一号の照合テスト."""
from __future__ import annotations

import pytest

from src.tokens.morse_tokens import (
    BLANK_CODE,
    BLANK_TOKEN_ID,
    DAKUTEN_CHAR,
    DAKUTEN_COMPOSE,
    EUROPEAN_CHAR_TO_CODE,
    EUROPEAN_TABLE,
    HANDAKUTEN_CHAR,
    HANDAKUTEN_COMPOSE,
    ID_TO_TOKEN,
    JAPANESE_CHAR_TO_CODES,
    JAPANESE_TABLE,
    TOKEN_TO_ID,
    UNIFIED_TOKENS,
    VOCAB_SIZE,
    text_to_codes,
)
from tests.data.eGov_morse_reference import (
    ALL_EUROPEAN,
    EGOV_EUROPEAN_DIGITS,
    EGOV_EUROPEAN_LETTERS,
    EGOV_EUROPEAN_PUNCT,
    EGOV_JAPANESE_AUX,
    EGOV_JAPANESE_KANA,
    EGOV_JAPANESE_SPECIAL,
    EGOV_PROSIGNS,
)


# ============================================================
# 欧文符号 vs e-Gov
# ============================================================
class TestEuropeanAgainstEGov:
    @pytest.mark.parametrize("char,code", sorted(EGOV_EUROPEAN_LETTERS.items()))
    def test_letter_matches_egov(self, char: str, code: str) -> None:
        assert EUROPEAN_TABLE[code] == char, (
            f"欧文 {char} の符号が e-Gov と不一致. 期待 {code!r}, 実装 "
            f"{EUROPEAN_CHAR_TO_CODE.get(char)!r}"
        )

    @pytest.mark.parametrize("char,code", sorted(EGOV_EUROPEAN_DIGITS.items()))
    def test_digit_matches_egov(self, char: str, code: str) -> None:
        assert EUROPEAN_TABLE[code] == char

    @pytest.mark.parametrize("char,code", sorted(EGOV_EUROPEAN_PUNCT.items()))
    def test_punct_matches_egov(self, char: str, code: str) -> None:
        assert EUROPEAN_TABLE[code] == char

    def test_all_egov_european_chars_synthesizable(self) -> None:
        for char in ALL_EUROPEAN:
            assert char in EUROPEAN_CHAR_TO_CODE, f"{char!r} が合成入力辞書に無い"


# ============================================================
# 和文符号 vs e-Gov
# ============================================================
class TestJapaneseAgainstEGov:
    @pytest.mark.parametrize("char,code", sorted(EGOV_JAPANESE_KANA.items()))
    def test_kana_matches_egov(self, char: str, code: str) -> None:
        assert JAPANESE_TABLE[code] == char

    @pytest.mark.parametrize("char,code", sorted(EGOV_JAPANESE_AUX.items()))
    def test_aux_matches_egov(self, char: str, code: str) -> None:
        assert JAPANESE_TABLE[code] == char

    @pytest.mark.parametrize("name,code", sorted(EGOV_JAPANESE_SPECIAL.items()))
    def test_special_matches_egov(self, name: str, code: str) -> None:
        expected = f"[{name}]"
        assert JAPANESE_TABLE[code] == expected

    def test_all_kana_synthesizable(self) -> None:
        for char in EGOV_JAPANESE_KANA:
            assert char in JAPANESE_CHAR_TO_CODES, f"{char!r} が合成入力辞書に無い"


# ============================================================
# プロサイン
# ============================================================
class TestProsigns:
    @pytest.mark.parametrize("name,code", sorted(EGOV_PROSIGNS.items()))
    def test_prosign_code_present_in_european_table(self, name: str, code: str) -> None:
        assert code in EUROPEAN_TABLE
        display = EUROPEAN_TABLE[code]
        if name == "AR":
            assert display == "+"
        elif name == "BT":
            assert display == "="
        else:
            assert display == f"[{name}]"


# ============================================================
# 統合トークン集合
# ============================================================
class TestUnifiedVocabulary:
    def test_blank_is_id_zero(self) -> None:
        assert BLANK_TOKEN_ID == 0
        assert ID_TO_TOKEN[0].code == BLANK_CODE
        assert TOKEN_TO_ID[BLANK_CODE] == 0

    def test_token_ids_are_consecutive_from_zero(self) -> None:
        ids = sorted(t.id for t in UNIFIED_TOKENS)
        assert ids == list(range(VOCAB_SIZE))

    def test_no_duplicate_codes(self) -> None:
        codes = [t.code for t in UNIFIED_TOKENS]
        assert len(codes) == len(set(codes))

    def test_no_duplicate_ids(self) -> None:
        ids = [t.id for t in UNIFIED_TOKENS]
        assert len(ids) == len(set(ids))

    def test_blank_code_not_collide_with_real_code(self) -> None:
        real_codes = {t.code for t in UNIFIED_TOKENS if t.id != BLANK_TOKEN_ID}
        assert BLANK_CODE not in real_codes

    def test_every_token_has_at_least_one_display(self) -> None:
        from src.tokens.morse_tokens import WORD_BREAK_CODE
        for token in UNIFIED_TOKENS:
            if token.id == BLANK_TOKEN_ID:
                continue
            if token.code == WORD_BREAK_CODE:
                # 語間トークンは表ではなく " " として変換器が直接処理
                continue
            in_eu = token.code in EUROPEAN_TABLE
            in_ja = token.code in JAPANESE_TABLE
            assert in_eu or in_ja, f"{token.code!r} がどちらの表にも無い"

    def test_european_table_codes_in_vocabulary(self) -> None:
        for code in EUROPEAN_TABLE:
            assert code in TOKEN_TO_ID

    def test_japanese_table_codes_in_vocabulary(self) -> None:
        for code in JAPANESE_TABLE:
            assert code in TOKEN_TO_ID

    def test_vocab_size_lower_bound(self) -> None:
        # 重複排除後の最小見積 = EU-only 16 + 共通 32 + JA-only 23 = 71 + blank = 72
        # 余裕を見て 70 以上であることを確認
        assert VOCAB_SIZE >= 70


# ============================================================
# 符号の文字種制限
# ============================================================
class TestCodeCharacters:
    @pytest.mark.parametrize("table_name,table", [
        ("european", EUROPEAN_TABLE),
        ("japanese", JAPANESE_TABLE),
    ])
    def test_codes_use_only_dot_and_dash(self, table_name: str, table: dict[str, str]) -> None:
        for code in table:
            for ch in code:
                assert ch in ("・", "-"), f"{table_name} の符号 {code!r} に許可外文字 {ch!r}"


# ============================================================
# 濁点・半濁点合成
# ============================================================
class TestDakutenComposition:
    def test_dakuten_char_consistent(self) -> None:
        assert DAKUTEN_CHAR == "゛"
        assert HANDAKUTEN_CHAR == "゜"

    def test_dakuten_targets_are_kana(self) -> None:
        for plain in DAKUTEN_COMPOSE:
            assert plain in EGOV_JAPANESE_KANA

    def test_handakuten_targets_are_h_row(self) -> None:
        assert set(HANDAKUTEN_COMPOSE.keys()) == {"ハ", "ヒ", "フ", "ヘ", "ホ"}

    def test_dakuten_h_row_overlap(self) -> None:
        h_row = {"ハ", "ヒ", "フ", "ヘ", "ホ"}
        for ch in h_row:
            assert ch in DAKUTEN_COMPOSE
            assert ch in HANDAKUTEN_COMPOSE

    def test_synthesis_of_composite_kana(self) -> None:
        # ガ = カ符号 + 濁点符号
        codes = JAPANESE_CHAR_TO_CODES["ガ"]
        assert codes == ("・-・・", "・・")

    def test_synthesis_of_handakuten_kana(self) -> None:
        codes = JAPANESE_CHAR_TO_CODES["パ"]
        assert codes == ("-・・・", "・・--・")


# ============================================================
# 同形符号の重複検出
# ============================================================
class TestSharedCodes:
    def test_a_and_i_share_code(self) -> None:
        assert EUROPEAN_CHAR_TO_CODE["A"] == JAPANESE_CHAR_TO_CODES["イ"][0]

    def test_e_and_he_share_code(self) -> None:
        assert EUROPEAN_CHAR_TO_CODE["E"] == JAPANESE_CHAR_TO_CODES["ヘ"][0]

    def test_i_and_dakuten_share_code(self) -> None:
        # 欧文 I と 和文 濁点 ゛ は同符号 ・・
        assert EUROPEAN_CHAR_TO_CODE["I"] == "・・"
        assert JAPANESE_CHAR_TO_CODES["゛"][0] == "・・"

    def test_sn_and_rata_share_code(self) -> None:
        # SN プロサインと ラタ は同符号 ・・・-・
        assert EUROPEAN_TABLE["・・・-・"] == "[SN]"
        assert JAPANESE_TABLE["・・・-・"] == "[ラタ]"


# ============================================================
# text_to_codes 公開 API
# ============================================================
class TestTextToCodes:
    def test_european_simple(self) -> None:
        assert text_to_codes("ABC", "european") == ["・-", "-・・・", "-・-・"]

    def test_european_lowercase_normalized(self) -> None:
        assert text_to_codes("abc", "european") == text_to_codes("ABC", "european")

    def test_european_emits_word_break_for_spaces(self) -> None:
        from src.tokens.morse_tokens import WORD_BREAK_CODE
        # 既定では語間 → WORD_BREAK トークン
        result = text_to_codes("A B C", "european")
        expected = ["・-", WORD_BREAK_CODE, "-・・・", WORD_BREAK_CODE, "-・-・"]
        assert result == expected

    def test_european_no_word_break_legacy(self) -> None:
        # emit_word_breaks=False で旧挙動 (スペース無視)
        assert text_to_codes("A B C", "european", emit_word_breaks=False) == \
            text_to_codes("ABC", "european", emit_word_breaks=False)

    def test_consecutive_spaces_collapse_to_single_word_break(self) -> None:
        from src.tokens.morse_tokens import WORD_BREAK_CODE
        result = text_to_codes("A   B", "european")
        assert result == ["・-", WORD_BREAK_CODE, "-・・・"]

    def test_trailing_space_is_stripped(self) -> None:
        from src.tokens.morse_tokens import WORD_BREAK_CODE
        result = text_to_codes("AB ", "european")
        assert result == ["・-", "-・・・"]
        assert WORD_BREAK_CODE not in result

    def test_japanese_simple(self) -> None:
        assert text_to_codes("イロハ", "japanese") == ["・-", "・-・-", "-・・・"]

    def test_japanese_dakuten_expands(self) -> None:
        # ガ → カ符号 + 濁点符号
        assert text_to_codes("ガ", "japanese") == ["・-・・", "・・"]

    def test_japanese_handakuten_expands(self) -> None:
        assert text_to_codes("パ", "japanese") == ["-・・・", "・・--・"]

    def test_japanese_long_vowel(self) -> None:
        assert text_to_codes("カー", "japanese") == ["・-・・", "・--・-"]


# ============================================================
# 段落と括弧 (2026-08-12 に運用者が発見した誤り)
# ============================================================
class TestDanrakuAndBrackets:
    """``・-・-・・`` は **段落 (。)** であって ``」`` ではない.

    2026-06-11 の最初のトークン定義から ``」`` と書かれていた。**段落の記号の
    字形が ``」`` に似ている**ため、表を書き写すときに閉じ括弧と取り違えた
    ものと思われる (「みんなの知識」は段落を ``└`` と表記している)。

    **``tests/data/eGov_morse_reference.py`` も同じ誤りを持っていたので、
    照合テストは全部通っていた。** 実装と参照が同じ間違いをすると、
    独立した検証にならない。**だからここは参照表を経由せず、値を直に書く。**

    本物の括弧は和文表に存在する:

    * 下向き括弧 ``「`` = ``-・--・-``
    * 上向き括弧 ``」`` = ``・-・・-・``

    **どちらもこの語彙には入れていない。** トークン ID は
    ``(符号長, 符号文字列)`` の辞書順で振られるので、6 要素と 7 要素の符号を
    足すと**それ以降の ID が全部ずれ、学習済みモデルの出力層が無意味になる**。
    足すのは再学習の機会に合わせること。
    """

    def test_段落は句点である(self) -> None:
        assert JAPANESE_TABLE["・-・-・・"] == "。"

    def test_句点から段落を引ける(self) -> None:
        assert text_to_codes("。", "japanese") == ["・-・-・・"]

    def test_区切点は読点のまま(self) -> None:
        """段落を直したついでに区切点まで動かしていないこと."""
        assert JAPANESE_TABLE["・-・-・-"] == "、"

    @pytest.mark.parametrize(
        "code,name",
        [("-・--・-", "下向き括弧 「"), ("・-・・-・", "上向き括弧 」")],
    )
    def test_本物の括弧はまだ語彙に無い(self, code: str, name: str) -> None:
        """**わざと入れていない。** 入れると ID がずれて再学習が要る.

        入れたくなったときは、このテストを消すのではなく
        **再学習まで込みで計画すること** (`{name}` は実在の符号である)。
        """
        assert code not in JAPANESE_TABLE, f"{name} を足すと ID がずれる"
        assert code not in TOKEN_TO_ID

    def test_鍵括弧は符号を持たない(self) -> None:
        """``「`` ``」`` の文字自体は送れない (符号を割り当てていないため)."""
        assert "「" not in JAPANESE_CHAR_TO_CODES
        assert "」" not in JAPANESE_CHAR_TO_CODES

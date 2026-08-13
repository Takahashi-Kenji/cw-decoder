"""日本語 → 送信可能なカタカナ変換のテスト."""
from __future__ import annotations

import pytest

from src.tx.profile import BilingualField, OperatorProfile
from src.tx.reading import (
    find_bad_chars,
    normalise_sendable,
    to_sendable_kana,
)


@pytest.fixture
def profile() -> OperatorProfile:
    p = OperatorProfile(
        callsign="JH0ILL",
        name=BilingualField("TARO", "タロウ"),
        qth=BilingualField("YOKOHAMA", "ヨコハマシ"),
    )
    # **経歴の欄はもう変換辞書に注がれない** (2026-08-12)。固有名詞の読みは
    # 運用者が読み辞書に入れる。以前は `field_readings()` が経歴から辞書を
    # 作っており、それが欧文の本文を壊す欠陥の発生源だった
    p.reading_dictionary = {
        "東京": "トウキヨウ", "神奈川県": "カナガワケン",
        "横浜市": "ヨコハマシ", "太郎": "タロウ",
    }
    return p


class TestNormalise:
    """小書きと句読点を送信可能な形へ倒す."""

    @pytest.mark.parametrize(
        ("small", "large"),
        [("ャ", "ヤ"), ("ュ", "ユ"), ("ョ", "ヨ"), ("ッ", "ツ"), ("ェ", "エ")],
    )
    def test_small_kana_becomes_large(self, small: str, large: str) -> None:
        """和文モールスに小書き文字は無い."""
        assert normalise_sendable(small) == large

    def test_period_stays_as_danraku(self) -> None:
        """`。` は**段落として符号表にある**。倒さずそのまま残す.

        2026-08-12 まで `、` に倒していた。「符号表に無い」と考えていたのは
        `・-・-・・` を `」` と誤記していたためで、段落は元からある
        (`src/tokens/morse_tokens.py` の注記)。
        """
        assert normalise_sendable("ハレ。") == "ハレ。"

    def test_full_width_period_becomes_danraku(self) -> None:
        """`．` (全角ピリオド) は同じ意味なので `。` に倒す."""
        assert normalise_sendable("ハレ．") == "ハレ。"

    def test_danraku_does_not_gain_word_gaps(self) -> None:
        """`。` の前後に語間を入れない (`、` と同じ扱い).

        段落を残すようにしたとき、句読点の整形が `、` しか見ておらず
        `コンニチハ 。 テンキ` と**語間が 2 つ増えていた** (実際に踏んだ)。
        語間は 7 単位あるので、これは電波の長さに直接効く。

        **整形は ``normalise_sendable`` ではなく ``to_sendable_kana`` にある。**
        """
        assert to_sendable_kana("コンニチハ 。 テンキ").text == "コンニチハ。テンキ"

    def test_full_width_space_becomes_word_gap(self) -> None:
        assert normalise_sendable("ア　イ") == "ア イ"

    def test_opening_bracket_is_kept(self) -> None:
        """`「` は欧文区間の印なので落とさない (2026-08-12 に仕様変更).

        以前は符号表に無いのでここで落としていたが、`「…」` を欧文区間として
        解釈するのは `encoder.split_segments` の仕事になった。ここで消すと
        印が届く前に無くなる (:class:`TestEuropeanSpanBrackets`)。"""
        assert normalise_sendable("「ア」") == "「ア」"

    def test_double_bracket_becomes_single_bracket(self) -> None:
        """`『』` (全角二重括弧) は `「」` に倒す.

        以前は `』` だけ `」` に倒し `『` は空文字に落としていたため非対称で、
        `『FT991』` が `FT991」` になり `FT` が「送信できない文字」として
        弾かれていた (2026-08-12 の最終レビューで指摘。IME で二重括弧を打つ
        運用者が踏む)。"""
        assert normalise_sendable("『FT991』") == "「FT991」"


class TestEuropeanSpanBrackets:
    """`「…」` は欧文区間の印なので、カナ変換で消してはいけない.

    **`「` は ``PUNCTUATION_MAP`` で空文字に落とされていた** (2026-08-12 に判明)。
    そのため `「FT991」` が `FT991」` になり、印が届く前に消えていた。
    区間として解釈するのは `encoder.split_segments` の仕事なので、
    ここでは**そのまま通す**。
    """

    def test_開き括弧が消えない(self) -> None:
        assert "「" in to_sendable_kana("「FT991」").text

    def test_閉じ括弧も残る(self) -> None:
        assert "」" in to_sendable_kana("「FT991」").text

    def test_括弧の中身が変わらない(self) -> None:
        assert "FT991" in to_sendable_kana("リグ ハ 「FT991」 デス").text

    def test_括弧を送れない文字として報告しない(self) -> None:
        """判定は encoder 側がモードを見て行う。ここでは通す."""
        result = to_sendable_kana("「FT991」")
        assert "「" not in "".join(b.char for b in result.bad_chars)

    def test_単独の閉じ括弧は従来どおり(self) -> None:
        """`」` は和文の終わりとして既に使われている. 壊さないこと."""
        assert to_sendable_kana("こんにちは」").text.endswith("」")


class TestFindBadChars:
    def test_plain_kana_is_fine(self) -> None:
        assert find_bad_chars("コンニチハ") == ()

    def test_digits_are_fine(self) -> None:
        assert find_bad_chars("599") == ()

    def test_kanji_is_reported_with_its_position(self) -> None:
        bad = find_bad_chars("ア晴レ")
        assert [(b.index, b.char) for b in bad] == [(1, "晴")]

    def test_markers_pass_through(self) -> None:
        """`{HORE}` は符号に展開されるので文字として弾かない."""
        assert find_bad_chars("{HORE}アイ{RATA}") == ()


class TestConversion:
    def test_basic_sentence(self, profile: OperatorProfile) -> None:
        result = to_sendable_kana("こんにちは。天気は晴れです", profile)
        assert result.text == "コンニチハ。テンキ ハ ハレ デス"
        assert result.sendable

    def test_word_gaps_are_inserted(self, profile: OperatorProfile) -> None:
        """和文 CW は語間を置いて送る。区切りを捨てると読みづらい 1 本になる."""
        result = to_sendable_kana("天気は晴れです", profile)
        assert " " in result.text

    def test_no_gap_around_punctuation(self, profile: OperatorProfile) -> None:
        """実録音のラベルも `ハレ、アツイ` の形である."""
        result = to_sendable_kana("晴れです。暑いです", profile)
        assert "、 " not in result.text
        assert " 、" not in result.text

    def test_reading_dictionary_wins(self, profile: OperatorProfile) -> None:
        result = to_sendable_kana("東京です", profile)
        assert "トウキヨウ" in result.text

    def test_dictionary_word_is_not_resplit(self, profile: OperatorProfile) -> None:
        """辞書の読みを塊として守る.

        守らないと `ヨコハマシ` が `ヨコハマ シ` に割れる (機械変換の区切りが
        そのまま語間になるため)。
        """
        result = to_sendable_kana("横浜市です", profile)
        assert "ヨコハマシ" in result.text
        assert "ヨコハマ シ" not in result.text

    def test_longer_dictionary_word_wins(self, profile: OperatorProfile) -> None:
        """`神奈川県` を `神奈川` + `県` に割らない."""
        result = to_sendable_kana("神奈川県です", profile)
        assert "カナガワケン" in result.text

    def test_profile_reading_is_used(self, profile: OperatorProfile) -> None:
        """自局の名前を誤読されるのが一番痛い (読み辞書に入れておく)."""
        result = to_sendable_kana("太郎です", profile)
        assert "タロウ" in result.text

    def test_経歴の欄は本文に当たらない(self, profile: OperatorProfile) -> None:
        """**欧文の本文の `TARO` が `タロウ` に化けないこと** (2026-08-12).

        以前は経歴の欄から「表示形 → 読み」の辞書を作って本文全体に当てて
        いたため、欧文の本文にカタカナが混ざってモードが和文に倒れ、
        **文中の欧文がまるごと送れなくなっていた** (2026-08-11 に実測)。
        経歴を 2 値にして、この仕組みごと消した。

        経歴には `name.japanese = "タロウ"` が入っている (上の fixture)。
        それでも本文の `TARO` は素通りすること。
        """
        result = to_sendable_kana("GM UR RST 599 NAME TARO QTH YOKOHAMA HW? K", profile)
        assert "TARO" in result.text
        assert "タロウ" not in result.text
        assert "YOKOHAMA" in result.text
        assert "ヨコハマシ" not in result.text

    def test_small_kana_in_source(self, profile: OperatorProfile) -> None:
        result = to_sendable_kana("ちょっと待って", profile)
        assert "ョ" not in result.text
        assert "ッ" not in result.text

    def test_digits_survive(self, profile: OperatorProfile) -> None:
        result = to_sendable_kana("信号は599です", profile)
        assert "599" in result.text
        assert result.sendable

    def test_latin_is_reported_not_dropped(self, profile: OperatorProfile) -> None:
        """黙って落とすと、相手には意味の通らない符号が届く."""
        result = to_sendable_kana("Hello", profile)
        assert not result.sendable
        assert "".join(b.char for b in result.bad_chars) == "Hello"

    def test_empty(self, profile: OperatorProfile) -> None:
        result = to_sendable_kana("", profile)
        assert result.text == ""
        assert result.sendable

    def test_works_without_a_profile(self) -> None:
        result = to_sendable_kana("天気")
        assert result.text
        assert result.sendable

    def test_hits_are_reported(self, profile: OperatorProfile) -> None:
        """画面で「辞書が当たった」根拠を出せるようにする."""
        result = to_sendable_kana("東京です", profile)
        assert ("東京", "トウキヨウ") in result.dictionary_hits


class TestNewlines:
    """改行を含む入力.

    **pykakasi は改行をまたいで内部の蓄積を捨てない。** 複数行をまとめて渡すと、
    改行の後に「それまでの全文」をもう一度返す (2026-08-11 に実運用で発覚)::

        convert("こんにちは\\nこんにちは")
          [0] orig="こんにちは"           kana="コンニチハ"
          [1] orig="\\n"                  kana=""
          [2] orig="こんにちはこんにちは"  kana="コンニチハコンニチハ"   <- 蓄積

    画面では「Enter を押すたびに送信文が増えていく」形で現れた。
    **CW に改行は無いので、行の切れ目は語間 1 つとして扱う。**
    """

    def test_改行で文が増えない(self) -> None:
        result = to_sendable_kana("こんにちは\nこんにちは")
        assert result.text == "コンニチハ コンニチハ"

    def test_改行は語間になる(self) -> None:
        assert to_sendable_kana("あ\nい").text == "ア イ"

    def test_連続した改行でも増えない(self) -> None:
        assert to_sendable_kana("あ\n\nい").text == "ア イ"

    def test_行数が増えても比例して増えない(self) -> None:
        """3 行なら 3 語。蓄積があると 6 語になる."""
        result = to_sendable_kana("あ\nい\nう")
        assert result.text == "ア イ ウ"

    def test_CRLF_でも同じ(self) -> None:
        assert to_sendable_kana("あ\r\nい").text == "ア イ"

    def test_末尾の改行は語を増やさない(self) -> None:
        assert to_sendable_kana("あ\n").text == "ア"

    def test_改行を含んでも送信できる(self) -> None:
        """改行そのものが「送れない文字」として残らないこと."""
        result = to_sendable_kana("こんにちは\nてんきです")
        assert result.sendable
        assert "\n" not in result.text


class TestUnknownMarkerSpacing:
    """``?`` は「値が無い」の印でもある (2026-08-12).

    差し込みで空の欄が ``?`` になるようにしたところ、句読点の整形が ``?`` の
    直前の空白を削っており、``ナマエ ハ ? デス`` が ``ナマエ ハ? デス`` に
    なっていた。**``ハ`` と ``?`` が語間なしで繋がり、別の語として届く。**
    """

    def test_疑問符の前の語間を残す(self) -> None:
        assert to_sendable_kana("ナマエ ハ ? デス").text == "ナマエ ハ ? デス"

    def test_欧文でも語間を残す(self) -> None:
        assert to_sendable_kana("? DE ? K").text == "? DE ? K"

    def test_空白なしで書いたものはそのまま(self) -> None:
        """``HW?`` は 1 語として書かれている。勝手に割らない."""
        assert to_sendable_kana("HW? K").text == "HW? K"

    def test_句読点は従来どおり詰める(self) -> None:
        """``?`` を外したせいで `、` `。` まで緩めていないこと."""
        assert to_sendable_kana("ハレ 、 アツイ").text == "ハレ、アツイ"
        assert to_sendable_kana("ハレ 。 アツイ").text == "ハレ。アツイ"

    def test_語間は符号の長さに効く(self) -> None:
        """語間は 7 単位あるので、消えると電波の長さが変わる."""
        from src.tx.encoder import build_sequence

        glued = build_sequence("ナマエ ハ? デス", 20.0).total_seconds
        spaced = build_sequence("ナマエ ハ ? デス", 20.0).total_seconds
        assert spaced > glued

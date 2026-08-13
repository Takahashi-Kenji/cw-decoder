"""送信テキスト → 符号列 → 要素列 のテスト."""
from __future__ import annotations

import numpy as np
import pytest

from src.tokens.morse_tokens import JAPANESE_CHAR_TO_CODES, WORD_BREAK_CODE, text_to_codes
from src.tx.encoder import (
    HORE,
    RATA,
    build_sequence,
    encode,
    find_unsendable,
    split_segments,
    wrap_japanese,
)
from src.tx.reading import to_sendable_kana


class TestSplitSegments:
    """和文の交信でもコールサインと RST は欧文で送る."""

    def test_plain_text_is_all_european(self) -> None:
        segments = split_segments("CQ DE JH0ILL")
        assert [s.mode for s in segments] == ["european"]

    def test_markers_switch_the_mode(self) -> None:
        segments = split_segments(f"JH0ILL {HORE}コンニチハ{RATA} K")
        assert [s.mode for s in segments] == ["european", "japanese", "european"]

    def test_hore_belongs_to_the_european_side(self) -> None:
        """`{HORE}` は欧文の符号として送られ、受信側が和文へ切り替える."""
        segments = split_segments(f"JH0ILL {HORE}アイ{RATA}")
        assert segments[0].text.endswith(HORE)
        assert segments[0].mode == "european"

    def test_rata_belongs_to_the_japanese_side(self) -> None:
        """`{RATA}` は和文側の終わりに付く (和文の符号として送り、欧文へ戻す)."""
        segments = split_segments(f"{HORE}アイ{RATA} K")
        japanese = [s for s in segments if s.mode == "japanese"]
        assert len(japanese) == 1
        assert japanese[0].text == f"アイ{RATA}"

    def test_empty(self) -> None:
        assert split_segments("") == []


class TestFindUnsendable:
    def test_european_text_is_fine(self) -> None:
        assert find_unsendable("CQ DE JH0ILL K") == ()

    def test_kana_inside_markers_is_fine(self) -> None:
        assert find_unsendable(f"{HORE}コンニチハ、テンキ{RATA}") == ()

    def test_kana_without_markers_is_accepted(self) -> None:
        """**マーカーが無くてもカタカナは送れる** (2026-08-11 の運用判断).

        既に和文モードで交信している最中は、毎回ホレを送り直す必要がない。
        符号は作れるのだから送れてよい、という運用者の判断による。
        """
        assert find_unsendable("コンニチハ") == ()

    def test_kanji_is_rejected(self) -> None:
        bad = find_unsendable(f"{HORE}ア晴{RATA}")
        assert [b.char for b in bad] == ["晴"]

    def test_position_points_at_the_original_text(self) -> None:
        text = f"{HORE}ア晴{RATA}"
        bad = find_unsendable(text)
        assert text[bad[0].index] == "晴"

    def test_lowercase_european_is_accepted(self) -> None:
        """符号化時に大文字化されるので小文字も送れる."""
        assert find_unsendable("cq de jh0ill") == ()

    def test_digits_in_both_modes(self) -> None:
        assert find_unsendable("599") == ()
        assert find_unsendable(f"{HORE}599{RATA}") == ()


class TestModeInference:
    """マーカーが無いときのモード推定.

    **モードは「どちらの符号表で読むか」であり、ホレ/ラタを送るかとは別物である。**
    以前は ``{HORE}`` を見つけるまで欧文のままだったため、囲みを外すとカタカナが
    まるごと「送れない文字」になっていた (2026-08-11 に実運用で判明)。

    **最初のマーカーより前の中身**でモードを決める。和文にしか無い文字が
    あれば和文、無ければ欧文。
    """

    def test_カタカナだけなら和文になる(self) -> None:
        assert [(s.text, s.mode) for s in split_segments("コンニチハ")] == [
            ("コンニチハ", "japanese")
        ]

    def test_欧文だけなら欧文のまま(self) -> None:
        assert [(s.text, s.mode) for s in split_segments("CQ DE JH0ILL K")] == [
            ("CQ DE JH0ILL K", "european")
        ]

    def test_数字だけなら欧文のまま(self) -> None:
        """数字は両方の表にある。**曖昧なものでモードを決めない。**"""
        assert [s.mode for s in split_segments("599")] == ["european"]

    def test_ラタだけで閉じる場合も和文で始まる(self) -> None:
        """和文の途中から書き始め、ラタで欧文に戻す使い方."""
        segments = [(s.text, s.mode) for s in split_segments(f"コンニチハ{RATA} K")]
        assert segments == [(f"コンニチハ{RATA}", "japanese"), (" K", "european")]

    def test_ホレがあれば従来どおり欧文で始まる(self) -> None:
        segments = [(s.text, s.mode) for s in split_segments(f"{HORE}ア{RATA}")]
        assert segments == [(HORE, "european"), (f"ア{RATA}", "japanese")]

    def test_マーカー無しの和文が符号になる(self) -> None:
        assert encode("コンニチハ") == encode(f"{HORE}コンニチハ{RATA}")[1:-1]

    def test_欧文と和文が混ざると欧文側が撥ねられる(self) -> None:
        """**黙って落とさない。** マーカーで分けるべき場面だと運用者に見せる."""
        bad = find_unsendable("JA1ABC コンニチハ")
        assert [b.char for b in bad] == list("JAABC")


class TestEuropeanSpan:
    """`「…」` は「ここは欧文で打つ」という印.

    運用者は実際の交信で `「FT991」` と書き、**欧文で短い単語を打つときは
    ホレ・ラタを使わない** (2026-08-12 の聞き取り)。

    * `「` は**出さない** (どちらの符号表にも無い)
    * 中身は**欧文の符号表**で送る
    * `」` は**和文の符号として出す** (区切りとして相手に届く)
    * 欧文の中では**括弧を両方落とす** (`」` は和文表にしかないため)
    """

    def test_和文の中の括弧が欧文区間になる(self) -> None:
        segments = [(s.text, s.mode) for s in split_segments(f"{HORE}リグ ハ 「FT991」 デス{RATA}")]
        assert ("FT991", "european") in segments

    def test_和文の中では閉じ括弧を段落で出す(self) -> None:
        """区間の閉じは**段落** (``・-・-・・``) として送る.

        2026-08-12 まで ``・-・-・・`` を ``」`` と誤記していたので、この
        テストも ``」`` を引いていた。**電波に出る符号は当時と同じ**で、
        名前だけが正しくなった (`src/tx/encoder.py` の ``DANRAKU``)。
        """
        codes = encode(f"{HORE}リグ ハ 「FT991」{RATA}")
        assert JAPANESE_CHAR_TO_CODES["。"][0] in codes

    def test_開き括弧は出さない(self) -> None:
        """どちらの符号表にも無いので出しようがない.

        `find_unsendable() == ()` だけでは「符号が増えていないか」までは
        分からない (エラーが出ないことしか見ていない、台帳が戒めているまさに
        その見方。2026-08-12 の最終レビューで指摘)。**欧文側の
        `test_欧文の中の括弧は符号を増やさない` に相当する encode の比較**を
        和文側にも足す。各語を `text_to_codes` で個別に符号化し、語間に
        `WORD_BREAK_CODE` を手で挟んだものと完全一致することを確かめる
        (`「` の分の符号が 1 つも増えていないこと、`」` がちょうど 1 つだけ
        出ることの両方を見る)。
        """
        codes = encode(f"{HORE}リグ ハ 「FT991」 デス{RATA}")
        expected = (
            text_to_codes(HORE, "european")
            + text_to_codes("リグ", "japanese")
            + [WORD_BREAK_CODE]
            + text_to_codes("ハ", "japanese")
            + [WORD_BREAK_CODE]
            + text_to_codes("FT991", "european")
            + text_to_codes("。", "japanese")   # 区間の閉じは段落
            + [WORD_BREAK_CODE]
            + text_to_codes("デス", "japanese")
            + text_to_codes(RATA, "japanese")
        )
        assert codes == expected
        assert find_unsendable(f"{HORE}リグ ハ 「FT991」 デス{RATA}") == ()

    def test_欧文の中では括弧を両方落とす(self) -> None:
        """`」` は和文表にしかない。欧文の文に残すと送れなくなる."""
        assert find_unsendable("RIG 「FT991」 ANT 「DP」 K") == ()

    def test_欧文の中の括弧は符号を増やさない(self) -> None:
        assert encode("RIG 「FT991」 K") == encode("RIG FT991 K")

    def test_単独の閉じ括弧は和文の終わり(self) -> None:
        """従来の使い方を壊さない."""
        assert find_unsendable(f"{HORE}コンニチハ」{RATA}") == ()

    def test_閉じていない括弧は印にしない(self) -> None:
        """**書き間違いを黙って通さない。** `「` が送れない文字として見える."""
        bad = find_unsendable(f"{HORE}リグ ハ 「FT991 デス{RATA}")
        assert "「" in "".join(b.char for b in bad)

    def test_中身が和文なら送れない(self) -> None:
        """欧文区間なのでカタカナが送れない. 黙って通さないこと."""
        bad = find_unsendable(f"{HORE}「コンニチハ」{RATA}")
        assert bad != ()

    def test_空の括弧(self) -> None:
        """中身が空なら `」` だけが出る.

        `inner` が空文字になるが、``split_segments`` の末尾で空文字列の
        segment だけを捨てる処理により消える。**落ちないことが要件**である。
        """
        assert find_unsendable(f"{HORE}アイ「」{RATA}") == ()
        assert encode(f"{HORE}アイ「」{RATA}") == encode(f"{HORE}アイ」{RATA}")

    def test_入れ子は考えない(self) -> None:
        """`「` から**次の** `」` までを 1 区間とする (設計書 §3.4).

        `「A「B」C」` は `A「B` が欧文区間になり、`C」` が続く。
        **区間の中に残った `「` は欧文表に無いので「送信できない文字」として見える。**
        入れ子を書くと気づけるということであり、これでよい。
        """
        segments = [(s.text, s.mode) for s in split_segments(f"{HORE}「A「B」C」{RATA}")]
        assert ("A「B", "european") in segments
        assert "「" in "".join(b.char for b in find_unsendable(f"{HORE}「A「B」C」{RATA}"))

    def test_括弧が複数あってもよい(self) -> None:
        assert find_unsendable(f"{HORE}リグ ハ 「FT991」 アンテナ ハ 「DP」{RATA}") == ()

    def test_ホレラタの振る舞いは変わらない(self) -> None:
        segments = [(s.text, s.mode) for s in split_segments(f"{HORE}ア{RATA}")]
        assert segments == [(HORE, "european"), (f"ア{RATA}", "japanese")]

    def test_区間の中にホレがまたがると印にしない(self) -> None:
        """`「…」` の中に ``{HORE}`` 等のマーカーが混じっているのは書き間違い.

        規則 5 (ホレ・ラタは区間の中では使わない) に反する書き方であり、
        黙って通さず「送信できない文字」として見せる (2026-08-12 のレビューで
        指摘された、過小評価していた懸念)。
        """
        bad = find_unsendable(f"「FT{HORE}991」")
        assert "「" in "".join(b.char for b in bad)


class TestEuropeanSpanWordGaps:
    """語間 (WORD_BREAK) が消えていないかを**所要秒数**で確かめる.

    符号列の要素数や `WORD_BREAK` の個数を数えても分からない
    (``encode("A B")`` は WORD_BREAK トークンを含まない。語間は
    ``build_element_sequence`` が要素の長さとして表現するため)。
    2026-08-12 のレビューで、`「…」` の直前・直後・連続する区間の間で
    語間が消える欠陥が 2 件見つかった (単体テストでは検知できず、
    経路をつないだテストで初めて見つかった)。
    """

    def test_変換を通しても語間が保たれる(self) -> None:
        """``to_sendable_kana`` の出力を ``build_sequence`` に通しても、
        手で書いた形と同じ秒数になること (実運用の経路: tx_dialog は
        ``to_sendable_kana()`` の出力をそのまま encoder に渡す)。"""
        converted = to_sendable_kana("リグ は 「FT991」 です").text
        handwritten = "リグ ハ 「FT991」 デス"
        assert build_sequence(converted, 20.0).total_seconds == pytest.approx(
            build_sequence(handwritten, 20.0).total_seconds
        )

    def test_連続する括弧の間の語間が保たれる(self) -> None:
        gap = build_sequence("「A」 「B」", 20.0).total_seconds
        with_gap = build_sequence("A B", 20.0).total_seconds
        without_gap = build_sequence("AB", 20.0).total_seconds
        assert gap == pytest.approx(with_gap)
        assert gap > without_gap

    def test_括弧の直後に語間がある場合(self) -> None:
        assert build_sequence("「A」 B", 20.0).total_seconds == pytest.approx(
            build_sequence("A B", 20.0).total_seconds
        )

    def test_括弧のすぐ内側の余白は語間にならない(self) -> None:
        """``「 A 」`` (括弧のすぐ内側に空白を書いた形) は ``「A」`` と同じ秒数になること.

        `「` は符号を出さないので、直前の語間の WORD_BREAK と `「` の直後の
        空白による WORD_BREAK が連続する。**実運用の UI は必ず
        ``to_sendable_kana`` を経由するので届かないが、``key_server`` に
        直接テキストが来る経路 (LAN 越しの直接入力等) では届く**
        (2026-08-12 の最終レビューで判明)。
        """
        padded = build_sequence("ア 「 A 」 イ", 20.0).total_seconds
        tight = build_sequence("ア 「A」 イ", 20.0).total_seconds
        assert padded == pytest.approx(tight)


class TestEncode:
    def test_produces_codes(self) -> None:
        codes = encode("CQ")
        assert codes == ["-・-・", "--・-"]

    def test_mixed_modes(self) -> None:
        codes = encode(f"K {HORE}ア{RATA}")
        assert len(codes) >= 4        # K, WORD_BREAK, HORE, ア, RATA

    def test_rejects_unsendable_before_encoding(self) -> None:
        """`text_to_codes` は表に無い文字で KeyError を投げる。
        その前に読める例外にして止める."""
        with pytest.raises(ValueError, match="送信できない文字"):
            encode("こんにちは")

    def test_empty(self) -> None:
        assert encode("") == []


class TestBuildSequence:
    def test_dot_length_follows_wpm(self) -> None:
        """20 WPM なら短点 60 ms."""
        seq = build_sequence("E", wpm=20.0)      # E は短点 1 つ
        assert seq.durations[0] == pytest.approx(0.06)

    def test_faster_is_shorter(self) -> None:
        slow = build_sequence("E", wpm=10.0)
        fast = build_sequence("E", wpm=40.0)
        assert slow.durations[0] > fast.durations[0]

    def test_dash_is_three_dots(self) -> None:
        seq = build_sequence("T", wpm=20.0)      # T は長音 1 つ
        assert seq.durations[0] == pytest.approx(0.18)

    def test_key_is_on_for_the_element(self) -> None:
        seq = build_sequence("E", wpm=20.0)
        assert bool(seq.is_on[0]) is True

    def test_lengths_match(self) -> None:
        seq = build_sequence("CQ DE K", wpm=20.0)
        assert len(seq.durations) == len(seq.is_on)

    def test_total_seconds(self) -> None:
        seq = build_sequence("CQ DE K", wpm=20.0)
        assert seq.total_seconds == pytest.approx(float(np.sum(seq.durations)))

    def test_word_gap_is_longer_than_char_gap(self) -> None:
        """語間 7 dot、字間 3 dot."""
        gaps = build_sequence("E E", wpm=20.0)
        off = gaps.durations[~gaps.is_on]
        assert off.max() == pytest.approx(0.42)      # 7 * 0.06

    def test_zero_wpm_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="wpm"):
            build_sequence("E", wpm=0.0)

    def test_empty_gives_an_empty_sequence(self) -> None:
        seq = build_sequence("", wpm=20.0)
        assert seq.durations.size == 0
        assert seq.total_seconds == 0.0

    def test_codes_are_kept_for_the_preview(self) -> None:
        """試聴と送信が同じ codes から作られることを担保する."""
        seq = build_sequence("CQ", wpm=20.0)
        assert seq.codes == ("-・-・", "--・-")


class TestWrapJapanese:
    def test_wraps(self) -> None:
        assert wrap_japanese("アイ") == f"{HORE}アイ{RATA}"

    def test_does_not_double_wrap(self) -> None:
        already = f"{HORE}アイ{RATA}"
        assert wrap_japanese(already) == already

    def test_empty_is_left_alone(self) -> None:
        assert wrap_japanese("") == ""

    def test_wrapped_text_is_sendable(self) -> None:
        """囲んでも送れる.

        **囲まなくても送れる** (:class:`TestModeInference`)。囲みの役目は
        「相手のデコーダに和文へ切り替えろと伝えること」であって、
        こちらが符号を作れるかどうかではない。
        """
        assert find_unsendable(wrap_japanese("コンニチハ")) == ()

    def test_wrapping_adds_the_prosigns(self) -> None:
        """囲むとホレとラタが**符号として増える** (相手の切替のため)."""
        assert len(encode(wrap_japanese("コンニチハ"))) == len(encode("コンニチハ")) + 2

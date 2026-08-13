"""無音による改行のテスト."""
from __future__ import annotations

import pytest

from src.infer.line_break import (
    DEFAULT_LINE_BREAK_GAP_S,
    render_committed,
    split_at_gaps,
)
from src.infer.sliding_window import CommittedToken
from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import EUROPEAN_CHAR_TO_CODE, TOKEN_TO_ID

SR = 8000


def _tokens(chars: str, gaps_s: list[float]) -> list[CommittedToken]:
    """各トークン 0.1 秒、間に指定秒の無音を挟んだ確定トークン列を作る."""
    dur = int(0.1 * SR)
    out: list[CommittedToken] = []
    cursor = 0
    for i, ch in enumerate(chars):
        if i > 0:
            cursor += int(gaps_s[i - 1] * SR)
        code = EUROPEAN_CHAR_TO_CODE[ch]
        out.append(
            CommittedToken(
                token_id=TOKEN_TO_ID[code],
                confidence=1.0,
                absolute_sample_start=cursor,
                absolute_sample_end=cursor + dur,
            )
        )
        cursor += dur
    return out


class TestSplitAtGaps:
    def test_empty_returns_empty(self) -> None:
        assert split_at_gaps([], int(3.0 * SR)) == []

    def test_below_threshold_stays_one_segment(self) -> None:
        segs = split_at_gaps(_tokens("ABC", [2.9, 0.5]), int(3.0 * SR))
        assert len(segs) == 1

    def test_at_threshold_splits(self) -> None:
        segs = split_at_gaps(_tokens("ABC", [3.0, 0.5]), int(3.0 * SR))
        assert [len(s) for s in segs] == [1, 2]

    def test_multiple_splits(self) -> None:
        segs = split_at_gaps(_tokens("ABC", [5.0, 5.0]), int(3.0 * SR))
        assert [len(s) for s in segs] == [1, 1, 1]

    def test_zero_gap_disables_splitting(self) -> None:
        segs = split_at_gaps(_tokens("ABC", [10.0, 10.0]), 0)
        assert len(segs) == 1


class TestRenderCommitted:
    def _conv(self) -> TokenConverter:
        return TokenConverter(mode="european", confidence_threshold=0.5)

    def test_no_break_below_threshold(self) -> None:
        text, _ = render_committed(
            _tokens("ABC", [2.9, 0.5]), self._conv(), int(3.0 * SR)
        )
        assert "\n" not in text
        assert text == "ABC"

    def test_breaks_at_gap(self) -> None:
        text, _ = render_committed(
            _tokens("ABC", [3.0, 0.5]), self._conv(), int(3.0 * SR)
        )
        assert text.split("\n") == ["A", "BC"]

    def test_stable_across_recomputation(self) -> None:
        """同じ入力を 2 回変換すると同じ結果になる (作り直しても確定が動かない)."""
        tokens = _tokens("ABC", [4.0, 0.5])
        a, _ = render_committed(tokens, self._conv(), int(3.0 * SR))
        b, _ = render_committed(tokens, self._conv(), int(3.0 * SR))
        assert a == b

    def test_append_only_as_tokens_grow(self) -> None:
        """トークンが増えても既存の行は変わらない (確定は追記のみ)."""
        tokens = _tokens("AB", [4.0])
        before, _ = render_committed(tokens, self._conv(), int(3.0 * SR))
        grown = tokens + _tokens("C", [])
        # 3 つ目を末尾に繋ぎ直す (時刻を連続させる)
        last = tokens[-1]
        grown = tokens + [
            CommittedToken(
                token_id=grown[-1].token_id,
                confidence=1.0,
                absolute_sample_start=last.absolute_sample_end + int(0.5 * SR),
                absolute_sample_end=last.absolute_sample_end + int(0.6 * SR),
            )
        ]
        after, _ = render_committed(grown, self._conv(), int(3.0 * SR))
        assert after.startswith(before)

    def test_mode_is_carried_across_segments(self) -> None:
        """自動モードで、区間をまたいでも和文モードが続く.

        引き継がないと区切りのたびに欧文へ戻ってしまう。
        """
        from src.tokens.morse_tokens import HORE_CODE, JAPANESE_CHAR_TO_CODES

        conv = TokenConverter(mode="auto", confidence_threshold=0.5)
        # ホレ (和文開始) → 4 秒無音 → 和文の「イ」
        i_code = JAPANESE_CHAR_TO_CODES["イ"][0]
        tokens = [
            CommittedToken(TOKEN_TO_ID[HORE_CODE], 1.0, 0, int(0.1 * SR)),
            CommittedToken(
                TOKEN_TO_ID[i_code], 1.0, int(4.1 * SR), int(4.2 * SR)
            ),
        ]
        text, final_mode = render_committed(tokens, conv, int(3.0 * SR))
        assert final_mode == "japanese"
        # 区切り後も和文として解釈されている (欧文なら 'A' になる)
        assert text.split("\n")[-1] == "イ"

    def test_default_threshold_is_three_seconds(self) -> None:
        assert DEFAULT_LINE_BREAK_GAP_S == pytest.approx(3.0)


class TestDanrakuBreak:
    """**段落 (``。``) でも行を分ける** (運用者の要望、2026-08-12).

    無音による区切りは 3 秒以上の間が空いたときにしか入らず、**1 回の送信が
    長いと 1 行に延々と続く**。段落は文の終わりを表す実在の符号なので、
    これで分ければ送信の途中でも読みやすい区切りが入る。

    ``、`` (区切点) では分けない。文の途中に何度も現れるので、分けると
    1 文が細切れになる。
    """

    @staticmethod
    def _japanese(chars: str) -> tuple[list[CommittedToken], TokenConverter]:
        """和文トークン列を作る (無音は挟まない = 無音による改行は起きない)."""
        from src.tokens.morse_tokens import JAPANESE_CHAR_TO_CODES

        dur = int(0.1 * SR)
        out: list[CommittedToken] = []
        for i, ch in enumerate(chars):
            code = JAPANESE_CHAR_TO_CODES[ch][0]
            out.append(
                CommittedToken(TOKEN_TO_ID[code], 1.0, i * dur, (i + 1) * dur)
            )
        return out, TokenConverter(mode="japanese", confidence_threshold=0.5)

    def test_段落で行が分かれる(self) -> None:
        tokens, conv = self._japanese("アイ。ウエ")
        text, _ = render_committed(tokens, conv, int(3.0 * SR))
        assert text == "アイ。\nウエ"

    def test_末尾の段落で空行を作らない(self) -> None:
        tokens, conv = self._japanese("アイ。")
        text, _ = render_committed(tokens, conv, int(3.0 * SR))
        assert text == "アイ。"

    def test_区切点では分けない(self) -> None:
        """``、`` は文の途中に何度も出るので分けない."""
        tokens, conv = self._japanese("アイ、ウエ")
        text, _ = render_committed(tokens, conv, int(3.0 * SR))
        assert text == "アイ、ウエ"

    def test_連続する段落で空行を作らない(self) -> None:
        tokens, conv = self._japanese("ア。。イ")
        text, _ = render_committed(tokens, conv, int(3.0 * SR))
        assert text == "ア。\n。\nイ"

    def test_無音の改行と両立する(self) -> None:
        """無音による改行と段落による改行が重なっても空行を作らない."""
        from src.tokens.morse_tokens import JAPANESE_CHAR_TO_CODES

        dur = int(0.1 * SR)
        codes = [JAPANESE_CHAR_TO_CODES[c][0] for c in "ア。イ"]
        # 「ア」「。」の後に 4 秒空けてから「イ」
        tokens = [
            CommittedToken(TOKEN_TO_ID[codes[0]], 1.0, 0, dur),
            CommittedToken(TOKEN_TO_ID[codes[1]], 1.0, dur, 2 * dur),
            CommittedToken(TOKEN_TO_ID[codes[2]], 1.0, 2 * dur + 4 * SR, 3 * dur + 4 * SR),
        ]
        conv = TokenConverter(mode="japanese", confidence_threshold=0.5)
        text, _ = render_committed(tokens, conv, int(3.0 * SR))
        assert text == "ア。\nイ"

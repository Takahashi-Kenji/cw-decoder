"""清書をいつ・どこまで送るかの判定テスト (2 モード)."""
from __future__ import annotations

from src.llm.auto import (
    BOUNDARY_CHARS,
    DEFAULT_LEAD_CHARS,
    AutoRefineState,
    lead_text,
    pending_text,
    plan_auto_refine,
    plan_refine_all,
    should_refine,
)


class TestPendingText:
    """一度清書した分は送り直さない (これが増分方式の要)."""

    def test_all_is_pending_at_first(self) -> None:
        assert pending_text("CQ DE JA1ABC", AutoRefineState()) == "CQ DE JA1ABC"

    def test_only_the_new_part_is_pending(self) -> None:
        state = AutoRefineState(refined_len=len("CQ DE "))
        assert pending_text("CQ DE JA1ABC", state) == "JA1ABC"

    def test_nothing_pending_when_fully_refined(self) -> None:
        text = "CQ DE JA1ABC"
        assert pending_text(text, AutoRefineState(refined_len=len(text))) == ""

    def test_shrunk_text_is_fully_pending(self) -> None:
        """クリアやモード変更で確定テキストが作り直されたら最初から清書し直す."""
        assert pending_text("SHORT", AutoRefineState(refined_len=100)) == "SHORT"


class TestLeadText:
    """未清書分の直前を文脈として添える (話が繋がらないと清書できない)."""

    def test_empty_at_first(self) -> None:
        assert lead_text("CQ DE JA1ABC", AutoRefineState()) == ""

    def test_returns_text_before_the_pending_part(self) -> None:
        state = AutoRefineState(refined_len=len("CQ DE "))
        assert lead_text("CQ DE JA1ABC", state) == "CQ DE "

    def test_is_capped(self) -> None:
        """長すぎると小さいモデルが参考部分まで出力してしまう."""
        state = AutoRefineState(refined_len=500)
        assert len(lead_text("あ" * 500, state)) == DEFAULT_LEAD_CHARS

    def test_does_not_overrun_a_shrunk_text(self) -> None:
        assert lead_text("SHORT", AutoRefineState(refined_len=100)) == "SHORT"


class TestPlanRefineAll:
    """まとめて清書 (手動ボタン)."""

    def test_sends_everything_pending_in_one_call(self) -> None:
        request = plan_refine_all("コンニチハ\nテンキ ハ ハレ", AutoRefineState())
        assert request is not None
        assert request.text == "コンニチハ\nテンキ ハ ハレ"
        assert request.reason == "manual"

    def test_does_not_resend_refined_part(self) -> None:
        state = AutoRefineState(refined_len=len("コンニチハ\n"))
        request = plan_refine_all("コンニチハ\nテンキ ハ ハレ", state)
        assert request is not None
        assert request.text == "テンキ ハ ハレ"
        assert request.lead == "コンニチハ\n"

    def test_none_when_nothing_pending(self) -> None:
        text = "コンニチハ"
        assert plan_refine_all(text, AutoRefineState(refined_len=len(text))) is None

    def test_none_for_whitespace_only(self) -> None:
        assert plan_refine_all("   \n  ", AutoRefineState()) is None

    def test_marks_everything_as_sent(self) -> None:
        text = "コンニチハ\nテンキ"
        request = plan_refine_all(text, AutoRefineState())
        assert request is not None
        assert request.refined_len == len(text)


class TestPlanAutoRefineOnLineBreak:
    """改行 = 送信の切れ目。そこで区切ると文が完結する."""

    def _plan(self, text, state=None, now=100.0, interval=20.0):
        return plan_auto_refine(
            text, state or AutoRefineState(), now=now, interval_s=interval
        )

    def test_fires_on_a_line_break(self) -> None:
        request = self._plan("コンニチハ\nテンキ ハ")
        assert request is not None
        assert request.reason == "line_break"
        assert request.text == "コンニチハ"

    def test_leaves_the_unfinished_line(self) -> None:
        """行の途中は残す (文の途中で切って清書させない)."""
        request = self._plan("コンニチハ\nテンキ ハ")
        assert request is not None
        assert "テンキ" not in request.text
        assert request.refined_len == len("コンニチハ\n")

    def test_sends_up_to_the_last_line_break(self) -> None:
        """1 回の更新で複数行ぶん確定したらまとめて送る."""
        request = self._plan("イチ\nニ\nサン")
        assert request is not None
        assert request.text == "イチ\nニ"
        assert request.refined_len == len("イチ\nニ\n")

    def test_line_break_ignores_the_interval(self) -> None:
        """切れ目はもともと数秒空いているので待たせない."""
        state = AutoRefineState(refined_len=0, last_time=100.0)
        request = self._plan("コンニチハ\nテンキ", state, now=100.1, interval=20.0)
        assert request is not None
        assert request.reason == "line_break"

    def test_carries_the_previous_line_as_context(self) -> None:
        state = AutoRefineState(refined_len=len("コンニチハ\n"))
        request = self._plan("コンニチハ\nテンキ ハ ハレ\nサヨウナラ", state)
        assert request is not None
        assert request.text == "テンキ ハ ハレ"
        assert request.lead == "コンニチハ\n"

    def test_no_refire_for_an_already_sent_line(self) -> None:
        text = "コンニチハ\n"
        state = AutoRefineState(refined_len=len(text))
        assert self._plan(text, state) is None

    def test_blank_line_alone_does_not_fire(self) -> None:
        assert self._plan("\n\n") is None


class TestPlanAutoRefineFallback:
    """改行が来ない長い送信のための保険."""

    def _plan(self, text, state=None, now=100.0, interval=20.0):
        return plan_auto_refine(
            text, state or AutoRefineState(), now=now, interval_s=interval
        )

    def test_first_send_skips_the_interval(self) -> None:
        request = self._plan("コンニチハ", now=0.5)
        assert request is not None
        assert request.reason == "interval"

    def test_waits_for_the_interval_without_a_line_break(self) -> None:
        state = AutoRefineState(refined_len=0, last_time=100.0)
        assert self._plan("コンニチハ", state, now=110.0, interval=20.0) is None

    def test_fires_after_the_interval(self) -> None:
        state = AutoRefineState(refined_len=0, last_time=100.0)
        request = self._plan("コンニチハ", state, now=120.0, interval=20.0)
        assert request is not None
        assert request.reason == "interval"

    def test_nothing_to_send(self) -> None:
        text = "コンニチハ"
        assert self._plan(text, AutoRefineState(refined_len=len(text))) is None


class TestShouldRefine:
    """間隔判定そのもの (保険経路で使う)."""

    def test_no_refine_without_pending(self) -> None:
        assert should_refine(
            has_pending=False, now=100.0, interval_s=20.0,
            state=AutoRefineState(last_time=0.0),
        ) is False

    def test_first_refine_skips_the_interval(self) -> None:
        assert should_refine(
            has_pending=True, now=0.5, interval_s=20.0, state=AutoRefineState(),
        ) is True

    def test_interval_boundary_is_inclusive(self) -> None:
        assert should_refine(
            has_pending=True, now=20.0, interval_s=20.0,
            state=AutoRefineState(last_time=0.0),
        ) is True


class TestIncrementalFlow:
    """1 回の QSO を通した流れ: 同じ文を二度と送らないこと."""

    def test_a_line_is_sent_only_once(self) -> None:
        state = AutoRefineState()
        text = "コンニチハ\nテンキ ハ ハレ"
        first = plan_auto_refine(text, state, now=0.0, interval_s=20.0)
        assert first is not None and first.text == "コンニチハ"
        state.refined_len, state.last_time = first.refined_len, 0.0

        text = "コンニチハ\nテンキ ハ ハレ\nサヨウナラ"
        second = plan_auto_refine(text, state, now=1.0, interval_s=20.0)
        assert second is not None
        assert second.text == "テンキ ハ ハレ"
        assert "コンニチハ" not in second.text        # 送り直していない
        assert second.lead == "コンニチハ\n"          # 参考としては渡す


class TestPunctuationBoundaries:
    """改行だけでは滅多に発火しないので、和文の区切点でも清書する.

    ``、`` は区切点、``」`` は終わり。どちらも符号表にある実在の文字。
    """

    def _plan(self, text, state=None, now=100.1, interval=20.0):
        return plan_auto_refine(
            text, state or AutoRefineState(last_time=100.0),
            now=now, interval_s=interval,
        )

    def test_fires_on_a_comma(self) -> None:
        request = self._plan("テンキ ハ ハレ、アツイ デス")
        assert request is not None
        assert request.reason == "punctuation"
        assert request.text == "テンキ ハ ハレ、"

    def test_fires_on_the_end_mark(self) -> None:
        """段落 (`。`) でも発火すること.

        2026-08-12 まで `」` と書いていた。`・-・-・・` を `」` と誤記していた
        ためで、**同じ符号を指している**。デコード結果の表記が `。` に
        変わったので、境界もそれに合わせないと文末で発火しなくなる。
        """
        request = self._plan("コンニチハ。ドウゾ")
        assert request is not None
        assert request.text == "コンニチハ。"

    def test_punctuation_is_kept_in_the_text(self) -> None:
        """句読点は文の一部なので送る (改行と違って落とさない)."""
        request = self._plan("テンキ ハ ハレ、アツイ")
        assert request is not None
        assert request.text.endswith("、")

    def test_newline_is_not_sent(self) -> None:
        """改行は行の区切りなので本文には入れない."""
        request = self._plan("コンニチハ\nテンキ")
        assert request is not None
        assert request.reason == "line_break"
        assert "\n" not in request.text

    def test_uses_the_last_boundary(self) -> None:
        """1 回の更新で複数の区切りが確定したらまとめて送る."""
        request = self._plan("イチ、ニ、サン")
        assert request is not None
        assert request.text == "イチ、ニ、"

    def test_waits_without_a_boundary(self) -> None:
        assert self._plan("テンキ ハ ハレ") is None

    def test_question_mark_is_not_a_boundary(self) -> None:
        """`?` は「読めなかった」印としても出る。境界にすると出まくる."""
        assert "?" not in BOUNDARY_CHARS
        assert self._plan("ドウゾ?モウ イチド") is None

    def test_punctuation_ignores_the_interval(self) -> None:
        state = AutoRefineState(refined_len=0, last_time=100.0)
        request = self._plan("テンキ ハ ハレ、アツイ", state, now=100.01)
        assert request is not None
        assert request.reason == "punctuation"

    def test_advances_past_the_punctuation(self) -> None:
        text = "テンキ ハ ハレ、アツイ"
        request = self._plan(text)
        assert request is not None
        assert request.refined_len == text.index("、") + 1

    def test_leading_punctuation_alone_does_not_fire(self) -> None:
        assert self._plan("、") is None

    def test_second_call_does_not_resend(self) -> None:
        state = AutoRefineState(last_time=100.0)
        text = "テンキ ハ ハレ、アツイ デス"
        first = self._plan(text, state)
        assert first is not None
        state.refined_len = first.refined_len

        text = "テンキ ハ ハレ、アツイ デス、シカタ ナイ"
        second = self._plan(text, state)
        assert second is not None
        assert second.text == "アツイ デス、"
        assert "テンキ" not in second.text
        assert second.lead == "テンキ ハ ハレ、"

    def test_punctuation_only_chunk_is_not_sent(self) -> None:
        """`、` だけを清書させても意味がなく、LLM の往復が無駄になる."""
        assert self._plan("、、　") is None

    def test_still_fires_when_there_is_content(self) -> None:
        request = self._plan("、ハレ、")
        assert request is not None
        assert "ハレ" in request.text

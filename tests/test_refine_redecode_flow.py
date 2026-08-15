"""清書前の全体再デコードの流れをテストする.

**方針** (運用者、2026-08-14): 交信しながら読む側と、記録として整える側を分ける。
短時間で読みたいのは交信のため、全体が欲しいのは LLM で正規化するため。

したがって再デコードの結果は**画面の確定テキストを置き換えない**。
清書 (LLM) の入力にだけ使う。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.infer.refine_buffer import RefineBuffer


class _Recorder:
    """再デコード要求と LLM 要求を記録する差し替え."""

    def __init__(self) -> None:
        self.redecode: list[tuple] = []
        self.llm: list[tuple] = []


class _FakeWindow:
    """``MainWindow`` の清書経路だけを取り出した最小の模型.

    Qt を起動せずに「どちらの経路を通るか」「どこまで清書済みか」を検証する。
    実物の分岐と同じ順序で書いてある。
    """

    def __init__(self, *, redecode_enabled: bool, has_model: bool) -> None:
        self.redecode_enabled = redecode_enabled
        self.has_model = has_model
        self.buffer = RefineBuffer(capacity_s=10.0, sample_rate=100)
        self.refined_until = 0
        self.pending_mode: str | None = None
        self.pending_until: int | None = None
        self.pending_replace = False
        self.refined_html: list[str] = []
        self.rec = _Recorder()

    def start_redecode_refine(self, *, whole: bool) -> bool:
        if self.pending_mode is not None:      # 再デコード中は次を出さない
            return True
        if not self.redecode_enabled or not self.has_model:
            return False
        audio, start = self.buffer.snapshot(None if whole else self.refined_until)
        if audio.size == 0:
            return False
        self.pending_mode = "replace" if whole else "append"
        self.rec.redecode.append((audio.size, start + audio.size, whole))
        return True

    def on_redecode_result(self, text: str, end_sample: int) -> None:
        mode = self.pending_mode
        self.pending_mode = None
        if mode is None or not text.strip():
            return
        self.pending_until = end_sample
        self.pending_replace = mode == "replace"
        self.rec.llm.append((text, end_sample))

    def on_llm_result(self, text: str) -> None:
        cleaned = text.strip()
        if self.pending_replace:
            self.refined_html = [cleaned] if cleaned else []
        elif cleaned:
            self.refined_html.append(cleaned)
        if self.pending_until is not None:
            self.refined_until = self.pending_until
            self.pending_until = None
        self.pending_replace = False


@pytest.fixture()
def window() -> _FakeWindow:
    w = _FakeWindow(redecode_enabled=True, has_model=True)
    w.buffer.push(np.ones(500, dtype=np.float32))
    return w


class TestWhichPathIsTaken:
    def test_disabled_falls_back(self) -> None:
        """再デコードを切ってあれば従来経路に落ちること."""
        w = _FakeWindow(redecode_enabled=False, has_model=True)
        w.buffer.push(np.ones(100, dtype=np.float32))
        assert w.start_redecode_refine(whole=False) is False

    def test_without_model_falls_back(self) -> None:
        """**学習済みモデルが無ければ使わない。** 未学習エンジンで読み直しても
        意味が無く、確定テキストをそのまま送る方がまし."""
        w = _FakeWindow(redecode_enabled=True, has_model=False)
        w.buffer.push(np.ones(100, dtype=np.float32))
        assert w.start_redecode_refine(whole=False) is False

    def test_without_audio_falls_back(self) -> None:
        w = _FakeWindow(redecode_enabled=True, has_model=True)
        assert w.start_redecode_refine(whole=False) is False


class TestAutoRefineSendsOnlyTheNewPart:
    """運用者: 「自動清書なら既に清書済みは不要」."""

    def test_first_time_sends_everything(self, window: _FakeWindow) -> None:
        assert window.start_redecode_refine(whole=False) is True
        size, end, whole = window.rec.redecode[0]
        assert size == 500 and end == 500 and whole is False

    def test_second_time_sends_only_the_increment(self, window: _FakeWindow) -> None:
        window.start_redecode_refine(whole=False)
        window.on_redecode_result("テンキ ハ ハレ", 500)
        window.on_llm_result("天気は晴れ")
        assert window.refined_until == 500

        window.buffer.push(np.ones(200, dtype=np.float32))
        window.start_redecode_refine(whole=False)
        size, end, _ = window.rec.redecode[-1]
        assert size == 200, "既に清書した区間まで送り直している"
        assert end == 700

    def test_results_are_appended(self, window: _FakeWindow) -> None:
        window.start_redecode_refine(whole=False)
        window.on_redecode_result("A", 500)
        window.on_llm_result("あ")
        window.buffer.push(np.ones(100, dtype=np.float32))
        window.start_redecode_refine(whole=False)
        window.on_redecode_result("B", 600)
        window.on_llm_result("い")
        assert window.refined_html == ["あ", "い"]


class TestRefineAllReplaces:
    """運用者: 「まとめての場合は全部で OK」.

    全体を送るので**積み上げてはいけない** (同じ内容が二重に並ぶ)。
    """

    def test_sends_the_whole_buffer(self, window: _FakeWindow) -> None:
        window.refined_until = 300          # 途中まで清書済みでも
        assert window.start_redecode_refine(whole=True) is True
        size, end, whole = window.rec.redecode[0]
        assert size == 500 and whole is True, "全体を送っていない"

    def test_result_replaces_instead_of_appending(self, window: _FakeWindow) -> None:
        window.refined_html = ["古い清書", "もっと古い清書"]
        window.start_redecode_refine(whole=True)
        window.on_redecode_result("全部", 500)
        window.on_llm_result("整えた全文")
        assert window.refined_html == ["整えた全文"]


class TestAutoRefineRespectsTheInterval:
    """**送る契機の判定は文字位置で行われる。**

    ``plan_auto_refine`` は「未清書分に区切り文字 (``\\n`` ``、`` ``。``) が
    あるか」で送る。間隔はその保険にすぎない。再デコード経路で文字位置を
    進めないと**未清書分が常に全文のまま**になり、和文は区切り文字を多く
    含むので毎回成立して連続で走る (2026-08-15 に運用者が「2 秒間隔で動く」
    と報告した不具合)。
    """

    def test_character_position_also_advances(self) -> None:
        from src.llm.auto import AutoRefineState, plan_auto_refine

        state = AutoRefineState()
        committed = "テンキ ハ ハレ、アツイ デス。"

        # 再デコード経路でも文字位置を進めていれば、同じ本文では再送しない
        state.refined_len = len(committed)
        state.last_time = 1000.0
        assert plan_auto_refine(
            committed, state, now=1001.0, interval_s=20.0
        ) is None, "同じ本文で送り直している"

    def test_not_advancing_would_fire_every_time(self) -> None:
        """**進めないとどうなるか** を明示しておく (回帰の意味を残すため)."""
        from src.llm.auto import AutoRefineState, plan_auto_refine

        state = AutoRefineState()          # refined_len = 0 のまま
        state.last_time = 1000.0
        committed = "テンキ ハ ハレ、アツイ デス。"
        assert plan_auto_refine(
            committed, state, now=1001.0, interval_s=20.0
        ) is not None, "区切り文字があるので間隔を待たずに成立する"


class TestOnlyOneRedecodeInFlight:
    """**再デコード中は次を出さない.**

    読み直しの往復の間は LLM がまだ動いていないので ``_llm_busy`` は False。
    これが無いと要求が積み重なる。
    """

    def test_second_request_is_ignored(self, window: _FakeWindow) -> None:
        assert window.start_redecode_refine(whole=False) is True
        first = len(window.rec.redecode)
        # 応答が返る前にもう一度呼ばれる (hop ごとに確定テキストが伸びるため)
        window.start_redecode_refine(whole=False)
        assert len(window.rec.redecode) == first, "要求が積み重なっている"

    def test_next_request_works_after_the_result(self, window: _FakeWindow) -> None:
        window.start_redecode_refine(whole=False)
        window.on_redecode_result("テンキ", 500)
        window.on_llm_result("天気")
        window.buffer.push(np.ones(100, dtype=np.float32))
        assert window.start_redecode_refine(whole=False) is True


class TestProgressIsTrackedByTime:
    """**清書済みの位置は時間で持つ。**

    再デコードするとテキストが作り直されるので、文字位置では管理できない。
    """

    def test_position_advances_only_after_the_llm_replies(
        self, window: _FakeWindow
    ) -> None:
        window.start_redecode_refine(whole=False)
        window.on_redecode_result("テンキ", 500)
        assert window.refined_until == 0, "LLM の応答前に進めている"
        window.on_llm_result("天気")
        assert window.refined_until == 500

    def test_empty_redecode_does_not_advance(self, window: _FakeWindow) -> None:
        """読み直した結果が空なら清書済みにしない (取りこぼしを防ぐ)."""
        window.start_redecode_refine(whole=False)
        window.on_redecode_result("   ", 500)
        assert window.rec.llm == []
        assert window.refined_until == 0

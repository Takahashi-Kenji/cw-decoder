"""主画面の構成を固定するテスト.

**運用者は実機で確認できないことがある** (リモート作業)。起動時に落ちる類の
失敗をテストで捕まえられるようにしておく。

見るのは 3 つ。

* **窓が組み上がること** — 参照漏れで ``AttributeError`` にならない
* **主画面に残すものが残っていること** — 交信中に触るものが消えていない
* **設定画面へ移したものが画面に出ていないこと** — コンパクトさの担保
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from src.app.main_window import CWDecoderWindow      # noqa: E402
from src.infer.engine import InferenceEngine         # noqa: E402
from src.infer.settings import AppSettings           # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp, tmp_path):
    engine = InferenceEngine.untrained("cpu")
    win = CWDecoderWindow(
        engine, AppSettings(), config_path=tmp_path / "settings.json"
    )
    yield win
    win.close()


class TestItBuilds:
    def test_window_is_constructed(self, window) -> None:
        """**参照漏れで落ちないこと。** ここが赤なら起動もしない."""
        assert window.windowTitle() == "CW デコーダ"


class TestKeptOnTheMainWindow:
    """交信中に触るものは主画面に残す (運用者の指定)."""

    @pytest.mark.parametrize(
        "name",
        [
            "mode_combo",          # モード
            "run_btn",             # 開始・停止 (統合)
            "decode_toggle_btn",   # デコード開始・停止
            "clear_decode_btn",    # クリア
            "record_btn",          # 録音開始
            "tx_btn",              # 送信
            "llm_refine_btn",      # まとめて清書
            "llm_clear_btn",       # 清書クリア
            "settings_btn",        # 設定…
            "level_meter",         # レベルメータ + スケルチ
            "spectrogram_panel",   # スペクトル + 濃さ・幅のスライダ
            "wpm_label",           # 受信 WPM
        ],
    )
    def test_visible(self, window, name: str) -> None:
        widget = getattr(window, name)
        assert widget is not None
        assert not widget.isHidden(), f"{name} が主画面から消えている"


class TestMovedToSettings:
    """設定画面へ移したものは主画面に出さない (コンパクトさの担保)."""

    @pytest.mark.parametrize(
        "name",
        [
            "threshold_slider",
            "show_spectrogram_check",
            "show_provisional_check",
            "word_correct_check",
            "word_correct_ja_check",
            "two_stage_check",
            "refine_redecode_check",
            "bpf_check",
            "bpf_center_spin",
            "bpf_bw_spin",
            "llm_provider_combo",
            "llm_model_edit",
            "llm_auto_check",
            "llm_compact_check",
            "llm_highlight_check",
            "start_btn",
            "stop_btn",
        ],
    )
    def test_hidden(self, window, name: str) -> None:
        widget = getattr(window, name)
        assert widget.isHidden(), f"{name} がまだ主画面に出ている"


class TestRunButton:
    """**開始と停止は 1 つのボタン** (運用者の要望)."""

    def test_label_reflects_the_state(self, window) -> None:
        assert window.run_btn.text() == "● 開始"
        window._sync_run_button(True)
        assert window.run_btn.text() == "■ 停止"
        assert window.run_btn.isChecked()
        window._sync_run_button(False)
        assert window.run_btn.text() == "● 開始"
        assert not window.run_btn.isChecked()

    def test_sync_does_not_retrigger(self, window) -> None:
        """外から状態を合わせるときに開始・停止を呼び直さないこと.

        呼び直すと停止 → 開始 → 停止 … と往復する。
        """
        calls: list[bool] = []
        window._on_run_toggled = lambda checked: calls.append(checked)
        window._sync_run_button(True)
        assert calls == []


class TestHiddenWidgetsFollowTheSettings:
    """**非表示にしたウィジェットも設定と揃えること.**

    ``_save_settings`` はそれらから値を読み戻すので、更新し忘れると
    設定画面で変えた値がその場で古い値に巻き戻る。2026-08-15 に実際に
    「チェックを外せない」「開き直すと戻っている」という形で表面化した。
    """

    BOOLEANS = [
        ("show_spectrogram", "show_spectrogram_check"),
        ("show_provisional", "show_provisional_check"),
        ("word_correct_enabled", "word_correct_check"),
        ("word_correct_ja_enabled", "word_correct_ja_check"),
        ("two_stage_commit_enabled", "two_stage_check"),
        ("refine_redecode_enabled", "refine_redecode_check"),
        ("bpf_enabled", "bpf_check"),
        ("llm_auto", "llm_auto_check"),
        ("llm_compact_prompt", "llm_compact_check"),
        ("llm_highlight_guesses", "llm_highlight_check"),
    ]

    @pytest.mark.parametrize(("field", "widget"), BOOLEANS)
    def test_widget_follows_the_setting(self, window, field: str, widget: str) -> None:
        for value in (False, True, False):
            setattr(window._settings, field, value)
            window._apply_settings_to_widgets()
            assert getattr(window, widget).isChecked() is value, field

    @pytest.mark.parametrize(("field", "widget"), BOOLEANS)
    def test_saving_does_not_revert_the_setting(
        self, window, field: str, widget: str
    ) -> None:
        """**保存が値を巻き戻さないこと** (これが今回の不具合の本体)."""
        original = getattr(window._settings, field)
        flipped = not original
        setattr(window._settings, field, flipped)
        window._apply_settings_to_widgets()
        window._save_settings()
        assert getattr(window._settings, field) is flipped, (
            f"{field} が保存で元に戻っている"
        )

    def test_numeric_settings_survive_saving(self, window) -> None:
        window._settings.confidence_threshold = 0.35
        window._settings.bpf_center_hz = 700.0
        window._settings.bpf_bandwidth_hz = 250.0
        window._apply_settings_to_widgets()
        window._save_settings()
        assert window._settings.confidence_threshold == pytest.approx(0.35)
        assert window._settings.bpf_center_hz == pytest.approx(700.0)
        assert window._settings.bpf_bandwidth_hz == pytest.approx(250.0)

    def test_llm_settings_survive_saving(self, window) -> None:
        window._settings.llm_provider = "claude"
        window._settings.llm_model = "claude-haiku-4-5"
        window._apply_settings_to_widgets()
        window._save_settings()
        assert window._settings.llm_provider == "claude"
        assert window._settings.llm_model == "claude-haiku-4-5"


class TestSettingsRoundTrip:
    def test_dialog_result_is_applied(self, window) -> None:
        """設定画面の結果が反映され、保留項目が数えられること."""
        before = window._settings
        after = type(before)(**{**vars(before), "commit_lag_s": 2.5})
        deferred = window._deferred_setting_names(before, after)
        assert "確定までの待ち" in deferred

    def test_no_change_means_nothing_deferred(self, window) -> None:
        s = window._settings
        assert window._deferred_setting_names(s, s) == []

    def test_two_stage_commit_is_deferred(self, window) -> None:
        """2 段階確定は ``SlidingWindowDecoder`` を作るときに固まる.

        受信中に変えても効かないので、通知にも設定画面の印にも出す
        (印の側は ``tests/test_settings_dialog.py`` が見ている)。
        """
        before = window._settings
        after = type(before)(
            **{**vars(before), "two_stage_commit_enabled": False}
        )
        assert "2 段階確定" in window._deferred_setting_names(before, after)


class TestDecodeTextCanBeCopied:
    """**受信中に本文を選べること** (運用者、2026-08-17「コピーできません」).

    表示は hop (0.5 秒) ごとに ``setHtml`` で文書を作り直していた。作り直すと
    選択もスクロール位置も消えるので、選んだ傍から解除されてコピーできない。
    """

    @staticmethod
    def _select(window, start: int, end: int) -> None:
        from PySide6.QtGui import QTextCursor

        cursor = window.text_view.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        window.text_view.setTextCursor(cursor)

    def test_selection_survives_a_refresh(self, window) -> None:
        """**内容が変わらない更新で選択を消さない.**

        無音の間も hop ごとに呼ばれるので、ここが本体。
        """
        window._committed_text = "CQ CQ DE JH0ILL"
        window._refresh_decode_display()
        self._select(window, 0, 5)
        window._refresh_decode_display()
        assert window.text_view.textCursor().selectedText() == "CQ CQ"

    def test_selection_survives_new_text(self, window) -> None:
        """**続きが届いても、選んだ範囲は選ばれたまま.**"""
        window._committed_text = "CQ CQ DE JH0ILL"
        window._refresh_decode_display()
        self._select(window, 0, 5)
        window._committed_text = "CQ CQ DE JH0ILL K"
        window._refresh_decode_display()
        assert window.text_view.textCursor().selectedText() == "CQ CQ"

    def test_text_is_selectable_by_mouse(self, window) -> None:
        """読み取り専用でも、マウスで選べる状態であること."""
        from PySide6.QtCore import Qt

        flags = window.text_view.textInteractionFlags()
        assert flags & Qt.TextInteractionFlag.TextSelectableByMouse

    def test_clearing_does_not_crash(self, window) -> None:
        """**選択したまま本文が消えても落ちないこと** (位置が文書外になる)."""
        window._committed_text = "CQ CQ DE JH0ILL"
        window._refresh_decode_display()
        self._select(window, 0, 15)
        window._committed_text = ""
        window._refresh_decode_display()
        assert window.text_view.toPlainText() == ""


class TestStatusBarDiagnostics:
    """ステータスバーの診断表示.

    **小数を切り捨てないこと.** hop の既定は 0.5 秒なのに ``hop=0s`` と出ていた。
    取扱説明書で「実効右文脈 = lag + hop ÷ 2」を説明するので、0 に見えると
    計算が合わなくなる (2026-08-16 発見)。
    """

    DIAG = {"window": 30.0, "hop": 0.5, "lag": 2.0, "decode_ms": 112.0}

    def test_hop_keeps_its_decimal(self, window) -> None:
        window._on_stream_diag(dict(self.DIAG))
        assert "hop=0.5s" in window.statusBar().currentMessage()

    def test_the_other_values_are_still_shown(self, window) -> None:
        window._on_stream_diag(dict(self.DIAG))
        message = window.statusBar().currentMessage()
        assert "window=30s" in message
        assert "lag=2.0s" in message
        assert "decode=112ms" in message

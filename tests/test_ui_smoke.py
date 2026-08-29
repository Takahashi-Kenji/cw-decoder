"""UI スモークテスト (オフスクリーン Qt プラットフォームで起動確認のみ).

実 GUI を表示せず、ウィンドウインスタンス化と signal/slot 接続が動作することを確認.
"""
from __future__ import annotations

import os

import pytest

# Qt は GUI 不可環境でも "offscreen" プラットフォームで動作する
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")


def test_main_window_constructs() -> None:
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    window = CWDecoderWindow(engine, AppSettings())
    assert window.windowTitle().startswith("CW")
    assert window.start_btn.isEnabled()
    assert not window.stop_btn.isEnabled()
    # デコードトグルは開始前は無効
    assert not window.decode_toggle_btn.isEnabled()
    window.close()


def test_threshold_slider_updates_label() -> None:
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    window = CWDecoderWindow(engine, AppSettings())
    window.threshold_slider.setValue(75)
    assert window.threshold_value_label.text() == "0.75"
    window.close()


def test_mode_combo_changes_state() -> None:
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    window = CWDecoderWindow(engine, AppSettings(mode="japanese"))
    # 設定の和文が反映される
    assert window.mode_combo.currentIndex() == 1
    assert window._current_mode() == "japanese"
    window.close()


def test_mode_combo_has_two_items() -> None:
    """モードコンボは欧文・和文の 2 択。**自動は選べない** (2026-08-29 要望).

    欧文ストリームラインで欧文読みが常に見えるようになり、自動切替に
    頼る必要が薄れたため、選択肢から外した。内部の切替機構
    (ワーカーの mode="auto") は温存している。
    """
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    window = CWDecoderWindow(engine, AppSettings())
    assert window.mode_combo.count() == 2
    window.mode_combo.setCurrentIndex(1)
    assert window._current_mode() == "japanese"
    window.close()


def test_mode_combo_auto_setting_falls_back_to_japanese() -> None:
    """旧設定の mode='auto' は**和文へ縮退**すること.

    自動を選んでいた人は和文の交信を受けたい人 (欧文だけなら欧文固定を
    選んでいる)。設定ファイル自体は load_settings の移行で書き換わるが、
    直接渡された場合もここで受け止める。
    """
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    settings = AppSettings()
    settings.mode = "auto"  # type: ignore[assignment]  # 旧設定の値
    window = CWDecoderWindow(engine, settings)
    assert window.mode_combo.currentIndex() == 1
    assert window._current_mode() == "japanese"
    window.close()


def test_decode_toggle_btn_text_changes() -> None:
    """デコードトグルの checked/unchecked でボタン文言が切り替わること."""
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    window = CWDecoderWindow(engine, AppSettings())
    # toggled シグナルに接続された _on_decode_toggled を直接呼ぶ
    window._on_decode_toggled(True)
    assert "停止" in window.decode_toggle_btn.text()
    window._on_decode_toggled(False)
    assert "デコード開始" in window.decode_toggle_btn.text()
    window.close()


def test_window_handles_live_signals() -> None:
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    w = CWDecoderWindow(engine, AppSettings())
    w._on_committed_text("CQ CQ")
    w._on_provisional_text("DE JA")
    w._on_stream_diag({"window": 30.0, "hop": 1.0, "lag": 2.5, "decode_ms": 42.0})
    assert "CQ CQ" in w._current_display_html()
    w.close()


def test_prosign_angle_brackets_escaped_not_stripped() -> None:
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    w = CWDecoderWindow(engine, AppSettings())
    w._on_committed_text("JA1QRP DE JA7QRS <KN>")
    html = w._current_display_html()
    assert "&lt;KN&gt;" in html
    assert "<KN>" not in html
    w.close()


def test_on_current_mode_stores_submode() -> None:
    """_on_current_mode は _auto_submode を更新するだけでステータスバーを書き換えない."""
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    w = CWDecoderWindow(engine, AppSettings())
    w.statusBar().showMessage("テスト前")
    w._on_current_mode("japanese")
    assert w._auto_submode == "japanese"
    # ステータスバーは書き換えない
    assert w.statusBar().currentMessage() == "テスト前"
    w._on_current_mode("european")
    assert w._auto_submode == "european"
    w.close()


def test_stream_diag_no_submode_in_fixed_mode() -> None:
    """欧文/和文固定モード中は _on_stream_diag がサブモードラベルを出力しない."""
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    w = CWDecoderWindow(engine, AppSettings())
    # 欧文固定モード (index 0)
    w.mode_combo.setCurrentIndex(0)
    diag = {"window": 30.0, "hop": 1.0, "lag": 2.5, "decode_ms": 42.0}
    w._on_stream_diag(diag)
    msg = w.statusBar().currentMessage()
    assert "自動" not in msg
    assert "window=" in msg
    w.close()


def test_llm_panel_and_controls_exist() -> None:
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    win = CWDecoderWindow(engine, AppSettings())
    assert win.llm_text_view is not None
    assert win.llm_provider_combo.count() == 3   # ollama/openai/claude
    assert win.llm_refine_btn is not None
    assert win.llm_auto_check is not None
    win.close()


def test_model_follows_provider_change() -> None:
    """プロバイダを claude にするとモデル欄が有効な Claude モデルへ追従する (M8).

    追従の担い手は設定画面 (``SettingsDialog``) になった。主画面の隠しコンボは
    値の置き場で、反映は ``_apply_settings_to_widgets`` が明示的に行う。
    """
    from PySide6.QtWidgets import QApplication
    from src.app.settings_dialog import SettingsDialog
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    dialog = SettingsDialog(AppSettings(llm_provider="ollama", llm_model="llama3.1"))
    dialog.llm_provider.setCurrentText("claude")
    model = dialog.llm_model.currentText()
    assert model.startswith("claude-")   # llama3.1 のまま残らない


def test_clear_decode_button_clears_body() -> None:
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    win = CWDecoderWindow(engine, AppSettings())
    win._on_committed_text("CQ CQ DE JA1ABC")
    win._on_provisional_text("K")
    assert "CQ CQ" in win._current_display_html()
    win._on_clear_decode()
    assert win._committed_text == ""
    assert win._provisional_text == ""
    assert "CQ CQ" not in win._current_display_html()
    win.close()


def test_llm_clear_button_clears_panel() -> None:
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings
    from src.llm.markup import OPEN_MARK, CLOSE_MARK

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    win = CWDecoderWindow(engine, AppSettings())
    win._on_llm_result(f"晴天 {OPEN_MARK}X{CLOSE_MARK}")
    assert win.llm_text_view.toPlainText() != ""
    win._on_llm_clear()
    assert win.llm_text_view.toPlainText() == ""
    win.close()


def test_llm_result_renders_red_for_marked_spans() -> None:
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings
    from src.llm.markup import OPEN_MARK, CLOSE_MARK

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    win = CWDecoderWindow(engine, AppSettings())
    win._on_llm_result(f"晴天 {OPEN_MARK}JH0ILL{CLOSE_MARK}")
    html = win.llm_text_view.toHtml()
    assert "cc0000" in html        # 赤 span が入っている
    win.close()


def test_worker_error_message_persists_after_stop() -> None:
    """_on_worker_error 後、ステータスバーに残るのはエラー本文であること.

    _on_stop() は末尾で無条件に「停止しました」を表示するため、エラー本文を
    _on_stop() より先に表示する素朴な実装だと、そのメッセージで上書きされて
    消えてしまう。--net-source 経路で最も起きやすい失敗 (送信側未起動・IP
    間違い) が画面から消える回帰の防止。
    """
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    class _StubWorker:
        """_on_stop() の本処理 (末尾の showMessage 含む) を通すためのスタブ."""

        def stop(self) -> None:
            pass

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    win = CWDecoderWindow(engine, AppSettings())
    # _on_stop() は self._worker が None だと即 return して何もしないため、
    # 動作中を模したスタブを差し込んで実際の停止経路 (showMessage 込み) を通す。
    win._worker = _StubWorker()
    win._worker_thread = None
    error_message = "start failed: 転送元に接続できません (127.0.0.1:45678)"
    win._on_worker_error(error_message)
    assert win.statusBar().currentMessage() == error_message
    win.close()


def test_on_start_with_invalid_net_source_recovers() -> None:
    """不正な --net-source 指定時、_on_start が例外を外へ投げず、
    self._worker を None に戻して「開始」を再度押せる状態を保つこと.

    実際の音声デバイス/スレッドには触れない: set_net_source() の検証は
    QThread 生成より前で行われるため、この経路は軽量なままテストできる。
    """
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    win = CWDecoderWindow(engine, AppSettings(), net_source="192.168.1.20:abc")
    win._on_start()
    assert win._worker is None
    assert "不正" in win.statusBar().currentMessage()
    # 開始ボタン等の状態は変更されておらず、再試行できる (無反応バグの再発防止)
    assert win.start_btn.isEnabled()
    assert not win.stop_btn.isEnabled()
    win.close()


# --- 未確定 (暫定) テキストの表示オプション ---
#
# 確定と暫定が混ざると読みにくいという運用上の指摘による (2026-08-04)。
# 既定はオフ。


def _make_window(show_provisional: bool = False):
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    QApplication.instance() or QApplication([])
    settings = AppSettings()
    settings.show_provisional = show_provisional
    return CWDecoderWindow(InferenceEngine.untrained(device="cpu"), settings)


def test_provisional_hidden_by_default() -> None:
    win = _make_window()
    try:
        assert win._settings.show_provisional is False, "既定はオフ"
        win._committed_text = "CQ DE JH0ILL"
        win._provisional_text = " K"
        html = win._current_display_html()
        assert "CQ DE JH0ILL" in html
        assert "999999" not in html, "既定では暫定用のグレー span を出さない"
    finally:
        win.close()


def test_provisional_shown_when_enabled() -> None:
    win = _make_window(show_provisional=True)
    try:
        win._committed_text = "CQ DE JH0ILL"
        win._provisional_text = " K"
        html = win._current_display_html()
        assert "999999" in html, "有効時はグレー span を出す"
        assert "K" in html
    finally:
        win.close()


def test_toggle_updates_setting_and_display() -> None:
    win = _make_window()
    try:
        win._committed_text = "CQ"
        win._provisional_text = " DE"
        win._on_show_provisional_toggled(True)
        assert win._settings.show_provisional is True
        assert "DE" in win._current_display_html()
        win._on_show_provisional_toggled(False)
        assert win._settings.show_provisional is False
        assert "DE" not in win._current_display_html()
    finally:
        win.close()


def _window(**settings_kwargs):
    """テスト用ウィンドウを組み立てる (呼び出し側で close すること)."""
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    QApplication.instance() or QApplication([])
    engine = InferenceEngine.untrained(device="cpu")
    return CWDecoderWindow(engine, AppSettings(**settings_kwargs))


def test_word_correction_is_applied_to_committed_text() -> None:
    """辞書補正が確定テキストに効き、補正箇所が橙色になること."""
    window = _window(word_correct_enabled=True)
    try:
        window._on_committed_text("CQ CQCQDE JF1GL K")
        assert window._committed_text == "CQ CQ CQ DE JF1GL K"
        assert window._committed_spans
        html = window._committed_html()
        assert "#c05000" in html          # 補正箇所の橙色
        assert "元: CQCQDE" in html       # 元の姿がツールチップに残る
    finally:
        window.close()


def test_word_correction_can_be_turned_off() -> None:
    """寄せも語彙分割も切れば、確定テキストに一切触らない."""
    window = _window(word_correct_enabled=False, word_split_enabled=False)
    try:
        window._on_committed_text("CQ CQCQDE JF1GL K")
        assert window._committed_text == "CQ CQCQDE JF1GL K"
        assert window._committed_spans == ()
        assert "#c05000" not in window._committed_html()
    finally:
        window.close()


def test_word_split_works_without_correction() -> None:
    """寄せを切っていても、つながった欧文は語彙で切る (既定 ON、2026-08-28).

    間隔では切れない局 (文字間 4.3 / 語間 5 dot) のため。運用者は寄せを
    切って使っているので、分割が寄せに連動していると効かない。
    """
    window = _window(word_correct_enabled=False, word_split_enabled=True)
    try:
        window._on_committed_text("CQ CQCQDE JF1GL K")
        assert window._committed_text == "CQ CQ CQ DE JF1GL K"
    finally:
        window.close()


def test_callsign_is_never_corrected() -> None:
    """存在しない局を自信ありげに作らないこと."""
    window = _window(word_correct_enabled=True)
    try:
        window._on_committed_text("DE JH0ILL RST 599")
        assert "JH0ILL" in window._committed_text
        assert "599" in window._committed_text
    finally:
        window.close()


def test_committed_html_escapes_prosign_brackets() -> None:
    """[SK] の角括弧が HTML として解釈されないこと (色分けで位置がずれても)."""
    window = _window(word_correct_enabled=True)
    try:
        window._on_committed_text("TNX CQCQDE <KN>")
        assert "&lt;KN&gt;" in window._committed_html()
    finally:
        window.close()


def test_newline_survives_correction_and_html() -> None:
    """無音による改行が補正でも HTML 化でも失われないこと."""
    window = _window(word_correct_enabled=True)
    try:
        window._on_committed_text("CQCQDE\nTNX")
        assert "\n" in window._committed_text
        assert "<br>" in window._committed_html()
    finally:
        window.close()


def test_display_scrolls_to_bottom() -> None:
    """受信中は常に最新行が見えること (setHtml は先頭に戻してしまう).

    ``show()`` と ``processEvents()`` が要る。表示していないとレイアウトが
    走らずスクロールバーの maximum が 0 のままで、何を assert しても通ってしまう。
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    window = _window()
    try:
        window.resize(500, 300)
        window.show()
        app.processEvents()
        window._on_committed_text("\n".join(f"LINE {i} TNX QSO" for i in range(200)))
        app.processEvents()
        scroll = window.text_view.verticalScrollBar()
        assert scroll.maximum() > 0, "スクロールできる長さが無いとテストにならない"
        assert scroll.value() == scroll.maximum()

        # 自動スクロールが無ければ先頭に戻ってしまうこと (この差が本体)
        window.text_view.setHtml(window._current_display_html())
        assert scroll.value() == 0
    finally:
        window.close()


def test_auto_refine_sends_only_the_new_part() -> None:
    """自動清書は増分。清書済みの分を送り直さないこと."""
    window = _window(llm_auto=True, llm_auto_interval_s=0.0)
    try:
        sent: list[tuple] = []
        window.request_llm_transform.connect(lambda *a: sent.append(a))

        window._on_committed_text("CQ DE")
        assert sent and sent[-1][0] == "CQ DE"
        # LLM から結果が返って清書済みになる
        window._on_llm_result("CQ DE")

        window._on_committed_text("CQ DE JA1ABC")
        assert sent[-1][0] == "JA1ABC"          # 増分だけ
        assert sent[-1][2].strip() == "CQ DE"   # 直前は参考として渡す
    finally:
        window.close()


def test_manual_refine_sends_everything() -> None:
    """手動ボタンは全体を清書し直す (増分ではない)."""
    window = _window()
    try:
        sent: list[tuple] = []
        window.request_llm_transform.connect(lambda *a: sent.append(a))
        window._on_committed_text("CQ DE JA1ABC")
        window._on_refine_clicked()
        assert sent[-1][0] == "CQ DE JA1ABC"
        assert sent[-1][2] == ""
    finally:
        window.close()


def test_incremental_results_accumulate() -> None:
    """増分の清書結果は積み上げる (置き換えると前の分が消える)."""
    window = _window(llm_auto=True, llm_auto_interval_s=0.0)
    try:
        window._on_committed_text("CQ DE")
        window._on_llm_result("こんにちは")
        window._on_committed_text("CQ DE JA1ABC")
        window._on_llm_result("JA1ABC です")
        html = window.llm_text_view.toHtml()
        assert "こんにちは" in html
        assert "JA1ABC" in html
    finally:
        window.close()


def test_guess_highlight_can_be_turned_off() -> None:
    """赤が多いと読みにくいときに切れること。マーカー記号は出さない."""
    window = _window()
    try:
        window.llm_highlight_check.setChecked(True)
        window._on_llm_result("こんにちは⟦。晴れです⟧")
        assert "#cc0000" in window.llm_text_view.toHtml()

        window.llm_highlight_check.setChecked(False)
        html = window.llm_text_view.toHtml()
        assert "#cc0000" not in html
        assert "⟦" not in html
        assert "晴れです" in html
    finally:
        window.close()


def test_llm_clear_restarts_from_the_beginning() -> None:
    """清書をクリアしたら、次は最初から清書し直すこと."""
    window = _window(llm_auto=True, llm_auto_interval_s=0.0)
    try:
        window._on_committed_text("CQ DE")
        window._on_llm_result("CQ DE")
        window._on_llm_clear()
        sent: list[tuple] = []
        window.request_llm_transform.connect(lambda *a: sent.append(a))
        window._on_committed_text("CQ DE JA1ABC")
        assert sent[-1][0] == "CQ DE JA1ABC"
    finally:
        window.close()


def test_settings_are_saved_to_the_given_path(tmp_path) -> None:
    """保存先を注入できること.

    注入できないと close() のたびに利用者の実設定を上書きする
    (実際に llm_auto_interval_s を 0.0 に書き戻していた)。
    """
    import json

    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    QApplication.instance() or QApplication([])
    target = tmp_path / "settings.json"
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(llm_auto_interval_s=42.0),
        config_path=target,
    )
    window.close()
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["llm_auto_interval_s"] == 42.0


def test_default_config_path_is_not_touched_by_tests(tmp_path) -> None:
    """conftest の安全網が効いていること (実設定を触らない)."""
    from pathlib import Path

    from src.infer.settings import DEFAULT_CONFIG_PATH

    assert Path.home() / ".cw-decorder" != Path(DEFAULT_CONFIG_PATH).parent


def test_操作できない幅に潰れた部品が無い(tmp_path) -> None:
    """**幅 0 の部品を作らない** (運用者が「モデル選択ができない」と報告、2026-08-12).

    窓を細くするために ``QSizePolicy.Ignored`` を付けたところ、伸縮の指定が
    無い ``llm_model_edit`` が**幅 0 px に潰れて選べなくなった**。
    ``Ignored`` は「sizeHint を無視して余りを取る」であり、余りが無ければ
    0 になる。**下限は明示すること。**

    見た目のテストは書きにくいが、**押せる・選べるかどうかは幅で測れる。**
    """
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QPushButton,
        QSlider,
        QSpinBox,
    )

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(),
        config_path=tmp_path / "settings.json",
    )
    try:
        window.show()
        # **一番細くした状態で見る。** 広げれば直るのでは意味が無い
        window.resize(window.minimumSizeHint().width(), 700)
        app.processEvents()

        # PySide6 の findChildren はタプルの型指定を受けないので種類ごとに集める
        widgets = []
        for kind in (QComboBox, QPushButton, QSlider, QSpinBox):
            widgets.extend(window.findChildren(kind))
        collapsed = [
            f"{type(w).__name__}({w.text() if isinstance(w, QPushButton) else w.objectName() or '?'})"
            for w in widgets
            if w.isVisible() and w.width() < 20
        ]
        assert not collapsed, f"幅が足りず操作できない部品: {collapsed}"
    finally:
        window.close()


def test_LLMのモデル欄は窓を細くしても選べる(tmp_path) -> None:
    """名指しの回帰テスト。**ここが 0 px になっていた.**"""
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    app = QApplication.instance() or QApplication([])
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(),
        config_path=tmp_path / "settings.json",
    )
    try:
        window.show()
        window.resize(window.minimumSizeHint().width(), 700)
        app.processEvents()

        assert window.llm_model_edit.width() >= 100
        assert window.llm_model_edit.count() > 0        # 候補がある
        assert window.device_combo.width() >= 100
    finally:
        window.close()


def test_設定画面でモデルだけ変えてもワーカーに届く(tmp_path) -> None:
    """**モデル単独の変更が再起動まで効かない**回帰の歯止め (2026-08-29).

    ``_apply_settings_to_widgets`` が ``setEditText`` するだけでは、繋がって
    いる信号 (editingFinished / activated) はどちらも人の操作でしか出ないため、
    ``_refresh_llm_provider`` が呼ばれずワーカーは旧モデルのままだった。
    """
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    QApplication.instance() or QApplication([])
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(llm_provider="ollama", llm_model="qwen3.5:4b"),
        config_path=tmp_path / "settings.json",
    )
    try:
        received: list[object] = []
        window.request_set_llm_provider.connect(received.append)
        window._settings.llm_model = "gemma4:e4b"     # 設定画面の OK 相当
        window._apply_settings_to_widgets()
        assert received, "ワーカーへプロバイダが送り直されていない"
        assert getattr(received[-1], "model", None) == "gemma4:e4b"
    finally:
        window.close()


def test_設定画面のモデル選択が既定に潰されない(tmp_path) -> None:
    """プロバイダも変えたとき、選んだモデルが**先頭の既定に巻き戻らない**こと.

    旧経路では ``setCurrentIndex`` → ``_on_llm_provider_changed`` が
    ``models[0]`` を書き込み、設定画面で選んだモデルを上書きしていた。
    """
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    QApplication.instance() or QApplication([])
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(llm_provider="claude", llm_model="claude-haiku-4-5"),
        config_path=tmp_path / "settings.json",
    )
    try:
        window._settings.llm_provider = "ollama"
        window._settings.llm_model = "gemma4:e4b"     # 既定 (先頭) ではない候補
        window._apply_settings_to_widgets()
        assert window._settings.llm_model == "gemma4:e4b"
        assert window.llm_model_edit.currentText() == "gemma4:e4b"
    finally:
        window.close()


def test_欧文ストリームラインはスペクトラムの上にある() -> None:
    """欧文ストリームライン (2026-08-29 要望): 1 行・読み取り専用・折返しなし、
    位置は本文とスペクトラムスコープの間."""
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    QApplication.instance() or QApplication([])
    window = CWDecoderWindow(InferenceEngine.untrained(device="cpu"), AppSettings())
    try:
        view = window.euro_stream_view
        assert isinstance(view, QPlainTextEdit)
        assert view.isReadOnly()
        assert view.lineWrapMode() == QPlainTextEdit.LineWrapMode.NoWrap
        layout = window.centralWidget().layout()
        assert 0 <= layout.indexOf(view) < layout.indexOf(window.spectrogram_panel)
    finally:
        window.close()


def test_欧文ストリームは差分追記で選択が壊れない() -> None:
    """前回の続きなら**末尾に追記**する (作り直すと選択・コピーが壊れる —
    decode-view-selection の既知の罠)。続きでなければ全置換."""
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QApplication

    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    from src.infer.settings import AppSettings

    QApplication.instance() or QApplication([])
    window = CWDecoderWindow(InferenceEngine.untrained(device="cpu"), AppSettings())
    try:
        window._on_european_stream("CQ DE")
        # 先頭 2 文字を選択しておく (選択の終端が文書末尾に触れていると、
        # Qt の仕様で末尾挿入時に選択が伸びる。それは「壊れ」ではないので
        # 現実的な「途中を選択したまま流れ続ける」形で見る)
        cursor = window.euro_stream_view.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        window.euro_stream_view.setTextCursor(cursor)
        window._on_european_stream("CQ DE JA1ABC")   # 続き → 追記
        assert window.euro_stream_view.toPlainText() == "CQ DE JA1ABC"
        assert window.euro_stream_view.textCursor().selectedText() == "CQ"
        window._on_european_stream("")               # クリア → 全置換
        assert window.euro_stream_view.toPlainText() == ""
    finally:
        window.close()

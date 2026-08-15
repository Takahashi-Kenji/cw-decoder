"""設定画面のテスト.

見るのは 3 つ。

* **往復して値が壊れないこと** — 開いて OK を押すだけで設定が変わらない
* **取り消しで元に戻ること** — 元の設定を書き換えていない
* **危ない値に歯止めがあること** — commit_lag 0 やリング 1 秒で受信を壊さない
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from src.app.settings_dialog import SettingsDialog      # noqa: E402
from src.infer.settings import AppSettings              # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestRoundTrip:
    def test_ok_without_editing_keeps_everything(self, qapp) -> None:
        """**開いて OK を押すだけで設定が変わらないこと.**

        ここが崩れると、設定画面を覗いただけで挙動が変わる。
        """
        original = AppSettings()
        dialog = SettingsDialog(original)
        dialog._on_accept()
        result = dialog.result_settings
        assert result is not None
        for name in vars(original):
            assert getattr(result, name) == getattr(original, name), name

    def test_cancel_leaves_the_original_untouched(self, qapp) -> None:
        original = AppSettings()
        before = dict(vars(original))
        dialog = SettingsDialog(original)
        dialog.hop_s.setValue(1.0)
        dialog.commit_lag_s.setValue(3.0)
        dialog.reject()
        assert dialog.result_settings is None
        assert vars(original) == before

    def test_edits_are_carried_over(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings())
        dialog.commit_lag_s.setValue(2.5)
        dialog.word_correct_ja_enabled.setChecked(False)
        dialog.llm_model.setText("gemma4:e4b")
        dialog._on_accept()
        result = dialog.result_settings
        assert result.commit_lag_s == pytest.approx(2.5)
        assert result.word_correct_ja_enabled is False
        assert result.llm_model == "gemma4:e4b"


class TestEffectiveRightContext:
    """**hop と lag は和で効く。** 片方だけ動かす事故を防ぐため計算結果を出す.

    過去に hop だけ縮めて右文脈を静かに失った経緯がある。
    """

    def test_default_shows_the_target(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings())
        assert "2.25" in dialog.effective_label.text()

    def test_changing_hop_alone_is_visible(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings())
        dialog.hop_s.setValue(1.0)          # lag はそのまま
        text = dialog.effective_label.text()
        assert "2.50" in text
        assert "目標" in text, "目標から外れたことが分からない"

    def test_matching_the_target_shows_no_warning(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings())
        dialog.hop_s.setValue(1.0)
        dialog.commit_lag_s.setValue(1.75)   # 1.75 + 0.5 = 2.25
        assert "目標" not in dialog.effective_label.text()


class TestGuardrails:
    """危ない値で受信を壊さないこと."""

    @pytest.mark.parametrize(
        ("field", "low", "high"),
        [
            ("commit_lag_s", 0.5, 5.0),
            ("hop_s", 0.1, 2.0),
            ("window_s", 5.0, 120.0),
            ("decode_left_context_s", 1.0, 30.0),
            ("confidence_threshold", 0.0, 1.0),
            ("refine_capacity_s", 30.0, 900.0),
        ],
    )
    def test_range_is_bounded(self, qapp, field: str, low: float, high: float) -> None:
        dialog = SettingsDialog(AppSettings())
        box = getattr(dialog, field)
        assert box.minimum() == pytest.approx(low)
        assert box.maximum() == pytest.approx(high)

    def test_zero_commit_lag_is_refused(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings())
        dialog.commit_lag_s.setValue(0.0)
        assert dialog.commit_lag_s.value() >= 0.5

    def test_empty_recording_dir_falls_back(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings())
        dialog.recording_dir.setText("   ")
        dialog._on_accept()
        assert dialog.result_settings.recording_dir == "data/real"

    def test_empty_checkpoint_becomes_none(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings(checkpoint_path="x.pt"))
        dialog.checkpoint_path.setText("")
        dialog._on_accept()
        assert dialog.result_settings.checkpoint_path is None


class TestTabs:
    def test_all_tabs_exist(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings())
        titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert titles == ["入力", "デコード", "確定", "補正", "清書", "表示"]

"""LAN 音声の受け取り先を**設定として**持てること.

**配布版には `--net-source` を渡す手段が無い。** アイコンを叩いて起動するので、
コマンドライン引数だけに置いてあると音声送出側 (`scripts/audio_send.py`) を
入れても受け取れない (2026-08-31 に運用者からの指摘で発覚)。

引数は残す。**引数があればそちらが勝つ** — 一時的に別の PC から受けたいときに、
保存された設定を書き換えずに済ませたいため (``--ckpt`` と同じ扱い)。
"""
from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("PySide6")

from src.app.settings_dialog import (                    # noqa: E402
    DEFERRED_SETTING_LABELS,
    SettingsDialog,
)
from src.infer.settings import (                         # noqa: E402
    AppSettings,
    load_settings,
    save_settings,
)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestSettingField:
    def test_default_is_empty(self) -> None:
        """既定は空 = LAN を使わない (マイク入力)."""
        assert AppSettings().net_source == ""

    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "settings.json"
        save_settings(replace(AppSettings(), net_source="192.168.0.10:45678"), path)
        assert load_settings(path).net_source == "192.168.0.10:45678"

    def test_missing_key_loads_as_empty(self, tmp_path) -> None:
        """旧い設定ファイル (この項目が無い) を読んでも落ちないこと."""
        path = tmp_path / "settings.json"
        path.write_text('{"settings_version": 21, "mode": "european"}', encoding="utf-8")
        assert load_settings(path).net_source == ""


class TestSettingsDialog:
    """入力タブから編集できること."""

    def test_field_exists_and_round_trips(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings(net_source="192.168.0.10"))
        assert dialog.net_source.text() == "192.168.0.10"
        dialog.net_source.setText("192.168.0.20:45678")
        dialog._on_accept()
        assert dialog.result_settings.net_source == "192.168.0.20:45678"

    def test_blank_means_microphone(self, qapp) -> None:
        dialog = SettingsDialog(AppSettings(net_source="192.168.0.10"))
        dialog.net_source.setText("   ")
        dialog._on_accept()
        assert dialog.result_settings.net_source == ""

    def test_is_deferred(self, qapp) -> None:
        """次の「開始」から効く項目なので ⟳ の一覧に入っていること."""
        assert "net_source" in DEFERRED_SETTING_LABELS


class TestCommandLineWins:
    """``--net-source`` は設定より強い (一時的な切替を保存値で汚さない)."""

    @pytest.mark.parametrize(
        ("argument", "saved", "expected"),
        [
            ("10.0.0.5", "192.168.0.10", "10.0.0.5"),   # 引数が勝つ
            (None, "192.168.0.10", "192.168.0.10"),    # 引数が無ければ設定
            (None, "", None),                            # どちらも無ければマイク
        ],
    )
    def test_resolution(self, argument: str | None, saved: str, expected: str | None) -> None:
        from src.app.main_window import resolve_net_source

        assert resolve_net_source(argument, replace(AppSettings(), net_source=saved)) == expected

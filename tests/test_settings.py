"""設定永続化のテスト."""
from __future__ import annotations

from pathlib import Path

from src.infer.settings import (
    CURRENT_SETTINGS_VERSION,
    AppSettings,
    load_settings,
    migrate_settings_dict,
    save_settings,
)


class TestAppSettings:
    def test_default_values(self) -> None:
        s = AppSettings()
        assert s.mode == "european"
        assert s.confidence_threshold == 0.5
        assert s.sample_rate == 8000

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        s = AppSettings(
            mode="japanese",
            confidence_threshold=0.7,
            input_device=3,
            checkpoint_path="models/best.pt",
        )
        save_settings(s, path)
        loaded = load_settings(path)
        assert loaded.mode == "japanese"
        assert loaded.confidence_threshold == 0.7
        assert loaded.input_device == 3
        assert loaded.checkpoint_path == "models/best.pt"

    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        s = load_settings(path)
        assert s == AppSettings()

    def test_malformed_file_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("this is not json", encoding="utf-8")
        s = load_settings(path)
        assert s == AppSettings()

    def test_extra_unknown_keys_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.json"
        path.write_text(
            '{"mode": "japanese", "fake_field": 123}', encoding="utf-8"
        )
        s = load_settings(path)
        assert s.mode == "japanese"
        # Should not raise

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "settings.json"
        save_settings(AppSettings(), path)
        assert path.exists()

    def test_streaming_defaults_present(self) -> None:
        s = AppSettings()
        assert s.settings_version == CURRENT_SETTINGS_VERSION
        assert s.window_s == 30.0
        assert s.hop_s == 0.5
        assert s.commit_lag_s == 2.0
        assert s.head_guard_s == 1.0
        # 効くのは commit_lag_s 単独ではなく commit_lag_s + hop_s/2 (実効右文脈)。
        # 2026-08-07 の held-out 実測で目標値が 3.0 → 2.25 秒になった
        assert s.commit_lag_s + s.hop_s / 2 == 2.25
        assert s.decode_left_context_s == 5.0
        assert s.commit_jitter_margin_s == 0.02


class TestStreamingDefaultsMigration:
    """旧既定のままの設定を新既定へ移行する (確定条件をブラウザ版に揃えた)."""

    def test_old_defaults_are_migrated(self) -> None:
        migrated, changed = migrate_settings_dict(
            {"settings_version": 6, "commit_lag_s": 2.5, "hop_s": 1.0,
             "squelch_threshold_db": -60.0}
        )
        # 実質無効だったスキッシュが有効な値になる
        assert migrated["squelch_threshold_db"] == -25.0
        assert changed is True
        assert migrated["commit_lag_s"] == 2.0
        assert migrated["hop_s"] == 0.5
        # 実効右文脈 = commit_lag + hop/2。ブラウザ版と同じ 2.25 秒になる
        assert migrated["commit_lag_s"] + migrated["hop_s"] / 2 == 2.25
        assert migrated["settings_version"] == CURRENT_SETTINGS_VERSION

    def test_user_customised_values_are_kept(self) -> None:
        """自分で値を変えていた人の設定は尊重する (置換表はそのためにある)."""
        migrated, _ = migrate_settings_dict(
            {"settings_version": 6, "commit_lag_s": 4.0, "hop_s": 2.0,
             "squelch_threshold_db": -45.0}
        )
        assert migrated["commit_lag_s"] == 4.0
        assert migrated["hop_s"] == 2.0
        assert migrated["squelch_threshold_db"] == -45.0


def test_送信の設定に既定がある() -> None:
    settings = AppSettings()
    assert settings.tx_endpoint == ""
    assert settings.tx_wpm == 20.0


def test_旧版の設定に送信の既定が入る() -> None:
    """**既にある設定ファイルを壊さない。** 足りない欄は既定で埋まる."""
    old = {"settings_version": 14, "mode": "european"}
    merged, changed = migrate_settings_dict(old)
    assert changed is True
    assert merged["settings_version"] == CURRENT_SETTINGS_VERSION
    assert merged["tx_endpoint"] == ""
    assert merged["tx_wpm"] == 20.0


def test_送信先を書いて読み戻せる(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_settings(AppSettings(tx_endpoint="192.168.0.10:45679", tx_wpm=22.0), path)
    restored = load_settings(path)
    assert restored.tx_endpoint == "192.168.0.10:45679"
    assert restored.tx_wpm == 22.0

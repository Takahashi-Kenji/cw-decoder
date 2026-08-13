"""settings.json マイグレーションのテスト."""
from __future__ import annotations

import json
from pathlib import Path

from src.infer.settings import (
    CURRENT_SETTINGS_VERSION,
    AppSettings,
    load_settings,
    migrate_settings_dict,
)


def test_missing_version_treated_as_v1_and_filled() -> None:
    # version フィールドが無い (= v1 相当) 旧 JSON
    raw = {"mode": "japanese", "chunk_duration_s": 1.5}
    migrated, changed = migrate_settings_dict(raw)
    assert migrated["settings_version"] == CURRENT_SETTINGS_VERSION
    # 新フィールドがデフォルトで補完される
    assert migrated["window_s"] == 30.0
    assert changed is True


def test_v1_legacy_chunk_duration_dropped() -> None:
    # v3 では chunk_duration_s はスキーマから削除済み — merged に含まれない
    raw = {"settings_version": 1, "chunk_duration_s": 1.5}
    migrated, changed = migrate_settings_dict(raw)
    assert "chunk_duration_s" not in migrated
    assert changed is True


def test_v1_legacy_chunk_fields_all_dropped() -> None:
    # v3 ではすべてのレガシーチャンクフィールドが merged から脱落する
    raw = {
        "settings_version": 1,
        "chunk_duration_s": 10.0,
        "chunk_overlap_s": 0.5,
        "auto_chunk_enabled": True,
        "auto_chunk_silence_sec": 1.2,
        "auto_chunk_min_buffer_sec": 2.0,
        "auto_chunk_silence_amplitude": 0.005,
        "live_continuous": False,
    }
    migrated, changed = migrate_settings_dict(raw)
    for legacy in (
        "chunk_duration_s", "chunk_overlap_s",
        "auto_chunk_enabled", "auto_chunk_silence_sec",
        "auto_chunk_min_buffer_sec", "auto_chunk_silence_amplitude",
        "live_continuous",
    ):
        assert legacy not in migrated
    assert changed is True


def test_v2_no_change() -> None:
    raw = AppSettings().to_dict()
    migrated, changed = migrate_settings_dict(raw)
    assert changed is False


def test_load_settings_applies_migration(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"chunk_duration_s": 1.5}), encoding="utf-8")
    s = load_settings(p)
    assert s.settings_version == CURRENT_SETTINGS_VERSION


def test_v2_to_v3_drops_legacy_keys_and_keeps_working():
    from src.infer.settings import migrate_settings_dict, CURRENT_SETTINGS_VERSION

    raw = {
        "mode": "european",
        "settings_version": 2,
        "auto_chunk_enabled": True,
        "auto_chunk_silence_sec": 1.2,
        "live_continuous": True,
        "chunk_duration_s": 5.0,
        "chunk_overlap_s": 0.5,
        "bpf_enabled": True,
    }
    merged, changed = migrate_settings_dict(raw)
    assert changed is True
    assert merged["settings_version"] == CURRENT_SETTINGS_VERSION
    for legacy in ("auto_chunk_enabled", "live_continuous", "chunk_duration_s",
                   "chunk_overlap_s", "auto_chunk_silence_sec"):
        assert legacy not in merged
    assert merged["bpf_enabled"] is True


def test_mode_auto_round_trips(tmp_path):
    from src.infer.settings import AppSettings, load_settings, save_settings

    p = tmp_path / "s.json"
    s = AppSettings()
    s.mode = "auto"
    save_settings(s, p)
    assert load_settings(p).mode == "auto"


def test_v3_settings_migrate_to_v4_with_llm_defaults():
    from src.infer.settings import migrate_settings_dict, CURRENT_SETTINGS_VERSION
    old = {"settings_version": 3, "mode": "auto"}
    migrated, changed = migrate_settings_dict(old)
    assert changed is True
    assert migrated["settings_version"] == CURRENT_SETTINGS_VERSION
    assert migrated["llm_enabled"] is False
    assert migrated["llm_provider"] == "ollama"
    assert migrated["llm_auto_interval_s"] == 20.0
    assert migrated["mode"] == "auto"   # 既存値は保持


def test_v4_llm_timeout_30_migrates_to_120():
    """旧既定 30 秒の llm_timeout_s は v5 で 120 秒へ更新される."""
    old = {"settings_version": 4, "llm_timeout_s": 30.0}
    migrated, changed = migrate_settings_dict(old)
    assert changed is True
    assert migrated["llm_timeout_s"] == 120.0


def test_user_customized_llm_timeout_is_preserved():
    """ユーザーが 30 以外に変更済みなら値を保持する."""
    old = {"settings_version": 4, "llm_timeout_s": 60.0}
    migrated, _ = migrate_settings_dict(old)
    assert migrated["llm_timeout_s"] == 60.0

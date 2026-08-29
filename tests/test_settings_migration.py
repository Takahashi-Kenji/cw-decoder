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


def test_mode_auto_becomes_japanese_on_load(tmp_path):
    """ファイル経由でも mode='auto' は和文へ縮退すること (2026-08-29).

    「自動」をモード選択から外したため。以前はそのまま往復していた。
    """
    from src.infer.settings import AppSettings, load_settings, save_settings

    p = tmp_path / "s.json"
    s = AppSettings()
    s.mode = "auto"
    save_settings(s, p)
    assert load_settings(p).mode == "japanese"


def test_v3_settings_migrate_to_v4_with_llm_defaults():
    from src.infer.settings import migrate_settings_dict, CURRENT_SETTINGS_VERSION
    old = {"settings_version": 3, "mode": "auto"}
    migrated, changed = migrate_settings_dict(old)
    assert changed is True
    assert migrated["settings_version"] == CURRENT_SETTINGS_VERSION
    assert migrated["llm_enabled"] is False
    assert migrated["llm_provider"] == "ollama"
    assert migrated["llm_auto_interval_s"] == 20.0
    assert migrated["mode"] == "japanese"   # auto は和文へ縮退 (2026-08-29)


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


class TestAutoModeRemoval:
    """モード選択から「自動」を外した (2026-08-29)。旧設定は和文へ縮退する.

    自動を選んでいた人は和文の交信を受けたい人 (欧文だけなら欧文固定を
    選んでいる)。内部の切替機構は温存しており、移行は設定値だけの話。
    """

    def test_auto_mode_migrates_to_japanese(self) -> None:
        data = {"settings_version": CURRENT_SETTINGS_VERSION, "mode": "auto"}
        migrated, changed = migrate_settings_dict(data)
        assert migrated["mode"] == "japanese"
        assert changed is True

    def test_fixed_modes_pass_through(self) -> None:
        for mode in ("european", "japanese"):
            data = {"settings_version": CURRENT_SETTINGS_VERSION, "mode": mode}
            migrated, changed = migrate_settings_dict(data)
            assert migrated["mode"] == mode
            assert changed is False


class TestJapaneseCorrectionDefaultOff:
    """和文の辞書補正は**既定 OFF** (2026-08-29 運用者指示).

    運用者:「和文の補正は補正が強すぎて、使えない。間違ったところを補完して
    あげればいいだけだ」。実測でも正しい和文 22 件のうち 4 件を壊していた:

        アンテナアゲマシタ      → アンテナ アタタカイ シタ
        コチラモアツイデス      → コチラハ アツイ デス   (助詞が変わる)
        コトシハアメガスクナク  → コトシ ハ アメ ガ ザイタク

    符号距離で寄せる設計そのものは正しいが、語の区切りを跨いで連結した符号列で
    測るため、``ヨウカイ`` (--・・-・-・・・-) と ``マイク`` (-・・-・-・・・-) が
    距離 1 になってしまう。**代わりに、確実な局所修正 (「。ス」→「デス」) を
    変換器の中で常時行う** (fix_danraku_su。辞書補正の ON/OFF と無関係に効く)。
    """

    def test_default_is_off(self) -> None:
        assert AppSettings().word_correct_ja_enabled is False

    def test_old_setting_left_at_the_old_default_is_turned_off(self) -> None:
        old = {"settings_version": 20, "word_correct_ja_enabled": True}
        migrated, changed = migrate_settings_dict(old)
        assert migrated["word_correct_ja_enabled"] is False
        assert changed is True

    def test_explicit_off_stays_off(self) -> None:
        old = {"settings_version": 20, "word_correct_ja_enabled": False}
        migrated, _ = migrate_settings_dict(old)
        assert migrated["word_correct_ja_enabled"] is False

    def test_european_correction_is_untouched(self) -> None:
        """**欧文の補正は切らない。** held-out で CER -2.09pt の実績がある."""
        assert AppSettings().word_correct_enabled is True

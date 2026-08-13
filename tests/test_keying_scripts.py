"""打鍵原稿生成 (自己打鍵データ収集用) のテスト."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.finetune.keying_scripts import (
    apply_script_label,
    estimate_duration_sec,
    generate_keying_script,
    write_script_files,
)
from src.tokens.morse_tokens import text_to_codes


class TestGenerateKeyingScript:
    def test_japanese_tokenizable(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(30):
            text = generate_keying_script(rng, "japanese")
            codes = text_to_codes(text, "japanese")
            assert codes, f"tokenize failed: {text!r}"

    def test_european_tokenizable(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(30):
            text = generate_keying_script(rng, "european")
            codes = text_to_codes(text, "european")
            assert codes, f"tokenize failed: {text!r}"

    def test_duration_within_ctc_range_at_24wpm(self) -> None:
        # 1件 3〜30 秒 (CTC範囲) に収まる長さであること
        rng = np.random.default_rng(2)
        for mode in ("japanese", "european"):
            for _ in range(30):
                text = generate_keying_script(rng, mode)
                dur = estimate_duration_sec(text, mode, wpm=24.0)
                assert 3.0 <= dur <= 30.0, f"{mode}: {dur:.1f}s for {text!r}"

    def test_reproducible(self) -> None:
        t1 = generate_keying_script(np.random.default_rng(7), "japanese")
        t2 = generate_keying_script(np.random.default_rng(7), "japanese")
        assert t1 == t2


class TestWriteScriptFiles:
    def test_writes_numbered_files(self, tmp_path: Path) -> None:
        paths = write_script_files(tmp_path, mode="japanese", count=5, seed=0)
        assert len(paths) == 5
        assert all(p.exists() for p in paths)
        assert paths[0].name == "script_ja_01.txt"

    def test_deterministic_content(self, tmp_path: Path) -> None:
        p1 = write_script_files(tmp_path / "a", mode="european", count=3, seed=9)
        p2 = write_script_files(tmp_path / "b", mode="european", count=3, seed=9)
        for a, b in zip(p1, p2, strict=True):
            assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


class TestApplyScriptLabel:
    def test_replaces_body_keeps_header(self, tmp_path: Path) -> None:
        txt = tmp_path / "20260701_120000_japanese.txt"
        txt.write_text(
            "mode: japanese\nsample_rate: 8000\n---\nゴミデコード\n",
            encoding="utf-8",
        )
        apply_script_label(txt, "コンバンハ、テンキ ハ ハレ")
        content = txt.read_text(encoding="utf-8")
        assert "mode: japanese" in content
        assert "コンバンハ、テンキ ハ ハレ" in content
        assert "ゴミデコード" not in content

    def test_rejects_untokenizable_text(self, tmp_path: Path) -> None:
        txt = tmp_path / "20260701_120001_japanese.txt"
        txt.write_text("mode: japanese\n---\n\n", encoding="utf-8")
        with pytest.raises(ValueError):
            apply_script_label(txt, "漢字はダメ")


class TestNormalizeLabelMarkers:
    def test_converts_display_prosigns_to_input_markers(self) -> None:
        from src.finetune.keying_scripts import normalize_label_markers
        assert normalize_label_markers("[ホレ]テンキ[ラタ]") == "{HORE}テンキ{RATA}"
        assert normalize_label_markers("TU 73 [SK]") == "TU 73 {SK}"

    def test_normalized_japanese_is_tokenizable(self) -> None:
        from src.finetune.keying_scripts import normalize_label_markers
        norm = normalize_label_markers("[ホレ]テンキ ハ ハレ[ラタ]")
        # 正規化後は text_to_codes を通る (KeyError にならない)
        codes = text_to_codes(norm, "japanese")
        assert len(codes) > 0

    def test_normalized_european_is_tokenizable(self) -> None:
        from src.finetune.keying_scripts import normalize_label_markers
        norm = normalize_label_markers("CQ DE JA0XYZ [SK]")
        codes = text_to_codes(norm, "european")
        assert len(codes) > 0

    def test_no_markers_passthrough(self) -> None:
        from src.finetune.keying_scripts import normalize_label_markers
        assert normalize_label_markers("CQ DE JA0XYZ K") == "CQ DE JA0XYZ K"

    def test_question_mark_is_left_as_is(self) -> None:
        from src.finetune.keying_scripts import normalize_label_markers
        # ? は疑問符符号として扱う (設計書 §4.7)。置換しない。
        assert normalize_label_markers("QRL?") == "QRL?"

    def test_unsupported_prosign_raises(self) -> None:
        from src.finetune.keying_scripts import normalize_label_markers
        with pytest.raises(ValueError, match=r"\[SN\]|\[KN\]|\[HH\]"):
            normalize_label_markers("R [SN] TU")

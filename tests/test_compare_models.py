"""モデル比較ツールの引数解釈と TER 計算のテスト."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compare_models import (
    parse_ckpt,
    parse_wav,
    read_reference,
    token_error_rate,
)
from src.tokens.morse_tokens import TOKEN_TO_ID, text_to_codes


class TestParseWav:
    def test_モードとパスを分ける(self) -> None:
        spec = parse_wav("japanese:rec/20260827.wav")
        assert spec.mode == "japanese"
        assert spec.path == Path("rec/20260827.wav")

    def test_前後の空白を許す(self) -> None:
        assert parse_wav(" european : a.wav ").path == Path("a.wav")

    def test_モードが無ければエラー(self) -> None:
        with pytest.raises(ValueError, match="モード"):
            parse_wav("rec.wav")

    def test_未知のモードはエラー(self) -> None:
        with pytest.raises(ValueError, match="european"):
            parse_wav("morse:rec.wav")

    def test_ドライブレターをモードと誤認しない(self) -> None:
        """**Windows の `C:\\...` を「モード C」と読んではいけない。**"""
        with pytest.raises(ValueError, match="モード"):
            parse_wav(r"C:\rec\20260827.wav")


class TestParseCkpt:
    def test_ラベルとパスを分ける(self) -> None:
        assert parse_ckpt("現行=models/full/best_infer.pt") == (
            "現行", Path("models/full/best_infer.pt"))

    def test_ラベル省略ならファイル名を使う(self) -> None:
        label, path = parse_ckpt("models/ft_v5_wide/cw_v5wide.onnx")
        assert label == "cw_v5wide"
        assert path == Path("models/ft_v5_wide/cw_v5wide.onnx")


class TestReadReference:
    def test_区切りの後を本文とする(self, tmp_path: Path) -> None:
        wav = tmp_path / "a.wav"
        wav.with_suffix(".txt").write_text(
            "mode: european\nsample_rate: 8000\n---\nCQ TEST\n", encoding="utf-8")
        assert read_reference(wav) == "CQ TEST"

    def test_区切りが無ければ全体を本文とする(self, tmp_path: Path) -> None:
        wav = tmp_path / "a.wav"
        wav.with_suffix(".txt").write_text("CQ TEST\n", encoding="utf-8")
        assert read_reference(wav) == "CQ TEST"

    def test_TXTが無ければNone(self, tmp_path: Path) -> None:
        assert read_reference(tmp_path / "a.wav") is None


class TestTokenErrorRate:
    def test_完全一致なら0(self) -> None:
        ids = [TOKEN_TO_ID[c] for c in text_to_codes("CQ TEST", "european")]
        assert token_error_rate("CQ TEST", ids, "european") == 0.0

    def test_1トークン違えば1件ぶん(self) -> None:
        ids = [TOKEN_TO_ID[c] for c in text_to_codes("CQ TEST", "european")]
        n = len(ids)
        assert token_error_rate("CQ TEST", ids[:-1], "european") == pytest.approx(1 / n)

    def test_空の正解ならNone(self) -> None:
        assert token_error_rate("", [1, 2], "european") is None

    def test_トークン化できない仮ラベルはNone(self) -> None:
        """アプリの録音 TXT はデコード結果 (``_`` を含む) が仮ラベルとして入る."""
        assert token_error_rate("AB_CD", [1, 2], "european") is None

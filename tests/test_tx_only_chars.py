"""送信専用符号 (TX_ONLY) のテスト.

受信語彙 (トークン集合) に触れずに送信だけ拡張していることを検証する。
設計書: docs/superpowers/specs/2026-08-13-tx-only-chars-design.md
"""
from __future__ import annotations

import hashlib

import pytest

from src.tokens.morse_tokens import (
    SPECIAL_INPUT_MARKERS,
    TOKEN_TO_ID,
    TX_INPUT_MARKERS,
    TX_ONLY_EUROPEAN_CHAR_TO_CODE,
    TX_ONLY_MARKERS,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
    text_to_codes,
)
from tests.data.eGov_morse_reference import (
    EGOV_EUROPEAN_PUNCT_TX_ONLY,
    EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB,
)


class TestReferenceMatch:
    """参照表 (実装と独立) との照合."""

    def test_欧文送信専用は参照表と一致(self) -> None:
        assert TX_ONLY_EUROPEAN_CHAR_TO_CODE == EGOV_EUROPEAN_PUNCT_TX_ONLY

    def test_和文括弧マーカーは参照表と一致(self) -> None:
        assert TX_ONLY_MARKERS == {
            "{KAKKO}": EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB["「"],
            "{TOJI}": EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB["」"],
        }


class TestTokenSetUnchanged:
    """受信語彙の不変 (これが崩れたら学習済みモデルが無意味になる)."""

    def test_語彙サイズ(self) -> None:
        assert VOCAB_SIZE == 73

    def test_wordbreak_id(self) -> None:
        assert WORD_BREAK_TOKEN_ID == 72

    def test_全トークンIDのハッシュ(self) -> None:
        digest = hashlib.sha256(repr(sorted(TOKEN_TO_ID.items())).encode()).hexdigest()
        assert digest[:16] == "d5b369163e2c0881"

    def test_合成器用マーカーは3個のまま(self) -> None:
        assert set(SPECIAL_INPUT_MARKERS) == {"{HORE}", "{RATA}", "{SK}"}


class TestTextToCodes:
    """text_to_codes のフラグ挙動."""

    def test_欧文の送信専用文字(self) -> None:
        assert text_to_codes("(A)", "european", include_tx_only=True) == [
            "-・--・", "・-", "-・--・-",
        ]

    def test_欧文の記号ぜんぶ(self) -> None:
        got = text_to_codes(":'\"×", "european", include_tx_only=True)
        assert got == ["---・・・", "・----・", "・-・・-・", "-・・-"]

    def test_和文括弧マーカー(self) -> None:
        assert text_to_codes("{KAKKO}アイ{TOJI}", "japanese", include_tx_only=True) == [
            "-・--・-", "--・--", "・-", "・-・・-・",
        ]

    def test_既定では欧文送信専用文字は従来どおりKeyError(self) -> None:
        with pytest.raises(KeyError):
            text_to_codes("(", "european")

    def test_既定では括弧マーカーは従来どおりKeyError(self) -> None:
        with pytest.raises(KeyError):
            text_to_codes("{KAKKO}", "japanese")

    def test_TX_INPUT_MARKERSは合成器マーカーを含む(self) -> None:
        assert set(TX_INPUT_MARKERS) == set(SPECIAL_INPUT_MARKERS) | {"{KAKKO}", "{TOJI}"}

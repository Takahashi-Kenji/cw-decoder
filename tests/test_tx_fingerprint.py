"""符号表の指紋のテスト.

指紋は**両 PC のリポジトリのずれを見つけるため**にある。決定的であること、
表が変われば変わることの 2 つが要件。
"""
from __future__ import annotations

import re

from src.tokens.morse_tokens import (
    EUROPEAN_CHAR_TO_CODE,
    JAPANESE_CHAR_TO_CODES,
    TX_ONLY_EUROPEAN_CHAR_TO_CODE,
    TX_ONLY_MARKERS,
)
from src.tx.fingerprint import FINGERPRINT_LENGTH, tokens_fingerprint


def test_決定的である() -> None:
    assert tokens_fingerprint() == tokens_fingerprint()


def test_16進の固定長である() -> None:
    value = tokens_fingerprint()
    assert len(value) == FINGERPRINT_LENGTH
    assert re.fullmatch(r"[0-9a-f]+", value)


def test_欧文表が変わると指紋が変わる(monkeypatch) -> None:
    before = tokens_fingerprint()
    monkeypatch.setitem(EUROPEAN_CHAR_TO_CODE, "A", "----")
    assert tokens_fingerprint() != before


def test_和文表が変わると指紋が変わる(monkeypatch) -> None:
    before = tokens_fingerprint()
    key = next(iter(JAPANESE_CHAR_TO_CODES))
    monkeypatch.setitem(JAPANESE_CHAR_TO_CODES, key, ("----",))
    assert tokens_fingerprint() != before


def test_送信専用欧文表が変わると指紋が変わる(monkeypatch) -> None:
    # 送れる文字の集合が変わるので、NN 語彙に無くても指紋には効く必要がある。
    # ここが漏れると、片方の PC だけ更新されても指紋が一致してしまう。
    before = tokens_fingerprint()
    key = next(iter(TX_ONLY_EUROPEAN_CHAR_TO_CODE))
    monkeypatch.setitem(TX_ONLY_EUROPEAN_CHAR_TO_CODE, key, "----")
    assert tokens_fingerprint() != before


def test_送信専用マーカー表が変わると指紋が変わる(monkeypatch) -> None:
    before = tokens_fingerprint()
    key = next(iter(TX_ONLY_MARKERS))
    monkeypatch.setitem(TX_ONLY_MARKERS, key, "----")
    assert tokens_fingerprint() != before

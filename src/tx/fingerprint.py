"""符号表の指紋.

なぜ要るか
----------
**アプリ (GPU PC) と打鍵 CLI (無線機 PC) は別のリポジトリの複製を読む。**
ずれると、画面が「送れる」と言った文字を打鍵側が撥ねる (逆はない — 打鍵側が
必ず再検証するため)。接続時にこの指紋を突き合わせ、違えば警告する。

``morse_tokens.py`` は符号定義の唯一の真正ソースであり、変更すると
``scripts/export_tokens.py`` の再実行が要る (CLAUDE.md)。
**このモジュールは読むだけで、あちらを一切変更しない。**
"""
from __future__ import annotations

import hashlib

from src.tokens.morse_tokens import (
    EUROPEAN_CHAR_TO_CODE,
    JAPANESE_CHAR_TO_CODES,
    SPECIAL_INPUT_MARKERS,
)

# 指紋の桁数。人が目で読んで違いに気づける程度で足りる。
FINGERPRINT_LENGTH = 8


def tokens_fingerprint() -> str:
    """符号表から決定的な短い指紋を作る.

    **並び順に依存させない。** 辞書の反復順は保証されるが、表の作られ方が
    変わったときに中身が同じなのに指紋が変わると偽の警告になる。
    """
    digest = hashlib.sha256()
    for char, code in sorted(EUROPEAN_CHAR_TO_CODE.items()):
        digest.update(f"E{char}={code}\n".encode())
    for char, codes in sorted(JAPANESE_CHAR_TO_CODES.items()):
        digest.update(f"J{char}={'|'.join(codes)}\n".encode())
    for marker, code in sorted(SPECIAL_INPUT_MARKERS.items()):
        digest.update(f"M{marker}={code}\n".encode())
    return digest.hexdigest()[:FINGERPRINT_LENGTH]


__all__ = ["FINGERPRINT_LENGTH", "tokens_fingerprint"]

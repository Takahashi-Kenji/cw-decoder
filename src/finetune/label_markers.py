"""ラベル表示マーカーの正規化 (torch 非依存の純関数).

デコード表示のラベル (``[ホレ]`` 等の表示プロサイン表記) を
``text_to_codes`` が受理する合成入力マーカー (``{HORE}`` 等) へ変換する。

``src.finetune.dataset`` 経由の torch 依存を避けるため、このモジュールは
標準ライブラリ以外に何も import しない。``onair_clip.py`` はこのモジュール
から直接 ``normalize_label_markers`` を import することで torch 非依存性を
保つ。
"""
from __future__ import annotations

# TokenConverter の表示表記 → text_to_codes が受理する合成入力マーカー
_MARKER_NORMALIZE: dict[str, str] = {
    "[ホレ]": "{HORE}",
    "[ラタ]": "{RATA}",
    "[SK]": "{SK}",
}
# 合成入力マーカーが未定義の表示プロサイン (採用不可)
_UNSUPPORTED_MARKERS: tuple[str, ...] = ("[SN]", "[KN]", "[HH]")


def normalize_label_markers(text: str) -> str:
    """デコード表示のラベルを text_to_codes が受理する形式へ正規化する.

    ``[ホレ]``→``{HORE}`` のように表示プロサインを合成入力マーカーへ変換する。
    ``?`` は疑問符符号 (``・・--・・``) として扱い、置換しない (設計書 §4.7)。
    合成入力マーカーが未定義の ``[SN]`` / ``[KN]`` / ``[HH]`` を含む場合は
    その区間を採用できないため ``ValueError``。
    """
    for bad in _UNSUPPORTED_MARKERS:
        if bad in text:
            raise ValueError(
                f"合成入力マーカー未定義のプロサインを含みます: {bad} — この区間は採用不可"
            )
    result = text
    for display, marker in _MARKER_NORMALIZE.items():
        result = result.replace(display, marker)
    return result


__all__ = ["normalize_label_markers"]

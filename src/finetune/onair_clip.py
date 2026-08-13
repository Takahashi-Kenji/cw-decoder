"""オンエア録音の切り出し・ラベル検証 (torch 非依存の純関数)."""
from __future__ import annotations

import numpy as np

from src.finetune.label_markers import normalize_label_markers
from src.tokens.morse_tokens import Mode, text_to_codes


def clip_segment(
    wave: np.ndarray, sample_rate: int, start_s: float, end_s: float
) -> np.ndarray:
    """波形から ``[start_s, end_s)`` 秒の区間を切り出す.

    範囲が波形長を超える / start >= end の場合は ``ValueError``。
    """
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"不正な区間: start={start_s}, end={end_s}")
    start = int(round(start_s * sample_rate))
    end = int(round(end_s * sample_rate))
    if end > wave.size:
        raise ValueError(
            f"区間が波形長を超えます: end={end_s}s > {wave.size / sample_rate:.2f}s"
        )
    return np.ascontiguousarray(wave[start:end], dtype=np.float32)


def validate_label(text: str, mode: Mode) -> str:
    """ラベルを正規化してトークン化可能か検証し、正規化済みテキストを返す.

    ``[ホレ]``→``{HORE}`` 等へ正規化した上で ``text_to_codes`` を通す。
    トークン化不能 (漢字混入・未対応プロサイン) や空文字なら ``ValueError``。
    """
    normalized = normalize_label_markers(text).strip()
    if not normalized:
        raise ValueError("ラベルが空です")
    try:
        codes = text_to_codes(normalized, mode)
    except KeyError as exc:
        raise ValueError(f"トークン化できない文字を含みます: {exc}") from exc
    if not codes:
        raise ValueError("符号列が空になりました")
    return normalized


__all__ = ["clip_segment", "validate_label"]

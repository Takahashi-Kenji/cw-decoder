"""WORD_BREAK トークンの抑制ポリシー (再学習なしのデコード側レバー).

keyed_val の誤り 110 件のうち 33 件が WORD_BREAK の過剰出力であるため、
argmax **前** のロジットバイアス β と argmax **後** の確信度閾値 τ の 2 つで
過剰出力を抑える。設計は
``docs/superpowers/specs/2026-07-31-word-break-threshold-design.md`` を参照.

このモジュールは掃引実験 (``scripts/sweep_word_break.py``) 専用であり、
現時点では本番の推論・評価経路 (``src/eval/harness.py`` 等) のどこからも
呼ばれていない。掃引の結論と不採用の理由は ``docs/word_break_threshold_result.md``
を参照.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.tokens.morse_tokens import WORD_BREAK_TOKEN_ID
from src.train.decode import CTCDecodeResult


@dataclass(frozen=True)
class WordBreakPolicy:
    """WORD_BREAK 抑制のパラメータ.

    Attributes:
        logit_bias: argmax 前に WORD_BREAK の log 確率へ加算する値 (nat).
            負で抑制、0 で無効. 掃引の対称性確認に使うため正も許容する.
        conf_threshold: argmax 後、この確信度**未満**の WORD_BREAK を除去する.
            0.0 で無効.
    """

    logit_bias: float = 0.0
    conf_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.logit_bias):
            raise ValueError(f"logit_bias must be finite, got {self.logit_bias}")
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError(
                f"conf_threshold must be in [0, 1], got {self.conf_threshold}"
            )

    @property
    def is_identity(self) -> bool:
        """何もしないポリシーか (現行挙動と完全一致する)."""
        return self.logit_bias == 0.0 and self.conf_threshold == 0.0


def apply_logit_bias(log_probs: Tensor, bias: float) -> Tensor:
    """``(B, T, V)`` log-softmax の WORD_BREAK 列に ``bias`` を加えて再正規化.

    WORD_BREAK 以外のトークン同士の確率比は保存される (同じ定数で割るため).

    Args:
        log_probs: log-softmax 済みの ``(B, T, V)``.
        bias: WORD_BREAK の log 確率へ加算する値 (nat). 負で抑制.

    Returns:
        再正規化済みの ``(B, T, V)``. ``bias == 0.0`` なら入力をそのまま返す.

    Raises:
        ValueError: 次元数が 3 でない / bias が非有限 /
            語彙に WORD_BREAK が含まれない (WORD_BREAK 追加前の古い ckpt).
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be 3D, got {tuple(log_probs.shape)}")
    if not math.isfinite(bias):
        raise ValueError(f"bias must be finite, got {bias}")
    if bias == 0.0:
        return log_probs
    vocab = log_probs.size(-1)
    if vocab <= WORD_BREAK_TOKEN_ID:
        raise ValueError(
            f"語彙サイズ {vocab} に WORD_BREAK (id={WORD_BREAK_TOKEN_ID}) が含まれない. "
            "WORD_BREAK 追加前の古いチェックポイントの可能性がある"
        )
    biased = log_probs.clone()
    biased[..., WORD_BREAK_TOKEN_ID] += bias
    return torch.log_softmax(biased, dim=-1)


def filter_word_breaks(result: CTCDecodeResult, threshold: float) -> CTCDecodeResult:
    """確信度が ``threshold`` 未満の WORD_BREAK **だけ** を除去する.

    WORD_BREAK 以外のトークンは確信度が低くても残す. 入力は変更しない.

    Args:
        result: ``ctc_greedy_decode`` の 1 サンプル分の結果.
        threshold: [0, 1]. 0.0 なら入力をそのまま返す.

    Returns:
        除去後の新しい ``CTCDecodeResult``.

    Raises:
        ValueError: threshold が [0, 1] 外 / token_ids と confidences の長さ不一致.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if len(result.token_ids) != len(result.confidences):
        raise ValueError(
            f"token_ids length {len(result.token_ids)} != confidences length "
            f"{len(result.confidences)}"
        )
    if threshold == 0.0:
        return result
    ids: list[int] = []
    confs: list[float] = []
    for tid, conf in zip(result.token_ids, result.confidences, strict=True):
        if tid == WORD_BREAK_TOKEN_ID and conf < threshold:
            continue
        ids.append(tid)
        confs.append(conf)
    return CTCDecodeResult(token_ids=ids, confidences=confs)


__all__ = ["WordBreakPolicy", "apply_logit_bias", "filter_word_breaks"]

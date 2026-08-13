"""CTC デコード (greedy) と確信度抽出.

要件 §3.4.3:

- CTC greedy decoding を基本
- トークンごとの事後確率 (確信度) を取得
- 確信度閾値未満は ``?`` 置換 (これは ``TokenConverter`` 側で処理)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.tokens.morse_tokens import BLANK_TOKEN_ID


@dataclass
class CTCDecodeResult:
    """1 サンプル分のデコード結果."""

    token_ids: list[int]            # blank/重複除去後の token ID 列
    confidences: list[float]        # 各 token の最大事後確率


def ctc_greedy_decode(
    log_probs: Tensor,
    input_lengths: Tensor | None = None,
    blank_id: int = BLANK_TOKEN_ID,
) -> list[CTCDecodeResult]:
    """``(B, T, V)`` の log-softmax 出力から CTC greedy decode.

    Args:
        log_probs: ``(B, T, V)`` の log-softmax (model 出力に ``log_softmax`` 適用).
        input_lengths: 各サンプルの有効時間長. ``None`` で全フレーム使用.
        blank_id: CTC blank の token ID.

    Returns:
        各バッチサンプルのデコード結果リスト.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be 3D, got {log_probs.shape}")
    b, t, _ = log_probs.shape

    # 各フレームの argmax と max prob
    probs = log_probs.exp()
    max_prob, argmax = probs.max(dim=-1)            # (B, T)

    if input_lengths is None:
        input_lengths = torch.full((b,), t, dtype=torch.long)

    results: list[CTCDecodeResult] = []
    am_cpu = argmax.detach().cpu().numpy()
    mp_cpu = max_prob.detach().cpu().numpy()
    il_cpu = input_lengths.detach().cpu().numpy()

    for i in range(b):
        length = int(il_cpu[i])
        ids: list[int] = []
        confs: list[float] = []
        prev = -1
        # 各 token の確信度はその token を出力したフレーム群の最大値
        # (連続重複を collapse する間、最大を取る)
        current_max = 0.0
        for j in range(length):
            tok = int(am_cpu[i, j])
            conf = float(mp_cpu[i, j])
            if tok == prev:
                if tok != blank_id:
                    current_max = max(current_max, conf)
                continue
            # 切り替わり: 前の token を確定 (blank でなければ)
            if prev != -1 and prev != blank_id:
                ids.append(prev)
                confs.append(current_max)
            prev = tok
            current_max = conf
        # 末尾の token を確定
        if prev != -1 and prev != blank_id:
            ids.append(prev)
            confs.append(current_max)
        results.append(CTCDecodeResult(token_ids=ids, confidences=confs))
    return results


__all__ = ["CTCDecodeResult", "ctc_greedy_decode"]

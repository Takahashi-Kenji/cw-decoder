"""CTC greedy デコード (フレーム位置付き) — **numpy だけで動く**.

``engine.py`` から切り出した。切り出した理由は配布物の大きさである:
PyTorch は 2.8 GB あり、インストーラの 7 割を占める。デコードに必要なのは
``exp`` と ``argmax`` だけなので、ここを numpy にしておけば
ONNX 推論の経路が torch を一切読み込まずに済む。

**このモジュールは torch を import してはいけない。**
``tests/test_no_torch_import.py`` が歯止めになっている。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tokens.morse_tokens import BLANK_TOKEN_ID, WORD_BREAK_TOKEN_ID


@dataclass(frozen=True)
class FrameToken:
    """1 つのデコード済みトークン (フレーム位置付き)."""

    token_id: int
    confidence: float
    frame_start: int    # この token を出力した最初のフレーム
    frame_end: int      # 最後のフレーム (inclusive)


def ctc_greedy_decode_frames(
    log_probs: np.ndarray,
    blank_id: int = BLANK_TOKEN_ID,
    word_break_bias: float = 0.0,
) -> list[list[FrameToken]]:
    """``(B, T, V)`` の log-softmax から ``FrameToken`` 列を返す.

    各 ``FrameToken`` は ``[frame_start, frame_end]`` の連続フレームから
    出力された 1 つのトークン。blank と重複は collapse 済み。

    確信度はそのランの中の**最大値**を採る (平均ではない)。ランの端は
    隣のトークンへ移る過渡で必ず下がるため、平均を採ると実際より低く出る。

    Args:
        word_break_bias: argmax の**前**に WORD_BREAK の log 確率へ足す値 (nat)。
            負で語間が出にくく、正で出やすくなる。

            **モードで最適値が違う。** held-out の実測 (2026-08-25) では、
            語間の出力数が 欧文 50/正解 50 とぴったりなのに対し、
            和文は 52/正解 29 と 1.8 倍に膨らむ。和文の挿入誤り 34 個のうち
            27 個が語間で、欧文と和文の TER 差はほぼこれで説明できる。
            掃引の結果は 欧文 −1.0 / 和文 −5.0
            (`docs/word_break_bias_by_mode.md`)。

            確信度はブラウザ版 (`web/src/decode/ctc.ts`) と同じく
            **バイアス後の値**を採る。両実装がずれると golden テストが破れる。
    """
    if log_probs.ndim != 3:
        raise ValueError(f"log_probs must be 3D, got {log_probs.shape}")

    if word_break_bias != 0.0:
        log_probs = log_probs.astype(np.float32, copy=True)
        log_probs[..., WORD_BREAK_TOKEN_ID] += word_break_bias
    probs = np.exp(log_probs.astype(np.float32, copy=False))
    am = probs.argmax(axis=-1)
    mp = probs.max(axis=-1)
    b, t = am.shape

    results: list[list[FrameToken]] = []
    for i in range(b):
        out: list[FrameToken] = []
        prev = -1
        run_start = 0
        run_max = 0.0
        for j in range(t):
            tok = int(am[i, j])
            conf = float(mp[i, j])
            if tok == prev:
                if tok != blank_id:
                    run_max = max(run_max, conf)
                continue
            # 切り替わり: 前のランを確定
            if prev != -1 and prev != blank_id:
                out.append(FrameToken(
                    token_id=prev, confidence=run_max,
                    frame_start=run_start, frame_end=j - 1,
                ))
            prev = tok
            run_start = j
            run_max = conf
        # 末尾
        if prev != -1 and prev != blank_id:
            out.append(FrameToken(
                token_id=prev, confidence=run_max,
                frame_start=run_start, frame_end=t - 1,
            ))
        results.append(out)
    return results


__all__ = ["FrameToken", "ctc_greedy_decode_frames"]

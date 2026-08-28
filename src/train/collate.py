"""DataLoader 用 collate_fn (可変長波形のパディング)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


# 詰め長をこの倍数に切り上げる (サンプル数)。8 kHz で 2.0 秒。
#
# **バッチごとに詰め長が変わると、GPU のキャッシュアロケータが形状ごとに
# ブロックを取り、古いブロックを再利用できずに積み上がる。** 実測では
# 4.3M パラメータ (17 MB) のモデルが専用 15,035 MB を抱えて 16 GB のカードを
# 埋め、2,794 MB がシステムメモリへ退避して**スループットが 2.21 → 0.87 sps**
# に落ちた。OOM で落ちないので静かに遅くなる。
#
# 量子化すると形状の種類が数千から十数個に減り、ブロックが再利用できる。
# 代償は 1 バッチあたり最大 2 秒ぶんの無駄な計算だが、``input_lengths`` は
# 実長から出すので**学習結果は変わらない** (CTC は詰めた分を見ない)。
#
# **既定はオフ (``pad_bucket=1``)。** 評価経路・ゴールデン・既存テストの挙動を
# 変えないため、学習ループ側から明示的に渡す。
PAD_BUCKET_SAMPLES: int = 16000


def cw_collate(
    batch: Sequence[tuple[Tensor, Tensor]],
    pad_bucket: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(waveform, token_ids)`` のシーケンスをミニバッチにまとめる.

    Returns:
        - ``waveforms_padded``: ``(B, T_max)`` float32. 0 パディング.
        - ``targets_concat``: 各サンプルの token_ids を一連で連結 (CTC convention).
        - ``wave_lengths``: ``(B,)`` int32. 実波形長.
        - ``target_lengths``: ``(B,)`` int32. 各サンプルの token 数.
    """
    if not batch:
        raise ValueError("Empty batch")
    waveforms = [item[0] for item in batch]
    token_ids = [item[1] for item in batch]

    wave_lengths = torch.tensor(
        [w.numel() for w in waveforms], dtype=torch.int32
    )
    target_lengths = torch.tensor(
        [t.numel() for t in token_ids], dtype=torch.int32
    )

    max_wave = int(wave_lengths.max().item())
    if pad_bucket > 1:
        max_wave = -(-max_wave // pad_bucket) * pad_bucket      # 切り上げ
    waveforms_padded = torch.zeros(
        len(batch), max_wave, dtype=torch.float32
    )
    for i, w in enumerate(waveforms):
        waveforms_padded[i, : w.numel()] = w

    targets_concat = torch.cat([t.to(torch.long) for t in token_ids])

    return waveforms_padded, targets_concat, wave_lengths, target_lengths


__all__ = ["cw_collate"]

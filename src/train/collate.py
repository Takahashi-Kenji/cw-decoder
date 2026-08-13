"""DataLoader 用 collate_fn (可変長波形のパディング)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def cw_collate(
    batch: Sequence[tuple[Tensor, Tensor]],
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
    waveforms_padded = torch.zeros(
        len(batch), max_wave, dtype=torch.float32
    )
    for i, w in enumerate(waveforms):
        waveforms_padded[i, : w.numel()] = w

    targets_concat = torch.cat([t.to(torch.long) for t in token_ids])

    return waveforms_padded, targets_concat, wave_lengths, target_lengths


__all__ = ["cw_collate"]

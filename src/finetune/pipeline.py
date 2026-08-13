"""FT 用パイプライン補助 (合成データとの混合、検証セット構築)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.finetune.dataset import RealSignalDataset
from src.synth.dataset import ConfigSampler, DefaultConfigSampler, MorseSynthDataset
from src.synth.noise import RealNoisePool
from src.tokens.morse_tokens import Mode


class MixedRealSynthDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """実信号と合成サンプルを比率混合する IterableDataset.

    合成のみで FT すると分布シフトが起きないため、実信号に偏らせつつ
    一部合成でカタストロフィックフォゲッティングを抑制する.
    """

    def __init__(
        self,
        real_dataset: RealSignalDataset,
        mode_mix: dict[Mode, float],
        real_ratio: float = 0.7,
        seed: int | None = None,
        max_samples: int | None = None,
        sample_rate: int = 8000,
        config_sampler: ConfigSampler | None = None,
        noise_pool: RealNoisePool | None = None,
        noise_prob: float = 0.8,
        noise_snr_range: tuple[float, float] = (-5.0, 15.0),
        tone_freq_range: tuple[float, float] | None = None,
        *,
        hand_keying: bool = False,
        extreme_tail: bool = True,
        electronic_keyer_prob: float = 0.25,
    ) -> None:
        if not 0.0 < real_ratio <= 1.0:
            raise ValueError(f"real_ratio must be in (0, 1], got {real_ratio}")
        self.real_dataset = real_dataset
        self.synth_dataset = MorseSynthDataset(
            mode_mix=mode_mix,
            config_sampler=config_sampler,
            seed=seed,
            max_samples=None,
            sample_rate=sample_rate,
            noise_pool=noise_pool,
            noise_prob=noise_prob,
            noise_snr_range=noise_snr_range,
            tone_freq_range=tone_freq_range,
            hand_keying=hand_keying,
            extreme_tail=extreme_tail,
            electronic_keyer_prob=electronic_keyer_prob,
        )
        self.real_ratio = real_ratio
        self.seed = seed
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        synth_iter = iter(self.synth_dataset)
        count = 0
        while self.max_samples is None or count < self.max_samples:
            if rng.random() < self.real_ratio:
                idx = int(rng.integers(0, len(self.real_dataset)))
                yield self.real_dataset[idx]
            else:
                yield next(synth_iter)
            count += 1


def split_train_validation(
    samples: list,
    validation_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list, list]:
    """サンプルを train / validation に分割 (再現可能)."""
    rng = np.random.default_rng(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * validation_ratio))
    val_idx = sorted(indices[:n_val])
    train_idx = sorted(indices[n_val:])
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


__all__ = [
    "MixedRealSynthDataset",
    "split_train_validation",
]

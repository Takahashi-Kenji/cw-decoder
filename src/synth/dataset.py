"""``torch.utils.data.IterableDataset`` ラッパと固定評価セット生成.

学習時の DataLoader でオンザフライ合成を並列に行うためのインターフェース.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.synth.keying import KeyingParams
from src.synth.noise import RealNoisePool, add_real_noise, effective_snr_db
from src.synth.synthesizer import SynthConfig, SynthResult, synthesize_random
from src.synth.text_generator import TextGenConfig
from src.tokens.morse_tokens import Mode


ConfigSampler = Callable[[np.random.Generator], SynthConfig]


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """[lo, hi] から対数一様に 1 つ引く.

    一様分布と違い小さい値に確率質量が集まる。手打ちの σ に使う理由は、
    上限 (長音 σ 1.30 dot) が人間にも解読できない領域であり、そこを一様に
    引くとモデルが「曖昧なら当てずっぽう」に振れてきれいな符号の精度を
    落とす恐れがあるため (設計書 §3.2)。lo は 0 より大きいこと。
    """
    if lo <= 0.0:
        raise ValueError(f"lo must be > 0 for log-uniform, got {lo}")
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


class DefaultConfigSampler:
    """要件 §3.2 の範囲で SynthConfig をランダムサンプリングする標準サンプラ.

    DataLoader (multiprocessing spawn) でピクル可能なよう module-level クラス
    として実装する.
    """

    def __init__(
        self,
        mode: Mode,
        tone_freq_range: tuple[float, float] = (400.0, 900.0),
        effective_snr_range: tuple[float, float] | None = None,
        *,
        hand_keying: bool = False,
        extreme_tail: bool = True,
        electronic_keyer_prob: float = 0.25,
    ) -> None:
        self.mode: Mode = mode
        # 実ノイズ混合 FT ではユーザーの受信ピッチ近傍 (例: 600±50Hz) に絞る
        self.tone_freq_range = tone_freq_range
        # 実効SNR (BPF後) モードの範囲. None なら従来通り nominal SNR で AWGN を加える.
        self.effective_snr_range = effective_snr_range
        # 手打ちの分布を使うか。False なら従来 (エレキー相当のみ) の分布に戻る。
        # 注意: この分岐を True にすると、common dict 化により乱数の描画順が
        # 従来 (Task 6 以前) と変わるため、同一 seed でも従来実装とビット単位では
        # 一致しなくなる (乱数列そのものは決定的なままなので再現性は保たれる)。
        self.hand_keying = hand_keying
        # 長音 σ の極端テール (0.70〜1.30 dot) を含めるか。設計書 §3.2 の A/B 用。
        self.extreme_tail = extreme_tail
        # 手打ち分布のうち、この確率で従来 (エレキー相当) の分布を引く。
        #
        # 各パラメータを独立に引くと、周辺分布にはエレキー相当の値が入っていても
        # 「間隔ちょうど 1/3/7 かつ σ が小さい」という**組み合わせ**は事実上出ない
        # (20,000 サンプルで 0 件を実測)。設計書が要求する「エレキー相当を部分集合
        # として残す」を満たすには、混合成分として明示的に引く必要がある。
        #
        # synth_val の固定評価セットはまさにこの一点なので、ここが空だと
        # 「実信号は上がったが合成が下がる」を再演することになる
        # (このプロジェクトは過去 3 回それで施策を不採用にしている)。
        if not 0.0 <= electronic_keyer_prob <= 1.0:
            raise ValueError(
                f"electronic_keyer_prob must be in [0, 1], got {electronic_keyer_prob}"
            )
        self.electronic_keyer_prob = electronic_keyer_prob

    def __call__(self, rng: np.random.Generator) -> SynthConfig:
        common = dict(
            wpm=float(rng.uniform(8.0, 50.0)),                 # 拡張: 10-40 → 8-50
            tone_freq_hz=float(rng.uniform(*self.tone_freq_range)),
            tone_drift_hz_per_sec=float(rng.uniform(-50.0, 50.0)),
            rise_fall_ms=float(rng.uniform(3.0, 10.0)),
            pre_silence_sec=float(rng.uniform(0.0, 0.3)),
            post_silence_sec=float(rng.uniform(0.0, 0.3)),
        )
        # エレキー相当を混合成分として引く (I-1)。hand_keying=False のときは
        # 常に従来分岐なのでコインは引かない (乱数列を無駄に変えないため)
        use_hand = self.hand_keying and rng.random() >= self.electronic_keyer_prob
        if use_hand:
            # 実測 (設計書 §2.2) に合わせた手打ちの分布。
            # 短点は正確 (σ 0.064〜0.107 dot) で長音だけ暴れる (σ 0.681〜1.195 dot)
            # という非対称が要点。σ は対数一様なのでエレキー相当も同じ頻度で引ける。
            dash_sigma_hi = 1.30 if self.extreme_tail else 0.70
            keying = KeyingParams(
                dash_dot_ratio=float(rng.uniform(2.5, 5.0)),
                element_jitter_sigma_ratio=0.0,   # 種別ごとの σ を使うので基準値は 0
                dot_jitter_sigma_ratio=log_uniform(rng, 0.02, 0.20),
                dash_jitter_sigma_ratio=log_uniform(rng, 0.05, dash_sigma_hi),
                intra_element_space_units=float(rng.uniform(1.0, 1.3)),
                intra_gap_jitter_sigma_ratio=log_uniform(rng, 0.05, 0.50),
                inter_char_space_units=float(rng.uniform(2.6, 3.2)),
                char_gap_jitter_sigma_ratio=log_uniform(rng, 0.05, 0.90),
                inter_word_space_units=float(rng.uniform(5.0, 16.0)),
                # 設計書は「0〜4」だが、対数一様は下限 0 を取れないので 0.01 を下限にしている
                word_gap_jitter_sigma_ratio=log_uniform(rng, 0.01, 4.00),
                **common,
            )
        else:
            keying = KeyingParams(
                dash_dot_ratio=float(rng.uniform(2.5, 4.0)),
                element_jitter_sigma_ratio=float(rng.uniform(0.05, 0.25)),  # 拡張: 上限0.20→0.25
                **common,
            )
        if self.effective_snr_range is not None:
            snr_db = float(rng.uniform(*self.effective_snr_range))
            snr_is_effective = True
        else:
            snr_db = float(rng.uniform(-10.0, 20.0))
            snr_is_effective = False
        return SynthConfig(
            mode=self.mode,
            keying=keying,
            snr_db=snr_db,
            qsb_depth=float(rng.uniform(0.0, 0.8)) if rng.random() < 0.5 else 0.0,
            qsb_period_s=float(rng.uniform(1.0, 8.0)),
            qrm_stations=int(rng.integers(0, 3)),
            qrn_rate_per_sec=float(rng.uniform(0.0, 10.0)) if rng.random() < 0.3 else 0.0,
            qrn_intensity=float(rng.uniform(0.5, 3.0)),
            filter_center_hz=keying.tone_freq_hz,
            filter_bandwidth_hz=float(rng.uniform(250.0, 500.0)),
            snr_is_effective=snr_is_effective,
        )


def default_config_sampler(
    mode: Mode,
    hand_keying: bool = False,
    extreme_tail: bool = True,
    electronic_keyer_prob: float = 0.25,
) -> ConfigSampler:
    """要件 §3.2 の範囲で SynthConfig をサンプリングする標準サンプラを返す."""
    return DefaultConfigSampler(
        mode,
        hand_keying=hand_keying,
        extreme_tail=extreme_tail,
        electronic_keyer_prob=electronic_keyer_prob,
    )


class MorseSynthDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """オンザフライ合成 IterableDataset.

    各 worker は独立した RNG ストリームを持つ. ``seed`` を指定すると
    全体の擬似乱数列が決定的になる (同一シード・同一 worker 数で同じ出力).
    """

    def __init__(
        self,
        mode_mix: dict[Mode, float],
        config_sampler: ConfigSampler | None = None,
        text_config: TextGenConfig | None = None,
        seed: int | None = None,
        max_samples: int | None = None,
        sample_rate: int = 8000,
        noise_pool: RealNoisePool | None = None,
        noise_prob: float = 0.8,
        noise_snr_range: tuple[float, float] = (-5.0, 15.0),
        noise_clean_snr_db: float = 40.0,
        tone_freq_range: tuple[float, float] | None = None,
        effective_snr_range: tuple[float, float] | None = None,
        *,
        hand_keying: bool = False,
        extreme_tail: bool = True,
        electronic_keyer_prob: float = 0.25,
    ) -> None:
        if not mode_mix:
            raise ValueError("mode_mix must be non-empty")
        if any(v < 0 for v in mode_mix.values()):
            raise ValueError("mode_mix weights must be non-negative")
        total = sum(mode_mix.values())
        if total <= 0:
            raise ValueError("mode_mix weights sum must be positive")

        self.mode_mix: dict[Mode, float] = {k: v / total for k, v in mode_mix.items()}
        self.text_config: TextGenConfig | None = text_config
        self.seed: int | None = seed
        self.max_samples: int | None = max_samples
        self.sample_rate: int = sample_rate
        # 実録音バンドノイズ混合 (Phase 4 FT 用). 混合時は合成 AWGN をほぼ
        # 無効化 (noise_clean_snr_db) し、実ノイズを目標 SNR で加算する.
        self.noise_pool: RealNoisePool | None = noise_pool
        self.noise_prob: float = noise_prob
        self.noise_snr_range: tuple[float, float] = noise_snr_range
        self.noise_clean_snr_db: float = noise_clean_snr_db

        if config_sampler is not None:
            # 注意: ここは全モードに同一インスタンスを配る。DefaultConfigSampler は
            # self.mode を保持するので、mode_mix が複数モードを含む状態で
            # DefaultConfigSampler を config_sampler として渡すと、全サンプルが
            # その 1 モードで合成される (和文が欧文モードで合成される等の無言の汚染)。
            # モード別の設定を変えたいときは config_sampler ではなく、
            # hand_keying / extreme_tail / tone_freq_range 等の引数を使うこと。
            self._samplers: dict[Mode, ConfigSampler] = {
                mode: config_sampler for mode in self.mode_mix
            }
        else:
            # default_config_sampler は tone_freq_range / effective_snr_range を
            # 渡せないため、per-mode で DefaultConfigSampler を直接構築する
            # (tone_freq_range, effective_snr_range いずれも None の場合は
            # DefaultConfigSampler(mode) と等価).
            self._samplers = {
                mode: DefaultConfigSampler(
                    mode,
                    tone_freq_range=tone_freq_range if tone_freq_range is not None else (400.0, 900.0),
                    effective_snr_range=effective_snr_range,
                    hand_keying=hand_keying,
                    extreme_tail=extreme_tail,
                    electronic_keyer_prob=electronic_keyer_prob,
                )
                for mode in self.mode_mix
            }

    def _make_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + worker_id * 100_003)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = self._make_rng()
        keys = list(self.mode_mix.keys())
        probs = np.asarray([self.mode_mix[k] for k in keys])
        count = 0
        while self.max_samples is None or count < self.max_samples:
            mode = keys[int(rng.choice(len(keys), p=probs))]
            config = self._samplers[mode](rng)
            use_real_noise = (
                self.noise_pool is not None and rng.random() < self.noise_prob
            )
            if use_real_noise:
                config.snr_db = self.noise_clean_snr_db
            result = synthesize_random(
                rng, config, self.text_config, sample_rate=self.sample_rate
            )
            samples = result.samples
            if use_real_noise and samples.size > 0:
                assert self.noise_pool is not None
                snr_db = float(rng.uniform(*self.noise_snr_range))
                segment = self.noise_pool.sample_segment(samples.size, rng)
                samples = add_real_noise(samples, segment, snr_db)
            yield (
                torch.from_numpy(samples.copy()),
                torch.from_numpy(result.token_ids.copy()),
            )
            count += 1


@dataclass
class EvalCell:
    """評価セット内の 1 セル."""

    snr_db: float
    wpm: float
    samples_per_cell: int


def make_fixed_eval_set(
    snr_grid: list[float],
    wpm_grid: list[float],
    samples_per_cell: int,
    seed: int,
    mode: Mode = "european",
    sample_rate: int = 8000,
) -> list[SynthResult]:
    """SNR × WPM の格子で固定評価セットを生成 (再現可能).

    手送りジッタは小さく (=機械キーイング相当) する.

    Args:
        snr_grid: SNR (dB) の値リスト.
        wpm_grid: WPM の値リスト.
        samples_per_cell: 各セルでのサンプル数.
        seed: マスターシード.
    """
    rng = np.random.default_rng(seed)
    results: list[SynthResult] = []
    for snr in snr_grid:
        for wpm in wpm_grid:
            for _ in range(samples_per_cell):
                config = SynthConfig(
                    mode=mode,
                    keying=KeyingParams(
                        wpm=wpm,
                        dash_dot_ratio=3.0,
                        element_jitter_sigma_ratio=0.02,
                        tone_freq_hz=600.0,
                        rise_fall_ms=5.0,
                        pre_silence_sec=0.05,
                        post_silence_sec=0.05,
                    ),
                    snr_db=snr,
                    qsb_depth=0.0,
                    qrm_stations=0,
                    qrn_rate_per_sec=0.0,
                    filter_center_hz=600.0,
                    filter_bandwidth_hz=400.0,
                )
                results.append(synthesize_random(rng, config, sample_rate=sample_rate))
    return results


@dataclass(frozen=True)
class RealNoiseEvalSample:
    """実ノイズ混合の固定評価サンプル 1 件."""

    samples: np.ndarray          # float32 波形 (実ノイズ混合済み)
    token_ids: np.ndarray        # int64 正解トークン列
    text: str                    # 正解テキスト
    mode: Mode
    wpm: float
    target_snr_db: float         # add_real_noise に渡した目標 SNR
    eff_snr_db: float            # 受信機 BPF 後の実効 SNR


def make_fixed_real_noise_eval_set(
    noise_pool: RealNoisePool,
    snr_grid: list[float],
    wpm_grid: list[float],
    samples_per_cell: int,
    seed: int,
    mode: Mode = "european",
    tone_center_hz: float = 494.0,
    filter_bandwidth_hz: float = 300.0,
    sample_rate: int = 8000,
) -> list[RealNoiseEvalSample]:
    """合成キーイング + 実録音ノイズの固定評価セットを生成 (決定的).

    実運用に近い評価のため、トーン中心・BPF 帯域はユーザー実測値
    (設計書 §2.6: 494 Hz, 帯域内ノイズ 300-550 Hz) を既定にする。
    ``target_snr_db`` は ``add_real_noise`` に渡す SNR。実録音ノイズは
    既に帯域内なので実効 SNR ≒ 目標だが、``effective_snr_db`` で実測して記録する。

    手送りジッタは小さく (機械キーイング相当) 固定し、評価の再現性を保つ。
    """
    rng = np.random.default_rng(seed)
    results: list[RealNoiseEvalSample] = []
    for snr in snr_grid:
        for wpm in wpm_grid:
            for _ in range(samples_per_cell):
                config = SynthConfig(
                    mode=mode,
                    keying=KeyingParams(
                        wpm=wpm,
                        dash_dot_ratio=3.0,
                        element_jitter_sigma_ratio=0.02,
                        tone_freq_hz=tone_center_hz,
                        rise_fall_ms=5.0,
                        pre_silence_sec=0.05,
                        post_silence_sec=0.05,
                    ),
                    snr_db=40.0,  # 高 SNR でクリーン合成 → 実ノイズを後段で加算
                    qsb_depth=0.0,
                    qrm_stations=0,
                    qrn_rate_per_sec=0.0,
                    filter_center_hz=tone_center_hz,
                    filter_bandwidth_hz=filter_bandwidth_hz,
                )
                res = synthesize_random(rng, config, sample_rate=sample_rate)
                clean = res.samples
                if clean.size == 0:
                    continue
                segment = noise_pool.sample_segment(clean.size, rng)
                mixed = add_real_noise(clean, segment, snr)
                # effective_snr_db は「実際に混合されたノイズ」のパワーを前提とする
                # (Task 2 の意味論)。noise_pool から切り出した生 segment は
                # add_real_noise が目標 SNR に合わせて内部でスケーリングするため、
                # そのままでは実効 SNR が目標から大きく外れる。実際に加算された
                # ノイズ成分 (mixed - clean) を使って実測する。
                actual_noise = (
                    mixed.astype(np.float64) - clean.astype(np.float64)
                )
                eff = effective_snr_db(
                    clean, actual_noise, tone_center_hz, filter_bandwidth_hz, sample_rate
                )
                results.append(RealNoiseEvalSample(
                    samples=mixed.astype(np.float32, copy=False),
                    token_ids=res.token_ids,
                    text=res.text,
                    mode=mode,
                    wpm=wpm,
                    target_snr_db=snr,
                    eff_snr_db=eff,
                ))
    return results


__all__ = [
    "ConfigSampler",
    "DefaultConfigSampler",
    "EvalCell",
    "MorseSynthDataset",
    "RealNoiseEvalSample",
    "default_config_sampler",
    "log_uniform",
    "make_fixed_eval_set",
    "make_fixed_real_noise_eval_set",
]

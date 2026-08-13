"""合成データジェネレータ統合.

テキスト → 符号列 → 波形 → QSB → QRM → QRN → AWGN → CW BPF の順で
処理し、トークン ID 列と波形のペアを返す.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.synth.keying import KeyingParams, codes_to_waveform
from src.synth.noise import add_awgn, add_awgn_effective, add_qrm, add_qrn, apply_cw_filter, apply_qsb
from src.synth.text_generator import TextGenConfig, generate_text
from src.tokens.morse_tokens import TOKEN_TO_ID, Mode, text_to_codes


@dataclass
class SynthConfig:
    """1 サンプルの合成パラメータ."""

    mode: Mode
    keying: KeyingParams = field(default_factory=KeyingParams)
    snr_db: float = 10.0
    qsb_depth: float = 0.0
    qsb_period_s: float = 2.0
    qrm_stations: int = 0
    qrn_rate_per_sec: float = 0.0
    qrn_intensity: float = 1.0
    filter_center_hz: float = 600.0
    filter_bandwidth_hz: float = 500.0
    apply_receiver_filter: bool = True
    snr_is_effective: bool = False  # True: snr_db を BPF後の実効SNRとして扱う


@dataclass
class SynthResult:
    """合成結果."""

    samples: np.ndarray              # float32
    sample_rate: int
    text: str
    codes: list[str]
    token_ids: np.ndarray            # int64
    code_start_samples: np.ndarray   # int64
    effective_wpm: float
    config: SynthConfig


def synthesize_from_text(
    text: str,
    config: SynthConfig,
    rng: np.random.Generator,
    sample_rate: int = 8000,
) -> SynthResult:
    """与えられたテキストから 1 サンプルを合成."""
    codes = text_to_codes(text, config.mode)
    token_ids = np.array([TOKEN_TO_ID[c] for c in codes], dtype=np.int64)

    if not codes:
        return SynthResult(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=sample_rate,
            text=text,
            codes=codes,
            token_ids=token_ids,
            code_start_samples=np.zeros(0, dtype=np.int64),
            effective_wpm=config.keying.wpm,
            config=config,
        )

    # 1. キーイング波形
    wave = codes_to_waveform(codes, config.keying, rng, sample_rate=sample_rate)
    samples = wave.samples

    # 2. QSB
    if config.qsb_depth > 0:
        samples = apply_qsb(
            samples,
            depth=config.qsb_depth,
            period_s=config.qsb_period_s,
            sample_rate=sample_rate,
            rng=rng,
        )

    # 3. QRM
    if config.qrm_stations > 0:
        samples = add_qrm(
            samples,
            num_stations=config.qrm_stations,
            sample_rate=sample_rate,
            rng=rng,
            self_freq_hz=config.keying.tone_freq_hz,
        )

    # 4. QRN
    if config.qrn_rate_per_sec > 0:
        samples = add_qrn(
            samples,
            rate_per_sec=config.qrn_rate_per_sec,
            intensity=config.qrn_intensity,
            sample_rate=sample_rate,
            rng=rng,
        )

    # 5. AWGN (実効モードなら BPF後の実効SNR目標で加算)
    if config.snr_is_effective:
        samples = add_awgn_effective(
            samples, config.snr_db,
            config.filter_center_hz, config.filter_bandwidth_hz,
            sample_rate, rng,
        )
    else:
        samples = add_awgn(samples, config.snr_db, rng)

    # 6. 受信機 BPF
    if config.apply_receiver_filter:
        samples = apply_cw_filter(
            samples,
            center_hz=config.filter_center_hz,
            bandwidth_hz=config.filter_bandwidth_hz,
            sample_rate=sample_rate,
        )

    return SynthResult(
        samples=samples,
        sample_rate=sample_rate,
        text=text,
        codes=codes,
        token_ids=token_ids,
        code_start_samples=wave.code_start_samples,
        effective_wpm=wave.effective_wpm,
        config=config,
    )


def synthesize_random(
    rng: np.random.Generator,
    config: SynthConfig,
    text_config: TextGenConfig | None = None,
    sample_rate: int = 8000,
) -> SynthResult:
    """ランダムテキストから 1 サンプルを合成."""
    text = generate_text(rng, config.mode, text_config)
    return synthesize_from_text(text, config, rng, sample_rate)


__all__ = [
    "SynthConfig",
    "SynthResult",
    "synthesize_from_text",
    "synthesize_random",
]

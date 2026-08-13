"""HF CW 通信路ノイズ・伝搬モデル.

要件 §3.2.3 の各種劣化を ndarray ベクトル演算で適用する.

- ``add_awgn``: 加法ホワイトガウシアンノイズ (SNR 指定)
- ``apply_qsb``: フェージング (振幅変調)
- ``add_qrm``: 別 CW 局の重畳
- ``add_qrn``: インパルスノイズ (空電)
- ``apply_cw_filter``: 狭帯域 CW 受信機フィルタ (バターワース BPF)
- ``RealNoisePool`` / ``add_real_noise``: 実録音バンドノイズの混合 (Phase 4)
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import signal as _scipy_signal


def signal_power(sig: np.ndarray) -> float:
    """RMS パワー (平均二乗値). 全ゼロ入力に対しては 0.0 を返す."""
    if sig.size == 0:
        return 0.0
    return float(np.mean(sig.astype(np.float64) ** 2))


def add_awgn(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """加法ホワイトガウシアンノイズ.

    Args:
        sig: 入力波形 (float).
        snr_db: 目標 SNR (dB). 大きいほどクリーン.
        rng: 乱数生成器.

    Returns:
        ノイズが加算された波形 (入力と同じ dtype).
    """
    sp = signal_power(sig)
    if sp == 0.0:
        # 無音の場合はノイズのみ
        noise_power = 1e-6 * 10 ** (-snr_db / 10)
    else:
        noise_power = sp / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, float(np.sqrt(noise_power)), size=sig.shape)
    return (sig + noise.astype(sig.dtype)).astype(sig.dtype)


def apply_qsb(
    sig: np.ndarray,
    depth: float,
    period_s: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """フェージング (QSB) — 周期的振幅変調.

    Args:
        depth: 0 (変調なし) 〜 1 (深く 0 まで落ちる).
        period_s: 周期 (秒). 1〜数秒が一般的.
    """
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth}")
    if period_s <= 0:
        raise ValueError(f"period_s must be > 0, got {period_s}")
    if depth == 0.0:
        return sig.copy()
    t = np.arange(len(sig), dtype=np.float64) / sample_rate
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    mod = (1.0 - depth) + depth * 0.5 * (1.0 + np.cos(2.0 * np.pi * t / period_s + phi))
    return (sig * mod.astype(sig.dtype)).astype(sig.dtype)


def add_qrm(
    sig: np.ndarray,
    num_stations: int,
    sample_rate: int,
    rng: np.random.Generator,
    amplitude_range: tuple[float, float] = (0.2, 0.6),
    freq_range_hz: tuple[float, float] = (300.0, 1100.0),
    self_freq_hz: float | None = None,
    self_freq_exclusion_hz: float = 50.0,
) -> np.ndarray:
    """別 CW 局の重畳 (QRM) を簡易シミュレート.

    本格的な実装では別途 codes_to_waveform を呼んで実 CW を重畳するが、
    実装簡素化のため、ここでは断続的キーイングされたトーンで近似する.

    Args:
        num_stations: 重畳する局数 (0 で無効).
        amplitude_range: 各局の振幅範囲 (本信号 RMS に対する比).
        freq_range_hz: 重畳局の周波数範囲.
        self_freq_hz: 自局周波数 (これに近い周波数は避ける).
        self_freq_exclusion_hz: 自局周波数から ±この帯域を避ける.
    """
    if num_stations <= 0:
        return sig.copy()
    sp = signal_power(sig)
    sig_rms = float(np.sqrt(sp)) if sp > 0 else 1.0
    n = len(sig)
    t = np.arange(n, dtype=np.float64) / sample_rate
    qrm_total = np.zeros(n, dtype=np.float64)

    for _ in range(num_stations):
        # 周波数選定 (自局周波数を避ける)
        for _try in range(10):
            f = float(rng.uniform(*freq_range_hz))
            if (
                self_freq_hz is None
                or abs(f - self_freq_hz) > self_freq_exclusion_hz
            ):
                break
        amp = float(rng.uniform(*amplitude_range)) * sig_rms

        # ランダムなオンオフパターン (5〜20 WPM 相当のジッタ付き矩形波)
        on_duration_sec = float(rng.uniform(0.05, 0.3))
        off_duration_sec = float(rng.uniform(0.05, 0.3))
        cycle_samples = int((on_duration_sec + off_duration_sec) * sample_rate)
        on_samples = int(on_duration_sec * sample_rate)
        if cycle_samples <= 0:
            continue
        phase_offset = int(rng.integers(0, cycle_samples))
        mod_index = (np.arange(n) + phase_offset) % cycle_samples
        envelope = (mod_index < on_samples).astype(np.float64)

        tone = np.sin(2.0 * np.pi * f * t)
        qrm_total += amp * envelope * tone

    return (sig + qrm_total.astype(sig.dtype)).astype(sig.dtype)


def add_qrn(
    sig: np.ndarray,
    rate_per_sec: float,
    intensity: float,
    sample_rate: int,
    rng: np.random.Generator,
    impulse_width_samples: int = 4,
) -> np.ndarray:
    """インパルスノイズ (QRN, 空電) を加算.

    Args:
        rate_per_sec: 平均インパルス発生率 (1 秒あたり).
        intensity: 各インパルスの振幅 (本信号 RMS に対する比).
        impulse_width_samples: 各インパルスの幅 (サンプル).
    """
    if rate_per_sec <= 0 or intensity <= 0:
        return sig.copy()
    sp = signal_power(sig)
    sig_rms = float(np.sqrt(sp)) if sp > 0 else 1.0
    n = len(sig)
    duration_sec = n / sample_rate
    num_impulses = int(rng.poisson(rate_per_sec * duration_sec))
    if num_impulses == 0:
        return sig.copy()

    out = sig.astype(np.float64).copy()
    positions = rng.integers(0, n, size=num_impulses)
    amplitudes = rng.normal(0.0, intensity * sig_rms, size=num_impulses)
    # 各インパルスは指数減衰
    decay = np.exp(-np.arange(impulse_width_samples) / max(1.0, impulse_width_samples / 3))
    for pos, amp in zip(positions, amplitudes, strict=True):
        end = min(pos + impulse_width_samples, n)
        seg = end - pos
        out[pos:end] += amp * decay[:seg]
    return out.astype(sig.dtype)


def apply_cw_filter(
    sig: np.ndarray,
    center_hz: float,
    bandwidth_hz: float,
    sample_rate: int,
    order: int = 4,
) -> np.ndarray:
    """CW 受信機の狭帯域 BPF (バターワース).

    Args:
        center_hz: 中心周波数.
        bandwidth_hz: 通過帯域幅 (250〜500 Hz 相当).
        order: フィルタ次数.
    """
    low = max(center_hz - bandwidth_hz / 2, 10.0)
    high = min(center_hz + bandwidth_hz / 2, sample_rate / 2 - 10.0)
    if low >= high:
        raise ValueError(
            f"Invalid passband: low={low}, high={high} (sample_rate={sample_rate})"
        )
    nyquist = sample_rate / 2.0
    sos = _scipy_signal.butter(
        order, [low / nyquist, high / nyquist], btype="bandpass", output="sos"
    )
    filtered = _scipy_signal.sosfiltfilt(sos, sig.astype(np.float64))
    return filtered.astype(sig.dtype)


def effective_snr_db(
    sig: np.ndarray,
    noise: np.ndarray,
    center_hz: float,
    bandwidth_hz: float,
    sample_rate: int,
) -> float:
    """受信機 BPF 後の帯域内パワー比から実効 SNR (dB) を求める.

    公称 SNR は BPF 前で定義されるが、AWGN は帯域外にも広がるため BPF 後の
    実効 SNR は公称より高くなる (設計書 §2.1: 約 +9.5 dB)。実録音ノイズは
    最初から帯域内なので実効 ≒ 公称。評価表の SNR 列をこの実効値で記録する。

    Args:
        sig: 信号波形 (BPF 前).
        noise: ノイズ波形 (BPF 前、``sig`` と同一サンプルレート).
        center_hz: 受信機 BPF の中心周波数.
        bandwidth_hz: 受信機 BPF の帯域幅.
        sample_rate: サンプルレート.

    Returns:
        実効 SNR (dB). BPF 後のノイズパワーが 0 なら ``inf``.
    """
    sig_f = apply_cw_filter(sig, center_hz, bandwidth_hz, sample_rate)
    noi_f = apply_cw_filter(noise, center_hz, bandwidth_hz, sample_rate)
    np_ = signal_power(noi_f)
    if np_ == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(signal_power(sig_f) / np_))


def add_awgn_effective(
    sig: np.ndarray,
    target_snr_db: float,
    center_hz: float,
    bandwidth_hz: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """BPF後の実効(帯域内)SNRが ``target_snr_db`` になる広帯域AWGNを加算する.

    ``add_awgn`` は SNR を BPF 前 (広帯域) で定義するため、受信機 BPF が
    帯域外ノイズを捨てる分だけ実効SNRが高くなる (設計書 §2.1: 帯域500で +9.5dB、
    帯域可変で下駄が変わる)。本関数は目標を「BPF後の帯域内SNR」で受け、信号と
    ノイズをそれぞれ BPF に通した後のパワー比が目標になるよう広帯域ノイズを
    スケールして **BPF前に** 加える。以降のパイプラインで BPF が適用されると
    実効SNRが目標に一致する (帯域可変でも確定)。

    Args:
        sig: 入力波形 (BPF前).
        target_snr_db: 目標の実効(BPF後・帯域内)SNR (dB).
        center_hz / bandwidth_hz: 後段で適用される受信機 BPF のパラメータ.
        sample_rate: サンプルレート.
        rng: 乱数生成器.

    Returns:
        ノイズを加えた波形 (入力と同じ dtype). 無音入力はそのまま複製を返す.
    """
    sig_bpf_power = signal_power(apply_cw_filter(sig, center_hz, bandwidth_hz, sample_rate))
    if sig_bpf_power == 0.0:
        return sig.copy()
    unit_noise = rng.standard_normal(sig.shape).astype(np.float32)
    noise_bpf_power = signal_power(
        apply_cw_filter(unit_noise, center_hz, bandwidth_hz, sample_rate)
    )
    if noise_bpf_power == 0.0:
        return sig.copy()
    target_lin = 10.0 ** (target_snr_db / 10.0)
    scale = float(np.sqrt(sig_bpf_power / (target_lin * noise_bpf_power)))
    return (sig + scale * unit_noise).astype(sig.dtype)


def add_real_noise(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """実録音ノイズを目標 SNR で加算.

    ``add_awgn`` の実ノイズ版. ノイズは信号と同じ長さ・サンプルレートで
    与える (``RealNoisePool.sample_segment`` を使う).

    Args:
        sig: 入力波形.
        noise: 加算するノイズ波形 (``sig`` と同じ shape).
        snr_db: 目標 SNR (dB). 信号パワー / スケール後ノイズパワー.

    Returns:
        ノイズが加算された波形 (入力と同じ dtype).
    """
    if sig.shape != noise.shape:
        raise ValueError(f"shape mismatch: sig={sig.shape}, noise={noise.shape}")
    noise_power = signal_power(noise)
    if noise_power == 0.0:
        return sig.copy()
    sp = signal_power(sig)
    # 無音入力は add_awgn と同じ微小基準パワーでノイズのみ乗せる
    ref_power = sp if sp > 0.0 else 1e-6
    target_noise_power = ref_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return (sig + scale * noise.astype(np.float64)).astype(sig.dtype)


class RealNoisePool:
    """実録音バンドノイズの WAV プール.

    受信機で録音した「信号のいない周波数」のノイズを保持し、合成キーイング
    波形に混合するランダムセグメントを切り出す. 録音はアプリの BPF を通した
    後の音声を想定 (= 推論時の前処理と同一経路).

    すべてのメソッドは ``np.random.Generator`` を受け取り再現可能.
    """

    def __init__(self, waves: Sequence[np.ndarray], sample_rate: int = 8000) -> None:
        cleaned = [np.asarray(w, dtype=np.float32) for w in waves if np.asarray(w).size > 0]
        if not cleaned:
            raise ValueError("waves is empty")
        self.sample_rate = sample_rate
        self._waves: list[np.ndarray] = cleaned
        lengths = np.array([w.size for w in cleaned], dtype=np.float64)
        # 長いファイルほど選ばれやすくする (時間的に一様なサンプリング)
        self._probs = lengths / lengths.sum()

    def __len__(self) -> int:
        return len(self._waves)

    @property
    def total_samples(self) -> int:
        return int(sum(w.size for w in self._waves))

    @classmethod
    def from_dir(cls, root: Path | str, sample_rate: int = 8000) -> "RealNoisePool":
        """ディレクトリ以下の WAV を再帰的に読み込む (mono 化・リサンプル込み)."""
        import soundfile as sf

        root = Path(root)
        waves: list[np.ndarray] = []
        for path in sorted(root.rglob("*.wav")):
            wave, sr = sf.read(path, dtype="float32", always_2d=False)
            if wave.ndim > 1:
                wave = wave[:, 0]
            if sr != sample_rate:
                from scipy.signal import resample_poly
                g = np.gcd(int(sr), sample_rate)
                wave = resample_poly(wave, sample_rate // g, int(sr) // g).astype(np.float32)
            if wave.size > 0:
                waves.append(wave.astype(np.float32, copy=False))
        if not waves:
            raise ValueError(f"no noise wav found in {root}")
        return cls(waves, sample_rate)

    def sample_segment(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """ランダムなファイル・オフセットからセグメントを切り出す.

        ファイル長を超える要求は循環 (wrap) で埋める.
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        idx = int(rng.choice(len(self._waves), p=self._probs))
        wave = self._waves[idx]
        start = int(rng.integers(0, wave.size))
        return np.take(wave, start + np.arange(num_samples), mode="wrap")


__all__ = [
    "RealNoisePool",
    "add_awgn",
    "add_awgn_effective",
    "add_qrm",
    "add_qrn",
    "add_real_noise",
    "apply_cw_filter",
    "apply_qsb",
    "effective_snr_db",
    "signal_power",
]

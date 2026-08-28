"""受信音から**動作点** (速度・間隔・コントラスト) を測る.

**なぜ要るか (2026-08-27 の実測)**

モデルは学習した動作点で強く、離れた動作点で弱い。「手打ち vs 機械打鍵」という
区別は成り立たず (要素長のばらつきはむしろ L4 の方が大きい)、違うのは動作点だった:

| 集合 | WPM | 文字間 (dot) | コントラスト |
|---|---|---|---|
| held-out (運用者の打鍵) | 28.8 | 4.38 | 28.4 dB |
| 旧録音 | 20.1 | 3.13 | 9.0 dB |
| L4 8/24 (5W) | 24.3 | 3.24 | 11.7 dB |
| L4 8/25 (50W) | 24.3 | 3.24 | 22.5 dB |

運用者の構想「動作点ごとのモデルを切り替える」の前提として、**受信音から
動作点を測れなければ手で切り替えるしかない**。このモジュールはその測定器。

**閾値は「ノイズ床と局所ピークの算術中点」で置く。** `envelope_on_off` の
「局所ピークの 50%」は、コントラスト 9.4 dB (5W) ではノイズ床がピークの 34% に
あって近すぎ、雑音を符号と読んで **120 件中 44 件が測れず、測れたものも
WPM 37.6 と誤測定した** (真値 24.3)。

**閾値の高さはそのまま間隔の測定を歪める** (低いと ON が広がり OFF が縮む)。
幾何平均を試したら、弱信号は直ったが held-out の文字間が 4.38 → 3.05 に潰れて
判別率が 86% → 80% に落ちた。中点 (床 + 局所ピーク) / 2 なら、床が低い
きれいな信号では元の 0.5 × ピークに一致し、床が高い弱信号でだけ閾値が上がる。
局所ピークは元実装と同じ 1 秒窓で取る (AGC・QSB に追従する)。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from src.infer.wpm import MAX_WPM, MIN_WPM, TARGET_SR, detect_tone, element_runs

# これ以下の ON は符号ではなく雑音の尖りとみなす (秒)。
# `envelope_on_off` が ON を絞るのに使っているのと同じ値。
MIN_ON_SEC = 0.020

# 間隔の分類境界 (dot 単位)。教科書は 要素間 1 / 文字間 3 / 語間 7。
INTRA_GAP_MAX_UNITS = 2.0
CHAR_GAP_MAX_UNITS = 5.0

# コントラストを測る包絡線の窓 (秒)
CONTRAST_WIN_S = 0.020

# これ未満のコントラストは弱信号として最初から適応閾値で測る (dB)。
# 元の方法 (局所ピークの 50%) は弱信号でも「成功」するが、雑音で短点が
# 刻まれて **WPM を 36 と返す** (L4 5W の真値は 15〜30)。嘘の数字を出さないため、
# 実測の分布 (旧録音 8.4 dB / L4 5W 9.0 dB / L4 50W 16.4 dB) の間に境界を置く。
WEAK_CONTRAST_DB = 12.0


@dataclass(frozen=True)
class OperatingPoint:
    """1 本の受信音の動作点."""

    wpm: float
    dash_dot_ratio: float
    intra_gap_units: float      # 同一符号内の要素間 (dot 比)
    char_gap_units: float       # 文字間 (dot 比)
    contrast_db: float          # 包絡線 95%tile / 10%tile
    n_dot: int
    n_dash: int
    robust: bool = False        # 適応閾値 (弱信号用) に落ちて測った

    def as_vector(self) -> np.ndarray:
        """3 軸 (WPM, 文字間, コントラスト)."""
        return np.array([self.wpm, self.char_gap_units, self.contrast_db], dtype=np.float64)

    def model_axes(self) -> np.ndarray:
        """**モデル選択に効く 2 軸** (文字間, コントラスト).

        WPM は入れない。合成が 8〜50 WPM を覆うのでどのモデルも速度には対応
        しており、L4 は意図的に 15〜30 WPM を混ぜている。効くのは学習データの
        間隔とコントラストの偏りである。
        """
        return np.array([self.char_gap_units, self.contrast_db], dtype=np.float64)


def merge_noise_spikes(
    runs: tuple[tuple[bool, float], ...],
    min_on_sec: float = MIN_ON_SEC,
) -> list[tuple[bool, float]]:
    """短すぎる ON を前後の無音に併合する.

    **これをやらないと単位長の推定が壊れる。** 実測 (`script_ja_03`) では
    20 ms 以下の尖りが 97 個あり、短点の平均が 40.2 ms から 13.6 ms に
    引き下げられていた。
    """
    merged: list[tuple[bool, float]] = []
    for is_on, sec in runs:
        if is_on and sec <= min_on_sec:
            is_on = False
        if merged and merged[-1][0] == is_on:
            merged[-1] = (is_on, merged[-1][1] + sec)
        else:
            merged.append((is_on, sec))
    return merged


def contrast_db(wave: np.ndarray, sample_rate: int, win_s: float = CONTRAST_WIN_S) -> float:
    """20 ms RMS 包絡線の 95%tile / 10%tile を dB で返す.

    5W (11.7 dB) と 50W (22.5 dB) と運用者の打鍵録音 (28.4 dB) を分ける軸。
    """
    n = max(1, int(win_s * sample_rate))
    m = len(wave) // n
    if m < 5:
        return 0.0
    env = np.sqrt((np.asarray(wave[: m * n], dtype=np.float64).reshape(m, n) ** 2).mean(axis=1))
    # **無音パディング (厳密なゼロ) を除いて測る。** 含めると短い録音で 10%tile が
    # 0 になり、0 dB を返してしまう (L4 8/24 の 39 件で実際に起きた)。
    env = env[env > 0.0]
    if env.size < 5:
        return 0.0
    hi = float(np.percentile(env, 95))
    lo = float(np.percentile(env, 10))
    if hi <= 0.0 or lo <= 0.0:
        return 0.0
    return 20.0 * np.log10(hi / lo)


def element_runs_robust(
    wave: np.ndarray, sample_rate: int, smooth_s: float = 0.010,
    band_hz: float = 80.0, win_s: float = 1.0, min_contrast: float = 2.0,
) -> tuple[tuple[bool, float], ...]:
    """包絡線から ON/OFF の並び ((ON か, 秒)) を取り出す. 弱信号に耐える版.

    トーンを検出して ±``band_hz`` で帯域通過 → Hilbert 包絡線 → ``smooth_s`` 平滑 →
    **1 秒窓ごとに床 (10%tile) とピーク (95%tile) を取り、その中点を閾値**にする。

    - 床を窓ごとに取るのは **AGC で床が動く**ため。ファイル全体の床では
      弱信号の窓で閾値がずれ、L4 8/24 の 120 件中 81 件が測れなかった
    - ピークを max でなく 95%tile にするのは雑音の尖りに引っ張られないため
    - 帯域を ±80 Hz に絞るのは弱信号の SNR を稼ぐため (L4 のトーンは 503 Hz で安定)。
      120 Hz 幅の立ち上がりは ~8 ms で、24 WPM の短点 50 ms には十分
    - ピーク/床 が ``min_contrast`` 未満の窓は信号なしとして全部 OFF にする

    きれいな信号 (床 ≈ 0) では `envelope_on_off` の「局所ピークの 50%」に一致する。
    """
    x = np.asarray(wave, dtype=np.float32)
    if sample_rate != TARGET_SR:
        g = np.gcd(sample_rate, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sample_rate // g).astype(np.float32)
    tone = detect_tone(x, TARGET_SR)
    sos = butter(4, [max(100.0, tone - band_hz), min(TARGET_SR / 2 - 100.0, tone + band_hz)],
                 btype="bandpass", fs=TARGET_SR, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    n = max(1, int(smooth_s * TARGET_SR))
    env = np.convolve(env, np.ones(n) / n, mode="same")

    # 窓ごとの床とピーク (**for ループを使わない**: 窓数 x 窓長に整形して一気に取る)
    win = max(1, int(win_s * TARGET_SR))
    n_win = int(np.ceil(env.size / win))
    padded = np.full(n_win * win, np.nan)
    padded[: env.size] = env
    padded[padded == 0.0] = np.nan            # 無音パディングは統計に入れない
    blocks = padded.reshape(n_win, win)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 全部 NaN の窓 (無音) は想定内
        lo = np.nanpercentile(blocks, 10, axis=1)
        hi = np.nanpercentile(blocks, 95, axis=1)
    lo = np.nan_to_num(lo, nan=0.0)
    hi = np.nan_to_num(hi, nan=0.0)
    has_signal = (lo > 0.0) & (hi / np.maximum(lo, 1e-12) >= min_contrast)
    if not has_signal.any():
        return ()
    thr = np.where(has_signal, 0.5 * (lo + hi), np.inf)
    mask = env > thr.repeat(win)[: env.size]
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    idx = np.concatenate(([0], changes, [mask.size]))
    lengths = np.diff(idx) / TARGET_SR
    values = mask[idx[:-1]]
    return tuple(zip(values.tolist(), lengths.tolist(), strict=True))


def measure_operating_point(wave: np.ndarray, sample_rate: int) -> OperatingPoint | None:
    """受信音の動作点を測る. **測れないときは ``None``** (嘘の数字を出さない).

    **コントラストが `WEAK_CONTRAST_DB` 以上なら `element_runs` (局所ピークの 50%、
    `envelope_on_off` と同じ) で測り、失敗したときだけ `element_runs_robust`
    (窓ごとの適応閾値) に落とす。弱信号は最初から適応閾値で測る。**
    適応閾値は弱信号に耐える代わりに縁の精度を落とす (held-out の文字間が
    4.41 → 3.64 に潰れた)。精度と頑健さの代償は、必要なときだけ払う。

    ``None`` になるのは両方で: 短点か長音が足りない / 間隔の標本が足りない /
    速度が 5〜40 WPM の外。
    """
    if wave.size < sample_rate:
        return None
    contrast = contrast_db(wave, sample_rate)
    if contrast >= WEAK_CONTRAST_DB:
        op = _measure_from_runs(merge_noise_spikes(element_runs(wave, sample_rate)),
                                wave, sample_rate, contrast)
        if op is not None:
            return op
    return _measure_from_runs(merge_noise_spikes(element_runs_robust(wave, sample_rate)),
                              wave, sample_rate, contrast, robust=True)


def _measure_from_runs(
    runs: list[tuple[bool, float]], wave: np.ndarray, sample_rate: int,
    contrast: float, robust: bool = False,
) -> OperatingPoint | None:
    ons = np.array([s for is_on, s in runs if is_on], dtype=np.float64)
    if ons.size < 10:
        return None
    split = (np.percentile(ons, 25) + np.percentile(ons, 90)) / 2.0
    dot, dash = ons[ons < split], ons[ons >= split]
    if dot.size < 3 or dash.size < 3:
        return None
    unit = float(dot.mean())
    if unit <= 0.0:
        return None
    wpm = 1.2 / unit
    if not (MIN_WPM <= wpm <= MAX_WPM):
        return None

    # 前後のパディングに隣接する OFF は除く (助走と末尾は間隔ではない)
    offs = [s for is_on, s in runs if not is_on][1:-1]
    gaps = np.asarray(offs, dtype=np.float64) / unit
    if gaps.size < 5:
        return None
    intra = gaps[gaps < INTRA_GAP_MAX_UNITS]
    char = gaps[(gaps >= INTRA_GAP_MAX_UNITS) & (gaps < CHAR_GAP_MAX_UNITS)]
    if intra.size == 0 or char.size == 0:
        return None

    return OperatingPoint(
        wpm=float(wpm),
        dash_dot_ratio=float(dash.mean() / unit),
        intra_gap_units=float(intra.mean()),
        char_gap_units=float(char.mean()),
        contrast_db=contrast,
        n_dot=int(dot.size),
        n_dash=int(dash.size),
        robust=robust,
    )


__all__ = [
    "CHAR_GAP_MAX_UNITS",
    "INTRA_GAP_MAX_UNITS",
    "MIN_ON_SEC",
    "OperatingPoint",
    "WEAK_CONTRAST_DB",
    "contrast_db",
    "element_runs_robust",
    "measure_operating_point",
    "merge_noise_spikes",
]

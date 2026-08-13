"""受信信号の速度 (WPM) を波形から測る.

なぜ要るか
----------
**相手の速さが分かれば、こちらの送信速度をそれに合わせられる** (運用者、2026-08-12)。
CW では相手より速く打たないのが作法であり、目安が画面に出ているだけで役に立つ。

どうやって測るか
----------------
包絡線が局所ピークの 50% を超える区間を ON とみなし、**短点の平均長から**
``WPM = 1.2 / 短点長`` で求める (PARIS 基準)。トークン列ではなく波形を見る。

CTC の出す ``frame_start``/``frame_end`` は**ピーク位置でほぼ同一** (幅 0〜20 ms) で
符号の実際の長さを持たないため、トークンからは速度を測れない
(``src/infer/line_break.py`` の注記と同じ理由)。

精度 (実測)
-----------
**運用者自身の実録音 90 件**を、台本に書かれた WPM と突き合わせた結果:

    誤差 平均 +3.5%  中央 +1.6%  ±10% に収まった件数 78/90

台本の WPM は**目標値**であり手打ちの実速度とは一致しないので、残差のうち
どこまでが測定誤差でどこからが実際の速度のばらつきかは分けられていない。
**「だいたいの速さ」の目安としては十分だが、精密な値として扱ってはいけない。**

やってはいけない補正 (2026-08-12 に一度採用しかけた)
----------------------------------------------------
ON はランプのぶん短く、OFF は同じだけ長く測れる。だから
``(短点長 + 符号内スペース) / 2`` を単位長とすれば偏りが消える — **合成音では
実際に消えた** (ランプ 10 ms でも誤差 +0.5%、無補正なら +33%)。

**しかし実録音では +32% と大幅に悪化した。** 手打ちの符号内スペースは教科書の
1 単位ではなく実測 1.18〜1.25 単位あり (``src/synth/keying.py`` の注記)、
1 単位を前提にした補正が破綻する。**合成音だけで判断すると悪いほうを採る。**

クリーン判定について
--------------------
``scripts/analyze_keying.py`` は 1 秒窓のコントラスト比が閾値以上の区間だけを
使う。合成器との突き合わせでノイズ区間を符号として拾わないための歯止めである。

**この歯止めを実運用の受信に当てると 1 件も測れない。** 実録音 90 件では
26 窓中 1 窓しか通らなかった (コントラスト比の中央値 4.2 < 閾値 6.0)。
帯域制限と正規化を通った信号はコントラストが低いためである。
そこで ``contrast_min`` を引数にし、**受信の表示では歯止めを使わない**
(``contrast_min=0.0``)。オフライン解析の既定は従来どおり 6.0 のまま。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

TARGET_SR = 8000

# オフライン解析でノイズ区間を捨てる既定のコントラスト比 (1 秒窓の 95%/20%)。
DEFAULT_CONTRAST_MIN = 6.0

# 受信の表示では歯止めを使わない (上の docstring 参照)。
LIVE_CONTRAST_MIN = 0.0

# 短点がこれだけ取れないと測らない。**少ない標本で数字を出すと嘘になる。**
MIN_DOTS = 6

# 測定できる速度の範囲。これを外れた値は測定の破綻とみなして捨てる。
# 実運用の CW は 5〜40 WPM に収まる (`AppSettings.tx_wpm` の範囲と揃えてある)。
MIN_WPM = 5.0
MAX_WPM = 40.0


@dataclass(frozen=True)
class OnOff:
    """包絡線から取り出した ON / OFF の長さ (秒) と、そのときのトーン."""

    on_sec: np.ndarray
    off_sec: np.ndarray
    tone_hz: float
    clean_sec: float
    # ON/OFF を時系列のまま並べたもの ((ON かどうか, 秒))。
    # 符号を組み立て直すのに使う (:func:`element_runs`)
    runs: tuple[tuple[bool, float], ...] = ()


@dataclass(frozen=True)
class WpmEstimate:
    """速度の推定値."""

    wpm: float
    dot_sec: float
    n_dot: int
    n_dash: int
    tone_hz: float


def detect_tone(wave: np.ndarray, sample_rate: int) -> float:
    """一番強い成分の周波数 (Hz) を返す. 200〜1500 Hz の範囲で探す."""
    spec = np.abs(np.fft.rfft(wave * np.hanning(wave.size)))
    freqs = np.fft.rfftfreq(wave.size, 1.0 / sample_rate)
    band = (freqs > 200.0) & (freqs < 1500.0)
    return float(freqs[band][np.argmax(spec[band])])


def envelope_on_off(
    wave: np.ndarray,
    sample_rate: int,
    *,
    contrast_min: float = DEFAULT_CONTRAST_MIN,
) -> OnOff:
    """波形から ON / OFF の長さの並びを取り出す.

    Args:
        wave: モノラル float32 波形.
        sample_rate: サンプリングレート. 8 kHz 以外はリサンプルする.
        contrast_min: 1 秒窓の 95%/20% コントラスト比がこれ未満の窓を捨てる.
            0 以下なら捨てない (受信の表示ではこちら)。

    Note:
        **測定値にはランプ由来の偏りがある。** 包絡線が局所ピークの 50% を
        超える区間を ON とみなすので、ON は公称よりランプ長ぶん短く、
        隣接する OFF は同じだけ長く測れる。偏りの大きさは
        ``ランプ長 / 短点長`` に比例する。
    """
    if sample_rate != TARGET_SR:
        g = np.gcd(sample_rate, TARGET_SR)
        wave = resample_poly(wave, TARGET_SR // g, sample_rate // g).astype(np.float32)

    tone = detect_tone(wave, TARGET_SR)
    sos = butter(
        4,
        [max(100.0, tone - 150.0), min(TARGET_SR / 2 - 100.0, tone + 150.0)],
        btype="bandpass", fs=TARGET_SR, output="sos",
    )
    env = np.abs(hilbert(sosfiltfilt(sos, wave)))
    smooth = max(1, int(0.005 * TARGET_SR))
    env = np.convolve(env, np.ones(smooth) / smooth, mode="same")

    win = int(1.0 * TARGET_SR)
    if contrast_min <= 0.0:
        good = np.ones(env.size, dtype=bool)
    else:
        n_win = max(1, env.size // win)
        good = np.zeros(env.size, dtype=bool)
        for i in range(n_win):
            seg = env[i * win:(i + 1) * win]
            hi, lo = np.percentile(seg, 95), np.percentile(seg, 20)
            if hi > 1e-4 and hi / max(lo, 1e-9) > contrast_min:
                good[i * win:(i + 1) * win] = True
        if not good.any():
            good[:] = True

    local_peak = np.maximum.reduceat(
        env, np.arange(0, env.size, win)
    ).repeat(win)[:env.size]
    mask = env > local_peak * 0.5

    changes = np.diff(mask.astype(np.int8))
    idx = np.concatenate(([0], np.where(changes != 0)[0] + 1, [mask.size]))
    lengths = np.diff(idx)
    values = mask[idx[:-1]].copy()
    # 5 ms 未満の OFF は物理的にありえないので穴とみなして埋める
    tiny = (~values) & (lengths < int(0.005 * TARGET_SR))
    tiny[0] = tiny[-1] = False
    values[tiny] = True

    keep = good[idx[:-1]]
    lens = lengths[keep] / TARGET_SR
    vals = values[keep]
    on = lens[vals]
    on = on[on > 0.020]
    off = lens[~vals]
    off = off[off < 2.0]
    return OnOff(
        on_sec=on, off_sec=off, tone_hz=tone,
        clean_sec=float(good.sum() / TARGET_SR),
        runs=tuple(zip(vals.tolist(), lens.tolist(), strict=True)),
    )


def element_runs(wave: np.ndarray, sample_rate: int) -> tuple[tuple[bool, float], ...]:
    """ON/OFF を**時系列のまま**返す ((ON かどうか, 秒) の並び).

    :func:`envelope_on_off` は ON と OFF を別の配列にして返すので、
    「どの ON がどの OFF に挟まれていたか」が失われる。**符号を組み立て直す
    には順番が要る** (`scripts/audit_labels.py`)。包絡線の処理を二重に
    持たないよう、同じ関数から取り出す。
    """
    return envelope_on_off(wave, sample_rate, contrast_min=LIVE_CONTRAST_MIN).runs


def split_dot_dash(on_sec: np.ndarray, split_sec: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """ON の並びを短点と長音に分ける.

    ``split_sec`` が ``None`` なら 25%tile と 90%tile の中点を境界にする
    (ヒストグラムの谷を外すことがあるので、オフライン解析では明示指定が望ましい)。
    """
    if on_sec.size == 0:
        return on_sec, on_sec
    if split_sec is None:
        split_sec = float((np.percentile(on_sec, 25) + np.percentile(on_sec, 90)) / 2.0)
    return on_sec[on_sec < split_sec], on_sec[on_sec >= split_sec]


def estimate_wpm(
    wave: np.ndarray,
    sample_rate: int,
    *,
    contrast_min: float = LIVE_CONTRAST_MIN,
    min_dots: int = MIN_DOTS,
) -> WpmEstimate | None:
    """受信信号の速度を測る. **測れないときは ``None`` を返す.**

    嘘の数字を出すくらいなら何も出さないほうがよい。次のいずれかなら ``None``:

    * 短点か長音が足りない (標本が少なすぎる)
    * 求まった速度が 5〜40 WPM の外 (測定の破綻)

    Args:
        wave: モノラル波形. 数秒以上あること.
        sample_rate: サンプリングレート.
        contrast_min: :func:`envelope_on_off` に渡す. 既定は歯止め無し.
        min_dots: これだけ短点が取れないと測らない.
    """
    if wave.size < sample_rate:          # 1 秒未満では話にならない
        return None
    seg = envelope_on_off(wave, sample_rate, contrast_min=contrast_min)
    dot, dash = split_dot_dash(seg.on_sec)
    if dot.size < min_dots or dash.size < 3:
        return None
    dot_sec = float(dot.mean())
    if dot_sec <= 0.0:
        return None
    wpm = 1.2 / dot_sec
    if not (MIN_WPM <= wpm <= MAX_WPM):
        return None
    return WpmEstimate(
        wpm=wpm, dot_sec=dot_sec,
        n_dot=int(dot.size), n_dash=int(dash.size), tone_hz=seg.tone_hz,
    )


__all__ = [
    "DEFAULT_CONTRAST_MIN",
    "LIVE_CONTRAST_MIN",
    "MAX_WPM",
    "MIN_DOTS",
    "MIN_WPM",
    "TARGET_SR",
    "OnOff",
    "WpmEstimate",
    "detect_tone",
    "element_runs",
    "envelope_on_off",
    "estimate_wpm",
    "split_dot_dash",
]

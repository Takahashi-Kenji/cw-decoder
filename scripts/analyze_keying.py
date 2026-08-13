"""録音・合成音からキーイングのタイミング統計を実測する.

合成器のパラメータ範囲を勘ではなく実測で決めるための道具。実録音の解析にも、
合成器が指定どおりの分散を持つ波形を作れているかの往復検証にも使う。

使い方::

    .venv/Scripts/python.exe scripts/analyze_keying.py path/to/recording.wav
    .venv/Scripts/python.exe scripts/analyze_keying.py path/to/rec.wav --split-ms 75

**ヒストグラムを必ず見ること。** ON 長の山が 2 つ (短点と長音) でなければ測定が
破綻している。最初にこの解析を書いたときはノイズ区間を符号として拾い、山が 3 つ
出ている状態で長短比 4.90 という誤った数値を出した。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.infer.wpm import (
    DEFAULT_CONTRAST_MIN,
    TARGET_SR,
    envelope_on_off,
    split_dot_dash,
)


@dataclass(frozen=True)
class KeyingStats:
    """キーイングのタイミング統計. 長さの σ は dot 長に対する比率で持つ."""

    tone_hz: float
    dot_sec: float  # 測定値にはランプ由来の偏りがある。analyze_wave の docstring を参照
    dot_sigma_dot: float
    dash_sec: float  # 測定値にはランプ由来の偏りがある。analyze_wave の docstring を参照
    dash_sigma_dot: float
    dash_dot_ratio: float
    wpm: float
    intra_gap_dot: float
    intra_gap_sigma_dot: float
    char_gap_dot: float
    char_gap_sigma_dot: float
    word_gap_dot: float
    n_dot: int
    n_dash: int
    clean_sec: float
    total_sec: float
    on_histogram_ms: list[tuple[int, int]]
    off_histogram_ms: list[tuple[int, int]]


def _histogram(values_sec: np.ndarray, bin_ms: int = 10) -> list[tuple[int, int]]:
    """10 ms 刻みのヒストグラムを (下限 ms, 件数) の列で返す (件数 0 の bin は省く)."""
    if values_sec.size == 0:
        return []
    edges = np.arange(0, values_sec.max() * 1000 + bin_ms * 2, bin_ms)
    counts, _ = np.histogram(values_sec * 1000.0, bins=edges)
    return [(int(edges[i]), int(c)) for i, c in enumerate(counts) if c > 0]


def analyze_wave(
    wave: np.ndarray, sample_rate: int, split_sec: float | None = None
) -> KeyingStats:
    """波形からタイミング統計を測る.

    Args:
        wave: モノラル float32 波形.
        sample_rate: サンプリングレート. 8 kHz 以外はリサンプルする.
        split_sec: 短点と長音を分ける境界 (秒). ``None`` なら分位点から自動推定する。
            自動推定はヒストグラムの谷を外すことがあるので、実録音では
            ヒストグラムを見て明示指定するのが望ましい。

    **測定値には系統的な偏りがあります (重要)。**

    この解析は包絡線が局所ピークの 50% を超える区間を ON とみなします。一方
    ``src/synth/keying.py`` の raised-cosine ランプは**要素の内側**に適用されるため
    (立上りは要素の先頭から、立下りは要素の末尾に向かって)、50% 交差で測った ON 長は
    公称値より **ランプ長ぶんそのまま短く**なります (``ramp/2`` ではなく ``ramp``)。
    その分だけ隣接する OFF (スペース) は長く測れます。

    偏りの大きさは ``rise_fall_ms / dot_sec`` に比例します:

    - ``rise_fall_ms=3.0`` / 20 WPM (dot 60 ms) → 短点が約 5% 短く出る
    - ``rise_fall_ms=5.0`` (既定) / 20 WPM → 約 8.3% 短く出る

    したがって合成音との突き合わせで公称値とのズレが出ても、まず**この偏りで説明
    できないか**を確認してください。合成器のバグとは限りません。偏りを避けたい
    比較では ``rise_fall_ms=0.0`` で合成すると影響が消えます。実録音の解析では
    送信側のランプ特性が不明なので、この偏りは避けられません (同じ条件で比べる
    限り相対比較には影響しません)。
    """
    if sample_rate != TARGET_SR:
        g = np.gcd(sample_rate, TARGET_SR)
        wave = resample_poly(wave, TARGET_SR // g, sample_rate // g).astype(np.float32)
    total_sec = wave.size / TARGET_SR

    # **包絡線から ON/OFF を取り出す処理は src/infer/wpm.py に置いてある。**
    # 受信中の速度表示 (`estimate_wpm`) と同じものを使う。ここに写すと、
    # 片方だけ直したときに「解析では出るのに画面では出ない」が起きる。
    seg = envelope_on_off(wave, TARGET_SR, contrast_min=DEFAULT_CONTRAST_MIN)
    tone, on, off, clean = seg.tone_hz, seg.on_sec, seg.off_sec, seg.clean_sec

    dot, dash = split_dot_dash(on, split_sec)
    if dot.size < 3 or dash.size < 3:
        raise ValueError(
            f"短点 {dot.size} 個 / 長音 {dash.size} 個しか取れなかった。"
            "測定が破綻している可能性が高い。ヒストグラムを見て --split-ms を指定すること"
        )
    dot_sec = float(dot.mean())
    units = off / dot_sec

    def _bucket(lo: float, hi: float) -> np.ndarray:
        return units[(units >= lo) & (units < hi)]

    intra, char, word = _bucket(0.0, 1.9), _bucket(1.9, 5.0), _bucket(5.0, 1e9)
    return KeyingStats(
        tone_hz=tone,
        dot_sec=dot_sec,
        dot_sigma_dot=float(dot.std() / dot_sec),
        dash_sec=float(dash.mean()),
        dash_sigma_dot=float(dash.std() / dot_sec),
        dash_dot_ratio=float(dash.mean() / dot_sec),
        wpm=float(1.2 / dot_sec),
        intra_gap_dot=float(intra.mean()) if intra.size else float("nan"),
        intra_gap_sigma_dot=float(intra.std()) if intra.size else float("nan"),
        char_gap_dot=float(char.mean()) if char.size else float("nan"),
        char_gap_sigma_dot=float(char.std()) if char.size else float("nan"),
        word_gap_dot=float(word.mean()) if word.size else float("nan"),
        n_dot=int(dot.size),
        n_dash=int(dash.size),
        clean_sec=clean,
        total_sec=float(total_sec),
        on_histogram_ms=_histogram(on),
        off_histogram_ms=_histogram(off),
    )


def _print_stats(name: str, s: KeyingStats) -> None:
    print(f"=== {name} ===")
    print(f"長さ {s.total_sec:.1f}s / クリーン {s.clean_sec:.1f}s / トーン {s.tone_hz:.0f} Hz")
    print()
    print("ON ヒスト (山が 2 つでなければ測定破綻を疑うこと):")
    for lo, c in s.on_histogram_ms:
        print(f"  {lo:3d}-{lo+10:3d}ms {'#' * min(c, 60)} ({c})")
    print("OFF ヒスト:")
    for lo, c in s.off_histogram_ms:
        print(f"  {lo:3d}-{lo+10:3d}ms {'#' * min(c, 60)} ({c})")
    print()
    print(f"短点 dot : {s.dot_sec*1000:6.1f} ms  σ {s.dot_sigma_dot:.3f} dot  n={s.n_dot}")
    print(f"長音 dash: {s.dash_sec*1000:6.1f} ms  σ {s.dash_sigma_dot:.3f} dot  n={s.n_dash}")
    print(f"長短比 dash/dot = {s.dash_dot_ratio:.2f}  (教科書 3.00)")
    print(f"実効 WPM = {s.wpm:.1f}")
    print()
    print("スペース (dot 単位):")
    print(f"  要素間: 平均 {s.intra_gap_dot:5.2f}  σ {s.intra_gap_sigma_dot:4.2f}  教科書 1.0")
    print(f"  文字間: 平均 {s.char_gap_dot:5.2f}  σ {s.char_gap_sigma_dot:4.2f}  教科書 3.0")
    print(f"  語間  : 平均 {s.word_gap_dot:5.2f}                    教科書 7.0")


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf

    p = argparse.ArgumentParser(description="録音からキーイングのタイミング統計を測る")
    p.add_argument("wav", type=Path, nargs="+", help="解析する WAV")
    p.add_argument(
        "--split-ms", type=float, default=None,
        help="短点と長音を分ける境界 (ms)。省略時は自動推定。"
             "ヒストグラムの谷を見て指定するのが望ましい",
    )
    args = p.parse_args(argv)

    if args.split_ms is None:
        print(
            "[警告] --split-ms が未指定のため短点/長音の境界を自動推定します。"
            "長音側のジッタが大きい録音 (実録音はまさにこの領域) では分類を誤ります。"
            "下のヒストグラムの谷を見て --split-ms を明示指定してください。",
            file=sys.stderr,
        )

    for path in args.wav:
        wave, sr = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave[:, 0]
        split = args.split_ms / 1000.0 if args.split_ms is not None else None
        _print_stats(path.name, analyze_wave(wave, sr, split))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

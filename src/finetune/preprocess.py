"""実録音を学習データに入れる前の前処理 (torch 非依存).

**未処理の録音を与えるとモデルが崩壊する。** 実測 (2026-08-03):

===========================  ========
条件                          TER
===========================  ========
そのまま                       97.03%
BPF 600 Hz / 400 Hz            14.44%
+ 前後 0.4 秒の無音            10.60%
===========================  ========

取り込み経路が増えるたびに同じ前処理を書き写すと必ず食い違うので、
**ここを唯一の実装にする** (`scripts/import_keying_recordings.py` と
`scripts/collect_l4.py` が共有する)。

振幅は int16 のスケール (±32768) で扱う。学習側が読む WAV が 16 bit PCM だからで、
0..1 正規化に直すと既存データと食い違う。
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

# BPF 後に揃えるピーク値 (int16 フルスケールの約 48%)。
# 録音レベルがまちまちでも学習側から見た振幅を揃えるため。
PEAK_SCALE = 0.48 * 32768

# アプリのライブ経路と同じ既定 (`src/infer/settings.py`)。
DEFAULT_BPF_CENTER_HZ = 600.0
DEFAULT_BPF_BANDWIDTH_HZ = 400.0

# 先頭に足す無音の長さ (秒)。
# オートキーヤーの録音は 5 ms から音が始まり先頭に無音が無いため、モデルが
# 立ち上がりを符号と誤読して先頭にゴミトークンを吐く。実測ではこの挿入だけで
# 誤り 297 件中 44 件を占めていた。
DEFAULT_PAD_SEC = 0.4


def bandpass(
    x: np.ndarray,
    sample_rate: int,
    center_hz: float = DEFAULT_BPF_CENTER_HZ,
    bandwidth_hz: float = DEFAULT_BPF_BANDWIDTH_HZ,
) -> np.ndarray:
    """アプリのライブ経路と同じ帯域に整形し、ピークを :data:`PEAK_SCALE` に揃える."""
    from scipy.signal import butter, sosfiltfilt

    low = max(center_hz - bandwidth_hz / 2, 50.0)
    high = min(center_hz + bandwidth_hz / 2, sample_rate / 2 - 100.0)
    sos = butter(4, [low / (sample_rate / 2), high / (sample_rate / 2)],
                 btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x)
    peak = float(np.abs(y).max())
    return y / peak * PEAK_SCALE if peak > 0 else y


def pad_silence(x: np.ndarray, sample_rate: int, pad_sec: float = DEFAULT_PAD_SEC) -> np.ndarray:
    """前後に無音を足す (:data:`DEFAULT_PAD_SEC` の注記を参照)."""
    if pad_sec <= 0:
        return x
    silence = np.zeros(int(sample_rate * pad_sec))
    return np.concatenate([silence, x, silence])


def read_wav_int16(path: Path) -> tuple[np.ndarray, int]:
    """16 bit PCM WAV を ±32768 スケールの float で読む (ステレオは平均)."""
    with wave.open(str(path)) as w:
        sample_rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        x = data.astype(np.float64)
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x, sample_rate


def write_wav_int16(path: Path, x: np.ndarray, sample_rate: int) -> None:
    """±32768 スケールの波形を 16 bit PCM WAV で書く."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())


__all__ = [
    "DEFAULT_BPF_BANDWIDTH_HZ",
    "DEFAULT_BPF_CENTER_HZ",
    "DEFAULT_PAD_SEC",
    "PEAK_SCALE",
    "bandpass",
    "pad_silence",
    "read_wav_int16",
    "write_wav_int16",
]

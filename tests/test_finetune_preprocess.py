"""録音の前処理 (BPF + 前後の無音) のテスト.

**未処理の録音を学習に入れると壊れる**ことが実測されている
(TER 97.03% → BPF で 14.44% → 前後 0.4 秒の無音追加で 10.60%)。
取り込み経路が増えるたびに同じ前処理を書き写すと必ず食い違うので、
ここを唯一の実装にする。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.finetune.preprocess import (
    PEAK_SCALE,
    bandpass,
    pad_silence,
    read_wav_int16,
    write_wav_int16,
)

SR = 8000


def _tone(hz: float, seconds: float = 0.5, amp: float = 1000.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * hz * t)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


class TestBandpass:
    def test_帯域内は残る(self) -> None:
        y = bandpass(_tone(600.0), SR, center_hz=600.0, bandwidth_hz=400.0)
        assert _rms(y) > 0.1 * PEAK_SCALE

    def test_帯域外は落ちる(self) -> None:
        inside = bandpass(_tone(600.0), SR, center_hz=600.0, bandwidth_hz=400.0)
        outside = bandpass(_tone(2500.0), SR, center_hz=600.0, bandwidth_hz=400.0)
        assert _rms(outside) < 0.1 * _rms(inside)

    def test_ピークを揃える(self) -> None:
        """録音レベルがまちまちでも学習側から見た振幅を揃える."""
        for amp in (100.0, 20000.0):
            y = bandpass(_tone(600.0, amp=amp), SR, 600.0, 400.0)
            assert float(np.abs(y).max()) == pytest.approx(PEAK_SCALE, rel=0.02)

    def test_無音はそのまま(self) -> None:
        """ピーク 0 で割らない."""
        y = bandpass(np.zeros(SR), SR, 600.0, 400.0)
        assert np.all(y == 0.0)


class TestPadSilence:
    def test_前後に足す(self) -> None:
        y = pad_silence(np.ones(SR), SR, 0.4)
        assert len(y) == SR + 2 * int(SR * 0.4)
        assert y[0] == 0.0 and y[-1] == 0.0

    def test_ゼロなら何もしない(self) -> None:
        x = np.ones(10)
        assert pad_silence(x, SR, 0.0) is x


class TestWavRoundTrip:
    def test_書いて読める(self, tmp_path: Path) -> None:
        x = _tone(600.0, seconds=0.1, amp=5000.0)
        path = tmp_path / "a.wav"
        write_wav_int16(path, x, SR)
        y, sr = read_wav_int16(path)
        assert sr == SR
        assert len(y) == len(x)
        assert np.abs(y - x).max() <= 1.0          # 量子化ぶんだけ

    def test_範囲外は潰さずクリップする(self, tmp_path: Path) -> None:
        path = tmp_path / "b.wav"
        write_wav_int16(path, np.array([40000.0, -40000.0]), SR)
        y, _ = read_wav_int16(path)
        assert y.max() <= 32767 and y.min() >= -32768

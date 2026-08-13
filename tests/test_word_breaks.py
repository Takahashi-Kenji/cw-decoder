"""音声エンベロープ語間検出のテスト."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.infer.word_breaks import (
    compute_envelope,
    detect_silence_mask,
    detect_word_breaks_from_audio,
    estimate_dot_samples,
)


@dataclass
class FakeFrameToken:
    """テスト用 FrameToken (TimedToken Protocol 互換)."""

    token_id: int
    confidence: float
    frame_start: int
    frame_end: int


def _make_cw_tone(
    on_dot_segments: list[tuple[float, bool]],
    dot_ms: float = 80.0,
    sample_rate: int = 8000,
    tone_hz: float = 600.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """``(dot_units, is_on)`` のセグメント列から CW 信号を生成.

    Example::

        # 1 dot ON, 3 dot OFF, 3 dot ON (= dot, char-gap, dash) - "A の半分"
        wave = _make_cw_tone([(1, True), (3, False), (3, True)])
    """
    samples_per_dot = int(sample_rate * dot_ms / 1000.0)
    chunks: list[np.ndarray] = []
    t_offset = 0
    for n_dots, is_on in on_dot_segments:
        n_samples = int(n_dots * samples_per_dot)
        t = np.arange(n_samples, dtype=np.float64) + t_offset
        wave = (np.sin(2 * np.pi * tone_hz * t / sample_rate) * amplitude
                if is_on else np.zeros(n_samples, dtype=np.float64))
        chunks.append(wave.astype(np.float32))
        t_offset += n_samples
    return np.concatenate(chunks)


class TestComputeEnvelope:
    def test_silence_yields_near_zero(self) -> None:
        wave = np.zeros(8000, dtype=np.float32)
        env = compute_envelope(wave)
        assert env.max() < 1e-5

    def test_tone_yields_nonzero(self) -> None:
        t = np.arange(8000) / 8000.0
        wave = np.sin(2 * np.pi * 600 * t).astype(np.float32)
        env = compute_envelope(wave)
        # |sin| の平均は 2/π ≒ 0.637
        # 短窓平均なので大まかに 0.4 〜 0.7
        assert 0.3 < env.mean() < 0.8


class TestDetectSilenceMask:
    def test_pure_silence_all_true(self) -> None:
        wave = np.zeros(2000, dtype=np.float32)
        mask = detect_silence_mask(wave)
        assert mask.all()

    def test_alternating_on_off_detects_off(self) -> None:
        # 5 dot ON, 5 dot OFF, 5 dot ON
        wave = _make_cw_tone([(5, True), (5, False), (5, True)])
        mask = detect_silence_mask(wave)
        # OFF 区間 (中央) は silent_mask True
        n = len(wave)
        center = n // 2
        center_window = mask[center - 100:center + 100]
        assert center_window.sum() > 100   # 過半数が無音


class TestEstimateDotSamples:
    def test_basic(self) -> None:
        # 1 dot ON, 1 dot OFF を繰り返す (要素間ギャップ多数)
        wave = _make_cw_tone(
            [(1, True), (1, False)] * 10,
            dot_ms=80.0,
        )
        mask = detect_silence_mask(wave)
        est = estimate_dot_samples(mask)
        # 1 dot = 80ms = 640 samples だが推定はばらつくため広めに
        assert 400 < est < 900


class TestDetectWordBreaksFromAudio:
    def test_clean_word_gap_detected(self) -> None:
        # 構成: dot, 1dot-off, dash (A) | 7dot-off | dot, 1dot-off, dash (再びA)
        wave = _make_cw_tone([
            (1, True), (1, False), (3, True),    # A
            (7, False),                            # 語間
            (1, True), (1, False), (3, True),    # A
        ])
        # token frame は CTC モデル相当の単一フレーム burst をエミュレート
        hop = 80
        tokens = [
            FakeFrameToken(0, 0.9, 1, 1),       # A1
            FakeFrameToken(0, 0.9, int((len(wave) - 320) / hop), int((len(wave) - 320) / hop)),  # A2
        ]
        flags = detect_word_breaks_from_audio(wave, tokens)
        assert flags == [False, True]

    def test_clean_char_gap_not_detected(self) -> None:
        # A (dot dash) + 3dot-off (char gap) + B (dash dot dot dot)
        wave = _make_cw_tone([
            (1, True), (1, False), (3, True),    # A
            (3, False),                            # 文字間
            (3, True), (1, False),
            (1, True), (1, False),
            (1, True),                             # B
        ])
        hop = 80
        a_end = int(5 * 640 / hop)
        b_start = a_end + int(3 * 640 / hop)
        tokens = [
            FakeFrameToken(0, 0.9, 1, 1),
            FakeFrameToken(1, 0.9, b_start, b_start),
        ]
        flags = detect_word_breaks_from_audio(wave, tokens)
        # 文字間 (3 dot) は False
        assert flags == [False, False]

    def test_empty_returns_empty(self) -> None:
        flags = detect_word_breaks_from_audio(np.zeros(0, dtype=np.float32), [])
        assert flags == []

"""動作点の測定 (`src/infer/operating_point.py`) のテスト.

合成器で速度と間隔を既知にして作った音から、同じ値が測れることを確かめる。
**測定値にはランプ由来の偏りがある** (ON はランプ長ぶん短く、OFF は長く出る)
ので、許容は広めに取る。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.infer.operating_point import (
    OperatingPoint,
    contrast_db,
    measure_operating_point,
    merge_noise_spikes,
)
from src.synth.keying import KeyingParams
from src.synth.synthesizer import SynthConfig, synthesize_from_text

SR = 8000
TEXT = "CQ CQ DE JA1ABC JA1ABC K"


def synth(wpm: float = 20.0, char_gap: float = 3.0, snr_db: float = 20.0,
          seed: int = 0) -> np.ndarray:
    cfg = SynthConfig(
        mode="european", snr_db=snr_db, snr_is_effective=True,
        keying=KeyingParams(wpm=wpm, inter_char_space_units=char_gap,
                            tone_freq_hz=600.0, pre_silence_sec=0.5, post_silence_sec=0.5),
    )
    return synthesize_from_text(TEXT, cfg, np.random.default_rng(seed)).samples


class TestMergeNoiseSpikes:
    def test_短いONを無音に併合する(self) -> None:
        runs = ((False, 0.1), (True, 0.005), (False, 0.1), (True, 0.06))
        merged = merge_noise_spikes(runs, min_on_sec=0.02)
        assert [m[0] for m in merged] == [False, True]
        assert [m[1] for m in merged] == pytest.approx([0.205, 0.06])

    def test_長いONはそのまま(self) -> None:
        runs = ((True, 0.06), (False, 0.06), (True, 0.18))
        assert merge_noise_spikes(runs) == list(runs)


class TestContrast:
    def test_強い信号ほど大きい(self) -> None:
        strong = contrast_db(synth(snr_db=25.0), SR)
        weak = contrast_db(synth(snr_db=5.0), SR)
        assert strong > weak + 5.0

    def test_無音ならゼロ(self) -> None:
        assert contrast_db(np.zeros(SR * 2, dtype=np.float32), SR) == 0.0


class TestMeasureOperatingPoint:
    def test_速度が測れる(self) -> None:
        for wpm in (15.0, 20.0, 30.0):
            op = measure_operating_point(synth(wpm=wpm), SR)
            assert op is not None
            # ランプ (5 ms) のぶん短点が短く測れるので速めに出る (30 WPM で +14%)
            assert op.wpm == pytest.approx(wpm, rel=0.2)

    def test_文字間が測れる(self) -> None:
        """**ここが動作点判別の要。** 3.0 と 4.4 (held-out) を区別できること."""
        tight = measure_operating_point(synth(char_gap=3.0), SR)
        wide = measure_operating_point(synth(char_gap=4.4), SR)
        assert tight is not None and wide is not None
        # **測定値には系統的な偏りがある**: OFF はランプ長ぶん長く、dot は短く出るので
        # 比は +12% ほど高い (4.4 → 実測 4.9)。実信号どうしの比較では同じ偏りを
        # 共有するので判別には影響しない。ここでは順序と差を見る。
        assert tight.char_gap_units == pytest.approx(3.0, abs=0.6)
        assert wide.char_gap_units == pytest.approx(4.4, abs=0.8)
        assert wide.char_gap_units - tight.char_gap_units > 0.9

    def test_短すぎる音はNone(self) -> None:
        assert measure_operating_point(np.zeros(SR // 2, dtype=np.float32), SR) is None

    def test_無音はNone(self) -> None:
        """**嘘の数字を出さない。**"""
        assert measure_operating_point(np.zeros(SR * 5, dtype=np.float32), SR) is None

    def test_ベクトルは3軸(self) -> None:
        op = OperatingPoint(wpm=20.0, dash_dot_ratio=3.0, intra_gap_units=1.0,
                            char_gap_units=3.0, contrast_db=20.0, n_dot=10, n_dash=5)
        assert op.as_vector().tolist() == [20.0, 3.0, 20.0]


class TestTwoStage:
    def test_きれいな信号は元の方法で測る(self) -> None:
        """適応閾値は縁の精度を落とすので、必要なときだけ使う."""
        op = measure_operating_point(synth(snr_db=25.0), SR)
        assert op is not None and op.robust is False

    def test_モデル選択の軸は2つ(self) -> None:
        op = OperatingPoint(wpm=20.0, dash_dot_ratio=3.0, intra_gap_units=1.0,
                            char_gap_units=3.0, contrast_db=20.0, n_dot=10, n_dash=5)
        assert op.model_axes().tolist() == [3.0, 20.0]

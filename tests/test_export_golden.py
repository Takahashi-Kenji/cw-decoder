"""ゴールデン fixture 出力の検証."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.export_golden import build_golden, load_wave_8k
from src.infer.engine import FrameToken
from src.tokens.converter import TokenConverter

SAMPLE = Path("sample_wav/oubun.wav")

# .f32 の振幅健全性検査の閾値.
# 上限 1.0: 音声として当然の条件 (float32 PCM はこの範囲に収まるはず).
# 下限 0.01: 「ほぼ無音を掴んでいないか」の検査。実測 (oubun 最大振幅 0.436、
# wabun 最大振幅 0.484) より 1 桁以上小さく、将来モデル/波形が変わって
# 再生成しても偽陽性にならない余裕を持たせてある。
MIN_PLAUSIBLE_AMPLITUDE = 0.01
MAX_PLAUSIBLE_AMPLITUDE = 1.0


class FakeEngine:
    """固定のトークン列を返すダミーエンジン."""

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]:
        # 欧文 "E" (・) と "T" (-) に相当する符号のトークン ID は
        # morse_tokens 側で決まるため、ここでは 1 と 2 を使う (形の検証のみ)
        return [
            FrameToken(token_id=1, confidence=0.95, frame_start=0, frame_end=3),
            FrameToken(token_id=2, confidence=0.90, frame_start=10, frame_end=13),
        ]


@pytest.mark.skipif(not SAMPLE.exists(), reason="サンプル音声が無い環境ではスキップ")
def test_load_wave_8k_resamples_and_monos() -> None:
    wave = load_wave_8k(SAMPLE, max_seconds=5.0)
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert 8000 * 4 < wave.size <= 8000 * 5 + 8000


def test_build_golden_shape() -> None:
    wave = np.zeros(8000, dtype=np.float32)
    entry = build_golden(FakeEngine(), "dummy", wave)
    assert entry["name"] == "dummy"
    assert entry["waveFile"] == "dummy.f32"
    assert entry["nSamples"] == 8000
    assert entry["tokenIds"] == [1, 2]
    assert len(entry["confidences"]) == 2
    assert isinstance(entry["textEuropean"], str)
    assert isinstance(entry["textJapanese"], str)


def test_golden_json_is_valid_if_present() -> None:
    """生成済みの golden.json が壊れていないこと."""
    path = Path("web/tests/fixtures/golden.json")
    if not path.exists():
        pytest.skip("未生成 (scripts/export_golden.py を実行すること)")
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert entries, "エントリが空"
    for entry in entries:
        wave_path = path.parent / str(entry["waveFile"])
        assert wave_path.exists(), f"{wave_path} が無い"
        assert wave_path.stat().st_size == int(entry["nSamples"]) * 4
        assert len(entry["tokenIds"]) == len(entry["confidences"])


def test_golden_json_text_matches_token_converter_if_present() -> None:
    """golden.json の tokenIds/confidences から TokenConverter で再変換した結果が
    記録済みの textEuropean/textJapanese と一致すること (内部整合性の検査).

    golden.json が手で編集されてトークン列とテキストが食い違った場合や、
    .f32 だけ再生成して golden.json を再生成し忘れた場合を検出する。
    """
    path = Path("web/tests/fixtures/golden.json")
    if not path.exists():
        pytest.skip("未生成 (scripts/export_golden.py を実行すること)")
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert entries, "エントリが空"
    for entry in entries:
        token_ids = entry["tokenIds"]
        confidences = entry["confidences"]
        european = TokenConverter("european").convert(token_ids, confidences)
        japanese = TokenConverter("japanese").convert(token_ids, confidences)
        assert european.text == entry["textEuropean"], (
            f"{entry['name']}: textEuropean が tokenIds の再変換結果と不一致"
        )
        assert japanese.text == entry["textJapanese"], (
            f"{entry['name']}: textJapanese が tokenIds の再変換結果と不一致"
        )


def test_golden_wave_files_are_healthy_if_present() -> None:
    """.f32 波形が NaN/Inf/全ゼロを含まず、妥当な振幅範囲にあること.

    バイト数だけの検査 (test_golden_json_is_valid_if_present) では、内容が
    壊れていても (無音・NaN 等) 素通りしてしまうため、波形そのものの健全性を見る。
    """
    path = Path("web/tests/fixtures/golden.json")
    if not path.exists():
        pytest.skip("未生成 (scripts/export_golden.py を実行すること)")
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert entries, "エントリが空"
    for entry in entries:
        wave_path = path.parent / str(entry["waveFile"])
        wave = np.fromfile(wave_path, dtype=np.float32)
        assert wave.size == int(entry["nSamples"]), f"{entry['name']}: サンプル数不一致"
        assert not np.isnan(wave).any(), f"{entry['name']}: NaN を含む"
        assert not np.isinf(wave).any(), f"{entry['name']}: Inf を含む"
        assert np.any(wave != 0), f"{entry['name']}: 全ゼロ (無音) 波形"
        max_abs = float(np.max(np.abs(wave)))
        assert MIN_PLAUSIBLE_AMPLITUDE < max_abs <= MAX_PLAUSIBLE_AMPLITUDE, (
            f"{entry['name']}: 振幅が想定範囲外 (max_abs={max_abs})"
        )

"""オンエア切り出しロジックのテスト."""
from __future__ import annotations

import numpy as np
import pytest

from src.finetune.onair_clip import clip_segment, validate_label


class TestClipSegment:
    def test_extracts_time_range(self) -> None:
        sr = 8000
        wave = np.arange(sr * 10, dtype=np.float32)  # 10 秒
        seg = clip_segment(wave, sr, start_s=2.0, end_s=5.0)
        assert seg.size == sr * 3
        assert seg[0] == 2.0 * sr

    def test_end_beyond_length_raises(self) -> None:
        wave = np.zeros(8000 * 3, dtype=np.float32)
        with pytest.raises(ValueError):
            clip_segment(wave, 8000, start_s=1.0, end_s=5.0)

    def test_start_after_end_raises(self) -> None:
        wave = np.zeros(8000 * 10, dtype=np.float32)
        with pytest.raises(ValueError):
            clip_segment(wave, 8000, start_s=5.0, end_s=2.0)


class TestValidateLabel:
    def test_normalizes_and_accepts_tokenizable(self) -> None:
        assert validate_label("[SK] TU", "european") == "{SK} TU"

    def test_rejects_untokenizable(self) -> None:
        with pytest.raises(ValueError):
            validate_label("漢字混入", "european")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            validate_label("   ", "european")

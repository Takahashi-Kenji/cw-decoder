"""録音マネージャのテスト."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.app.recorder import Recorder


class TestRecorder:
    def test_initially_not_recording(self, tmp_path: Path) -> None:
        r = Recorder(out_dir=tmp_path)
        assert not r.is_recording
        assert r.duration_s == 0.0

    def test_start_then_add_blocks(self, tmp_path: Path) -> None:
        r = Recorder(out_dir=tmp_path, sample_rate=8000)
        r.start()
        r.add_block(np.zeros(4000, dtype=np.float32))
        r.add_block(np.zeros(4000, dtype=np.float32))
        assert r.is_recording
        assert r.duration_s == 1.0

    def test_add_block_ignored_when_not_recording(self, tmp_path: Path) -> None:
        r = Recorder(out_dir=tmp_path)
        r.add_block(np.zeros(800, dtype=np.float32))  # 録音前
        assert r.duration_s == 0.0

    def test_save_creates_wav_and_txt(self, tmp_path: Path) -> None:
        r = Recorder(out_dir=tmp_path, sample_rate=8000)
        r.start()
        rng = np.random.default_rng(0)
        for _ in range(5):
            r.add_block(rng.standard_normal(800).astype(np.float32) * 0.1)
        wav_path = r.save_and_reset(decoded_text="ABC DEF", mode="european")
        assert wav_path is not None
        assert wav_path.exists()
        txt_path = wav_path.with_suffix(".txt")
        assert txt_path.exists()
        # WAV を読み返して長さ確認
        data, sr = sf.read(wav_path)
        assert sr == 8000
        assert data.shape[0] == 4000

    def test_save_empty_returns_none(self, tmp_path: Path) -> None:
        r = Recorder(out_dir=tmp_path)
        r.start()
        # ブロック追加なし
        assert r.save_and_reset() is None

    def test_metadata_includes_decoded_text(self, tmp_path: Path) -> None:
        r = Recorder(out_dir=tmp_path, sample_rate=8000)
        r.start()
        r.add_block(np.zeros(800, dtype=np.float32))
        wav_path = r.save_and_reset(decoded_text="ハロー", mode="japanese")
        assert wav_path is not None
        txt = wav_path.with_suffix(".txt").read_text(encoding="utf-8")
        assert "japanese" in txt
        assert "ハロー" in txt

    def test_reset_after_save(self, tmp_path: Path) -> None:
        r = Recorder(out_dir=tmp_path, sample_rate=8000)
        r.start()
        r.add_block(np.zeros(800, dtype=np.float32))
        r.save_and_reset()
        assert r.duration_s == 0.0

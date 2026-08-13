"""受信音声の録音マネージャ (Phase 4 ファインチューニング用データ収集を兼ねる).

録音開始時刻からのバッファを WAV (8 kHz, 16-bit PCM) として保存し、
同名 .txt にデコード済みテキストを書き出す.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 8000


class Recorder:
    """録音バッファ管理. ``add_block`` で蓄積し、``save_and_reset`` で WAV 保存."""

    def __init__(
        self,
        out_dir: Path | str = Path("data/real"),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.sample_rate = sample_rate
        self._blocks: list[np.ndarray] = []
        self._active = False
        self._start_time: _dt.datetime | None = None

    @property
    def is_recording(self) -> bool:
        return self._active

    @property
    def duration_s(self) -> float:
        total = sum(b.size for b in self._blocks)
        return total / self.sample_rate

    def start(self) -> None:
        self._blocks = []
        self._active = True
        self._start_time = _dt.datetime.now()

    def stop(self) -> None:
        self._active = False

    def add_block(self, block: np.ndarray) -> None:
        if not self._active:
            return
        self._blocks.append(block.astype(np.float32, copy=False))

    def save_and_reset(
        self,
        decoded_text: str = "",
        mode: str = "european",
        note: str = "",
    ) -> Path | None:
        """録音内容を WAV と .txt に保存. バッファをクリア.

        Returns: 保存した WAV のパス. 録音内容が空なら ``None``.
        """
        if not self._blocks:
            self._blocks = []
            self._start_time = None
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = (self._start_time or _dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
        stem = f"{timestamp}_{mode}"
        wav_path = self.out_dir / f"{stem}.wav"
        txt_path = self.out_dir / f"{stem}.txt"

        samples = np.concatenate(self._blocks).astype(np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples = (samples / peak * 0.9).astype(np.float32)
        sf.write(wav_path, samples, self.sample_rate, subtype="PCM_16")

        meta_lines = [
            f"mode: {mode}",
            f"sample_rate: {self.sample_rate}",
            f"duration_s: {samples.size / self.sample_rate:.3f}",
            f"timestamp: {timestamp}",
        ]
        if note:
            meta_lines.append(f"note: {note}")
        meta_lines.append("---")
        meta_lines.append(decoded_text)
        txt_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

        self._blocks = []
        self._start_time = None
        return wav_path


__all__ = ["Recorder"]

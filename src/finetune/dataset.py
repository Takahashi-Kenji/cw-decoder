"""実信号データセット (WAV + テキスト).

Phase 3 の ``Recorder`` が出力する形式を主に想定:

::

    data/real/20260612_180000_european.wav
    data/real/20260612_180000_european.txt   ← 同じ stem の TXT

TXT は次のメタデータヘッダ + 区切り行 ``---`` + 正解テキスト:

::

    mode: european
    sample_rate: 8000
    duration_s: 3.500
    timestamp: 20260612_180000
    note: <任意>
    ---
    CQ DE JA0XYZ K

ファイル名から自動でモードを判定できる場合は TXT 不要だが、推奨は **TXT に正解テキストを必ず記述**.
モデルの decode 結果が初期保存されるので、人手で確認・修正してから FT に使う運用.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.tokens.morse_tokens import TOKEN_TO_ID, Mode, text_to_codes


@dataclass(frozen=True)
class RealSignalSample:
    """1 件の実信号サンプル (メタデータ + パス)."""

    wav_path: Path
    txt_path: Path
    mode: Mode
    text: str                       # 人手確認済み正解テキスト
    sample_rate: int = 8000

    @property
    def stem(self) -> str:
        return self.wav_path.stem


def _parse_txt(path: Path) -> tuple[dict[str, str], str]:
    """TXT ファイルを (ヘッダ dict, 本文) に分割.

    区切り行 ``---`` の前をヘッダ、後を本文として扱う.
    区切りが無ければ全体を本文、ヘッダ空.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    sep_idx: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            sep_idx = i
            break
    if sep_idx is None:
        return {}, "\n".join(lines).strip()
    header: dict[str, str] = {}
    for line in lines[:sep_idx]:
        if ":" in line:
            k, _, v = line.partition(":")
            header[k.strip()] = v.strip()
    body = "\n".join(lines[sep_idx + 1:]).strip()
    return header, body


def _infer_mode_from_filename(stem: str) -> Mode | None:
    low = stem.lower()
    if "japanese" in low or "_ja_" in low or "_ja." in low:
        return "japanese"
    if "european" in low or "_eu_" in low or "_eu." in low:
        return "european"
    return None


def discover_real_samples(
    root: Path | str,
    mode_filter: Mode | None = None,
    require_text: bool = True,
) -> list[RealSignalSample]:
    """``root`` 以下の WAV / TXT ペアを列挙.

    Args:
        root: 探索ルート (再帰).
        mode_filter: 指定モードのみ返す.
        require_text: TXT が無い / 本文空のサンプルを除外.

    Returns:
        サンプルのリスト (stem 昇順).
    """
    root = Path(root)
    if not root.exists():
        return []
    samples: list[RealSignalSample] = []
    for wav_path in sorted(root.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        header: dict[str, str] = {}
        body = ""
        if txt_path.exists():
            header, body = _parse_txt(txt_path)
        # モード判定
        mode_str = header.get("mode", "").lower()
        if mode_str in ("european", "japanese"):
            mode: Mode = mode_str  # type: ignore[assignment]
        else:
            inferred = _infer_mode_from_filename(wav_path.stem)
            if inferred is None:
                continue
            mode = inferred
        if mode_filter is not None and mode != mode_filter:
            continue
        if require_text and not body:
            continue
        sr = int(header.get("sample_rate", 8000) or 8000)
        samples.append(RealSignalSample(
            wav_path=wav_path,
            txt_path=txt_path if txt_path.exists() else wav_path.with_suffix(".txt"),
            mode=mode,
            text=body,
            sample_rate=sr,
        ))
    return samples


class RealSignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """``RealSignalSample`` のリストから ``(waveform, token_ids)`` を返す Dataset.

    波形はメモリ常駐 (合計 < 数 GB が前提). 大量データを扱う場合は別途
    ストリーミング Dataset を実装すること.
    """

    def __init__(
        self,
        samples: Sequence[RealSignalSample],
        target_sample_rate: int = 8000,
    ) -> None:
        if not samples:
            raise ValueError("samples is empty")
        self.target_sample_rate = target_sample_rate
        self._samples: list[RealSignalSample] = list(samples)
        # 事前に波形とトークン ID を読み込む.
        # ``_kept_samples`` は実際に採用したサンプルのみを保持し、
        # ``_waveforms`` / ``_token_ids`` とインデックスが一致する
        # (スキップが起きても ``sample_at`` がずれないため).
        self._kept_samples: list[RealSignalSample] = []
        self._waveforms: list[torch.Tensor] = []
        self._token_ids: list[torch.Tensor] = []
        for sample in self._samples:
            wave = self._load_wav(sample.wav_path, target_sample_rate)
            codes = text_to_codes(sample.text, sample.mode)
            ids = [TOKEN_TO_ID[c] for c in codes]
            if not ids:
                # 空テキストはスキップ (CTC が空ターゲットを嫌う)
                continue
            self._kept_samples.append(sample)
            self._waveforms.append(torch.from_numpy(wave))
            self._token_ids.append(torch.tensor(ids, dtype=torch.long))

    @staticmethod
    def _load_wav(path: Path, expected_sr: int) -> np.ndarray:
        wave, sr = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave[:, 0]
        if sr != expected_sr:
            # サンプリングレート不一致は polyphase で 8 kHz へ
            from scipy.signal import resample_poly
            g = np.gcd(sr, expected_sr)
            wave = resample_poly(wave, expected_sr // g, sr // g).astype(np.float32)
        return wave.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self._waveforms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._waveforms[idx], self._token_ids[idx]

    def sample_at(self, idx: int) -> RealSignalSample:
        """``__getitem__(idx)`` に対応する元サンプル (メタデータ) を返す.

        評価時に正解テキスト・モードを引くために使う. 採用済みサンプルのみを
        参照するので、空テキストのスキップが起きてもずれない.
        """
        return self._kept_samples[idx]


__all__ = [
    "RealSignalDataset",
    "RealSignalSample",
    "discover_real_samples",
]

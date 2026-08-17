"""推論エンジンの選択 (ONNX / PyTorch).

**torch の import は関数の中に閉じ込めてある。** これが本モジュールの主目的で、
``.onnx`` を使う限り torch はプロセスに読み込まれない。
配布用インストーラは torch (2.8 GB) を同梱せず、ONNX だけを入れる。

トップレベルで torch を import すると、PyInstaller の依存解析が torch を
引き込んで配布物が 10 倍以上に膨らむ。**ここに import 文を上げないこと。**
``tests/test_no_torch_import.py`` が歯止めになっている。
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from src.infer.ctc import FrameToken

ONNX_SUFFIX = ".onnx"


@runtime_checkable
class DecodeEngine(Protocol):
    """デコード側が engine に求めるもの (これだけ)."""

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]:
        """``(T_wave,)`` float32 8 kHz を ``FrameToken`` 列にする."""
        ...

    @property
    def frame_hop_samples(self) -> int:
        """1 フレームあたりのサンプル数."""
        ...


def is_onnx_model(path: Path | str | None) -> bool:
    """``path`` が ONNX モデルを指しているか."""
    return path is not None and Path(path).suffix.lower() == ONNX_SUFFIX


def load_engine(
    model_path: Path | str,
    device: str = "cpu",
    threads: int = 0,
) -> DecodeEngine:
    """拡張子で経路を選んでエンジンを作る.

    * ``.onnx`` → ``OnnxInferenceEngine`` (torch を読み込まない)
    * それ以外 (``.pt``) → ``InferenceEngine`` (PyTorch)

    Args:
        model_path: モデルファイル。
        device: ``cpu`` / ``cuda`` / ``auto``。**ONNX 経路では CPU 固定**
            (このモデルは小さく、GPU にしてもレイヤごとの起動費用が勝つ)。
        threads: CPU スレッド数。0 で各実装の既定。
    """
    path = Path(model_path)
    if is_onnx_model(path):
        from src.infer.onnx_engine import OnnxInferenceEngine

        return OnnxInferenceEngine(path, threads=threads)

    # --- ここから下は PyTorch 経路。**import を関数の外に出さないこと** ---
    import torch

    from src.infer.engine import InferenceEngine

    dev = resolve_torch_device(device)
    if dev.type == "cpu" and threads > 0:
        torch.set_num_threads(threads)
    return InferenceEngine.from_checkpoint(path, device=dev)


def resolve_torch_device(preference: str):  # -> torch.device
    """``cpu`` / ``cuda`` / ``auto`` を torch の device に解決する.

    **PyTorch 経路専用。** 呼ぶと torch が読み込まれる。
    """
    import torch

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)


def describe_engine(engine: DecodeEngine) -> str:
    """ログに出す 1 行 (どちらの経路で動いているかを必ず残す)."""
    name = type(engine).__name__
    if name == "OnnxInferenceEngine":
        return f"[decode] backend=onnx model={getattr(engine, 'model_path', '?')}"
    device = getattr(engine, "device", "?")
    return f"[decode] backend=torch device={device}"


__all__ = [
    "DecodeEngine",
    "describe_engine",
    "is_onnx_model",
    "load_engine",
    "resolve_torch_device",
]

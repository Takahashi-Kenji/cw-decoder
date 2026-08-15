"""ONNX エクスポートの検証: PyTorch 推論と ONNX 推論のトークン列が一致すること."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.export_onnx import (
    MEL_MAX_ABS_ERR,
    OnnxDecoder,
    export_onnx,
    load_real_sample_wave,
)
from src.infer.engine import InferenceEngine, ctc_greedy_decode_with_frames
from src.train.onnx_mel import ConvMelExtractor

CHECKPOINT = Path("models/full/best_infer.pt")
SAMPLE_WAV = Path("sample_wav/oubun.wav")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="学習済みチェックポイントが無い環境ではスキップ"
)


def _tone_wave(seconds: float = 3.0) -> np.ndarray:
    """CW に近いトーンバースト波形 (決定的)."""
    sr = 8000
    t = np.arange(int(sr * seconds)) / sr
    envelope = (np.sin(2 * np.pi * 4.0 * t) > 0).astype(np.float32)
    return (np.sin(2 * np.pi * 600.0 * t) * envelope).astype(np.float32)


def test_wrapper_matches_pytorch_engine() -> None:
    """OnnxDecoder (PyTorch のまま) の出力が InferenceEngine と一致すること."""
    from src.train.checkpoint import build_model_from_checkpoint

    engine = InferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
    model = build_model_from_checkpoint(CHECKPOINT, map_location=torch.device("cpu"))
    wrapper = OnnxDecoder(model, ConvMelExtractor()).eval()

    wave = _tone_wave()
    expected = [t.token_id for t in engine.decode_chunk(wave)]

    with torch.no_grad():
        log_probs = wrapper(torch.from_numpy(wave).unsqueeze(0))
    got = [t.token_id for t in ctc_greedy_decode_with_frames(log_probs)[0]]

    assert got == expected


def test_exported_onnx_matches_pytorch_engine(tmp_path: Path) -> None:
    """エクスポートした ONNX を onnxruntime で推論し、トークン列が一致すること."""
    import onnxruntime as ort

    out = tmp_path / "cw.onnx"
    export_onnx(CHECKPOINT, out)
    assert out.exists()

    engine = InferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
    wave = _tone_wave()
    expected = [t.token_id for t in engine.decode_chunk(wave)]

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    log_probs = sess.run(None, {"wave": wave[None, :]})[0]
    got = [
        t.token_id
        for t in ctc_greedy_decode_with_frames(torch.from_numpy(log_probs))[0]
    ]

    assert got == expected


@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="実音声サンプルが無い環境ではスキップ")
def test_mel_matches_reference_on_real_audio() -> None:
    """実音声で MelExtractor と ConvMelExtractor のメル出力が MEL_MAX_ABS_ERR 未満で一致すること.

    tests/test_onnx_mel.py は合成波形しか使っていない (Task 1)。ここでは実音声で
    設計書 §5 の「エクスポート時の検証 (必須)」が要求する許容誤差を検証する。
    ONNX グラフはメル単体を出力しないため、比較対象は ConvMelExtractor (PyTorch のまま) とする。
    """
    from src.train.preprocessing import MelExtractor

    wave = load_real_sample_wave(SAMPLE_WAV)
    wave_t = torch.from_numpy(wave).unsqueeze(0)

    ref = MelExtractor().eval()
    conv = ConvMelExtractor().eval()
    with torch.no_grad():
        ref_out = ref(wave_t)
        conv_out = conv(wave_t)

    assert ref_out.shape == conv_out.shape
    assert torch.max(torch.abs(conv_out - ref_out)).item() < MEL_MAX_ABS_ERR


def test_exported_onnx_accepts_variable_length(tmp_path: Path) -> None:
    """時間軸が動的軸になっていること (長さ違いで再エクスポート不要)."""
    import onnxruntime as ort

    out = tmp_path / "cw.onnx"
    export_onnx(CHECKPOINT, out)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

    for seconds in (1.0, 5.0, 8.5):
        wave = _tone_wave(seconds)
        log_probs = sess.run(None, {"wave": wave[None, :]})[0]
        assert log_probs.shape[0] == 1
        # VOCAB_SIZE (src/tokens/morse_tokens.py の唯一の真正なソース) は 73。
        # ブリーフ記載の 72 は旧トークン数に基づく既定値のズレ (詳細は task-2-report.md)。
        assert log_probs.shape[2] == 73
        assert log_probs.shape[1] == len(wave) // 80 + 1

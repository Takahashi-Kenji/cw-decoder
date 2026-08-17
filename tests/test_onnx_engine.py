"""ONNX 経路が PyTorch 経路と同じ結果を出すことを固定する.

**配布物から torch を外す判断の根拠がここにある。** 同じ音に同じトークン列を
返さないなら、外してはいけない。

ONNX モデルもサンプル音声も git 管理外なので、無い環境ではスキップする
(新規クローンや CI と同じ扱い)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.infer.backend import is_onnx_model, load_engine
from src.infer.mel_params import MelConfig

_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = _ROOT / "web" / "public" / "model" / "cw.onnx"
CKPT_PATH = _ROOT / "models" / "full" / "best_infer.pt"
SAMPLE_WAV = _ROOT / "sample_wav" / "oubun.wav"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _load_wave(path: Path, seconds: float = 10.0) -> np.ndarray:
    import soundfile as sf

    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave[:, 0]
    wave = wave[: int(sr * seconds)]
    if sr != MelConfig().sample_rate:
        import soxr

        wave = soxr.resample(wave, sr, MelConfig().sample_rate, quality="HQ")
    return np.ascontiguousarray(wave, dtype=np.float32)


class TestBackendSelection:
    def test_suffix_decides_the_path(self) -> None:
        assert is_onnx_model("models/cw.onnx")
        assert is_onnx_model(Path("a/b/CW.ONNX"))       # 大文字でも
        assert not is_onnx_model("models/best_infer.pt")
        assert not is_onnx_model(None)

    def test_missing_onnx_fails_loudly(self, tmp_path: Path) -> None:
        """**黙って PyTorch に落ちないこと。** 落ちると配布物に torch が要る."""
        with pytest.raises(FileNotFoundError):
            load_engine(tmp_path / "nope.onnx")


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="ONNX モデルが無い環境ではスキップ")
class TestOnnxEngine:
    def test_empty_input_gives_nothing(self) -> None:
        engine = load_engine(ONNX_PATH)
        assert engine.decode_chunk(np.zeros(0, dtype=np.float32)) == []

    def test_frame_hop_matches_mel_config(self) -> None:
        """フレーム位置の換算がずれると、確定の境界が全部ずれる."""
        engine = load_engine(ONNX_PATH)
        assert engine.frame_hop_samples == MelConfig().hop_length

    def test_threads_setting_is_accepted(self) -> None:
        engine = load_engine(ONNX_PATH, threads=2)
        assert engine.decode_chunk(np.zeros(8000, dtype=np.float32)) is not None


@pytest.mark.skipif(
    not (ONNX_PATH.exists() and CKPT_PATH.exists() and SAMPLE_WAV.exists()),
    reason="ONNX・チェックポイント・サンプル音声が揃った環境でのみ実行",
)
class TestMatchesTorchBackend:
    def test_same_token_ids_on_real_audio(self) -> None:
        """**ここが本丸。** 実音声でトークン列が完全一致すること."""
        wave = _load_wave(SAMPLE_WAV)
        onnx_tokens = load_engine(ONNX_PATH).decode_chunk(wave)
        torch_tokens = load_engine(CKPT_PATH).decode_chunk(wave)

        assert [t.token_id for t in onnx_tokens] == [t.token_id for t in torch_tokens]

    def test_same_frame_positions(self) -> None:
        """位置もずれないこと (ずれるとストリーミングの継ぎ目が壊れる)."""
        wave = _load_wave(SAMPLE_WAV)
        onnx_tokens = load_engine(ONNX_PATH).decode_chunk(wave)
        torch_tokens = load_engine(CKPT_PATH).decode_chunk(wave)

        assert [(t.frame_start, t.frame_end) for t in onnx_tokens] == \
               [(t.frame_start, t.frame_end) for t in torch_tokens]

    def test_confidence_is_close(self) -> None:
        """確信度は閾値判定に使うので、実装差で動いてはいけない."""
        wave = _load_wave(SAMPLE_WAV)
        onnx_tokens = load_engine(ONNX_PATH).decode_chunk(wave)
        torch_tokens = load_engine(CKPT_PATH).decode_chunk(wave)

        for o, t in zip(onnx_tokens, torch_tokens, strict=True):
            assert o.confidence == pytest.approx(t.confidence, abs=1e-3)

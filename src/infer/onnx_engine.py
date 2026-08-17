"""ONNX Runtime による推論エンジン (**torch を使わない**).

``InferenceEngine`` (PyTorch 版) と同じ振る舞いをする差し替え品である。
配布用のインストーラから PyTorch (2.8 GB) を外すために用意した。

**精度は変わらない。** ``scripts/export_onnx.py`` の自己検証が、実音声に対して
ONNX と PyTorch のトークン列が完全一致することを毎回確かめている。
``tests/test_onnx_engine.py`` も同じ性質を固定している。

ONNX グラフには**メル変換が焼き込まれている**ので、入力は波形そのもの
(``(1, T_wave)`` の float32 8 kHz)、出力は ``log_probs (1, T_frames, V)``。
ブラウザ版と同じグラフ・同じ前処理を使うことになる。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.infer.ctc import FrameToken, ctc_greedy_decode_frames
from src.infer.mel_params import MelConfig
from src.tokens.morse_tokens import BLANK_TOKEN_ID


class OnnxInferenceEngine:
    """``cw.onnx`` を読み、単一チャンクの波形から ``FrameToken`` 列を返す."""

    def __init__(
        self,
        model_path: Path | str,
        threads: int = 0,
        mel_config: MelConfig | None = None,
        blank_id: int = BLANK_TOKEN_ID,
    ) -> None:
        # 起動を遅くしないため、必要になってから読む
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX モデルが見つかりません: {path}")

        options = ort.SessionOptions()
        if threads > 0:
            # 0 のままだと onnxruntime が既定 (論理コア数) を使う。
            # 設定画面の「CPU スレッド数」をそのまま効かせる。
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.config = mel_config or MelConfig()
        self.blank_id = blank_id
        self.model_path = path

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]:
        """1 つの音声チャンクをデコードする.

        Args:
            waveform: ``(T_wave,)`` の float32 配列 (8 kHz サンプル)。

        Returns:
            ``FrameToken`` の列 (CTC greedy decode 結果)。
        """
        if waveform.size == 0:
            return []
        wave = np.ascontiguousarray(waveform, dtype=np.float32)[None, :]
        log_probs = self._session.run(None, {self._input_name: wave})[0]
        return ctc_greedy_decode_frames(log_probs, blank_id=self.blank_id)[0]

    @property
    def frame_hop_samples(self) -> int:
        """1 フレームあたりのサンプル数 (= mel hop_length)."""
        return self.config.hop_length


__all__ = ["OnnxInferenceEngine"]

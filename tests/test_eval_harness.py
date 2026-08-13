"""評価ハーネスのテスト (小さな seed 固定モデルで構造を検証)."""
from __future__ import annotations

import numpy as np
import torch

from src.eval.harness import decode_wave, evaluate_synth_noise
from src.synth.dataset import RealNoiseEvalSample
from src.train.model import CWModel, ModelConfig
from src.train.preprocessing import MelExtractor
from src.tokens.morse_tokens import VOCAB_SIZE


def _model_and_mel():
    torch.manual_seed(0)
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE))
    model.train(False)
    return model, MelExtractor()


class TestDecodeWave:
    def test_returns_token_id_list(self) -> None:
        model, mel = _model_and_mel()
        wave = torch.zeros(8000, dtype=torch.float32)  # 1 秒
        ids = decode_wave(model, mel, wave, torch.device("cpu"))
        assert isinstance(ids, list)
        assert all(isinstance(x, int) for x in ids)


class TestEvaluateSynthNoise:
    def _samples(self) -> list[RealNoiseEvalSample]:
        rng = np.random.default_rng(0)
        out = []
        for mode, eff in [("european", 10.0), ("european", 0.0), ("japanese", 10.0)]:
            out.append(RealNoiseEvalSample(
                samples=rng.standard_normal(8000).astype(np.float32) * 0.1,
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                text="ABC", mode=mode, wpm=20.0, target_snr_db=eff, eff_snr_db=eff,
            ))
        return out

    def test_structure_and_bins_populated(self) -> None:
        model, mel = _model_and_mel()
        report = evaluate_synth_noise(model, mel, self._samples(), torch.device("cpu"))
        assert report.overall.n_samples == 3
        # eff_snr ビン (10.0 が2件, 0.0 が1件) が集計される
        assert report.report.by_eff_snr[10.0].n_samples == 2
        assert report.report.by_eff_snr[0.0].n_samples == 1
        # モード別
        bm = report.by_mode()
        assert bm["european"].n_samples == 2
        assert bm["japanese"].n_samples == 1

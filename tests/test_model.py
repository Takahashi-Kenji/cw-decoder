"""CW デコーダモデルのテスト."""
from __future__ import annotations

import pytest
import torch

from src.train.model import CWModel, ModelConfig


class TestModelConfig:
    def test_default_freq_factor(self) -> None:
        cfg = ModelConfig()
        # デフォルト stride=(2,2,1) なので周波数縮約 = 4
        assert cfg.cnn_freq_factor == 4

    def test_lstm_input_size(self) -> None:
        cfg = ModelConfig(n_mels=64)
        # 64 → /2 → 32 → /2 → 16 → /1 → 16. 最終ch 64 × 16 = 1024
        assert cfg.lstm_input_size == 64 * 16


class TestForward:
    def test_output_shape_matches_input_time(self) -> None:
        model = CWModel(ModelConfig(n_mels=64, vocab_size=72))
        mel = torch.randn(2, 64, 101)
        out = model(mel)
        assert out.shape == (2, 101, 72)

    def test_handles_4d_input(self) -> None:
        model = CWModel()
        mel = torch.randn(2, 1, 64, 50)
        out = model(mel)
        assert out.shape[0] == 2 and out.shape[1] == 50

    def test_output_dtype_matches_input(self) -> None:
        model = CWModel()
        mel = torch.randn(1, 64, 50)
        assert model(mel).dtype == torch.float32

    def test_gradients_flow(self) -> None:
        model = CWModel()
        mel = torch.randn(2, 64, 30, requires_grad=False)
        out = model(mel)
        out.sum().backward()
        # 何らかの勾配が CNN 重みに流れる
        for p in model.cnn.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_runs_on_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = CWModel().cuda()
        mel = torch.randn(2, 64, 100, device="cuda")
        out = model(mel)
        assert out.is_cuda
        assert out.shape == (2, 100, 72)


class TestParameterBudget:
    def test_param_count_in_target_range(self) -> None:
        model = CWModel()
        n = model.num_parameters()
        # 要件 §3.3.2: 数百万 = 1M 〜 20M 程度
        assert 1_000_000 < n < 20_000_000

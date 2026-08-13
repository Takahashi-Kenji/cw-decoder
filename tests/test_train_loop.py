"""学習ループ・collate・チェックポイント・ロガーのテスト."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.synth.dataset import MorseSynthDataset, make_fixed_eval_set
from src.train.checkpoint import (
    build_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.train.collate import cw_collate
from src.train.logger import CSVLogger
from src.train.loop import compute_input_lengths, evaluate, train_step
from src.train.model import CWModel, ModelConfig
from src.train.preprocessing import MelExtractor


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCollate:
    def test_pads_and_concatenates(self) -> None:
        a_wave = torch.randn(100)
        b_wave = torch.randn(200)
        a_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        b_ids = torch.tensor([4, 5], dtype=torch.long)
        waveforms, targets, wave_lens, target_lens = cw_collate([(a_wave, a_ids), (b_wave, b_ids)])
        assert waveforms.shape == (2, 200)
        # a は 100 サンプル目以降ゼロパディング
        assert torch.all(waveforms[0, 100:] == 0)
        assert targets.tolist() == [1, 2, 3, 4, 5]
        assert wave_lens.tolist() == [100, 200]
        assert target_lens.tolist() == [3, 2]

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError):
            cw_collate([])


class TestComputeInputLengths:
    def test_basic(self) -> None:
        # wave_lengths=[80, 800], hop=80 → mel frames=[2, 11]
        wl = torch.tensor([80, 800], dtype=torch.long)
        result = compute_input_lengths(wl, hop_length=80, max_frames=20)
        assert result.tolist() == [2, 11]

    def test_clamps_to_max_frames(self) -> None:
        wl = torch.tensor([10000], dtype=torch.long)
        result = compute_input_lengths(wl, hop_length=80, max_frames=50)
        assert result.tolist() == [50]


class TestTrainStep:
    def test_one_step_reduces_loss(self, device: torch.device) -> None:
        torch.manual_seed(42)
        model = CWModel(ModelConfig(n_mels=64, vocab_size=72)).to(device)
        mel_ext = MelExtractor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device=device.type, enabled=False)
        criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        # 同じバッチで複数ステップ → loss が下がる
        ds = MorseSynthDataset(
            mode_mix={"european": 1.0}, seed=0, max_samples=4
        )
        batch = cw_collate(list(ds))
        first_loss = train_step(
            model, mel_ext, batch, optimizer, scaler, criterion, device
        ).loss
        for _ in range(5):
            train_step(model, mel_ext, batch, optimizer, scaler, criterion, device)
        last_loss = train_step(
            model, mel_ext, batch, optimizer, scaler, criterion, device
        ).loss
        assert first_loss is not None and last_loss is not None
        assert last_loss < first_loss

    def test_amp_step_on_cuda(self, device: torch.device) -> None:
        if device.type != "cuda":
            pytest.skip("AMP only meaningful on CUDA")
        torch.manual_seed(0)
        model = CWModel().to(device)
        mel_ext = MelExtractor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device="cuda", enabled=True)
        criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        ds = MorseSynthDataset(
            mode_mix={"european": 1.0}, seed=0, max_samples=2
        )
        batch = cw_collate(list(ds))
        result = train_step(
            model, mel_ext, batch, optimizer, scaler, criterion, device,
            amp_dtype=torch.float16,
        )
        assert result.loss is not None
        assert result.loss > 0


class TestCheckpoint:
    def test_roundtrip(self, tmp_path: Path, device: torch.device) -> None:
        torch.manual_seed(0)
        model = CWModel(ModelConfig(n_mels=64, vocab_size=72)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device=device.type, enabled=False)

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer, scaler, step=100, epoch=2, best_metric=0.3)
        assert path.exists()

        # 新しいモデルに復元
        model2 = CWModel(ModelConfig(n_mels=64, vocab_size=72)).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        state = load_checkpoint(path, model2, optimizer2, map_location=device)
        assert state["step"] == 100
        assert state["epoch"] == 2
        assert state["best_metric"] == 0.3

        # 重みが一致
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            torch.testing.assert_close(p1.cpu(), p2.cpu())

    def test_build_from_checkpoint(self, tmp_path: Path, device: torch.device) -> None:
        model = CWModel(ModelConfig(n_mels=64, vocab_size=72)).to(device)
        path = tmp_path / "infer.pt"
        save_checkpoint(path, model, None, None, step=0, epoch=0)
        rebuilt = build_model_from_checkpoint(path, map_location="cpu")
        assert isinstance(rebuilt, CWModel)
        assert rebuilt.config.vocab_size == 72


class TestCSVLogger:
    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        logger.log(step=1, loss=2.5, lr=0.001)
        logger.log(step=2, loss=2.0, lr=0.001)
        content = path.read_text(encoding="utf-8").splitlines()
        assert content[0] == "step,loss,lr"
        assert content[1] == "1,2.5,0.001"
        assert content[2] == "2,2.0,0.001"

    def test_resumes_with_existing_header(self, tmp_path: Path) -> None:
        path = tmp_path / "log.csv"
        path.write_text("step,loss\n1,3.0\n", encoding="utf-8")
        logger = CSVLogger(path)
        logger.log(step=2, loss=2.5)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[2] == "2,2.5"


class TestEvaluate:
    def test_runs_on_fixed_eval_set(self, device: torch.device) -> None:
        torch.manual_seed(0)
        model = CWModel(ModelConfig(n_mels=64, vocab_size=72)).to(device)
        mel_ext = MelExtractor().to(device)
        samples = make_fixed_eval_set(
            snr_grid=[10.0], wpm_grid=[20.0], samples_per_cell=2, seed=42
        )
        report = evaluate(model, mel_ext, samples, device, mode="european")
        # 未学習モデルなので TER は高いが、レポート構造は正しい
        assert report.overall.n_samples == 2
        assert 0.0 <= report.overall.ter <= 5.0
        assert 10.0 in report.by_snr

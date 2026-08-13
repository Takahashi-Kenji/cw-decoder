"""学習チェックポイント保存・読込."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.train.model import CWModel, ModelConfig


def save_checkpoint(
    path: Path | str,
    model: CWModel,
    optimizer: Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
    epoch: int,
    best_metric: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """モデル + オプティマイザ + スケーラ状態を保存."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "step": step,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    path: Path | str,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """チェックポイントを読み込み、与えられたコンポーネントに状態を復元."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if scaler is not None and "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    return state


def build_model_from_checkpoint(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> CWModel:
    """チェックポイントから ``CWModel`` を再構築 (推論用)."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    config = ModelConfig(**state["model_config"])
    model = CWModel(config)
    model.load_state_dict(state["model_state"])
    return model


__all__ = ["build_model_from_checkpoint", "load_checkpoint", "save_checkpoint"]

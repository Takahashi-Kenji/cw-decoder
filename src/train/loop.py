"""学習・評価ループ (AMP / 勾配クリッピング / 評価指標集計)."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from src.tokens.morse_tokens import BLANK_TOKEN_ID, ID_TO_TOKEN
from src.train.decode import ctc_greedy_decode
from src.train.metrics import EvalRecord, EvalReport, bin_snr, bin_wpm
from src.train.model import CWModel
from src.train.preprocessing import MelExtractor

Batch = tuple[Tensor, Tensor, Tensor, Tensor]
"""``(waveforms, targets, wave_lengths, target_lengths)``"""


@dataclass
class TrainStepResult:
    loss: float | None
    grad_norm: float | None
    n_tokens: int


def compute_input_lengths(
    wave_lengths: Tensor, hop_length: int, max_frames: int
) -> Tensor:
    """波形長 → mel フレーム数. ``max_frames`` でクリップ."""
    return torch.minimum(
        wave_lengths.to(torch.long) // hop_length + 1,
        torch.tensor(max_frames, dtype=torch.long, device=wave_lengths.device),
    )


def _set_eval_mode(module: nn.Module) -> None:
    """``module.train(False)`` 経由で推論モードに."""
    module.train(False)


def _set_train_mode(module: nn.Module) -> None:
    module.train(True)


def train_step(
    model: CWModel,
    mel_extractor: MelExtractor,
    batch: Batch,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.CTCLoss,
    device: torch.device,
    grad_clip: float = 5.0,
    amp_dtype: torch.dtype = torch.float16,
) -> TrainStepResult:
    """1 ミニバッチの学習ステップ.

    AMP 有効時は ``autocast`` + ``GradScaler``、勾配クリップを適用.
    NaN/Inf の損失はスキップ.
    """
    _set_train_mode(model)
    waveforms, targets, wave_lengths, target_lengths = batch
    waveforms = waveforms.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    wave_lengths = wave_lengths.to(device, non_blocking=True)
    target_lengths = target_lengths.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler.is_enabled()
    with torch.amp.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=use_amp
    ):
        mel = mel_extractor(waveforms)
        logits = model(mel)
        log_probs = F.log_softmax(logits.float(), dim=-1)

    input_lengths = compute_input_lengths(
        wave_lengths, mel_extractor.config.hop_length, log_probs.size(1)
    )
    loss = criterion(
        log_probs.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
    )

    if not torch.isfinite(loss):
        return TrainStepResult(
            loss=None, grad_norm=None, n_tokens=int(target_lengths.sum().item())
        )

    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return TrainStepResult(
        loss=float(loss.item()),
        grad_norm=float(grad_norm.item()) if grad_norm is not None else None,
        n_tokens=int(target_lengths.sum().item()),
    )


@torch.no_grad()
def evaluate(
    model: CWModel,
    mel_extractor: MelExtractor,
    samples: Iterable,
    device: torch.device,
    mode: str = "european",
    snr_step: float = 5.0,
    wpm_step: float = 5.0,
) -> EvalReport:
    """``SynthResult`` のシーケンスに対し TER/CER を集計."""
    from src.tokens.converter import TokenConverter

    _set_eval_mode(model)
    converter = TokenConverter(mode=mode, confidence_threshold=0.0)
    report = EvalReport()
    hop = mel_extractor.config.hop_length

    for sample in samples:
        wave = torch.from_numpy(sample.samples).unsqueeze(0).to(device)
        mel = mel_extractor(wave)
        logits = model(mel)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        input_lengths = compute_input_lengths(
            torch.tensor([wave.size(-1)], device=device), hop, log_probs.size(1)
        )
        results = ctc_greedy_decode(log_probs, input_lengths, blank_id=BLANK_TOKEN_ID)
        pred = results[0]

        ref_tokens = sample.token_ids.tolist()
        ref_text = sample.text
        pred_text = converter.convert(pred.token_ids).text

        snr_bin = bin_snr(float(sample.config.snr_db), step=snr_step)
        wpm_bin = bin_wpm(float(sample.effective_wpm), step=wpm_step)

        record = EvalRecord(
            ref_tokens=ref_tokens,
            pred_tokens=pred.token_ids,
            ref_text=ref_text,
            pred_text=pred_text,
            snr_db=float(sample.config.snr_db),
            wpm=float(sample.effective_wpm),
        )
        report.add(record, snr_bin=snr_bin, wpm_bin=wpm_bin)

    return report


def decode_token_ids_to_text(token_ids: list[int]) -> list[str]:
    """token ID → 符号文字列 (デバッグ用)."""
    return [ID_TO_TOKEN[i].code for i in token_ids if i in ID_TO_TOKEN]


__all__ = [
    "Batch",
    "TrainStepResult",
    "compute_input_lengths",
    "decode_token_ids_to_text",
    "evaluate",
    "train_step",
]

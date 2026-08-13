"""再利用可能な評価ロジック (decode / keyed_val / synth_val).

``scripts/eval_model.py`` と ``scripts/finetune.py`` / ``scripts/inspect_real_labels.py``
が共有する。decode 経路を一本化して測定の一貫性を保つ。
"""
from __future__ import annotations

import torch

from src.finetune.dataset import RealSignalDataset
from src.synth.dataset import RealNoiseEvalSample
from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import BLANK_TOKEN_ID
from src.train.decode import ctc_greedy_decode
from src.train.loop import compute_input_lengths
from src.train.metrics import DetailedEvalReport, EvalRecord, bin_snr
from src.train.model import CWModel
from src.train.preprocessing import MelExtractor


@torch.no_grad()
def decode_wave(
    model: CWModel,
    mel_extractor: MelExtractor,
    wave: torch.Tensor,
    device: torch.device,
) -> list[int]:
    """波形 (1D tensor) を greedy CTC でデコードし token ID 列を返す."""
    t = wave.unsqueeze(0).to(device)
    log_probs = torch.nn.functional.log_softmax(model(mel_extractor(t)).float(), dim=-1)
    input_lengths = compute_input_lengths(
        torch.tensor([wave.numel()], device=device),
        mel_extractor.config.hop_length,
        log_probs.size(1),
    )
    return ctc_greedy_decode(log_probs, input_lengths, blank_id=BLANK_TOKEN_ID)[0].token_ids


@torch.no_grad()
def evaluate_real_dataset(
    model: CWModel,
    mel_extractor: MelExtractor,
    dataset: RealSignalDataset,
    device: torch.device,
) -> DetailedEvalReport:
    """実信号データセット (keyed_val) を評価."""
    model.train(False)
    report = DetailedEvalReport()
    for i in range(len(dataset)):
        wave, target = dataset[i]
        meta = dataset.sample_at(i)
        pred_ids = decode_wave(model, mel_extractor, wave, device)
        pred_text = TokenConverter(mode=meta.mode, confidence_threshold=0.0).convert(pred_ids).text
        report.add(
            EvalRecord(
                ref_tokens=target.tolist(), pred_tokens=pred_ids,
                ref_text=meta.text, pred_text=pred_text,
            ),
            name=meta.stem, mode=meta.mode,
        )
    return report


@torch.no_grad()
def evaluate_synth_noise(
    model: CWModel,
    mel_extractor: MelExtractor,
    samples: list[RealNoiseEvalSample],
    device: torch.device,
) -> DetailedEvalReport:
    """実ノイズ混合の固定評価セット (synth_val) を評価. 実効SNR別に集計."""
    model.train(False)
    report = DetailedEvalReport()
    for s in samples:
        wave = torch.from_numpy(s.samples)
        pred_ids = decode_wave(model, mel_extractor, wave, device)
        pred_text = TokenConverter(mode=s.mode, confidence_threshold=0.0).convert(pred_ids).text
        report.add(
            EvalRecord(
                ref_tokens=s.token_ids.tolist(), pred_tokens=pred_ids,
                ref_text=s.text, pred_text=pred_text, eff_snr_db=s.eff_snr_db,
            ),
            name="", mode=s.mode, eff_snr_bin=bin_snr(s.eff_snr_db),
        )
    return report


__all__ = ["decode_wave", "evaluate_real_dataset", "evaluate_synth_noise"]

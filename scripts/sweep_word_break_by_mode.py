"""語間スペースの抑制を**モード別に**掃引する (再学習なしのデコード側レバー).

なぜモード別か
--------------
held-out (2026-08-25 実測) で、語間スペースの出し方がモードで真逆だった::

                正解の語間      モデルの出力
    欧文        50 個            50 個   ← ぴったり
    和文        29 個            52 個   ← **1.8 倍に膨らむ**

和文の挿入誤り 34 個のうち 27 個が語間スペースであり、欧文と和文の TER 差
(13.06% 対 27.15%) はほぼこれで説明できる。しかも**過剰挿入は 6 要素の符号の
直後に 3 倍集中**していた (本文中の出現率 11.5% に対し、挿入直前では 33.3%)。
6 要素は和文にしか無い長さで、和文にしか無い誤りを生んでいる。

2026-07-31 の掃引 (`scripts/sweep_word_break.py`) は**モード共通**で振ったため、
keyed_val が改善しても synth_val が悪化して不採用になった。今回は和文だけに
かけるので、欧文側を壊さずに済む可能性がある。

使い方::

    python scripts/sweep_word_break_by_mode.py --ckpt models/full/best_infer.pt \\
        --keyed-dir data/keying_scripts --bias 0 -0.5 -1 -1.5 -2 -3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.dataset import RealSignalDataset, discover_real_samples  # noqa: E402
from src.tokens.morse_tokens import BLANK_TOKEN_ID, VOCAB_SIZE, WORD_BREAK_TOKEN_ID  # noqa: E402
from src.train.checkpoint import load_checkpoint                          # noqa: E402
from src.train.decode import ctc_greedy_decode                            # noqa: E402
from src.train.loop import compute_input_lengths                          # noqa: E402
from src.train.metrics import AggregateMetrics, EvalRecord                # noqa: E402
from src.train.model import CWModel, ModelConfig                          # noqa: E402
from src.train.preprocessing import MelExtractor                          # noqa: E402


@torch.no_grad()
def decode_with_bias(
    model: CWModel,
    mel_extractor: MelExtractor,
    wave: torch.Tensor,
    device: torch.device,
    bias: float,
) -> list[int]:
    """``bias`` を WORD_BREAK の log 確率に足してから greedy デコードする.

    ``bias=0`` のとき :func:`src.eval.harness.decode_wave` と完全に一致する。
    """
    t = wave.unsqueeze(0).to(device)
    log_probs = torch.nn.functional.log_softmax(model(mel_extractor(t)).float(), dim=-1)
    if bias != 0.0:
        log_probs = log_probs.clone()
        log_probs[..., WORD_BREAK_TOKEN_ID] += bias
    input_lengths = compute_input_lengths(
        torch.tensor([wave.numel()], device=device),
        mel_extractor.config.hop_length,
        log_probs.size(1),
    )
    return ctc_greedy_decode(log_probs, input_lengths, blank_id=BLANK_TOKEN_ID)[0].token_ids


def _text(tokens: list[int], mode: str) -> str:
    """トークン列を表示テキストにする (**スペースを含めた CER** を測るため).

    語間を抑えると TER は下がってもテキストが繋がって読みにくくなりうる。
    運用者の元々の苦情が「文字は取れているのに語がくっついて読めない」なので、
    ここを見ずに採否を決めてはいけない。
    """
    from src.tokens.converter import TokenConverter
    conv = TokenConverter(mode, confidence_threshold=0.0)
    return conv.convert(tokens, [1.0] * len(tokens)).text


@torch.no_grad()
def _synth_val_ter(model, mel, device, args, bias: float) -> float:
    """同じバイアスで synth_val (合成 + 実ノイズ) の TER を測る."""
    from src.synth.dataset import make_fixed_real_noise_eval_set
    from src.synth.noise import RealNoisePool

    pool = RealNoisePool.from_dir(args.noise_dir)
    total_err = total_ref = 0
    for mode in ("european", "japanese"):
        for s in make_fixed_real_noise_eval_set(
            noise_pool=pool, snr_grid=[10.0, 5.0, 0.0, -5.0], wpm_grid=[17.0, 25.0],
            samples_per_cell=args.samples_per_cell, seed=20260718, mode=mode,
            tone_center_hz=494.0, filter_bandwidth_hz=300.0,
        ):
            wave = torch.from_numpy(s.samples.astype("float32"))
            pred = decode_with_bias(model, mel, wave, device, bias)
            ref = list(s.token_ids)
            total_err += EvalRecord(ref_tokens=ref, pred_tokens=pred,
                                    ref_text="", pred_text="").token_distance
            total_ref += len(ref)
    return total_err / max(total_ref, 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="語間スペース抑制のモード別掃引")
    p.add_argument("--ckpt", type=Path, default=Path("models/full/best_infer.pt"))
    p.add_argument("--keyed-dir", type=Path, default=Path("data/keying_scripts"))
    p.add_argument("--bias", type=float, nargs="+",
                   default=[0.0, -0.5, -1.0, -1.5, -2.0, -3.0])
    p.add_argument("--noise-dir", type=Path, default=None,
                   help="指定すると synth_val も同じバイアスで測る (採否の 3 条件目)")
    p.add_argument("--samples-per-cell", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("models/eval/wb_by_mode.json"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.train(False)
    mel = MelExtractor().to(device)

    dataset = RealSignalDataset(discover_real_samples(args.keyed_dir))
    print(f"[scan] {len(dataset)} 件", flush=True)

    results: dict[str, dict[str, float]] = {}
    for bias in args.bias:
        by_mode: dict[str, AggregateMetrics] = {}
        wb_pred = {"european": 0, "japanese": 0}
        wb_ref = {"european": 0, "japanese": 0}
        for i in range(len(dataset)):
            wave, target = dataset[i]
            sample = dataset.sample_at(i)
            pred = decode_with_bias(model, mel, wave, device, bias)
            ref = target.tolist()
            by_mode.setdefault(sample.mode, AggregateMetrics()).add(
                EvalRecord(ref_tokens=ref, pred_tokens=pred,
                           ref_text=_text(ref, sample.mode),
                           pred_text=_text(pred, sample.mode))
            )
            wb_pred[sample.mode] += pred.count(WORD_BREAK_TOKEN_ID)
            wb_ref[sample.mode] += ref.count(WORD_BREAK_TOKEN_ID)
        row = {m: metrics.ter for m, metrics in by_mode.items()}
        row.update({f"{m}_cer": metrics.cer for m, metrics in by_mode.items()})
        if args.noise_dir is not None:
            row["synth_val"] = _synth_val_ter(model, mel, device, args, bias)
        results[f"{bias:+.1f}"] = row
        eu, ja = row.get("european", 0.0), row.get("japanese", 0.0)
        sv = row.get("synth_val")
        sv_txt = "" if sv is None else f"   synth {sv * 100:6.2f}%"
        print(f"bias {bias:+5.1f}   欧文 TER {eu * 100:6.2f}% CER {row['european_cer']*100:6.2f}%"
              f"   和文 TER {ja * 100:6.2f}% CER {row['japanese_cer']*100:6.2f}%{sv_txt}   "
              f"語間 欧文 {wb_pred['european']:3d}/{wb_ref['european']:3d}  "
              f"和文 {wb_pred['japanese']:3d}/{wb_ref['japanese']:3d}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

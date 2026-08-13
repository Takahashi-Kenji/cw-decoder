"""チェックポイント評価 CLI (Phase B B0).

任意の ckpt に対し synth_val (合成+実ノイズ、実効SNR別) と keyed_val (打鍵録音) を
評価し JSON 保存する。--baseline 指定で改善前後を比較表示する。

使い方::

    python scripts/eval_model.py --ckpt models/full/best.pt \\
        --noise-dir data/keying_scripts --keyed-dir data/keying_scripts \\
        --out models/eval/baseline.json

    python scripts/eval_model.py --ckpt models/ft/best.pt \\
        --noise-dir data/keying_scripts --keyed-dir data/real/val \\
        --out models/eval/ft.json --baseline models/eval/baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.compare import compare_reports                                   # noqa: E402
from src.eval.harness import evaluate_real_dataset, evaluate_synth_noise       # noqa: E402
from src.finetune.dataset import RealSignalDataset, discover_real_samples      # noqa: E402
from src.synth.dataset import make_fixed_real_noise_eval_set                   # noqa: E402
from src.synth.noise import RealNoisePool                                      # noqa: E402
from src.tokens.morse_tokens import VOCAB_SIZE                                 # noqa: E402
from src.train.checkpoint import load_checkpoint                              # noqa: E402
from src.train.metrics import DetailedEvalReport, bin_snr                     # noqa: E402
from src.train.model import CWModel, ModelConfig                             # noqa: E402
from src.train.preprocessing import MelExtractor                            # noqa: E402


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="チェックポイント評価 (synth_val + keyed_val)")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--noise-dir", type=Path, default=None, help="synth_val 用ノイズ WAV ディレクトリ")
    p.add_argument("--keyed-dir", type=Path, default=None, help="keyed_val 用 WAV+TXT ディレクトリ")
    p.add_argument("--out", type=Path, default=Path("models/eval/eval.json"))
    p.add_argument("--baseline", type=Path, default=None, help="比較元 JSON")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--tone-center", type=float, default=494.0)
    p.add_argument("--bpf-bandwidth", type=float, default=300.0)
    p.add_argument("--wpm", type=float, nargs="+", default=[17.0, 25.0])
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0, -5.0])
    p.add_argument("--samples-per-cell", type=int, default=25)
    return p


def _section_dict(report: "DetailedEvalReport") -> "dict[str, Any]":
    d = report.to_dict()
    d.pop("samples", None)  # synth_val はサンプル別を含めない (件数が多い)
    d["confusion"] = report.analysis.confusion_to_dict()
    return d


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.train(False)
    mel = MelExtractor().to(device)

    report: dict = {"ckpt": str(args.ckpt), "seed": args.seed}

    # ---- synth_val ----
    if args.noise_dir is not None:
        pool = RealNoisePool.from_dir(args.noise_dir)
        synth_report = DetailedEvalReport()   # 欧文/和文を 1 レポートに統合
        for mode in ("european", "japanese"):
            samples = make_fixed_real_noise_eval_set(
                noise_pool=pool, snr_grid=args.snr, wpm_grid=args.wpm,
                samples_per_cell=args.samples_per_cell, seed=args.seed, mode=mode,
                tone_center_hz=args.tone_center, filter_bandwidth_hz=args.bpf_bandwidth,
            )
            r = evaluate_synth_noise(model, mel, samples, device)
            synth_report = _merge_reports(synth_report, r)
        section = _section_dict(synth_report)
        section["config"] = {
            "tone_center_hz": args.tone_center, "bpf_bandwidth_hz": args.bpf_bandwidth,
            "wpm_grid": args.wpm, "snr_grid": args.snr,
            "samples_per_cell": args.samples_per_cell,
        }
        report["synth_val"] = section
        for line in synth_report.summary_lines():
            print(f"[synth] {line}", flush=True)

    # ---- keyed_val ----
    if args.keyed_dir is not None:
        samples = discover_real_samples(args.keyed_dir)
        if samples:
            dataset = RealSignalDataset(samples)
            keyed_report = evaluate_real_dataset(model, mel, dataset, device)
            report["keyed_val"] = keyed_report.to_dict() | {
                "confusion": keyed_report.analysis.confusion_to_dict()
            }
            for line in keyed_report.summary_lines():
                print(f"[keyed] {line}", flush=True)
        else:
            print(f"[warn] keyed-dir にサンプルがありません: {args.keyed_dir}", flush=True)

    if "synth_val" not in report and "keyed_val" not in report:
        print("[err] synth_val も keyed_val も評価できませんでした", flush=True)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[out] {args.out}", flush=True)

    # ---- baseline 比較 ----
    if args.baseline is not None:
        try:
            base = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] baseline を読めません ({exc}) — 比較をスキップ", flush=True)
        else:
            _print_comparison(compare_reports(base, report))
    return 0


def _merge_reports(a: DetailedEvalReport, b: DetailedEvalReport) -> DetailedEvalReport:
    """2 つの DetailedEvalReport を統合 (samples を足し合わせ、eff_snr ビンを再構築)."""
    merged = DetailedEvalReport()
    for src_report in (a, b):
        for s in src_report.samples:
            eff_bin = None if s.record.eff_snr_db is None else bin_snr(s.record.eff_snr_db)
            merged.add(s.record, name=s.name, mode=s.mode, eff_snr_bin=eff_bin)
    return merged


def _print_comparison(cmp: dict) -> None:
    for section, data in cmp.items():
        print(f"\n=== {section}: baseline → current ===", flush=True)
        ov = data["overall"]
        if "ter_baseline" in ov:
            print(f"Overall  TER {ov['ter_baseline']*100:.1f}% → {ov['ter_current']*100:.1f}% "
                  f"({ov['ter_delta']*100:+.1f})", flush=True)
        if data["by_eff_snr"]:
            print("By EffSNR:", flush=True)
            for b, d in sorted(data["by_eff_snr"].items(), key=lambda kv: float(kv[0])):
                delta = d["ter_delta"]
                s = "n/a" if delta == "n/a" else f"{delta*100:+.1f}"
                print(f"  {b:>6}dB  TER delta {s}", flush=True)
        imp = data["token_recall_improvements"][:5]
        if imp:
            print("Top token recall improvements:", flush=True)
            for t in imp:
                print(f"  {t.get('code')}: recall {t['recall_baseline']*100:.0f}% "
                      f"→ {t['recall_current']*100:.0f}% ({t['recall_delta']*100:+.0f})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

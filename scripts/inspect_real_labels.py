"""実信号ラベル検品 CLI.

録音 (WAV+TXT) を既存モデルでデコードし、正解ラベルと照合して TER 降順に
「打鍵ミス / 誤ラベル疑い」を提示する。符号長別・token 別 recall も出力し、
新規録音を聴かずにラベル品質を検査する運用ツール。

使い方::

    python scripts/inspect_real_labels.py --data-dir data/keying_scripts \\
        --ckpt models/full/best_infer.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.harness import decode_wave                                   # noqa: E402
from src.finetune.dataset import RealSignalDataset, discover_real_samples  # noqa: E402
from src.finetune.label_inspection import recall_by_code_length            # noqa: E402
from src.tokens.converter import TokenConverter                            # noqa: E402
from src.tokens.morse_tokens import VOCAB_SIZE                             # noqa: E402
from src.train.checkpoint import load_checkpoint                           # noqa: E402
from src.train.metrics import DetailedEvalReport, EvalRecord, token_label  # noqa: E402
from src.train.model import CWModel, ModelConfig                          # noqa: E402
from src.train.preprocessing import MelExtractor                         # noqa: E402


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="実信号ラベル検品")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--top-n", type=int, default=10, help="疑い上位の表示件数")
    return p


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    samples = discover_real_samples(args.data_dir)
    if not samples:
        print(f"[err] サンプルが見つかりません: {args.data_dir}", flush=True)
        return 2
    dataset = RealSignalDataset(samples)

    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.train(False)
    mel = MelExtractor().to(device)

    report = DetailedEvalReport()
    rows: list[tuple[str, float, str, str]] = []
    for i in range(len(dataset)):
        wave, target = dataset[i]
        meta = dataset.sample_at(i)
        pred_ids = decode_wave(model, mel, wave, device)
        pred_text = TokenConverter(mode=meta.mode, confidence_threshold=0.0).convert(
            pred_ids
        ).text
        rec = EvalRecord(
            ref_tokens=target.tolist(), pred_tokens=pred_ids,
            ref_text=meta.text, pred_text=pred_text,
        )
        report.add(rec, name=meta.stem, mode=meta.mode)
        rows.append((meta.stem, report.samples[-1].ter, meta.text, pred_text))

    rows.sort(key=lambda r: -r[1])
    print(f"\n=== TER 降順 (上位 {args.top_n} = 打鍵ミス/誤ラベル疑い) ===")
    for stem, ter, ref, pred in rows[: args.top_n]:
        print(f"\n--- {stem}  TER={ter * 100:.1f}%")
        print(f"    ref : {ref}")
        print(f"    pred: {pred}")

    print("\n=== 符号長別 recall ===")
    for n, (ref, recall) in sorted(recall_by_code_length(report.analysis).items()):
        print(f"  長さ{n}: ref={ref:4d}  recall={recall:5.1f}%")

    print()
    for line in report.summary_lines(top_n=args.top_n):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

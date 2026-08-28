"""語間バイアスを**選んだのと別のデータで**測り直す.

なぜ要るか
----------
掃引は held-out 20 件の上で bias を選び、その同じ 20 件で成績を報告していた。
**評価セットでハイパーパラメータを調整すると成績は必ず良く出る。**
20 件では調整の自由度に対してデータが少なく、偶然を拾いうる。

ここでは held-out を 2 つに割り、**片方で bias を選び、もう片方で測る**。
両向き (A で選んで B、B で選んで A) を出し、選んだ側と測った側の差を見る。
差が大きければ、その bias は偶然を拾っている。

使い方::

    python scripts/validate_word_break_bias.py --ckpt models/full/best_infer.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.sweep_word_break_by_mode import decode_with_bias                # noqa: E402
from src.finetune.dataset import RealSignalDataset, discover_real_samples    # noqa: E402
from src.tokens.morse_tokens import VOCAB_SIZE                               # noqa: E402
from src.train.checkpoint import load_checkpoint                             # noqa: E402
from src.train.metrics import AggregateMetrics, EvalRecord                   # noqa: E402
from src.train.model import CWModel, ModelConfig                             # noqa: E402
from src.train.preprocessing import MelExtractor                             # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="語間バイアスの独立確認")
    p.add_argument("--ckpt", type=Path, default=Path("models/full/best_infer.pt"))
    p.add_argument("--keyed-dir", type=Path, default=Path("data/keying_scripts"))
    p.add_argument("--bias", type=float, nargs="+",
                   default=[0.0, -1.0, -2.0, -3.0, -5.0, -8.0])
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.train(False)
    mel = MelExtractor().to(device)

    dataset = RealSignalDataset(discover_real_samples(args.keyed_dir))
    # **モードごとに交互に割る。** 前半/後半で割ると難易度が偏りうる
    # (原稿は生成順に長さが変わる) ので、モード内の並び順で偶奇に分ける。
    halves: dict[tuple[str, str], list[int]] = {}
    seen: dict[str, int] = {}
    for i in range(len(dataset)):
        mode = dataset.sample_at(i).mode
        k = seen.get(mode, 0)
        seen[mode] = k + 1
        halves.setdefault((mode, "A" if k % 2 == 0 else "B"), []).append(i)

    # (mode, half, bias) → TER
    ter: dict[tuple[str, str, float], float] = {}
    for bias in args.bias:
        agg: dict[tuple[str, str], AggregateMetrics] = {}
        for (mode, half), idxs in halves.items():
            m = AggregateMetrics()
            for i in idxs:
                wave, target = dataset[i]
                pred = decode_with_bias(model, mel, wave, device, bias)
                m.add(EvalRecord(ref_tokens=target.tolist(), pred_tokens=pred,
                                 ref_text="", pred_text=""))
            agg[(mode, half)] = m
            ter[(mode, half, bias)] = m.ter
        print(f"bias {bias:+5.1f}  " + "  ".join(
            f"{mode[:2]}-{half} {agg[(mode, half)].ter * 100:6.2f}%"
            for mode, half in sorted(agg)), flush=True)

    print()
    print("選んだ側 → 測った側 (差が大きいほど、その bias は偶然を拾っている)")
    for mode in sorted({m for m, _ in halves}):
        for pick, check in (("A", "B"), ("B", "A")):
            best = min(args.bias, key=lambda b: ter[(mode, pick, b)])
            print(f"  {mode:9s} {pick} で選ぶと bias {best:+5.1f} "
                  f"({pick} 側 {ter[(mode, pick, best)] * 100:6.2f}%) → "
                  f"{check} 側では {ter[(mode, check, best)] * 100:6.2f}%   "
                  f"({check} 側の最良は {min(ter[(mode, check, b)] for b in args.bias) * 100:6.2f}%)",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

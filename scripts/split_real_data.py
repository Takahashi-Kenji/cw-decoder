"""実録音を train / val に取り分ける (**本文単位**).

**件数で割ってはいけない。** L4 の 2 セッション (`20260824_5w_18m` と
`20260825_50w_18m`) は**同じ 700 本文を 5W と 50W で録ったもの**で、
件数で割ると同じ本文が train と val の両方に入る。val が train の
言い換えになり、best の選び方が壊れる。

**held-out と本文が重なるものは学習から外す。** 実測で held-out 20 件のうち
2 件が L4 の本文と重なっていた (物差しの 10%)。学習に入れると、その分だけ
held-out の成績が良く見える。

使い方::

    python scripts/split_real_data.py \\
      --src data/l4/20260824_5w_18m --src data/l4/20260825_50w_18m \\
      --exclude data/keying_scripts \\
      --val-texts 120 --out-dir data/ft/v4 --seed 42
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.dataset import RealSignalSample, discover_real_samples  # noqa: E402


def plan_split(
    samples: list[RealSignalSample],
    val_texts: int,
    seed: int,
    excluded_texts: frozenset[str] = frozenset(),
) -> tuple[list[RealSignalSample], list[RealSignalSample], list[RealSignalSample]]:
    """``(train, val, excluded)`` に分ける.

    **本文が同じサンプルは必ず同じ側へ行く。** 除外本文はどちらにも入れず、
    3 本目のリストとして返す (何件落としたかを呼び出し側が印字するため)。
    """
    by_text: dict[str, list[RealSignalSample]] = defaultdict(list)
    excluded: list[RealSignalSample] = []
    for s in samples:
        text = s.text.strip()
        if text in excluded_texts:
            excluded.append(s)
        else:
            by_text[text].append(s)

    texts = sorted(by_text)                       # 決定的にするため先に整列
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    val_set = {texts[i] for i in order[:min(val_texts, len(texts))]}

    train = [s for t in texts if t not in val_set for s in by_text[t]]
    val = [s for t in texts if t in val_set for s in by_text[t]]
    return train, val, excluded


def copy_pairs(samples: list[RealSignalSample], out_dir: Path) -> int:
    """WAV + TXT を ``out_dir`` へ複製する (stem が衝突したら親フォルダ名を前置)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        stem = s.wav_path.stem
        if (out_dir / f"{stem}.wav").exists():
            stem = f"{s.wav_path.parent.name}_{stem}"
        shutil.copy2(s.wav_path, out_dir / f"{stem}.wav")
        shutil.copy2(s.txt_path, out_dir / f"{stem}.txt")
    return len(samples)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="実録音を本文単位で train / val に取り分ける")
    p.add_argument("--src", type=Path, action="append", required=True,
                   help="取り分け元 (再帰)。複数指定可")
    p.add_argument("--exclude", type=Path, action="append", default=[],
                   help="**ここと本文が重なるサンプルは train にも val にも入れない。** "
                        "held-out を渡すこと")
    p.add_argument("--val-texts", type=int, default=120, help="val に回す本文の数")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="件数だけ出して複製しない")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)

    samples: list[RealSignalSample] = []
    for src in args.src:
        found = discover_real_samples(src)
        print(f"[scan] {src}: {len(found)} 件", flush=True)
        samples.extend(found)
    if not samples:
        raise SystemExit("取り分け元にサンプルがありません")

    excluded_texts: set[str] = set()
    for ex in args.exclude:
        found = discover_real_samples(ex)
        excluded_texts.update(s.text.strip() for s in found)
        print(f"[scan] 除外元 {ex}: {len(found)} 件 / 本文 {len(excluded_texts)}", flush=True)

    train, val, dropped = plan_split(
        samples, args.val_texts, args.seed, frozenset(excluded_texts))
    n_train_text = len({s.text.strip() for s in train})
    n_val_text = len({s.text.strip() for s in val})
    print(f"[split] train {len(train)} 件 (本文 {n_train_text}) / "
          f"val {len(val)} 件 (本文 {n_val_text}) / "
          f"除外 {len(dropped)} 件 (held-out と本文が重なる)", flush=True)
    assert not ({s.text.strip() for s in train} & {s.text.strip() for s in val})

    if args.dry_run:
        return 0
    copy_pairs(train, args.out_dir / "train")
    copy_pairs(val, args.out_dir / "val")
    print(f"[done] {args.out_dir}/train と {args.out_dir}/val に複製した", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

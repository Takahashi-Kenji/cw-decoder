"""複数の集合の動作点を測り、**集合を分離できるか**を出す (計画 v5 P3).

モデルを切り替えて使うには、受信音からどの動作点かを当てられなければならない。
ここでは (WPM, 文字間, コントラスト) の 3 軸で**最近傍セントロイド**による
leave-one-out 判別率を出す。閾値を手で調整しないので、「分離できるか」を
公平に測れる。

使い方::

    python scripts/measure_operating_points.py \\
      --set held_out=data/keying_scripts --set old=data/ft/real_train \\
      --set l4_weak=data/l4/20260824_5w_18m --set l4_strong=data/l4/20260825_50w_18m \\
      --limit 120

判定 (計画 v5 P3): **90% 以上**正しく振り分けられること。

**判別の目標は「どの集合か」ではなく「どの動作点か」。** L4 は意図的に
15〜30 WPM を混ぜているので WPM で集合は当てられないし、`old` と `l4_weak`
(文字間 3.1〜3.3・コントラスト 9 dB) は**同じ動作点**である (wabun が両方で
学習して成功した)。`--group` で集合を動作点にまとめ、`--axes model` で
モデル選択に効く 2 軸 (文字間, コントラスト) だけを使う::

    ... --group weak_tight=old,l4_weak --group strong_tight=l4_strong \
        --group strong_wide=held_out --axes model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.dataset import discover_real_samples          # noqa: E402
from src.infer.operating_point import measure_operating_point   # noqa: E402


def measure_dir(path: Path, limit: int, axes: str = "all") -> tuple[np.ndarray, int, int]:
    """ディレクトリ内の WAV を測り、(N, k) の行列・測れなかった件数・適応閾値に落ちた件数を返す."""
    rows: list[np.ndarray] = []
    failed = robust = 0
    for s in discover_real_samples(path)[:limit]:
        if "noise_sample" in s.wav_path.name:
            continue
        wave, sr = sf.read(s.wav_path, dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave[:, 0]
        op = measure_operating_point(wave, sr)
        if op is None:
            failed += 1
            continue
        robust += int(op.robust)
        rows.append(op.model_axes() if axes == "model" else op.as_vector())
    k = 2 if axes == "model" else 3
    return (np.array(rows) if rows else np.zeros((0, k))), failed, robust


def loo_nearest_centroid(sets: dict[str, np.ndarray]) -> dict[str, float]:
    """leave-one-out の最近傍セントロイド判別率を集合ごとに返す.

    各軸は全体の標準偏差で正規化する (WPM と dB を同じ尺度で扱うため)。
    """
    names = list(sets)
    allx = np.concatenate([sets[n] for n in names])
    scale = allx.std(axis=0)
    scale[scale == 0] = 1.0
    acc: dict[str, float] = {}
    for n in names:
        x = sets[n]
        hit = 0
        for i in range(len(x)):
            cents = {}
            for m in names:
                pts = sets[m] if m != n else np.delete(sets[m], i, axis=0)
                if len(pts) == 0:
                    continue
                cents[m] = pts.mean(axis=0)
            d = {m: np.linalg.norm((x[i] - c) / scale) for m, c in cents.items()}
            if min(d, key=d.get) == n:
                hit += 1
        acc[n] = hit / len(x) if len(x) else 0.0
    return acc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="動作点を測り、集合を分離できるかを出す")
    p.add_argument("--set", action="append", required=True, help="名前=ディレクトリ (複数)")
    p.add_argument("--limit", type=int, default=120, help="各集合で測る最大件数")
    p.add_argument("--group", action="append", default=[],
                   help="動作点=集合1,集合2 (複数)。指定した集合をまとめて判別する")
    p.add_argument("--axes", choices=("all", "model"), default="all",
                   help="all: WPM/文字間/コントラスト, model: 文字間/コントラスト (モデル選択に効く 2 軸)")
    args = p.parse_args(argv)

    sets: dict[str, np.ndarray] = {}
    print(f"{'集合':14} {'n':>4} {'測れず':>5} {'適応':>4} {'WPM':>6} {'文字間':>6} {'コントラスト':>8}")
    for spec in args.set:
        name, _, d = spec.partition("=")
        full, failed, robust = measure_dir(Path(d), args.limit, axes="all")
        if len(full) == 0:
            print(f"{name:14} 測れず"); continue
        med = np.median(full, axis=0)
        print(f"{name:14} {len(full):4d} {failed:5d} {robust:4d} {med[0]:6.1f} {med[1]:6.2f} {med[2]:7.1f}dB")
        sets[name] = full[:, 1:] if args.axes == "model" else full

    if args.group:
        grouped: dict[str, np.ndarray] = {}
        for spec in args.group:
            gname, _, members = spec.partition("=")
            parts = [sets[m] for m in members.split(",") if m in sets]
            if parts:
                grouped[gname] = np.concatenate(parts)
        sets = grouped

    if len(sets) < 2:
        print("集合が 2 つ未満なので判別率は出せない"); return 1
    acc = loo_nearest_centroid(sets)
    print()
    axes_label = "文字間 / コントラスト" if args.axes == "model" else "WPM / 文字間 / コントラスト"
    print(f"leave-one-out 最近傍セントロイド判別率 ({axes_label}):")
    for n, a in acc.items():
        print(f"  {n:14} {a * 100:5.1f}%")
    total = sum(acc[n] * len(sets[n]) for n in sets) / sum(len(v) for v in sets.values())
    print(f"  {'全体':14} {total * 100:5.1f}%   (合格 90%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

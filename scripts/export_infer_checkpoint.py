"""推論専用の軽量チェックポイントを書き出す.

学習用チェックポイント (`best.pt`) には optimizer_state (Adam の
exp_avg / exp_avg_sq) と scaler_state が同梱されており、モデル重みの
約 3 倍 (≈49 MB) になる。推論・学習再開のいずれにもこれらは必須でない
ため、`model_state` (fp32) と `model_config` のみを残した軽量版を
書き出す。サイズは ≈17 MB に収まり、Git に直接コミットできる。

使い方:
    python scripts/export_infer_checkpoint.py \
        --src models/full/best.pt --dst models/full/best_infer.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

# 推論側ローダ (build_model_from_checkpoint) が参照するキーのみ残す。
# step は来歴確認用に保持 (サイズは無視できる)。
_KEEP_KEYS = ("model_state", "model_config", "step")


def export_infer_checkpoint(src: Path, dst: Path) -> tuple[float, float]:
    """軽量チェックポイントを書き出し、(元 MB, 新 MB) を返す."""
    # 中身はテンソル + dict + 数値のみなので weights_only=True で安全に読める。
    state = torch.load(src, map_location="cpu", weights_only=True)
    missing = [k for k in ("model_state", "model_config") if k not in state]
    if missing:
        raise KeyError(f"チェックポイントに必須キーがありません: {missing}")
    slim = {k: state[k] for k in _KEEP_KEYS if k in state}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    return src.stat().st_size / 1e6, dst.stat().st_size / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("models/full/best.pt"))
    parser.add_argument("--dst", type=Path, default=Path("models/full/best_infer.pt"))
    args = parser.parse_args()

    src_mb, dst_mb = export_infer_checkpoint(args.src, args.dst)
    print(f"[export] {args.src} ({src_mb:.1f} MB) -> {args.dst} ({dst_mb:.1f} MB)")


if __name__ == "__main__":
    main()

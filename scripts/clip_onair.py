"""オンエア録音の区間切り出し CLI.

長い QSO 録音から時間区間を切り出し、正解ラベルを付けて data/real/onair/ に
実信号 TXT 形式で保存する。ラベルは正規化 + トークン化検証を通す。
自信のない区間は保存しない運用 (誤ラベルは指標を壊すため)。

使い方::

    python scripts/clip_onair.py --wav rec/qso1.wav --mode european \\
        --start 12.0 --end 20.0 --label "CQ DE JA0XYZ K"
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

from src.finetune.onair_clip import clip_segment, validate_label  # noqa: E402


def _fmt_time_tag(seconds: float) -> str:
    """秒数を自動ファイル名向けの安全な文字列に変換する (小数第 1 位まで保持).

    小数点は ``p`` に置換する (例: ``12.4`` → ``12p4``)。切り捨て秒のみだと
    12.4s と 12.9s のような区間が同じ stem になり上書きされるため、
    自動生成 stem では常に小数第 1 位まで含める。
    """
    return f"{seconds:.1f}".replace(".", "p").replace("-", "m")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="オンエア録音の区間切り出し")
    p.add_argument("--wav", type=Path, required=True, help="長い QSO 録音 WAV")
    p.add_argument("--mode", choices=["european", "japanese"], required=True)
    p.add_argument("--start", type=float, required=True, help="開始秒")
    p.add_argument("--end", type=float, required=True, help="終了秒")
    p.add_argument("--label", type=str, required=True, help="正解テキスト")
    p.add_argument("--out-dir", type=Path, default=Path("data/real/onair"))
    p.add_argument("--stem", type=str, default=None, help="出力ファイル名 (省略時自動)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)

    # ラベル検証を先に (不能なら音声を書かずに終了)
    try:
        label = validate_label(args.label, args.mode)  # type: ignore[arg-type]
    except ValueError as exc:
        print(f"[err] ラベル不正: {exc}", flush=True)
        return 2

    wave, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave[:, 0]
    if sr != 8000:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), 8000)
        wave = resample_poly(wave, 8000 // g, int(sr) // g).astype(np.float32)
        sr = 8000

    try:
        seg = clip_segment(wave, sr, args.start, args.end)
    except ValueError as exc:
        print(f"[err] 区間不正: {exc}", flush=True)
        return 2

    peak = float(np.max(np.abs(seg))) if seg.size else 0.0
    if peak > 0:
        seg = (seg / peak * 0.9).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or (
        f"{args.wav.stem}_{_fmt_time_tag(args.start)}"
        f"_{_fmt_time_tag(args.end)}_{args.mode}"
    )
    wav_path = args.out_dir / f"{stem}.wav"
    txt_path = args.out_dir / f"{stem}.txt"
    sf.write(wav_path, seg, sr, subtype="PCM_16")
    txt_path.write_text(
        f"mode: {args.mode}\n"
        f"sample_rate: {sr}\n"
        f"duration_s: {seg.size / sr:.3f}\n"
        f"source: {args.wav.name} [{args.start:.1f}-{args.end:.1f}s]\n"
        "---\n"
        f"{label}\n",
        encoding="utf-8",
    )
    print(f"[done] {wav_path} ({seg.size / sr:.1f}s)  label={label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

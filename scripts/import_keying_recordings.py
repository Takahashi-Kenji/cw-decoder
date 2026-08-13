"""打鍵録音を学習用データセットとして取り込む CLI.

``keying_corpus`` の原稿に対応する録音 (``t01.wav`` … の連番) を、原稿本文を
ラベルとした WAV+TXT ペアとして ``data/real/train/`` に配置する。

使い方::

    python scripts/import_keying_recordings.py \\
        --european-dir data/欧文 --japanese-dir data/和文

    # 配置せず対応と時間差の確認だけ
    python scripts/import_keying_recordings.py --european-dir data/欧文 --dry-run

録音は原稿と同じ順の連番 (``t01.wav`` = 原稿 1 件目) であることを前提とする。
取り違えを検出するため、実録音長と原稿の見積もり長の差を必ず表示し、
差が大きいものは警告する (前後の無音があるため実録音の方が長いのが正常)。

**BPF を必ず通す**: モデルはライブ運用でも BPF 通過後の音を受け取るため、
学習データも同じ帯域に揃える。オートキーヤーから直接録音した音 (BPF 未通過)
をそのまま与えると TER 97% まで崩壊することを確認済み。既存の打鍵録音は
アプリ経由で録ったため BPF 済みで、帯域内エネルギーが 99.9% ある。
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.keying_corpus import (  # noqa: E402
    EUROPEAN_SCRIPTS,
    JAPANESE_SCRIPTS,
    KeyingScript,
)
from src.finetune.keying_scripts import estimate_duration_sec  # noqa: E402
from src.tokens.morse_tokens import text_to_codes  # noqa: E402

# 実録音長 − 見積もり長 の許容範囲 (秒)。録音には前後の無音が入るため
# 実測が数秒長いのが正常。下振れ・大幅な上振れは対応ずれを疑う。
_DELTA_MIN_SEC = -1.0
_DELTA_MAX_SEC = 12.0


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        x = data.astype(np.float64)
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x, sr


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())


def _bandpass(x: np.ndarray, sr: int, center_hz: float, bandwidth_hz: float) -> np.ndarray:
    """アプリのライブ経路と同じ帯域に整形し、ピークを揃える."""
    from scipy.signal import butter, sosfiltfilt

    low = max(center_hz - bandwidth_hz / 2, 50.0)
    high = min(center_hz + bandwidth_hz / 2, sr / 2 - 100.0)
    sos = butter(4, [low / (sr / 2), high / (sr / 2)], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x)
    peak = float(np.abs(y).max())
    return y / peak * 0.48 * 32768 if peak > 0 else y


def _pad(x: np.ndarray, sr: int, pad_sec: float) -> np.ndarray:
    """前後に無音を足す.

    オートキーヤーの録音は 5ms から音が始まり先頭に無音が無いため、モデルが
    立ち上がりを符号と誤読して先頭にゴミトークン (``U`` 等) を吐く。実測では
    この挿入だけで誤り 297 件中 44 件を占めていた。
    """
    if pad_sec <= 0:
        return x
    silence = np.zeros(int(sr * pad_sec))
    return np.concatenate([silence, x, silence])


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def _import_one(
    script: KeyingScript,
    wav: Path,
    out_dir: Path,
    dry_run: bool,
    bpf: tuple[float, float] | None,
    pad_sec: float,
) -> float:
    """録音 1 件を取り込み、実録音長 − 見積もり長 (秒) を返す."""
    text_to_codes(script.text, script.mode)  # ラベルが必ずトークン化できること
    actual = _wav_duration_sec(wav)
    estimated = estimate_duration_sec(script.text, script.mode, script.wpm)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"script_{script.name}.wav"
        x, sample_rate = _read_wav(wav)
        if bpf is not None:
            x = _bandpass(x, sample_rate, *bpf)
        _write_wav(dest, _pad(x, sample_rate, pad_sec), sample_rate)
        bpf_note = "none" if bpf is None else f"{bpf[0]:.0f}Hz/{bpf[1]:.0f}Hz"
        dest.with_suffix(".txt").write_text(
            f"mode: {script.mode}\n"
            f"sample_rate: {sample_rate}\n"
            f"wpm: {script.wpm:.0f}\n"
            f"source: {wav.as_posix()}\n"
            f"bpf: {bpf_note}\n"
            f"pad_sec: {pad_sec:.2f}\n"
            "---\n"
            f"{script.text}\n",
            encoding="utf-8",
        )
    return actual - estimated


def _process(
    scripts: tuple[KeyingScript, ...],
    src_dir: Path | None,
    out_dir: Path,
    dry_run: bool,
    bpf: tuple[float, float] | None,
    pad_sec: float,
) -> tuple[int, list[str]]:
    if src_dir is None:
        return 0, []
    wavs = sorted(src_dir.glob("*.wav"))
    warnings: list[str] = []
    if len(wavs) != len(scripts):
        warnings.append(
            f"{src_dir}: WAV {len(wavs)} 件に対し原稿 {len(scripts)} 件 (連番の対応を確認)"
        )
    count = 0
    for script, wav in zip(scripts, wavs):
        delta = _import_one(script, wav, out_dir, dry_run, bpf, pad_sec)
        flag = "" if _DELTA_MIN_SEC <= delta <= _DELTA_MAX_SEC else "  ← 要確認"
        print(f"  {wav.name:10s} -> script_{script.name}  差 {delta:+5.1f}s{flag}")
        if flag:
            warnings.append(f"{wav.name} -> {script.name}: 時間差 {delta:+.1f}s")
        count += 1
    return count, warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="打鍵録音を学習用データセットへ取り込む")
    p.add_argument("--european-dir", type=Path, default=None)
    p.add_argument("--japanese-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("data/real/train"))
    p.add_argument("--dry-run", action="store_true", help="配置せず確認だけ")
    p.add_argument(
        "--bpf-center", type=float, default=600.0, help="BPF 中心 Hz (アプリ既定 600)"
    )
    p.add_argument(
        "--bpf-bandwidth", type=float, default=400.0, help="BPF 帯域 Hz (アプリ既定 400)"
    )
    p.add_argument(
        "--no-bpf",
        action="store_true",
        help="BPF をかけずそのまま配置する (録音時に既に BPF を通した場合のみ)",
    )
    p.add_argument(
        "--pad-sec",
        type=float,
        default=0.4,
        help="前後に足す無音の秒数 (先頭ゴミトークン対策、0 で無効)",
    )
    args = p.parse_args(argv)

    if args.european_dir is None and args.japanese_dir is None:
        p.error("--european-dir か --japanese-dir のどちらかを指定してください")

    bpf = None if args.no_bpf else (args.bpf_center, args.bpf_bandwidth)
    total = 0
    warnings: list[str] = []
    for label, src, scripts in (
        ("欧文", args.european_dir, EUROPEAN_SCRIPTS),
        ("和文", args.japanese_dir, JAPANESE_SCRIPTS),
    ):
        if src is None:
            continue
        print(f"=== {label} ({src}) ===")
        count, warns = _process(
            scripts, src, args.out_dir, args.dry_run, bpf, args.pad_sec
        )
        total += count
        warnings.extend(warns)

    print()
    for w in warnings:
        print(f"[warn] {w}")
    action = "確認" if args.dry_run else f"{args.out_dir} に配置"
    print(f"[done] {total} 件を{action}")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())

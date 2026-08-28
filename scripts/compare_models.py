"""同じ音声を複数のモデルでデコードして並べる.

**受信音の入り口は 1 つしか掴めない。** ローカルの音声デバイスも LAN の音声
サーバーも、2 つのアプリから同時には開けない。だからライブで 2 モデルを
比べることはできない。**1 回録って、その WAV を両方に食わせる。**

録音はアプリの [● 録音開始] で作れる (保存先は設定の `recording_dir`)。

使い方::

    python scripts/compare_models.py --wav japanese:rec/20260827.wav \\
      --ckpt 現行=models/full/best_infer.pt \\
      --ckpt 新=models/ft_v5_wide/cw_v5wide.onnx

正解テキストがあるとき (WAV と同じ stem の .txt、`---` の後が本文)::

    python scripts/compare_models.py --wav japanese:data/keying_scripts/script_ja_01.wav \\
      --ckpt 現行=models/full/best_infer.pt --ckpt 新=models/ft_v5_wide/best_infer.pt

TER も出る。**正解が無いときは目で比べる。**
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.infer.backend import load_engine                        # noqa: E402
from src.infer.word_breaks import detect_word_breaks_from_audio  # noqa: E402
from src.tokens.converter import TokenConverter                  # noqa: E402
from src.tokens.morse_tokens import TOKEN_TO_ID, text_to_codes   # noqa: E402
from src.train.metrics import levenshtein_distance               # noqa: E402


@dataclass(frozen=True)
class WavSpec:
    mode: str
    path: Path


def parse_wav(entry: str) -> WavSpec:
    """``モード:パス`` を解く. モードが無い / 未知ならエラー."""
    mode, sep, path = entry.partition(":")
    # Windows のドライブレター (C:\...) をモード扱いしない
    if not sep or len(mode.strip()) == 1:
        raise ValueError(f"モードが指定されていません (例 japanese:rec.wav): {entry}")
    mode = mode.strip()
    if mode not in ("european", "japanese"):
        raise ValueError(f"モードは european か japanese: {mode}")
    return WavSpec(mode=mode, path=Path(path.strip()))


def parse_ckpt(entry: str) -> tuple[str, Path]:
    """``ラベル=パス`` を解く. ``=`` が無ければファイル名をラベルにする."""
    label, sep, path = entry.partition("=")
    if not sep:
        return Path(entry).stem, Path(entry)
    return label.strip(), Path(path.strip())


def read_reference(wav_path: Path) -> str | None:
    """WAV と同じ stem の TXT から正解本文を読む (``---`` の後)."""
    txt = wav_path.with_suffix(".txt")
    if not txt.exists():
        return None
    body = txt.read_text(encoding="utf-8")
    return body.split("---", 1)[1].strip() if "---" in body else body.strip()


def token_error_rate(ref_text: str, pred_ids: list[int], mode: str) -> float | None:
    """正解本文と予測トークン列から TER を出す.

    **アプリの録音の TXT はモデルのデコード結果が仮ラベルとして入っている**
    (``_`` などトークン化できない印を含む)。それは正解ではないので、
    トークン化できなければ TER を出さない (None)。
    """
    try:
        ref = [TOKEN_TO_ID[c] for c in text_to_codes(ref_text, mode)]
    except KeyError:
        return None
    if not ref:
        return None
    return levenshtein_distance(pred_ids, ref) / len(ref)


def load_wave(path: Path, target_sr: int = 8000) -> np.ndarray:
    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave[:, 0]
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = np.gcd(sr, target_sr)
        wave = resample_poly(wave, target_sr // g, sr // g).astype(np.float32)
    return wave.astype(np.float32, copy=False)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="同じ音声を複数のモデルでデコードして並べる")
    p.add_argument("--wav", action="append", required=True,
                   help="モード:WAV パス (複数指定可)。例 japanese:rec/20260827.wav")
    p.add_argument("--ckpt", action="append", required=True,
                   help="ラベル=チェックポイント (複数指定可)。.pt でも .onnx でも良い")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bias", type=float, default=-1.0,
                   help="語間バイアス (アプリの欧文既定 −1.0 / 和文 −5.0)。アプリと同じ値にすること")
    p.add_argument("--envelope", action="store_true",
                   help="包絡線から語間を補う (アプリの表示経路では使っていない。比較用)")
    p.add_argument("--out", type=Path, default=None, help="結果を書き出すファイル (UTF-8)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    wavs = [parse_wav(w) for w in args.wav]
    ckpts = [parse_ckpt(c) for c in args.ckpt]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    engines = {}
    for label, path in ckpts:
        if not path.exists():
            raise SystemExit(f"見つかりません: {path}")
        # **アプリと同じ経路で読む** (`load_engine` が拡張子で .onnx / .pt を選ぶ)。
        # ここを PyTorch 専用にすると ONNX を比較できない。
        engines[label] = load_engine(path, device=str(device))
        engines[label].word_break_bias = args.bias
        print(f"[init] {label}: {path} (bias {args.bias})", flush=True)

    lines: list[str] = []
    for spec in wavs:
        if not spec.path.exists():
            print(f"[err] 見つかりません: {spec.path}")
            continue
        wave = load_wave(spec.path)
        ref = read_reference(spec.path)
        header = (f"=== {spec.path.name} ({spec.mode}, {len(wave)/8000:.1f}秒) ===")
        print(f"\n{header}")
        lines.append(header)
        if ref:
            print(f"  {'正解':10}: {ref}")
            lines.append(f"  正解: {ref}")
        for label, _ in ckpts:
            engine = engines[label]
            tokens = engine.decode_chunk(wave)
            ids = [t.token_id for t in tokens]
            # **アプリの表示経路と同じにする**: 語間はモデルの WORD_BREAK トークンだけ。
            # 以前は包絡線検出 (`detect_word_breaks_from_audio`) を混ぜており、アプリで
            # 語間ゼロの録音がここでは出過ぎて見えた (2026-08-28)。--envelope で従来どおり。
            converter = TokenConverter(mode=spec.mode, confidence_threshold=0.5)
            if args.envelope:
                wb = detect_word_breaks_from_audio(
                    wave, tokens, sample_rate=8000, hop_samples=engine.frame_hop_samples)
                text = converter.convert_timed(tokens, word_break_flags=wb).text
            else:
                text = converter.convert(ids, [t.confidence for t in tokens]).text
            ter = token_error_rate(ref, ids, spec.mode) if ref else None
            if ref and ter is None and label == ckpts[0][0]:
                print("  (TXT の本文はトークン化できないので正解としては使わない)")
            suffix = "" if ter is None else f"   [TER {ter*100:.2f}%]"
            print(f"  {label:10}: {text}{suffix}")
            lines.append(f"  {label}: {text}{suffix}")

    if args.out is not None:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

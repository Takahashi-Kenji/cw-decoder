"""ユーザー提供 WAV をモデルで decode して結果を表示.

正解テキストは未提供想定なので、デコード結果をそのまま出力.
スペース処理は確認用に複数バリアントを表示.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.infer.engine import InferenceEngine                                 # noqa: E402
from src.infer.word_breaks import detect_word_breaks_from_audio              # noqa: E402
from src.tokens.converter import TokenConverter                              # noqa: E402


def _load_wav(path: Path, target_sr: int = 8000) -> np.ndarray:
    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave[:, 0]
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = np.gcd(sr, target_sr)
        wave = resample_poly(wave, target_sr // g, sr // g).astype(np.float32)
    return wave.astype(np.float32, copy=False), sr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    # 既定は配布モデル (git 追跡対象) と同じ best_infer.pt にする。
    # best.pt は学習用チェックポイントで .gitignore 対象であり、clone 直後には
    # 存在しない。加えて手元に残っていても ONNX (= ブラウザ版) の出力元とは
    # 別世代のことがあり、ブラウザ版との突き合わせで「別モデル同士を比べて
    # 食い違いに見える」事故が起きる (2026-08-06 に実際に踏んだ)。
    p.add_argument("--ckpt", type=Path, default=Path("models/full/best_infer.pt"))
    p.add_argument("--wav", type=Path, action="append", required=True,
                   help="モード:WAV パス. 複数指定可. 例: european:sample_wav/oubun.wav")
    p.add_argument("--out", type=Path, default=Path("decode_result.txt"),
                   help="結果出力ファイル (UTF-8)")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = InferenceEngine.from_checkpoint(args.ckpt, device=device)
    print(f"[init] checkpoint={args.ckpt}, device={device}\n")

    out_lines: list[str] = [f"checkpoint: {args.ckpt}", ""]

    for entry in args.wav:
        mode_path = str(entry)
        if ":" not in mode_path:
            print(f"[err] mode が指定されていません: {mode_path}")
            continue
        mode_str, _, path_str = mode_path.partition(":")
        mode = mode_str.strip()
        path = Path(path_str.strip())
        if mode not in ("european", "japanese"):
            print(f"[err] unknown mode: {mode}")
            continue
        if not path.exists():
            print(f"[err] not found: {path}")
            continue

        wave, original_sr = _load_wav(path)
        duration = len(wave) / 8000.0
        peak = float(np.abs(wave).max())
        print(f"=== {path.name} ===")
        print(f"  mode={mode}  original_sr={original_sr}Hz  "
              f"resampled_to=8000Hz  duration={duration:.1f}s  peak={peak:.3f}")

        # 推論
        tokens = engine.decode_chunk(wave)
        converter = TokenConverter(mode=mode, confidence_threshold=0.0)
        ids = [t.token_id for t in tokens]

        # 詰めた出力
        text_no_space = converter.convert(ids).text

        # フレームギャップで自動スペース
        text_auto = converter.convert_timed(tokens).text

        # 音声エンベロープでスペース
        wb = detect_word_breaks_from_audio(
            wave, tokens, sample_rate=8000, hop_samples=engine.frame_hop_samples
        )
        text_audio = converter.convert_timed(tokens, word_break_flags=wb).text

        print(f"  num_tokens={len(tokens)}")
        print(f"  [詰めて出力]    : {text_no_space.encode('ascii', errors='replace').decode()}")
        print(f"  [フレーム自動]  : {text_auto.encode('ascii', errors='replace').decode()}")
        print(f"  [音声エンベロープ]: {text_audio.encode('ascii', errors='replace').decode()}\n")
        out_lines.extend([
            f"=== {path.name} ===",
            f"  mode: {mode}",
            f"  original_sr: {original_sr} Hz",
            f"  duration: {duration:.1f} s",
            f"  peak: {peak:.3f}",
            f"  num_tokens: {len(tokens)}",
            f"  [詰めて出力]      : {text_no_space}",
            f"  [フレーム自動]    : {text_auto}",
            f"  [音声エンベロープ]: {text_audio}",
            "",
        ])

    args.out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[out] full result (UTF-8): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

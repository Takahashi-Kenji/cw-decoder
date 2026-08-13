"""Phase 1 で生成した試聴サンプル (data/samples/) を学習済みモデルで decode し
正解との TER / CER を集計するスクリプト.

これは学習・固定eval いずれにも使われていない、独立した検証データ.
"""
from __future__ import annotations

import argparse
import json
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
from src.train.metrics import error_rate, levenshtein_distance                # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    # 既定は配布モデル best_infer.pt (理由は decode_user_wav.py と同じ)
    p.add_argument("--ckpt", type=Path, default=Path("models/full/best_infer.pt"))
    p.add_argument("--dir", type=Path, default=Path("data/samples"))
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = InferenceEngine.from_checkpoint(args.ckpt, device=device)
    print(f"[init] checkpoint={args.ckpt}, device={device}")

    wav_paths = sorted(args.dir.glob("*.wav"))
    print(f"[init] {len(wav_paths)} samples found in {args.dir}\n")

    total_ter_num = 0
    total_ter_den = 0
    total_cer_num = 0
    total_cer_den = 0
    per_scenario: dict[str, list[tuple[float, float]]] = {}

    print(f"{'idx':>3} {'mode':10s} {'scenario':14s} {'snr':>5s} {'wpm':>4s} {'TER':>6s} {'CER':>6s}  {'ref → pred'}")
    print("-" * 120)

    for wav_path in wav_paths:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        meta = json.loads(txt_path.read_text(encoding="utf-8"))
        ref_text = meta["text"]
        mode = meta["config"]["mode"]
        snr = meta["config"]["snr_db"]
        scenario = meta.get("scenario", "?")
        wpm = meta["config"]["keying"]["wpm"]

        wave, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if sr != 8000:
            from scipy.signal import resample_poly
            g = np.gcd(sr, 8000)
            wave = resample_poly(wave, 8000 // g, sr // g).astype(np.float32)

        tokens = engine.decode_chunk(wave)
        converter = TokenConverter(mode=mode, confidence_threshold=0.0)
        ids = [t.token_id for t in tokens]
        # 音声エンベロープから語間検出
        word_breaks = detect_word_breaks_from_audio(
            wave, tokens, sample_rate=8000, hop_samples=engine.frame_hop_samples
        )
        pred_text = converter.convert_timed(tokens, word_break_flags=word_breaks).text

        # 比較用にもスペース有り/無しを保持
        ref_clean = ref_text.replace(" ", "")
        pred_clean = pred_text.replace(" ", "")
        ref_with_space = ref_text
        pred_with_space = pred_text

        # 符号列単位での TER は token_ids 同士で比較
        ref_codes = meta.get("codes", [])
        from src.tokens.morse_tokens import TOKEN_TO_ID
        ref_ids = [TOKEN_TO_ID[c] for c in ref_codes]

        ter_dist = levenshtein_distance(ids, ref_ids)
        cer_dist = levenshtein_distance(pred_clean, ref_clean)

        ter = ter_dist / max(len(ref_ids), 1)
        cer = cer_dist / max(len(ref_clean), 1)

        total_ter_num += ter_dist
        total_ter_den += len(ref_ids)
        total_cer_num += cer_dist
        total_cer_den += len(ref_clean)
        per_scenario.setdefault(scenario, []).append((ter, cer))

        # スペース込みの CER も計算 (語間復元の評価)
        cer_with_space_dist = levenshtein_distance(pred_with_space, ref_with_space)
        cer_with_space = cer_with_space_dist / max(len(ref_with_space), 1)

        idx = wav_path.stem.split("_")[0]
        ref_disp = ref_with_space[:35] + ("…" if len(ref_with_space) > 35 else "")
        pred_disp = pred_with_space[:35] + ("…" if len(pred_with_space) > 35 else "")
        print(
            f"{idx:>3} {mode:10s} {scenario:14s} {snr:+5.0f} {wpm:4.0f} "
            f"{ter*100:5.1f}% {cer_with_space*100:5.1f}%  '{ref_disp}' → '{pred_disp}'"
        )

    print("\n" + "=" * 80)
    print(f"OVERALL  n={len(wav_paths)}  "
          f"TER={total_ter_num/max(total_ter_den,1)*100:5.2f}%  "
          f"CER={total_cer_num/max(total_cer_den,1)*100:5.2f}%")
    print("\nBy scenario:")
    for scenario, vals in sorted(per_scenario.items()):
        ters = [v[0] for v in vals]
        cers = [v[1] for v in vals]
        print(f"  {scenario:14s}  n={len(vals):2d}  "
              f"mean TER={sum(ters)/len(ters)*100:5.2f}%  "
              f"mean CER={sum(cers)/len(cers)*100:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

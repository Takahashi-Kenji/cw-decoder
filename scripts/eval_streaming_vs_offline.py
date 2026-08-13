"""ライブ (スライディングウィンドウ) と オフライン全体デコードの一致率検証.

Usage:
    python scripts/eval_streaming_vs_offline.py --ckpt models/full/best.pt \
        --wav-dir data/real
"""
from __future__ import annotations

import argparse
import difflib
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.infer.engine import InferenceEngine
from src.infer.sliding_window import SlidingWindowDecoder


def token_match_ratio(a: list[int], b: list[int]) -> float:
    """2 つのトークン ID 列の一致率 (difflib ratio)."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _offline_ids(engine: InferenceEngine, wave: np.ndarray) -> list[int]:
    return [t.token_id for t in engine.decode_chunk(wave)]


def _streaming_ids(
    engine: InferenceEngine, wave: np.ndarray, sample_rate: int,
    window_s: float, hop_s: float, commit_lag_s: float, head_guard_s: float,
    decode_left_context_s: float,
) -> tuple[list[int], float, float]:
    d = SlidingWindowDecoder(
        engine, window_s=window_s, hop_s=hop_s, commit_lag_s=commit_lag_s,
        head_guard_s=head_guard_s, decode_left_context_s=decode_left_context_s,
        sample_rate=sample_rate,
    )
    block = int(0.05 * sample_rate)
    hop_samples = int(hop_s * sample_rate)
    since = 0
    max_redecode_ms = 0.0
    t0 = time.perf_counter()
    for i in range(0, wave.size, block):
        d.push(wave[i:i + block])
        since += min(block, wave.size - i)
        if since >= hop_samples:
            since = 0
            r0 = time.perf_counter()
            d.redecode()
            max_redecode_ms = max(max_redecode_ms, (time.perf_counter() - r0) * 1000.0)
    # ストリーム終端: finalize で残り全部を確定 (commit_lag 圏内も確定, レビュー #4)
    view = d.finalize()
    elapsed = time.perf_counter() - t0
    rtf = elapsed / (wave.size / sample_rate)
    return [t.token_id for t in view.committed], rtf, max_redecode_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav-dir", default="data/real")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--hop-s", type=float, default=1.0)
    ap.add_argument("--commit-lag-s", type=float, default=2.5)
    ap.add_argument("--head-guard-s", type=float, default=1.0)
    ap.add_argument("--decode-left-context-s", type=float, default=5.0)
    args = ap.parse_args()

    engine = InferenceEngine.from_checkpoint(args.ckpt, device="cpu")
    sr = 8000
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    if not wavs:
        print(f"[warn] {args.wav_dir} に WAV がありません")
        return
    print(f"{'file':40s} {'match%':>8s} {'rtf':>6s} {'maxdec_ms':>10s}")
    for wav in wavs:
        data, file_sr = sf.read(wav, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            import soxr
            data = soxr.resample(data, file_sr, sr).astype(np.float32)
        off = _offline_ids(engine, data)
        on, rtf, max_dec_ms = _streaming_ids(
            engine, data, sr, args.window_s, args.hop_s,
            args.commit_lag_s, args.head_guard_s, args.decode_left_context_s,
        )
        ratio = token_match_ratio(off, on)
        # RTF と「単発 redecode が hop 時間に収まるか」の両方を確認
        flag = "OK" if ratio >= 0.95 and rtf <= 0.2 else "NG"
        print(f"{wav.name:40s} {ratio*100:7.1f}% {rtf:6.2f} {max_dec_ms:9.0f}  {flag}")


if __name__ == "__main__":
    main()

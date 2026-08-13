"""ブラウザ側の検証用ゴールデン fixture を出力する.

移植で最大のリスクは「動くが微妙に精度が落ちている」ことなので、Python の
推論結果を固定データとして出し、ブラウザ側で完全一致を確認できるようにする。

波形は **8 kHz mono float32 に変換済み**のものを出す。リサンプルを検証対象から
切り離すことで、不一致が出たときに「リサンプルのせいか、デコードのせいか」で
悩まずに済む。

使い方:
    python scripts/export_golden.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from src.infer.engine import FrameToken
from src.tokens.converter import TokenConverter


class ChunkEngine(Protocol):
    """``InferenceEngine`` 互換の最小インタフェース (テストで差し替えるため)."""

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]: ...


SAMPLE_SOURCES: tuple[tuple[str, str], ...] = (
    ("oubun", "sample_wav/oubun.wav"),
    ("wabun", "sample_wav/wabun.wav"),
)
MAX_SECONDS = 30.0
OUTPUT_DIR = Path("web/tests/fixtures")


def load_wave_8k(path: Path, max_seconds: float | None = None) -> np.ndarray:
    """音声を 8 kHz mono float32 で読み込む.

    ステートフルな soxr を使う (ステートレスなリサンプルはブロック境界に歪みを
    生じるが、ここは一括変換なので境界問題は起きない).
    """
    import soundfile as sf
    import soxr

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data[:, 0]
    if sr != 8000:
        mono = soxr.resample(mono, sr, 8000).astype(np.float32)
    if max_seconds is not None:
        mono = mono[: int(max_seconds * 8000)]
    return np.ascontiguousarray(mono, dtype=np.float32)


def build_golden(engine: ChunkEngine, name: str, wave: np.ndarray) -> dict[str, object]:
    """1 ファイル分のゴールデンエントリを作る."""
    tokens: list[FrameToken] = engine.decode_chunk(wave)
    token_ids = [t.token_id for t in tokens]
    confidences = [round(float(t.confidence), 6) for t in tokens]

    european = TokenConverter("european").convert(token_ids, confidences)
    japanese = TokenConverter("japanese").convert(token_ids, confidences)

    return {
        "name": name,
        "waveFile": f"{name}.f32",
        "nSamples": int(wave.size),
        "tokenIds": token_ids,
        "confidences": confidences,
        "textEuropean": european.text,
        "textJapanese": japanese.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/full/best_infer.pt"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    from src.infer.engine import InferenceEngine

    engine = InferenceEngine.from_checkpoint(args.checkpoint, device="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for name, source in SAMPLE_SOURCES:
        path = Path(source)
        if not path.exists():
            print(f"スキップ (見つからない): {path}")
            continue
        wave = load_wave_8k(path, MAX_SECONDS)
        (args.out_dir / f"{name}.f32").write_bytes(wave.tobytes())
        entry = build_golden(engine, name, wave)
        entries.append(entry)
        print(f"{name}: {len(entry['tokenIds'])} トークン, {wave.size} サンプル")

    out_json = args.out_dir / "golden.json"
    out_json.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )
    print(f"書き出し: {out_json}")


if __name__ == "__main__":
    main()

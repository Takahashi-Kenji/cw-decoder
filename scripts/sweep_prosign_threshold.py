"""自動モードのプロサイン閾値を掃引する CLI.

**なぜ専用のスクリプトが要るか**

既存の評価 (`scripts/eval_model.py`) はこの変更を見られない。
`src/eval/harness.py:53,77` が ``TokenConverter(mode=meta.mode,
confidence_threshold=0.0)`` を使っており、**ラベルの固定モード**で変換し、
**閾値を 0.0 で無効化**しているため、自動モードの経路が一度も通らない。
TER はトークン単位なのでそもそも変換段の影響を受けない。

そこでこのスクリプトは **auto モードで変換した結果の文字誤り率 (CER)** を測る。

**測りたいリスク**

プロサイン閾値を下げると、拾えるホレが増える一方で**誤検出でモードが飛ぶ**
危険が増す。1 個の偽ホレで後続の全文字が別表で解釈されるため、欧文側の
悪化が起きやすい。よって和文・欧文の**両方**を見る。

使い方::

    python scripts/sweep_prosign_threshold.py --ckpt models/full/best_infer.pt \\
        --keyed-dir data/keying_scripts

推論は 1 ファイル 1 回だけ実行してトークン列をキャッシュし、掃引は変換のみ
やり直す (変換は軽いので全格子点が数秒で終わる)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.dataset import discover_real_samples  # noqa: E402
from src.infer.engine import InferenceEngine  # noqa: E402
from src.tokens.converter import TokenConverter  # noqa: E402
from src.tokens.morse_tokens import HORE_CODE, ID_TO_TOKEN, RATA_CODE  # noqa: E402
from src.train.metrics import error_rate  # noqa: E402


def _load_wave(path: Path) -> torch.Tensor:
    import wave

    import numpy as np

    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        x = data.astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return torch.from_numpy(x)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="自動モードのプロサイン閾値を掃引")
    p.add_argument("--ckpt", type=Path, default=Path("models/full/best_infer.pt"))
    p.add_argument("--keyed-dir", type=Path, default=Path("data/keying_scripts"))
    p.add_argument(
        "--confidence-threshold", type=float, default=0.5,
        help="通常トークンの閾値 (掃引しない)",
    )
    p.add_argument("--out", type=Path, default=None, help="結果 JSON の出力先")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    samples = discover_real_samples(args.keyed_dir)
    if not samples:
        print(f"[err] サンプルがありません: {args.keyed_dir}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    engine = InferenceEngine.from_checkpoint(str(args.ckpt), device=device)

    # ---- 推論は 1 回だけ。以降は変換のみ掃引する ----
    cached: list[dict] = []
    for s in samples:
        # decode_chunk は decode_wave と同じ greedy 経路だが、確信度も返す。
        toks = engine.decode_chunk(_load_wave(s.wav_path).numpy())
        cached.append({
            "name": s.wav_path.stem,
            "mode": s.mode,
            "ref": s.text,
            "ids": [t.token_id for t in toks],
            "confs": [t.confidence for t in toks],
        })

    grid = [None, 0.50, 0.45, 0.40, 0.35, 0.30, 0.20, 0.10]
    print(f"サンプル {len(cached)} 件 (欧文 "
          f"{sum(1 for c in cached if c['mode']=='european')} / 和文 "
          f"{sum(1 for c in cached if c['mode']=='japanese')})")
    print(f"通常トークンの閾値は {args.confidence_threshold} 固定\n")
    print(f"{'prosign':>8s} {'欧文CER':>8s} {'和文CER':>8s} {'全体CER':>8s} "
          f"{'和文で切替':>10s} {'欧文で誤切替':>12s}")

    rows = []
    for th in grid:
        conv = TokenConverter(
            mode="auto",
            confidence_threshold=args.confidence_threshold,
            prosign_threshold=th,
        )
        agg: dict[str, list[tuple[str, str]]] = {"european": [], "japanese": []}
        ja_switched = eu_switched = 0
        for c in cached:
            res = conv.convert(c["ids"], c["confs"], initial_mode="european")
            agg[c["mode"]].append((c["ref"], res.text))
            # 途中で一度でも和文に入ったかを、出力に [ホレ] が現れたかで判定する
            entered_ja = "[ホレ]" in res.text
            if c["mode"] == "japanese" and entered_ja:
                ja_switched += 1
            if c["mode"] == "european" and entered_ja:
                eu_switched += 1

        def cer_of(pairs: list[tuple[str, str]]) -> float:
            """参照文字数で重み付けした文字誤り率."""
            if not pairs:
                return float("nan")
            errs = sum(error_rate(h, r) * max(len(r), 1) for r, h in pairs)
            total = sum(max(len(r), 1) for r, _ in pairs)
            return errs / total if total else float("nan")

        eu = cer_of(agg["european"])
        ja = cer_of(agg["japanese"])
        allc = cer_of(agg["european"] + agg["japanese"])
        n_ja = len(agg["japanese"])
        n_eu = len(agg["european"])
        label = "既定(従来)" if th is None else f"{th:.2f}"
        print(f"{label:>8s} {eu*100:7.2f}% {ja*100:7.2f}% {allc*100:7.2f}% "
              f"{ja_switched:6d}/{n_ja:<3d} {eu_switched:8d}/{n_eu:<3d}")
        rows.append({
            "prosign_threshold": th, "cer_european": eu, "cer_japanese": ja,
            "cer_all": allc, "japanese_switched": ja_switched,
            "european_false_switch": eu_switched,
            "n_japanese": n_ja, "n_european": n_eu,
        })

    print("\n和文で切替 = 和文録音のうち和文モードに入った件数 (多いほど良い)")
    print("欧文で誤切替 = 欧文録音のうち和文モードに入ってしまった件数 (少ないほど良い)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"ckpt": str(args.ckpt), "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

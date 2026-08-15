"""実録音のキーイング統計を合成で再現し、どの要因が劣化を生むかを切り分ける.

使い方の流れ:

1. ``scripts/analyze_keying.py <wav> --split-ms <谷の位置>`` で実録音の統計を測る
   (実効 WPM・長短比・要素間/文字間/語間・種別ごとの σ)
2. その値をこのスクリプトの引数に渡し、要因を 1 つずつ実測値に寄せた合成を作って
   デコード誤り率を比べる
3. 合成で再現できた劣化が実録音の劣化に届かなければ、**その要因は原因ではない**

**なぜこれが要るか**: このプロジェクトは「もっともらしい仮説を立てて学習を回し、
10 時間かけて悪化させる」ことを 3 回やっている。合成器は仮説を数分で否定できる
道具であり、学習を回す前に必ず通すべきである (docs/model_baseline_and_improvement_plan.md §8)。

実例 (2026-08-14, data/real/20260814_052617_japanese.wav):

    実効 WPM 18.2 / 長短比 3.00 / 間隔 0.66・2.54・5.63 / σ 短点 .216 長音 .413 / SNR 13dB

    基準 (教科書間隔 + σ0.15 + 20dB)   0.51%
    + 実測の間隔                       0.51%   +0.00pt  ← 間隔は無関係だった
    + 実測のジッタ                     1.02%   +0.51pt
    + 実測の間隔とジッタ               3.05%   +2.54pt
    + 実測 SNR 13dB                    2.54%   +2.03pt

    実録音そのものは体感 25〜30%。**合成では 1 桁足りない** = 測れる統計量では
    実信号の難しさを説明できない、という結論になった。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.infer.engine import InferenceEngine                         # noqa: E402
from src.synth.keying import KeyingParams                            # noqa: E402
from src.synth.synthesizer import SynthConfig, synthesize_from_text  # noqa: E402
from src.tokens.morse_tokens import Mode                             # noqa: E402
from src.train.metrics import error_rate                             # noqa: E402

# 和文・欧文それぞれの評価文。実際の QSO に出る語で構成する。
TEXTS: dict[str, list[str]] = {
    "japanese": [
        "コチラノ テンキハ クモリデス",
        "キオンハ 22ド デ ヒンヤリ シテオリマス",
        "サクヤハ ソチラノ ホウハ オオアメデ タイヘン ダツタヨウデスネ",
        "ホンジツハ アリガトウ ゴザイマシタ",
        "ワタシノ ナマエハ タロウデス",
        "コチラノ リグハ 100ワツト デ アンテナハ ダイポ-ルデス",
        "ソチラノ シンゴウハ ツヨク ハイツテオリマス",
        "コレカラモ ヨロシク オネガイ イタシマス",
    ],
    "european": [
        "CQ CQ DE JA0ABC K",
        "UR RST 599 5NN QTH TOKYO",
        "TNX FB QSO ES 73 GL",
        "NAME TARO AGE 55 HW?",
        "RIG IS 100W ANT IS DIPOLE",
        "WX CLOUDY TEMP 22C",
        "PSE QSL VIA BURO TNX",
        "SRI QRM AGN PSE RPT",
    ],
}

# 教科書どおりの間隔 (dot 単位)
BOOK_SPACING = (1.00, 3.00, 7.00)


def _keying(
    wpm: float, ratio: float, spacing: tuple[float, float, float],
    sigma: float, per_kind: dict[str, float] | None, tone: float,
) -> KeyingParams:
    intra, char, word = spacing
    return KeyingParams(
        wpm=wpm,
        dash_dot_ratio=ratio,
        element_jitter_sigma_ratio=sigma,
        intra_element_space_units=intra,
        inter_char_space_units=char,
        inter_word_space_units=word,
        tone_freq_hz=tone,
        rise_fall_ms=5.0,
        pre_silence_sec=0.3,
        post_silence_sec=0.3,
        **(per_kind or {}),
    )


def _run_arm(
    engine: InferenceEngine, mode: Mode, texts: list[str],
    keying_kwargs: dict, snr_db: float, impair: bool, tone: float,
) -> float:
    """1 条件を全テキストで評価して TER (トークン加重平均) を返す."""
    total_err = 0.0
    total_n = 0
    for i, text in enumerate(texts):
        # 腕ごとに同じ seed を使い、変えた要因以外の乱数を揃える
        rng = np.random.default_rng(2000 + i)
        cfg = SynthConfig(
            mode=mode,
            keying=_keying(tone=tone, **keying_kwargs),
            snr_db=snr_db,
            qsb_depth=0.4 if impair else 0.0,
            qsb_period_s=4.0,
            qrn_rate_per_sec=3.0 if impair else 0.0,
            qrn_intensity=1.5,
            filter_center_hz=tone,
            filter_bandwidth_hz=400.0,
        )
        res = synthesize_from_text(text, cfg, rng)
        pred = [t.token_id for t in engine.decode_chunk(res.samples)]
        ref = res.token_ids.tolist()
        total_err += error_rate(pred, ref) * len(ref)
        total_n += len(ref)
    return total_err / total_n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=Path("models/full/best_infer.pt"))
    p.add_argument("--mode", choices=["japanese", "european"], default="japanese")
    p.add_argument("--wpm", type=float, required=True, help="analyze_keying.py の実効 WPM")
    p.add_argument("--ratio", type=float, default=3.00, help="実測の長短比 dash/dot")
    p.add_argument("--tone", type=float, default=600.0, help="実測のトーン周波数 (Hz)")
    p.add_argument("--spacing", type=float, nargs=3, metavar=("要素間", "文字間", "語間"),
                   required=True, help="実測の間隔 (dot 単位)")
    p.add_argument("--sigma-dot", type=float, required=True, help="実測の短点 σ (dot 比)")
    p.add_argument("--sigma-dash", type=float, required=True, help="実測の長音 σ (dot 比)")
    p.add_argument("--sigma-gap", type=float, nargs=3, default=(0.27, 0.42, 0.50),
                   metavar=("要素間", "文字間", "語間"), help="実測の間隔 σ (dot 比)")
    p.add_argument("--snr", type=float, required=True, help="実測の実効 SNR (dB)")
    p.add_argument("--clean-snr", type=float, default=20.0, help="基準条件の SNR (dB)")
    p.add_argument("--base-sigma", type=float, default=0.15, help="基準条件のジッタ σ")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    engine = InferenceEngine.from_checkpoint(args.ckpt, device=device)
    texts = TEXTS[args.mode]
    meas_spacing = tuple(args.spacing)
    gap_i, gap_c, gap_w = args.sigma_gap
    meas_jitter = dict(
        dot_jitter_sigma_ratio=args.sigma_dot,
        dash_jitter_sigma_ratio=args.sigma_dash,
        intra_gap_jitter_sigma_ratio=gap_i,
        char_gap_jitter_sigma_ratio=gap_c,
        word_gap_jitter_sigma_ratio=gap_w,
    )

    print(f"[init] ckpt={args.ckpt} device={device} mode={args.mode}")
    print(f"       wpm={args.wpm} ratio={args.ratio} tone={args.tone}Hz\n")

    # (表示名, 間隔, ジッタ, SNR, QSB/QRN)
    arms: list[tuple[str, tuple, dict | None, float, bool]] = [
        ("1 基準 (教科書間隔・弱ジッタ・高SNR)", BOOK_SPACING, None, args.clean_snr, False),
        ("2 + 実測の間隔",                      meas_spacing, None, args.clean_snr, False),
        ("3 + 実測のジッタ",                    BOOK_SPACING, meas_jitter, args.clean_snr, False),
        ("4 + 実測の間隔とジッタ",              meas_spacing, meas_jitter, args.clean_snr, False),
        ("5 + 実測 SNR",                        meas_spacing, meas_jitter, args.snr, False),
        ("6 + QSB/QRN",                         meas_spacing, meas_jitter, args.snr, True),
        ("7 SNR のみ (他は教科書)",             BOOK_SPACING, None, args.snr, False),
    ]

    print(f"{'条件':<38} {'TER':>8} {'差':>10}")
    print("-" * 58)
    base: float | None = None
    for name, spacing, jitter, snr, impair in arms:
        ter = _run_arm(
            engine, args.mode, texts,
            dict(wpm=args.wpm, ratio=args.ratio, spacing=spacing,
                 sigma=args.base_sigma, per_kind=jitter),
            snr, impair, args.tone,
        )
        if base is None:
            base, diff = ter, "  (基準)"
        else:
            diff = f"{(ter - base) * 100:+9.2f}pt"
        print(f"{name:<38} {ter * 100:>7.2f}% {diff:>10}")

    print("\n実録音そのものの TER と比べること。合成が届かなければ、"
          "測れる統計量では原因を説明できていない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

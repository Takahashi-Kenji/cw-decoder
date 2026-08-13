"""試聴用 CW サンプル生成 CLI.

代表的な条件 (SNR / WPM / QSB の有無) を組合せて、運用者が試聴して
「実際の CW らしさ」を確認するための WAV ファイル群を生成する.

使い方::

    python scripts/generate_samples.py --out data/samples --european 12 --japanese 12 --seed 42

各 WAV と同名 ``.txt`` に正解テキスト・トークン列・パラメータを書き出す.
出力ディレクトリには ``README.md`` も生成される.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

# プロジェクトルートを sys.path に追加 (pip install -e していなくても動くように)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.synth.keying import KeyingParams                  # noqa: E402
from src.synth.synthesizer import SynthConfig, synthesize_random  # noqa: E402
from src.tokens.morse_tokens import Mode                   # noqa: E402

SAMPLE_RATE = 8000

# 代表的なシナリオ (SNR dB, QSB depth, QSB period s, ラベル)
SCENARIOS: list[tuple[float, float, float, str]] = [
    (20.0, 0.0, 0.0, "clean"),
    (10.0, 0.0, 0.0, "mid"),
    (10.0, 0.6, 3.0, "mid_qsb"),
    (0.0, 0.0, 0.0, "noisy"),
    (0.0, 0.5, 4.0, "noisy_qsb"),
    (-5.0, 0.0, 0.0, "veryNoisy"),
]

# 代表的な WPM
WPM_CHOICES: list[float] = [15.0, 20.0, 25.0, 30.0]


def make_config(
    mode: Mode,
    snr: float,
    qsb_depth: float,
    qsb_period: float,
    wpm: float,
    rng: np.random.Generator,
) -> SynthConfig:
    return SynthConfig(
        mode=mode,
        keying=KeyingParams(
            wpm=wpm,
            dash_dot_ratio=float(rng.uniform(2.8, 3.2)),
            element_jitter_sigma_ratio=float(rng.uniform(0.05, 0.10)),
            tone_freq_hz=float(rng.uniform(550.0, 700.0)),
            tone_drift_hz_per_sec=0.0,
            rise_fall_ms=5.0,
            pre_silence_sec=0.15,
            post_silence_sec=0.15,
        ),
        snr_db=snr,
        qsb_depth=qsb_depth,
        qsb_period_s=qsb_period,
        qrm_stations=0,
        qrn_rate_per_sec=0.0,
        filter_center_hz=600.0,
        filter_bandwidth_hz=400.0,
    )


def write_sample(out_dir: Path, idx: int, mode: Mode, scenario_label: str, wpm: float, result) -> Path:
    snr_int = int(round(result.config.snr_db))
    fname = f"{idx:03d}_{mode}_{snr_int:+03d}dB_{int(wpm)}wpm_{scenario_label}"
    wav_path = out_dir / f"{fname}.wav"
    txt_path = out_dir / f"{fname}.txt"

    # 16-bit PCM. クリップ防止のためピーク正規化 (但し SNR を維持するため過度に増幅しない).
    samples = result.samples
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        scale = min(0.9 / peak, 4.0)  # 最大 4 倍まで増幅
        samples = (samples * scale).astype(np.float32)
    sf.write(wav_path, samples, SAMPLE_RATE, subtype="PCM_16")

    meta = {
        "text": result.text,
        "codes": result.codes,
        "token_ids": result.token_ids.tolist(),
        "effective_wpm": result.effective_wpm,
        "scenario": scenario_label,
        "config": {
            "mode": result.config.mode,
            "snr_db": result.config.snr_db,
            "qsb_depth": result.config.qsb_depth,
            "qsb_period_s": result.config.qsb_period_s,
            "filter_center_hz": result.config.filter_center_hz,
            "filter_bandwidth_hz": result.config.filter_bandwidth_hz,
            "keying": asdict(result.config.keying),
        },
    }
    txt_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return wav_path


def write_readme(out_dir: Path, entries: list[tuple[str, str]]) -> None:
    lines = [
        "# 試聴サンプル (Phase 1)",
        "",
        "`python scripts/generate_samples.py` で生成. 運用者が試聴して",
        "「実際の CW らしさ」を確認するための合成音声 (8 kHz, 16-bit PCM).",
        "",
        "各 WAV と同名の `.txt` に正解テキスト・トークン列・合成パラメータを",
        "JSON 形式で記録. ファイル名規則: ",
        "`<idx>_<mode>_<snr>dB_<wpm>wpm_<scenario>.wav`.",
        "",
        "| ファイル | テキスト |",
        "|---|---|",
    ]
    for wav_name, text in entries:
        # テーブル内パイプ衝突回避
        display_text = text.replace("|", "\\|")
        lines.append(f"| `{wav_name}` | {display_text} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="試聴用 CW サンプル生成")
    parser.add_argument("--out", type=Path, default=Path("data/samples"), help="出力ディレクトリ")
    parser.add_argument("--european", type=int, default=12, help="欧文サンプル数")
    parser.add_argument("--japanese", type=int, default=12, help="和文サンプル数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    entries: list[tuple[str, str]] = []
    idx = 0

    for mode, count in [("european", args.european), ("japanese", args.japanese)]:
        for n in range(count):
            scenario_idx = n % len(SCENARIOS)
            wpm_idx = (n // len(SCENARIOS)) % len(WPM_CHOICES)
            snr, depth, period, label = SCENARIOS[scenario_idx]
            wpm = WPM_CHOICES[wpm_idx]
            config = make_config(mode, snr, depth, period, wpm, rng)  # type: ignore[arg-type]
            result = synthesize_random(rng, config, sample_rate=SAMPLE_RATE)
            wav_path = write_sample(out_dir, idx, mode, label, wpm, result)  # type: ignore[arg-type]
            entries.append((wav_path.name, result.text))
            print(f"[{idx:03d}] {wav_path.name}  -- {result.text}")
            idx += 1

    write_readme(out_dir, entries)
    print(f"\n生成完了: {idx} サンプル → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

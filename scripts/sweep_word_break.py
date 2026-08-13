"""WORD_BREAK 抑制パラメータ (β, τ) の掃引 CLI.

推論を 1 サンプル 1 回だけ行って log_probs を CPU にキャッシュし、以降の
グリッド探索は純後処理で回す。設計は
``docs/superpowers/specs/2026-07-31-word-break-threshold-design.md``.

使い方::

    # Step 0: 診断のみ (正解WB と偽陽性WB の確信度・ラン長分布)
    python scripts/sweep_word_break.py --ckpt models/full/best.pt \\
        --keyed-dir data/keying_scripts --diagnose-only

    # Step 1: keyed_val で 2 次元グリッド掃引
    python scripts/sweep_word_break.py --ckpt models/full/best.pt \\
        --keyed-dir data/keying_scripts \\
        --out models/eval/word_break_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.dataset import RealSignalDataset, discover_real_samples   # noqa: E402
from src.infer.engine import ctc_greedy_decode_with_frames                  # noqa: E402
from src.infer.word_break_policy import (                                   # noqa: E402
    WordBreakPolicy,
    apply_logit_bias,
    filter_word_breaks,
)
from src.synth.dataset import RealNoiseEvalSample, make_fixed_real_noise_eval_set  # noqa: E402
from src.synth.noise import RealNoisePool                                   # noqa: E402
from src.tokens.converter import TokenConverter                             # noqa: E402
from src.tokens.morse_tokens import (                                       # noqa: E402
    BLANK_TOKEN_ID,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
)
from src.train.decode import ctc_greedy_decode                              # noqa: E402
from src.train.loop import compute_input_lengths                            # noqa: E402
from src.train.metrics import DetailedEvalReport, EvalRecord, align_sequences, bin_snr  # noqa: E402
from src.train.model import CWModel, ModelConfig                            # noqa: E402
from src.train.preprocessing import MelExtractor                            # noqa: E402


@dataclass(frozen=True)
class CachedSample:
    """1 サンプル分の推論結果キャッシュ (掃引中は再計算しない)."""

    log_probs: torch.Tensor       # (T, V) CPU float32、有効長で切り詰め済み
    ref_tokens: list[int]
    ref_text: str
    mode: str
    name: str
    eff_snr_db: float | None = None


@torch.no_grad()
def cache_keyed_samples(
    model: CWModel,
    mel_extractor: MelExtractor,
    dataset: RealSignalDataset,
    device: torch.device,
) -> list[CachedSample]:
    """keyed_val の全サンプルを 1 回だけ推論して log_probs をキャッシュ."""
    model.train(False)
    out: list[CachedSample] = []
    for i in range(len(dataset)):
        wave, target = dataset[i]
        meta = dataset.sample_at(i)
        t = wave.unsqueeze(0).to(device)
        log_probs = torch.log_softmax(model(mel_extractor(t)).float(), dim=-1)
        length = int(compute_input_lengths(
            torch.tensor([wave.numel()], device=device),
            mel_extractor.config.hop_length,
            log_probs.size(1),
        )[0])
        out.append(CachedSample(
            log_probs=log_probs[0, :length].cpu(),
            ref_tokens=target.tolist(),
            ref_text=meta.text,
            mode=meta.mode,
            name=meta.stem,
        ))
    return out


@torch.no_grad()
def cache_synth_samples(
    model: CWModel,
    mel_extractor: MelExtractor,
    samples: list[RealNoiseEvalSample],
    device: torch.device,
) -> list[CachedSample]:
    """synth_val の固定評価セットを 1 回だけ推論してキャッシュ."""
    model.train(False)
    out: list[CachedSample] = []
    for s in samples:
        wave = torch.from_numpy(s.samples)
        t = wave.unsqueeze(0).to(device)
        log_probs = torch.log_softmax(model(mel_extractor(t)).float(), dim=-1)
        length = int(compute_input_lengths(
            torch.tensor([wave.numel()], device=device),
            mel_extractor.config.hop_length,
            log_probs.size(1),
        )[0])
        out.append(CachedSample(
            log_probs=log_probs[0, :length].cpu(),
            ref_tokens=s.token_ids.tolist(),
            ref_text=s.text,
            mode=s.mode,
            name="",
            eff_snr_db=s.eff_snr_db,
        ))
    return out


def decode_cached(log_probs: torch.Tensor, policy: WordBreakPolicy) -> list[int]:
    """キャッシュ済み ``(T, V)`` にポリシーを適用してトークン ID 列を返す."""
    lp = apply_logit_bias(log_probs.unsqueeze(0), policy.logit_bias)
    result = ctc_greedy_decode(lp, blank_id=BLANK_TOKEN_ID)[0]
    return filter_word_breaks(result, policy.conf_threshold).token_ids


def _percentiles(values: list[float]) -> dict[str, float]:
    """空リストでも落ちない分位数の要約."""
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def diagnose(samples: list[CachedSample]) -> dict[str, Any]:
    """baseline の WORD_BREAK を正解/偽陽性に分け、確信度とラン長を集計.

    確信度分布が分離していれば τ に見込みがある. 完全に重なっていれば τ は
    原理的に効かない. ラン長分布は案C (フレームラン長による抑制) の有望度を示す.
    """
    correct_conf: list[float] = []
    false_conf: list[float] = []
    correct_run: list[float] = []
    false_run: list[float] = []

    for s in samples:
        frame_tokens = ctc_greedy_decode_with_frames(
            s.log_probs.unsqueeze(0), blank_id=BLANK_TOKEN_ID
        )[0]
        pred_ids = [ft.token_id for ft in frame_tokens]
        kind_by_pred: dict[int, str] = {
            op.pred_index: op.kind
            for op in align_sequences(pred_ids, s.ref_tokens)
            if op.pred_index is not None
        }
        for idx, ft in enumerate(frame_tokens):
            if ft.token_id != WORD_BREAK_TOKEN_ID:
                continue
            run = float(ft.frame_end - ft.frame_start + 1)
            if kind_by_pred.get(idx) == "equal":
                correct_conf.append(ft.confidence)
                correct_run.append(run)
            else:
                false_conf.append(ft.confidence)
                false_run.append(run)

    return {
        "correct_word_breaks": {
            "confidence": _percentiles(correct_conf),
            "frame_run_length": _percentiles(correct_run),
        },
        "false_word_breaks": {
            "confidence": _percentiles(false_conf),
            "frame_run_length": _percentiles(false_run),
        },
    }


def print_diagnosis(diag: dict[str, Any]) -> None:
    """診断結果を人が読める形で出力."""
    for key, label in (("correct_word_breaks", "正解WB"), ("false_word_breaks", "偽陽性WB")):
        d = diag[key]
        c, r = d["confidence"], d["frame_run_length"]
        if c["n"] == 0:
            print(f"[diag] {label}: 0 個", flush=True)
            continue
        print(
            f"[diag] {label} n={c['n']:3d}  "
            f"conf median={c['median']:.3f} (p25={c['p25']:.3f} p75={c['p75']:.3f})  "
            f"run median={r['median']:.1f} (p25={r['p25']:.1f} p75={r['p75']:.1f})",
            flush=True,
        )


# 設計書 §5 Step 1 のグリッド
BETA_GRID: list[float] = [0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0]
TAU_GRID: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# レビュー指摘 (Task 5): 最適点 (beta=-3.0, tau=0.95) の勝ち幅は僅か 1 トークン誤りで、
# 一方 beta=-6.0 は tau=0.0〜0.6 の 7 点にわたり TER が平坦でパラメータ変動に頑健。
# Task 7 でどちらを既定値にするか判断する材料として、synth_val でも両方を比較する。
ALT_BETA: float = -6.0
ALT_TAU: float = 0.3


def evaluate_cached(
    samples: list[CachedSample], policy: WordBreakPolicy
) -> DetailedEvalReport:
    """キャッシュに対しポリシーを適用して評価レポートを作る."""
    report = DetailedEvalReport()
    for s in samples:
        pred_ids = decode_cached(s.log_probs, policy)
        pred_text = TokenConverter(
            mode=s.mode, confidence_threshold=0.0
        ).convert(pred_ids).text
        report.add(
            EvalRecord(
                ref_tokens=list(s.ref_tokens), pred_tokens=pred_ids,
                ref_text=s.ref_text, pred_text=pred_text,
                eff_snr_db=s.eff_snr_db,
            ),
            name=s.name,
            mode=s.mode,
            eff_snr_bin=None if s.eff_snr_db is None else bin_snr(s.eff_snr_db),
        )
    return report


def _grid_point(samples: list[CachedSample], beta: float, tau: float) -> dict[str, Any]:
    """1 格子点を評価して要約を返す."""
    report = evaluate_cached(
        samples, WordBreakPolicy(logit_bias=beta, conf_threshold=tau)
    )
    totals = report.analysis.totals
    wb = report.analysis.counts[WORD_BREAK_TOKEN_ID]
    return {
        "beta": beta,
        "tau": tau,
        "ter": totals["ter"],
        "substitutions": totals["substitutions"],
        "deletions": totals["deletions"],
        "insertions": totals["insertions"],
        "word_break_insertions": wb.inserted,
        "word_break_deletions": wb.deleted,
        "by_mode": {k: v.ter for k, v in report.by_mode().items()},
    }


def sweep(
    samples: list[CachedSample], betas: list[float], taus: list[float]
) -> list[dict[str, Any]]:
    """2 次元グリッドを掃引して各点の要約を返す."""
    points: list[dict[str, Any]] = []
    for beta in betas:
        for tau in taus:
            points.append(_grid_point(samples, beta, tau))
        print(f"[sweep] beta={beta:g} 完了 ({len(taus)} 点)", flush=True)
    return points


def print_sweep_table(points: list[dict[str, Any]], baseline_ter: float) -> None:
    """β 行 × τ 列の TER 表を出力 (baseline からの差分 pt)."""
    taus = sorted({p["tau"] for p in points})
    betas = sorted({p["beta"] for p in points}, reverse=True)
    by_key = {(p["beta"], p["tau"]): p for p in points}
    header = "  beta\\tau " + " ".join(f"{t:>6.2f}" for t in taus)
    print(header, flush=True)
    for beta in betas:
        cells = []
        for tau in taus:
            delta = (by_key[(beta, tau)]["ter"] - baseline_ter) * 100
            cells.append(f"{delta:>+6.1f}")
        print(f"  {beta:>8.1f} " + " ".join(cells), flush=True)


def best_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    """TER 最小の格子点 (同点なら β・τ が 0 に近い方を選ぶ)."""
    return min(points, key=lambda p: (p["ter"], abs(p["beta"]), p["tau"]))


def neighbors(points: list[dict[str, Any]], point: dict[str, Any]) -> list[dict[str, Any]]:
    """最適点から β 方向・τ 方向に 1 点ずつ隣接する格子点 (最大 4 点)."""
    betas = sorted({p["beta"] for p in points}, reverse=True)
    taus = sorted({p["tau"] for p in points})
    by_key = {(p["beta"], p["tau"]): p for p in points}
    bi, ti = betas.index(point["beta"]), taus.index(point["tau"])
    out: list[dict[str, Any]] = []
    for db, dt in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        b, t = bi + db, ti + dt
        if 0 <= b < len(betas) and 0 <= t < len(taus):
            out.append(by_key[(betas[b], taus[t])])
    return out


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WORD_BREAK 抑制パラメータの掃引")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--keyed-dir", type=Path, required=True, help="keyed_val の WAV+TXT ディレクトリ")
    p.add_argument("--out", type=Path, default=Path("models/eval/word_break_sweep.json"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--diagnose-only", action="store_true", help="Step 0 診断だけ実行する")
    p.add_argument("--noise-dir", type=Path, default=None,
                   help="synth_val 用ノイズ WAV ディレクトリ (指定時のみ synth_val も評価)")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--tone-center", type=float, default=494.0)
    p.add_argument("--bpf-bandwidth", type=float, default=300.0)
    p.add_argument("--wpm", type=float, nargs="+", default=[17.0, 25.0])
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0, -5.0])
    p.add_argument("--samples-per-cell", type=int, default=25)
    return p


def load_model(ckpt: Path, device: torch.device) -> CWModel:
    """チェックポイントを読み込む.

    語彙サイズが現在の ``VOCAB_SIZE`` と違う場合、``WORD_BREAK_TOKEN_ID`` が
    別の符号を指してしまい **黙って無関係なトークンにバイアスをかける** ことになる.
    そうならないよう、モデルへ流し込む前に検査して明示的に落とす.
    """
    # save_checkpoint が入れるのはテンソル・plain dict・数値のみなので
    # weights_only=True で読める (既存の load_checkpoint より安全側に倒す)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    ckpt_vocab = int(state.get("model_config", {}).get("vocab_size", -1))
    if ckpt_vocab != VOCAB_SIZE:
        raise ValueError(
            f"チェックポイントの語彙サイズ {ckpt_vocab} が現在の VOCAB_SIZE "
            f"{VOCAB_SIZE} と異なる. WORD_BREAK (id={WORD_BREAK_TOKEN_ID}) の意味が "
            "ずれるため掃引できない"
        )
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    model.load_state_dict(state["model_state"])
    model.train(False)
    return model


def _passes_synth_check(degradation_pt: float) -> bool:
    """synth_val の悪化幅 (pt) が事前登録した合格ライン以内かを判定する純関数.

    事前登録した合格ライン: 悪化 +1.00pt 以内 (ちょうど +1.00pt は合格).
    """
    return degradation_pt <= 1.0


def _synth_check(
    synth_cached: list[CachedSample],
    base_ter: float,
    beta: float,
    tau: float,
    label: str,
) -> dict[str, Any]:
    """指定した (beta, tau) を synth_val に適用し baseline との悪化幅を測る."""
    point_ter = evaluate_cached(
        synth_cached, WordBreakPolicy(logit_bias=beta, conf_threshold=tau)
    ).analysis.totals["ter"]
    degradation_pt = (point_ter - base_ter) * 100
    result = {
        "beta": beta,
        "tau": tau,
        "baseline_ter": base_ter,
        "point_ter": point_ter,
        "degradation_pt": degradation_pt,
        "passes": _passes_synth_check(degradation_pt),
    }
    print(
        f"[synth] baseline {base_ter*100:.2f}% → {label} (beta={beta:g} tau={tau:g}) "
        f"{point_ter*100:.2f}% ({degradation_pt:+.2f}pt) 条件 "
        f"{'満たす' if result['passes'] else '満たさない'}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, device)
    mel = MelExtractor().to(device)

    samples = discover_real_samples(args.keyed_dir)
    if not samples:
        print(f"[err] keyed-dir にサンプルがありません: {args.keyed_dir}", flush=True)
        return 2
    cached = cache_keyed_samples(model, mel, RealSignalDataset(samples), device)
    print(f"[cache] keyed_val {len(cached)} 件をキャッシュ", flush=True)

    diag = diagnose(cached)
    print_diagnosis(diag)
    if args.diagnose_only:
        return 0

    points = sweep(cached, BETA_GRID, TAU_GRID)
    baseline = next(p for p in points if p["beta"] == 0.0 and p["tau"] == 0.0)
    print(f"[base] beta=0 tau=0 の TER = {baseline['ter']*100:.2f}%", flush=True)
    print_sweep_table(points, baseline["ter"])

    best = best_point(points)
    nb = neighbors(points, best)
    plateau = all(p["ter"] < baseline["ter"] for p in nb)
    improvement_pt = (baseline["ter"] - best["ter"]) * 100
    print(
        f"[best] beta={best['beta']:g} tau={best['tau']:g}  "
        f"TER {best['ter']*100:.2f}%  改善 {improvement_pt:+.2f}pt  "
        f"WB挿入 {baseline['word_break_insertions']}→{best['word_break_insertions']}  "
        f"プラトー条件 {'満たす' if plateau else '満たさない'}",
        flush=True,
    )

    synth_check: dict[str, Any] | None = None
    synth_check_alt: dict[str, Any] | None = None
    if args.noise_dir is not None:
        pool = RealNoisePool.from_dir(args.noise_dir)
        synth_cached: list[CachedSample] = []
        for mode in ("european", "japanese"):
            synth_cached.extend(cache_synth_samples(model, mel, make_fixed_real_noise_eval_set(
                noise_pool=pool, snr_grid=args.snr, wpm_grid=args.wpm,
                samples_per_cell=args.samples_per_cell, seed=args.seed, mode=mode,
                tone_center_hz=args.tone_center, filter_bandwidth_hz=args.bpf_bandwidth,
            ), device))
        print(f"[cache] synth_val {len(synth_cached)} 件をキャッシュ", flush=True)
        base_ter = evaluate_cached(synth_cached, WordBreakPolicy()).analysis.totals["ter"]

        # synth_check_alt (下記) と同じく beta/tau を含めて自己記述的にする。
        # 既存キー (baseline_ter/best_point_ter/degradation_pt/passes) は
        # docs/word_break_sweep.json 等の既存成果物が参照するため名前を保ったまま残す
        # (_synth_check の "point_ter" を "best_point_ter" に付け替えるだけ)。
        best_check = _synth_check(synth_cached, base_ter, best["beta"], best["tau"], "最適点")
        synth_check = dict(best_check)
        synth_check["best_point_ter"] = synth_check.pop("point_ter")

        # レビュー指摘: beta=-6.0 はグリッド上で τ 変動に頑健な代替運用点。
        # Task 7 でどちらを既定値にするか判断できるよう、synth_val でも並べて測る。
        synth_check_alt = _synth_check(synth_cached, base_ter, ALT_BETA, ALT_TAU, "代替点")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({
            "ckpt": str(args.ckpt),
            "keyed_dir": str(args.keyed_dir),
            "diagnosis": diag,
            "baseline_point": baseline,
            "best_point": best,
            "plateau": plateau,
            "improvement_pt": improvement_pt,
            "points": points,
            "synth_check": synth_check,
            "synth_check_alt": synth_check_alt,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

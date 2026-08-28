"""実信号ファインチューニングスクリプト (Phase 4).

使い方::

    # 実信号を data/real/ に集めた上で:
    python scripts/finetune.py --data-dir data/real --resume models/full/best.pt \\
        --ckpt-dir models/ft --steps 2000 --lr 1e-4

    # 実信号 + 合成データ混合 (合成 30% でカタストロフィックフォゲッティング抑制):
    python scripts/finetune.py --data-dir data/real --resume models/full/best.pt \\
        --ckpt-dir models/ft --steps 2000 --lr 1e-4 --mix-synth --real-ratio 0.7

    # 実録音バンドノイズを合成キーイングに混合 (data/noise/ に BPF 後のノイズ WAV):
    python scripts/finetune.py --data-dir data/real --resume models/full/best.pt \\
        --ckpt-dir models/ft --steps 2000 --lr 1e-4 --mix-synth --real-ratio 0.5 \\
        --noise-dir data/noise
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.harness import evaluate_real_dataset, evaluate_synth_noise      # noqa: E402
from src.finetune.lead_in import report_lead_in, resolve_lead_in_range  # noqa: E402
from src.finetune.dataset import RealSignalDataset, discover_real_samples  # noqa: E402
from src.finetune.pipeline import (                                         # noqa: E402
    MixedRealSynthDataset,
    split_train_validation,
)
from src.tokens.morse_tokens import BLANK_TOKEN_ID, ID_TO_TOKEN, VOCAB_SIZE  # noqa: E402
from src.train.checkpoint import load_checkpoint, save_checkpoint            # noqa: E402
from src.train.collate import cw_collate                                     # noqa: E402
from src.train.logger import CSVLogger                                       # noqa: E402
from src.train.loop import train_step                                        # noqa: E402
from src.train.metrics import DetailedEvalReport                             # noqa: E402
from src.train.model import CWModel, ModelConfig                             # noqa: E402
from src.train.preprocessing import MelExtractor                             # noqa: E402
from src.train.run_meta import build_run_meta, write_run_meta                 # noqa: E402


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="実信号ファインチューニング")
    p.add_argument("--data-dir", type=Path, default=Path("data/real"))
    p.add_argument("--resume", type=Path, required=True, help="出発点チェックポイント")
    p.add_argument("--ckpt-dir", type=Path, default=Path("models/ft"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-ratio", type=float, default=0.2)
    p.add_argument(
        "--eval-dir", type=Path, default=None,
        help="固定 validation ディレクトリ (指定時 --data-dir 全件を train、ここを val)",
    )
    p.add_argument("--mode-filter", type=str, default=None, choices=["european", "japanese"])
    p.add_argument("--mix-synth", action="store_true", help="合成データを混合する")
    p.add_argument("--real-ratio", type=float, default=0.7, help="混合時の実データ比率")
    p.add_argument("--best-saturation", type=float, default=0.01,
                   help="val TER がこれ未満なら飽和とみなし、best.pt を毎回更新する "
                        "(= last を採る)。飽和した val の揺れで古い重みに固定されない。"
                        "0 で従来どおり (strict な最小値)")
    p.add_argument("--char-space", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="合成の文字間 (dot 単位) をこの範囲で振る。**指定しなければ従来どおり 3.0 固定。** "
                        "実測: 学習した全サンプルが 3.0 固定なのに held-out は 4.38 だった")
    p.add_argument("--intra-space", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="合成の要素間 (dot 単位) をこの範囲で振る。既定は 1.0 固定")
    p.add_argument("--word-space", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="合成の語間 (dot 単位) をこの範囲で振る。既定は 7.0 固定。"
                        "**語間 5 dot 以下の局では baseline も p2b も語間がゼロになる** (2026-08-28)")
    p.add_argument("--pre-silence", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="合成の先頭の無音 (秒) をこの範囲で振る。既定は 0〜0.3 秒")
    p.add_argument("--real-lead-in-min", type=float, default=0.0,
                   help="実データの先頭に足す助走 (ノイズ床) の最短 (秒)")
    p.add_argument("--real-lead-in-max", type=float, default=2.0,
                   help="実データの先頭に足す助走の最長 (秒)。**0 で無効。** "
                        "**学習側だけに効く** (eval には効かせない)")
    p.add_argument(
        "--noise-dir", type=Path, default=None,
        help="実録音バンドノイズ WAV のディレクトリ (合成キーイングに混合。--mix-synth を自動有効化)",
    )
    p.add_argument("--noise-prob", type=float, default=0.8, help="合成サンプルへの実ノイズ適用率")
    p.add_argument("--noise-snr-min", type=float, default=-5.0, help="実ノイズ混合時の最小 SNR (dB)")
    p.add_argument("--noise-snr-max", type=float, default=15.0, help="実ノイズ混合時の最大 SNR (dB)")
    p.add_argument(
        "--tone-center", type=float, default=600.0,
        help="実ノイズ混合時の合成トーン中心周波数 (Hz)。録音時の BPF 中心に合わせる",
    )
    p.add_argument("--tone-span", type=float, default=50.0, help="トーン周波数の ± 揺らぎ幅 (Hz)")
    p.add_argument(
        "--eval-details-out", type=Path, default=None,
        help="詳細評価 JSON の出力先 (既定: <ckpt-dir>/ft_eval_details.json)",
    )
    p.add_argument(
        "--confusion-out", type=Path, default=None,
        help="confusion matrix JSON の出力先 (既定: <ckpt-dir>/ft_confusion.json)",
    )
    p.add_argument(
        "--eval-top-n", type=int, default=10,
        help="評価ログに表示する誤りの多い token の件数",
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    # 既定は従来分布。手打ち分布はオプトイン (2026-08-07)。
    # フル学習での検証で実手打ちの TER が 2 倍以上悪化したため
    # (docs/hand_keying_full_train_result.md)。
    p.add_argument("--hand-keying", action="store_true",
                   help="手打ちの合成分布を使う (既定は従来分布。手打ち分布は実測で悪化した)")
    p.add_argument("--no-extreme-tail", action="store_true",
                   help="長音ジッタ σ の上限を 1.30 から 0.70 に下げる (設計書 §3.2 の A/B 用)")
    p.add_argument("--electronic-keyer-prob", type=float, default=0.25,
                   help="手打ち分布のうち、この確率でエレキー相当 (従来分布) を引く")
    # --- 採否判定と学習曲線 (2026-08-24) ---
    p.add_argument("--synth-val-noise-dir", type=Path, default=None,
                   help="synth_val を FT の前後で測るための実ノイズディレクトリ。"
                        "指定すると「synth_val 悪化 3pt 超で不採用」を自動判定する")
    p.add_argument("--synth-val-samples-per-cell", type=int, default=10,
                   help="synth_val のセルあたり件数 (小さいほど速い)")
    p.add_argument("--synth-val-max-degrade", type=float, default=3.0,
                   help="synth_val TER の許容悪化幅 (pt)")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="学習に使う実サンプルの件数を間引く (件数 vs TER の学習曲線用)")
    p.add_argument("--subsample-seed", type=int, default=None,
                   help="間引きの乱数シード (既定は --seed と同じ)")
    return p


@dataclass(frozen=True)
class SynthValVerdict:
    """synth_val の採否判定."""

    passed: bool
    delta_pt: float
    message: str


def synth_val_verdict(
    before: float, after: float, max_degrade_pt: float
) -> SynthValVerdict:
    """FT 前後の synth_val TER から採否を言う.

    採用基準の 1 つ「**synth_val 悪化 3pt 超で不採用**」を自動化する。
    起点 (``--resume`` の ckpt) を同じ run の中で測るので、外部の基準ファイルが要らない。
    """
    delta_pt = (after - before) * 100.0
    passed = delta_pt <= max_degrade_pt
    verdict = "合格" if passed else "**不採用**"
    return SynthValVerdict(
        passed=passed,
        delta_pt=delta_pt,
        message=(f"synth_val TER {before * 100:.2f}% → {after * 100:.2f}% "
                 f"({delta_pt:+.2f}pt、許容 {max_degrade_pt:.1f}pt) {verdict}"),
    )


def subsample_samples(samples: list, n: int | None, seed: int) -> list:
    """学習サンプルを ``n`` 件に間引く (件数 vs TER の学習曲線用).

    **元の順序を保つ。** 順序が変わると DataLoader の並びまで変わり、
    間引き以外の差が混ざる。
    """
    if n is None or n >= len(samples):
        return list(samples)
    if n <= 0:
        raise SystemExit(f"--max-train-samples は 1 以上にしてください: {n}")
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(samples), size=n, replace=False).tolist())
    return [samples[i] for i in idx]


def resolve_train_eval_samples(
    data_dir: Path,
    eval_dir: Path | None,
    mode_filter: "str | None",
    eval_ratio: float,
    seed: int,
    on_skip: "Callable[[Path, str], None] | None" = None,
) -> tuple[list, list]:
    """train / eval のサンプルリストを決める.

    ``eval_dir`` 指定時は ``data_dir`` 全件を train、``eval_dir`` 全件を固定 val とする
    (改善前後を同じ val で比較できる)。未指定なら従来どおり乱数分割。
    """
    samples = discover_real_samples(data_dir, mode_filter=mode_filter, on_skip=on_skip)  # type: ignore[arg-type]
    if eval_dir is not None:
        eval_samples = discover_real_samples(eval_dir, mode_filter=mode_filter, on_skip=on_skip)  # type: ignore[arg-type]
        return samples, eval_samples
    return split_train_validation(samples, validation_ratio=eval_ratio, seed=seed)


@torch.no_grad()
def evaluate_real(
    model: CWModel,
    mel_extractor: MelExtractor,
    eval_dataset: RealSignalDataset,
    device: torch.device,
) -> DetailedEvalReport:
    """実信号評価セットに対し TER/CER と token 別エラーを集計 (harness へ委譲)."""
    return evaluate_real_dataset(model, mel_extractor, eval_dataset, device)


def _with_suffix(path: Path, suffix: str) -> Path:
    """``a/b.json`` + ``_step0`` → ``a/b_step0.json``."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def save_eval_details(
    report: DetailedEvalReport,
    details_path: Path,
    confusion_path: Path,
    step: int,
) -> None:
    """詳細評価と confusion matrix を JSON 保存 (人間が読める整形付き)."""
    for path in (details_path, confusion_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    details = report.to_dict()
    details["step"] = step
    details_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    confusion = report.analysis.confusion_to_dict()
    confusion["step"] = step
    confusion_path.write_text(
        json.dumps(confusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def should_update_best(ter: float, best_ter: float | None, saturation: float) -> bool:
    """best.pt を更新するか.

    **val が飽和したら「最新 = best」にする。** full_v5 では L4 val が 0.06% まで
    飽和し、評価ごとに ±4pt 揺れた結果、best.pt は「たまたま低く出た評価」
    (step 98,000) で固定され、held-out で last.pt より 12pt 悪かった。
    飽和した val に判定能力は無い。``saturation=0`` で従来と完全に同じ挙動。
    """
    if best_ter is None:
        return True
    if saturation > 0 and ter < saturation:
        return True
    return ter < best_ter


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[init] GPU={torch.cuda.get_device_name(0)}", flush=True)

    torch.manual_seed(args.seed)

    # ---- 実信号サンプル収集 ----
    print(f"[scan] {args.data_dir}", flush=True)
    train_samples, eval_samples = resolve_train_eval_samples(
        data_dir=args.data_dir, eval_dir=args.eval_dir,
        mode_filter=args.mode_filter, eval_ratio=args.eval_ratio, seed=args.seed,
        on_skip=lambda path, why: print(f"[skip] {path.name} ({why})", flush=True),
    )
    if not train_samples:
        print(f"[err] 学習サンプルが見つかりません: {args.data_dir}", flush=True)
        return 2
    if not eval_samples:
        print(f"[err] 評価サンプルが見つかりません: {args.eval_dir or args.data_dir}", flush=True)
        return 2
    n_found = len(train_samples)
    train_samples = subsample_samples(
        train_samples, args.max_train_samples,
        seed=args.seed if args.subsample_seed is None else args.subsample_seed,
    )
    if len(train_samples) != n_found:
        print(f"[scan] train を {n_found} 件から {len(train_samples)} 件へ間引き", flush=True)
    print(f"[scan] train={len(train_samples)} eval={len(eval_samples)}", flush=True)

    # **助走は学習側だけに効かせる。** eval_real に効かせると物差しが動く。
    lead_in_range = resolve_lead_in_range(args.real_lead_in_min, args.real_lead_in_max)
    train_real = RealSignalDataset(
        train_samples, lead_in_range=lead_in_range, seed=args.seed)
    report_lead_in(train_real, lead_in_range)
    eval_real = RealSignalDataset(eval_samples)

    # ---- 実ノイズプール ----
    noise_pool = None
    tone_freq_range: tuple[float, float] | None = None
    if args.noise_dir is not None:
        from src.synth.noise import RealNoisePool

        noise_pool = RealNoisePool.from_dir(args.noise_dir)
        dur_s = noise_pool.total_samples / noise_pool.sample_rate
        print(
            f"[noise] {args.noise_dir}: {len(noise_pool)} files, {dur_s:.1f}s total",
            flush=True,
        )
        if not args.mix_synth:
            print("[noise] --noise-dir 指定のため --mix-synth を自動有効化", flush=True)
            args.mix_synth = True
        # 合成トーンを録音時の BPF 中心近傍に制約 (ユーザーセットアップに整合)
        tone_freq_range = (args.tone_center - args.tone_span, args.tone_center + args.tone_span)

    # ---- DataLoader ----
    if args.mix_synth:
        mode_mix = (
            {"european": 1.0, "japanese": 0.0}
            if args.mode_filter == "european"
            else {"european": 0.0, "japanese": 1.0}
            if args.mode_filter == "japanese"
            else {"european": 0.5, "japanese": 0.5}
        )
        dataset = MixedRealSynthDataset(
            real_dataset=train_real,
            mode_mix=mode_mix,
            real_ratio=args.real_ratio,
            seed=args.seed,
            noise_pool=noise_pool,
            noise_prob=args.noise_prob,
            noise_snr_range=(args.noise_snr_min, args.noise_snr_max),
            tone_freq_range=tone_freq_range,
            char_space_range=None if args.char_space is None else tuple(args.char_space),
            intra_space_range=None if args.intra_space is None else tuple(args.intra_space),
            pre_silence_range=None if args.pre_silence is None else tuple(args.pre_silence),
            word_space_range=None if args.word_space is None else tuple(args.word_space),
            hand_keying=args.hand_keying,
            extreme_tail=not args.no_extreme_tail,
            electronic_keyer_prob=args.electronic_keyer_prob,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=cw_collate,
            pin_memory=device.type == "cuda",
        )
    else:
        loader = DataLoader(
            train_real,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=cw_collate,
            pin_memory=device.type == "cuda",
            shuffle=True,
        )

    # ---- モデル ----
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    mel_extractor = MelExtractor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    criterion = torch.nn.CTCLoss(blank=BLANK_TOKEN_ID, zero_infinity=True)

    print(f"[resume] {args.resume}", flush=True)
    load_checkpoint(args.resume, model, map_location=device)

    train_log = CSVLogger(args.ckpt_dir / "ft_train.csv")
    eval_log = CSVLogger(args.ckpt_dir / "ft_eval.csv")

    details_out: Path = args.eval_details_out or (args.ckpt_dir / "ft_eval_details.json")
    confusion_out: Path = args.confusion_out or (args.ckpt_dir / "ft_confusion.json")

    # 実験台帳へ転記するための情報 (ckpt の extra と meta.json の両方に入れる)
    run_meta = build_run_meta(
        argv if argv is not None else sys.argv,
        cwd=_PROJECT_ROOT,
        seed=args.seed,
        n_train=len(train_samples),
        n_eval=len(eval_samples),
        resume=str(args.resume),
        real_ratio=args.real_ratio if args.mix_synth else None,
        mix_synth=args.mix_synth,
        hand_keying=args.hand_keying,
    )
    write_run_meta(args.ckpt_dir, run_meta)

    # synth_val は FT の**前**にも測る。起点を同じ run の中で持っておくと、
    # 「synth_val 悪化 3pt 超で不採用」を外部の基準ファイル無しに判定できる。
    synth_val_before: float | None = None
    if args.synth_val_noise_dir is not None:
        synth_val_before = _measure_synth_val(model, mel_extractor, device, args)
        print(f"[synth 0] synth_val TER={synth_val_before * 100:.2f}%", flush=True)

    # 初期評価 (改善前後比較の基準となるため、専用ファイルにも残す)
    init_report = evaluate_real(model, mel_extractor, eval_real, device)
    for line in init_report.summary_lines(top_n=args.eval_top_n):
        print(f"[eval     0] {line}", flush=True)
    eval_log.log(
        step=0,
        ter=init_report.overall.ter,
        cer=init_report.overall.cer,
        n_samples=init_report.overall.n_samples,
    )
    save_eval_details(init_report, details_out, confusion_out, step=0)
    save_eval_details(
        init_report,
        _with_suffix(details_out, "_step0"),
        _with_suffix(confusion_out, "_step0"),
        step=0,
    )
    print(f"[eval     0] details -> {details_out}, {confusion_out}", flush=True)
    best_ter: float = init_report.overall.ter

    # ---- 学習ループ ----
    print(f"[ft] start: steps={args.steps}, lr={args.lr}, batch={args.batch_size}", flush=True)
    t0 = time.time()
    step = 0
    loader_iter = iter(loader)
    ema_loss: float | None = None
    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        result = train_step(
            model, mel_extractor, batch, optimizer, scaler, criterion, device
        )
        step += 1
        if result.loss is not None:
            ema_loss = result.loss if ema_loss is None else 0.95 * ema_loss + 0.05 * result.loss
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"[step {step:5d}] loss(ema)={ema_loss:.3f} grad={result.grad_norm:.2f} "
                f"sps={step / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
            train_log.log(
                step=step, loss=result.loss, loss_ema=ema_loss,
                grad_norm=result.grad_norm, lr=optimizer.param_groups[0]["lr"],
            )
        if step % args.eval_interval == 0 or step == args.steps:
            report = evaluate_real(model, mel_extractor, eval_real, device)
            ter = report.overall.ter
            for line in report.summary_lines(top_n=args.eval_top_n):
                print(f"[eval {step:5d}] {line}", flush=True)
            eval_log.log(
                step=step, ter=ter, cer=report.overall.cer,
                n_samples=report.overall.n_samples,
            )
            # 最新の詳細評価で上書き (step0 版と比較して改善内訳を見る)
            save_eval_details(report, details_out, confusion_out, step=step)
            # 評価指標更新後に last.pt 保存 (best_metric バグ回避)
            new_best = should_update_best(ter, best_ter, args.best_saturation)
            if new_best:
                best_ter = ter
            save_checkpoint(
                args.ckpt_dir / "last.pt", model, optimizer, scaler,
                step=step, epoch=0, best_metric=best_ter, extra=run_meta,
            )
            if new_best:
                save_checkpoint(
                    args.ckpt_dir / "best.pt", model, optimizer, scaler,
                    step=step, epoch=0, best_metric=best_ter, extra=run_meta,
                )
                print(f"[ckpt {step:5d}] new best TER={ter * 100:.2f}%", flush=True)

    print(f"[done] {step} steps, final best TER={best_ter * 100:.2f}%", flush=True)

    # ---- 採否判定 ----
    if synth_val_before is not None:
        after = _measure_synth_val(model, mel_extractor, device, args)
        verdict = synth_val_verdict(
            before=synth_val_before, after=after,
            max_degrade_pt=args.synth_val_max_degrade,
        )
        print(f"[verdict] {verdict.message}", flush=True)
        print("[verdict] held-out のモード別 TER は scripts/eval_model.py で測ること "
              "(**評価セットは学習にも val にも使わない**)", flush=True)
    return 0


def _measure_synth_val(
    model: CWModel,
    mel_extractor: MelExtractor,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """synth_val (合成 + 実ノイズ) の TER を測る. eval_model.py と同じ経路を使う."""
    from src.synth.dataset import make_fixed_real_noise_eval_set
    from src.synth.noise import RealNoisePool

    pool = RealNoisePool.from_dir(args.synth_val_noise_dir)
    total_errors = 0
    total_tokens = 0
    for mode in ("european", "japanese"):
        samples = make_fixed_real_noise_eval_set(
            noise_pool=pool, snr_grid=[10.0, 5.0, 0.0, -5.0], wpm_grid=[17.0, 25.0],
            samples_per_cell=args.synth_val_samples_per_cell, seed=args.seed, mode=mode,
            tone_center_hz=args.tone_center, filter_bandwidth_hz=300.0,
        )
        report = evaluate_synth_noise(model, mel_extractor, samples, device)
        total_errors += report.overall.total_token_errors
        total_tokens += report.overall.total_ref_tokens
    return total_errors / total_tokens if total_tokens else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

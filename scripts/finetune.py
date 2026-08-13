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
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.harness import evaluate_real_dataset                           # noqa: E402
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
    return p


def resolve_train_eval_samples(
    data_dir: Path,
    eval_dir: Path | None,
    mode_filter: "str | None",
    eval_ratio: float,
    seed: int,
) -> tuple[list, list]:
    """train / eval のサンプルリストを決める.

    ``eval_dir`` 指定時は ``data_dir`` 全件を train、``eval_dir`` 全件を固定 val とする
    (改善前後を同じ val で比較できる)。未指定なら従来どおり乱数分割。
    """
    samples = discover_real_samples(data_dir, mode_filter=mode_filter)  # type: ignore[arg-type]
    if eval_dir is not None:
        eval_samples = discover_real_samples(eval_dir, mode_filter=mode_filter)  # type: ignore[arg-type]
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
    )
    if not train_samples:
        print(f"[err] 学習サンプルが見つかりません: {args.data_dir}", flush=True)
        return 2
    if not eval_samples:
        print(f"[err] 評価サンプルが見つかりません: {args.eval_dir or args.data_dir}", flush=True)
        return 2
    print(f"[scan] train={len(train_samples)} eval={len(eval_samples)}", flush=True)

    train_real = RealSignalDataset(train_samples)
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
            new_best = best_ter is None or ter < best_ter
            if new_best:
                best_ter = ter
            save_checkpoint(
                args.ckpt_dir / "last.pt", model, optimizer, scaler,
                step=step, epoch=0, best_metric=best_ter,
            )
            if new_best:
                save_checkpoint(
                    args.ckpt_dir / "best.pt", model, optimizer, scaler,
                    step=step, epoch=0, best_metric=best_ter,
                )
                print(f"[ckpt {step:5d}] new best TER={ter * 100:.2f}%", flush=True)

    print(f"[done] {step} steps, final best TER={best_ter * 100:.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

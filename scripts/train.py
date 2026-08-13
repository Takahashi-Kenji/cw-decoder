"""学習スクリプト (CLI).

使い方::

    # スモーク学習 (動作確認、数分)
    python scripts/train.py --steps 200 --batch-size 8 --num-workers 0 \\
        --eval-interval 100 --ckpt-dir models/smoke

    # フル学習 (RTX 5060 Ti, 一晩 8 時間以内)
    python scripts/train.py --steps 100000 --batch-size 32 --num-workers 4 \\
        --eval-interval 2000 --ckpt-dir models/full

ログ: ``<ckpt-dir>/train.csv`` (各ステップ) と ``<ckpt-dir>/eval.csv`` (各評価).
チェックポイント: ``<ckpt-dir>/last.pt`` (毎評価時) と ``<ckpt-dir>/best.pt`` (TER 最良).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# プロジェクトルートを sys.path に追加
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.synth.dataset import MorseSynthDataset, make_fixed_eval_set  # noqa: E402
from src.synth.noise import RealNoisePool                            # noqa: E402
from src.tokens.morse_tokens import BLANK_TOKEN_ID, VOCAB_SIZE       # noqa: E402
from src.train.checkpoint import load_checkpoint, save_checkpoint    # noqa: E402
from src.train.collate import cw_collate                              # noqa: E402
from src.train.logger import CSVLogger                                # noqa: E402
from src.train.loop import evaluate, train_step                      # noqa: E402
from src.train.model import CWModel, ModelConfig                     # noqa: E402
from src.train.preprocessing import MelExtractor                     # noqa: E402


def resolve_effective_snr_range(
    eff_min: float | None, eff_max: float | None
) -> tuple[float, float] | None:
    """--eff-snr-min/max を検証して実効SNR範囲を返す.

    両方 None なら None (現行 nominal 動作)。片方のみはエラー、min>=max もエラー。
    """
    if eff_min is None and eff_max is None:
        return None
    if eff_min is None or eff_max is None:
        raise ValueError("--eff-snr-min と --eff-snr-max は両方指定してください")
    if eff_min >= eff_max:
        raise ValueError(f"--eff-snr-min ({eff_min}) は --eff-snr-max ({eff_max}) より小さくしてください")
    return (float(eff_min), float(eff_max))


def validate_noise_params(
    noise_dir: Path | None, noise_prob: float, snr_min: float, snr_max: float
) -> None:
    """--noise-* パラメータを検証する (noise_dir 指定時のみ)."""
    if noise_dir is None:
        return
    if not 0.0 <= noise_prob <= 1.0:
        raise ValueError(f"--noise-prob は [0, 1] の範囲で指定してください (got {noise_prob})")
    if snr_min >= snr_max:
        raise ValueError(
            f"--noise-snr-min ({snr_min}) は --noise-snr-max ({snr_max}) より小さくしてください"
        )


def resolve_noise_pool(noise_dir: Path | None) -> RealNoisePool | None:
    """--noise-dir から RealNoisePool を読み込む. None なら None (実ノイズ混合なし)."""
    if noise_dir is None:
        return None
    return RealNoisePool.from_dir(noise_dir)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """全 param_group に学習率を再適用する.

    ``optimizer.load_state_dict`` は保存時の param_groups (lr を含む) で現在の
    設定を上書きするため、resume 時に ``--lr`` を下げても無視されてしまう。
    FT では低い学習率が要点なので、復元直後にこれを呼んで打ち消す。
    """
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CW デコーダ学習")
    p.add_argument("--steps", type=int, default=200, help="総学習ステップ")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mode-european-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", type=Path, default=Path("models/smoke"))
    p.add_argument("--resume", type=Path, default=None, help="チェックポイントから再開")
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-samples-per-cell", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--no-amp", action="store_true", help="混合精度を無効化")
    p.add_argument("--eff-snr-min", type=float, default=None,
                   help="実効SNR学習範囲の下限 (dB)。--eff-snr-max と両方指定で実効モード")
    p.add_argument("--eff-snr-max", type=float, default=None,
                   help="実効SNR学習範囲の上限 (dB)")
    p.add_argument("--noise-dir", type=Path, default=None,
                   help="実録音ノイズ WAV のディレクトリ (指定で実ノイズ混合 FT を有効化)")
    p.add_argument("--noise-prob", type=float, default=0.8,
                   help="合成サンプルへの実ノイズ適用率 (--noise-dir 指定時)")
    p.add_argument("--noise-snr-min", type=float, default=-5.0,
                   help="実ノイズ混合時の最小 SNR (dB)")
    p.add_argument("--noise-snr-max", type=float, default=15.0,
                   help="実ノイズ混合時の最大 SNR (dB)")
    # --- 手打ちキーイングの合成分布 (2026-08-06 追加、2026-08-07 に既定オフ) ---
    #
    # **既定は従来分布。手打ち分布はオプトイン。**
    # 2026-08-07 に 102,000 step のフル学習で検証したところ、手打ち分布は実手打ち
    # 録音の TER を 25.95% → 52.35% と 2 倍以上悪化させた (合成側も悪化)。
    # 詳細は docs/hand_keying_full_train_result.md。
    # 分布を狭めた再挑戦の余地はあるので機能自体は残すが、既定にはしない。
    p.add_argument("--hand-keying", action="store_true",
                   help="手打ちの合成分布を使う (既定は従来分布。手打ち分布は実測で悪化した)")
    p.add_argument("--no-extreme-tail", action="store_true",
                   help="長音ジッタ σ の上限を 1.30 から 0.70 に下げる")
    p.add_argument("--electronic-keyer-prob", type=float, default=0.25,
                   help="手打ち分布のうち、この確率でエレキー相当 (従来分布) を引く")
    p.add_argument("--device", type=str, default="cuda")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[init] GPU={torch.cuda.get_device_name(0)}", flush=True)

    torch.manual_seed(args.seed)

    # ---- データ ----
    ratio_eu = args.mode_european_ratio
    mode_mix = {"european": ratio_eu, "japanese": 1.0 - ratio_eu}
    eff_range = resolve_effective_snr_range(args.eff_snr_min, args.eff_snr_max)
    if eff_range is not None:
        print(f"[init] effective SNR range = {eff_range} dB", flush=True)
    validate_noise_params(args.noise_dir, args.noise_prob, args.noise_snr_min, args.noise_snr_max)
    noise_pool = resolve_noise_pool(args.noise_dir)
    if noise_pool is not None:
        dur_s = noise_pool.total_samples / noise_pool.sample_rate
        print(
            f"[init] real noise: {args.noise_dir} ({len(noise_pool)} files, {dur_s:.1f}s), "
            f"prob={args.noise_prob}, snr=({args.noise_snr_min},{args.noise_snr_max})",
            flush=True,
        )
    dataset = MorseSynthDataset(
        mode_mix=mode_mix, seed=args.seed, effective_snr_range=eff_range,
        noise_pool=noise_pool, noise_prob=args.noise_prob,
        noise_snr_range=(args.noise_snr_min, args.noise_snr_max),
        hand_keying=args.hand_keying,
        extreme_tail=not args.no_extreme_tail,
        electronic_keyer_prob=args.electronic_keyer_prob,
    )
    print(
        f"[init] 合成分布: {'手打ち' if args.hand_keying else '従来 (エレキー相当のみ)'}"
        f"  極端テール={'あり' if not args.no_extreme_tail else 'なし'}"
        f"  エレキー混合率={args.electronic_keyer_prob}",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=cw_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # 固定評価セット (再現可能). WPM は 10/20/24/30 を含み実運用域をカバー
    eval_samples_eu = make_fixed_eval_set(
        snr_grid=[10.0, 0.0, -5.0],
        wpm_grid=[10.0, 20.0, 24.0, 30.0],
        samples_per_cell=args.eval_samples_per_cell,
        seed=args.seed + 1,
        mode="european",
    )

    # ---- モデル ----
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    n_params = model.num_parameters()
    print(f"[init] model params = {n_params:,} (~{n_params * 4 / 1e6:.1f} MB fp32)", flush=True)

    mel_extractor = MelExtractor().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    criterion = torch.nn.CTCLoss(blank=BLANK_TOKEN_ID, zero_infinity=True)

    start_step = 0
    best_ter: float | None = None
    if args.resume is not None and args.resume.exists():
        # 語彙が拡張されている可能性に対応 (例: WORD_BREAK 追加)
        raw_state = torch.load(args.resume, map_location=device, weights_only=False)
        old_vocab = int(raw_state["model_config"].get("vocab_size", VOCAB_SIZE))
        if old_vocab != VOCAB_SIZE:
            print(
                f"[resume] vocab size changed: {old_vocab} -> {VOCAB_SIZE}, "
                f"expanding classifier", flush=True,
            )
            old_w = raw_state["model_state"]["classifier.weight"]
            old_b = raw_state["model_state"]["classifier.bias"]
            hidden = old_w.shape[1]
            new_w = torch.zeros(VOCAB_SIZE, hidden, dtype=old_w.dtype, device=old_w.device)
            new_b = torch.zeros(VOCAB_SIZE, dtype=old_b.dtype, device=old_b.device)
            copy_n = min(VOCAB_SIZE, old_vocab)
            new_w[:copy_n] = old_w[:copy_n]
            new_b[:copy_n] = old_b[:copy_n]
            raw_state["model_state"]["classifier.weight"] = new_w
            raw_state["model_state"]["classifier.bias"] = new_b
        model.load_state_dict(raw_state["model_state"])
        if "optimizer_state" in raw_state and old_vocab == VOCAB_SIZE:
            # 語彙不変時のみ optimizer state も復元 (寸法不一致を避ける)
            optimizer.load_state_dict(raw_state["optimizer_state"])
            # 復元は保存時の lr で上書きするため、CLI 指定を再適用する
            apply_lr(optimizer, args.lr)
        if "scaler_state" in raw_state and use_amp:
            try:
                scaler.load_state_dict(raw_state["scaler_state"])
            except (RuntimeError, KeyError):
                pass
        start_step = int(raw_state.get("step", 0))
        best_ter = raw_state.get("best_metric")
        print(
            f"[resume] step={start_step} best_ter={best_ter} "
            f"lr={optimizer.param_groups[0]['lr']:g}",
            flush=True,
        )

    train_log = CSVLogger(args.ckpt_dir / "train.csv")
    eval_log = CSVLogger(args.ckpt_dir / "eval.csv")

    # ---- 学習ループ ----
    print(f"[train] start: total_steps={args.steps}, amp={use_amp}", flush=True)
    t0 = time.time()
    step = start_step
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
            sps = step / max(elapsed, 1e-6)
            loss_str = f"{ema_loss:.3f}" if ema_loss is not None else "nan"
            print(
                f"[step {step:6d}] loss(ema)={loss_str}  "
                f"grad={result.grad_norm:.2f} "
                f"sps={sps:.2f}",
                flush=True,
            )
            train_log.log(
                step=step,
                loss=result.loss if result.loss is not None else "",
                loss_ema=ema_loss if ema_loss is not None else "",
                grad_norm=result.grad_norm,
                lr=optimizer.param_groups[0]["lr"],
                elapsed_sec=elapsed,
            )

        if step % args.eval_interval == 0 or step == args.steps:
            report = evaluate(
                model, mel_extractor, eval_samples_eu, device, mode="european"
            )
            ter = report.overall.ter
            cer = report.overall.cer
            print(f"[eval  {step:6d}] {report.summary_lines()[0]}", flush=True)
            for line in report.summary_lines()[1:]:
                print(f"          {line}", flush=True)
            eval_log.log(step=step, ter=ter, cer=cer, n_samples=report.overall.n_samples)

            # 評価結果を先に反映してから last.pt 保存
            # (こうしないと resume 時に best_metric が古い値で読まれる)
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
                print(f"[ckpt  {step:6d}] new best TER={ter * 100:.2f}%", flush=True)

    total = time.time() - t0
    print(f"[done] {step} steps in {total:.1f}s ({step / max(total, 1e-6):.2f} sps)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

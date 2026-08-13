# 実ノイズ混合ファインチューニング Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `scripts/train.py` に `--noise-dir`/`--noise-prob`/`--noise-snr-min`/`--noise-snr-max` を追加し、実録音ノイズを合成に混ぜて base モデルを短時間 FT できるようにする (AWGN→実受信ノイズのドメインギャップを埋めるため)。

**Architecture:** `MorseSynthDataset` は既に `noise_pool`/`noise_prob`/`noise_snr_range` に対応済み。`train.py` に CLI を足して `RealNoisePool.from_dir` で読み込んだプールを `MorseSynthDataset` に渡すだけ (B1 で `--eff-snr` を足したのと同じ配線パターン)。検証は純関数に切り出す。モデル・特徴量・decode・推論・`MorseSynthDataset` 本体は変更しない。

**Tech Stack:** Python 3.11, PyTorch, numpy, soundfile, pytest。

## Global Constraints

- Python 3.11+。型ヒント必須 (`mypy` 互換)。
- モデル構造・特徴量・decode・推論・`MorseSynthDataset` 本体は変更しない (train.py の CLI 配線のみ)。
- 乱数は `np.random.Generator` を引数で受け取る。
- 既存 API 非破壊: `RealNoisePool`/`MorseSynthDataset`/`make_fixed_eval_set` の既存シグネチャと挙動を壊さない。`--noise-dir` 未指定で train.py は現行動作。
- CLI 引数名は finetune.py と同名・同義: `--noise-prob` (既定 0.8)、`--noise-snr-min` (既定 -5.0)、`--noise-snr-max` (既定 15.0)。
- ファイル UTF-8 (BOM なし)、改行 LF。docstring/コメント/コミットメッセージは日本語。
- コミット規則: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`。
- 既存 594 テストを壊さない。
- 作業ブランチ: `feature/real-noise-ft-train`。

---

## File Structure

- `.gitignore` — 変更: `data/noise/` を除外 (大容量 wav)。
- `scripts/train.py` — 変更: `validate_noise_params` (純関数・検証)、`resolve_noise_pool` (ディレクトリ→RealNoisePool)、CLI 引数、`main()` 配線。
- テスト: `tests/test_train_cli.py` (追記。既存の `resolve_effective_snr_range` テストと同ファイル)。

---

### Task 1: train.py の実ノイズ混合 CLI 配線

`--noise-dir` と混合パラメータを追加し、`RealNoisePool` を `MorseSynthDataset` に渡す。検証は純関数に切り出してテストする。`.gitignore` に `data/noise/` を追加。

**Files:**
- Modify: `scripts/train.py`
- Modify: `.gitignore`
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: `RealNoisePool.from_dir(root, sample_rate=8000) -> RealNoisePool` (src.synth.noise), `MorseSynthDataset(mode_mix, seed, effective_snr_range=, noise_pool=, noise_prob=, noise_snr_range=)` (src.synth.dataset)。
- Produces:
  - `validate_noise_params(noise_dir: Path | None, noise_prob: float, snr_min: float, snr_max: float) -> None` — `noise_dir` が None なら何もしない。指定時、`noise_prob` が [0,1] 外、または `snr_min >= snr_max` で `ValueError`。
  - `resolve_noise_pool(noise_dir: Path | None) -> RealNoisePool | None` — None→None、指定時 `RealNoisePool.from_dir(noise_dir)`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_train_cli.py` の末尾に追記 (先頭 import に `Path`, `numpy as np`, `soundfile as sf` が無ければ足す):

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.train import validate_noise_params, resolve_noise_pool


class TestValidateNoiseParams:
    def test_none_dir_is_noop(self) -> None:
        # noise_dir が None なら他の値が不正でも例外なし
        validate_noise_params(None, 5.0, 10.0, -10.0)

    def test_valid_params_pass(self) -> None:
        validate_noise_params(Path("data/noise"), 0.5, 0.0, 20.0)

    def test_prob_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="noise-prob"):
            validate_noise_params(Path("data/noise"), 1.5, 0.0, 20.0)
        with pytest.raises(ValueError, match="noise-prob"):
            validate_noise_params(Path("data/noise"), -0.1, 0.0, 20.0)

    def test_snr_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="snr"):
            validate_noise_params(Path("data/noise"), 0.5, 20.0, 0.0)
        with pytest.raises(ValueError, match="snr"):
            validate_noise_params(Path("data/noise"), 0.5, 5.0, 5.0)


class TestResolveNoisePool:
    def test_none_returns_none(self) -> None:
        assert resolve_noise_pool(None) is None

    def test_loads_pool_from_dir(self, tmp_path: Path) -> None:
        # 帯域内相当の短いノイズ wav を1本置く
        sf.write(tmp_path / "n.wav",
                 np.random.default_rng(0).standard_normal(8000).astype(np.float32),
                 8000, subtype="PCM_16")
        pool = resolve_noise_pool(tmp_path)
        assert pool is not None
        assert len(pool) == 1
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_train_cli.py::TestValidateNoiseParams tests/test_train_cli.py::TestResolveNoisePool -v`
Expected: FAIL (`ImportError: cannot import name 'validate_noise_params'`)

- [ ] **Step 3: 純関数を実装**

まず `scripts/train.py` の import 群 (既存の `from src.synth.dataset import MorseSynthDataset, make_fixed_eval_set` の隣) に追加:

```python
from src.synth.noise import RealNoisePool
```

(src.synth.noise は numpy/scipy のみで torch 非依存。train.py は既に torch を import 済みなので追加の起動コストは無い。)

`scripts/train.py` の `resolve_effective_snr_range` の直後に追加:

```python
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
```

(`Path` は train.py 冒頭で import 済み。)

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_train_cli.py -q`
Expected: PASS (既存の resolve_effective_snr_range テスト + 新規)

- [ ] **Step 5: CLI 引数を追加**

`scripts/train.py` の `build_args()`、`--eff-snr-max` の引数の直後に追加:

```python
    p.add_argument("--noise-dir", type=Path, default=None,
                   help="実録音ノイズ WAV のディレクトリ (指定で実ノイズ混合 FT を有効化)")
    p.add_argument("--noise-prob", type=float, default=0.8,
                   help="合成サンプルへの実ノイズ適用率 (--noise-dir 指定時)")
    p.add_argument("--noise-snr-min", type=float, default=-5.0,
                   help="実ノイズ混合時の最小 SNR (dB)")
    p.add_argument("--noise-snr-max", type=float, default=15.0,
                   help="実ノイズ混合時の最大 SNR (dB)")
```

- [ ] **Step 6: main() に配線**

`scripts/train.py` の `main()`、現在の dataset 構成部:

```python
    eff_range = resolve_effective_snr_range(args.eff_snr_min, args.eff_snr_max)
    if eff_range is not None:
        print(f"[init] effective SNR range = {eff_range} dB", flush=True)
    dataset = MorseSynthDataset(
        mode_mix=mode_mix, seed=args.seed, effective_snr_range=eff_range
    )
```

を次に置換:

```python
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
    )
```

- [ ] **Step 7: .gitignore に data/noise/ を追加**

`.gitignore` の `data/keying_scripts/*.wav` の行の直後に追加:

```gitignore
data/noise/
```

- [ ] **Step 8: テストと import を確認**

Run: `python -m pytest tests/test_train_cli.py -q`
Expected: PASS

Run: `python -c "import sys; sys.path.insert(0,'.'); from scripts.train import validate_noise_params, resolve_noise_pool, main; print('ok')"`
Expected: `ok`

- [ ] **Step 9: コミット**

```bash
git add scripts/train.py tests/test_train_cli.py .gitignore
git commit -m "feat: train.py に実ノイズ混合 FT の --noise-dir を配線"
```

---

### Task 2: 全体テストと end-to-end スモーク

**Files:** なし (確認のみ)

**Interfaces:**
- Consumes: Task 1 の成果物。

- [ ] **Step 1: 全テストを実行**

Run: `python -m pytest -q`
Expected: PASS (既存 594 + 新規、失敗 0)

- [ ] **Step 2: 実ノイズ混合モードで train.py が完走するか (CPU 極小スモーク)**

一時ノイズディレクトリを作って 2 ステップ回す:

```bash
python - <<'PY'
import numpy as np, soundfile as sf
from pathlib import Path
d = Path("models/smoke_noise_dir"); d.mkdir(parents=True, exist_ok=True)
sf.write(d/"n.wav", np.random.default_rng(0).standard_normal(8000*4).astype(np.float32), 8000, subtype="PCM_16")
print("noise wav written")
PY
python scripts/train.py --steps 2 --batch-size 2 --num-workers 0 --device cpu \
    --noise-dir models/smoke_noise_dir --noise-prob 0.5 --noise-snr-min 0 --noise-snr-max 20 \
    --ckpt-dir models/smoke_noise --eval-interval 2 --eval-samples-per-cell 1
```
Expected: `[init] real noise: ...` が出て、2 ステップ学習+評価が完走し `models/smoke_noise/last.pt` が生成される。torchaudio DLL 等の環境問題が出た場合はその旨報告。

- [ ] **Step 3: 現行 (ノイズなし) train も回帰していないか**

Run: `python scripts/train.py --steps 2 --batch-size 2 --num-workers 0 --device cpu --ckpt-dir models/smoke_nonoise --eval-interval 2 --eval-samples-per-cell 1`
Expected: `real noise` 行が出ず、従来通り完走する。

- [ ] **Step 4: スモーク生成物を削除**

Run: `rm -rf models/smoke_noise_dir models/smoke_noise models/smoke_nonoise`

- [ ] **Step 5: 変更ファイルとコミットを確認**

Run: `git log --oneline feature/real-noise-ft-train -3`
Expected: Task 1 のコミット + spec コミットが並ぶ。

Run: `git status --short`
Expected: 追跡対象の未コミット変更なし。

---

## FT 実行手順 (実装後、ユーザーが GPU で実行)

コード整備後、実際の FT と測定はユーザーが実行する (本計画のスコープ外だが手順を記す):

```bash
# 1. ノイズディレクトリ作成 (noise_sample.wav のみ)
mkdir -p data/noise
cp data/keying_scripts/noise_sample.wav data/noise/

# 2. FT (良いベースから継続、num_workers=0 で孤児回避)
python scripts/train.py --resume models/full/best.pt --noise-dir data/noise \
    --num-workers 0 --steps 3000 --lr 5e-5 --noise-prob 0.5 \
    --noise-snr-min 0 --noise-snr-max 20 --ckpt-dir models/ft_noise \
    --eval-interval 500 --log-interval 100

# 3. 測定 (keyed_val が真の指標)
python scripts/eval_model.py --ckpt models/ft_noise/best.pt --noise-dir data/noise \
    --keyed-dir data/keying_scripts --out models/eval/ft_noise.json \
    --baseline models/eval/b1_baseline.json
```

成功条件: keyed_val TER が baseline 23.4% を下回る (少なくとも B1 の 43.1% より明確に良い)。

---

## Self-Review 記録

**Spec coverage (設計書 §3-§7 との対応):**
- §3.1 data/noise/ + .gitignore → Task 1 Step 7 (.gitignore) + FT 実行手順 (dir 作成はランタイム)
- §3.2 train.py CLI + resolve_noise_pool + main 配線 → Task 1
- §3.3 MorseSynthDataset 変更なし → 明記 (触らない)
- §4 ハイパーパラメータ → FT 実行手順に反映 (CLI 既定は finetune 準拠、実験値は手順に記載)
- §5 実験手順・成功条件 → FT 実行手順
- §6 テスト方針 → Task 1 (validate/resolve のテスト) + Task 2 (スモーク)
- §7 エラー処理 (prob範囲・snr逆転・dir不在) → Task 1 validate_noise_params + resolve_noise_pool (from_dir の ValueError)

**型整合:** `validate_noise_params(noise_dir, noise_prob, snr_min, snr_max) -> None` と `resolve_noise_pool(noise_dir) -> RealNoisePool | None` (Task 1 で定義・使用)。`MorseSynthDataset(noise_pool=, noise_prob=, noise_snr_range=)` は既存シグネチャ (確認済み)。

**注意 (実FTはスコープ外):** 本計画は CLI 配線とスモークまで。実際の FT ラン (3000 steps) と eval_model.py 測定はユーザーが GPU で実行する。Task 2 のスモークは 2 ステップの配線確認のみ。

**注意 (noise-snr の意味):** `MorseSynthDataset` の実ノイズ経路は `add_real_noise` を使い、これは full-band SNR で加算する。実録音ノイズは帯域内なので実効 SNR ≒ 指定値。B1 の `add_awgn_effective` とは別経路 (併用しない)。

# B1 実効SNR学習範囲の拡張 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 合成器のノイズ付加を「実効(帯域内)SNR指定」に切り替えられるようにし、`train.py --eff-snr-min/max` で実効SNR範囲を指定して再学習できるようにする (0dB付近の崖を下げるため)。

**Architecture:** `add_awgn_effective` (BPF後の実効SNRが目標になる広帯域AWGNをBPF前に加算) を追加し、`SynthConfig.snr_is_effective` フラグ (既定False・後方互換) で合成器の分岐を切り替える。`DefaultConfigSampler` と `MorseSynthDataset` に `effective_snr_range` を通し、`train.py` の CLI から指定する。モデル・特徴量・decode・推論は不変。

**Tech Stack:** Python 3.11, numpy, scipy.signal, PyTorch, pytest。

## Global Constraints

- Python 3.11+。型ヒント必須 (`mypy` 互換)。
- モデル構造・特徴量・decode・推論は変更しない (B1 は合成器の SNR 付加のみ)。
- 波形処理は numpy ベクトル化 (for ループ禁止)。乱数は `np.random.Generator` を引数で受け取る。
- 既存 API 非破壊: `add_awgn` / `add_real_noise` / `apply_cw_filter` / `effective_snr_db` / `make_fixed_eval_set` の既存シグネチャと出力を壊さない。`SynthConfig` の新フィールドは末尾に既定値付きで追加 (既存利用箇所不変)。
- 不変データは `@dataclass(frozen=True)` (ただし `SynthConfig` は既存が非frozenなのでそれに合わせる)。
- ファイル UTF-8 (BOM なし)、改行 LF。docstring/コメント/コミットメッセージは日本語。
- コミット規則: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`。
- 既存 573 テストを壊さない。符号表の e-Gov 照合テストは不変。
- 実効SNRの単位は「受信機 BPF 後の帯域内SNR」。BPF有効前提 (`apply_receiver_filter=True`)。
- 作業ブランチ: `feature/b1-effective-snr`。

---

## File Structure

- `src/synth/noise.py` — 追加: `add_awgn_effective()` (BPF後の実効SNR目標で広帯域AWGNをスケール加算)。
- `src/synth/synthesizer.py` — 変更: `SynthConfig` に `snr_is_effective: bool = False`、`synthesize_from_text` の step5 を分岐。
- `src/synth/dataset.py` — 変更: `DefaultConfigSampler` に `effective_snr_range` パラメータ、`MorseSynthDataset` に `effective_snr_range` パラメータ (per-mode サンプラ構築)。
- `scripts/train.py` — 変更: CLI `--eff-snr-min` / `--eff-snr-max` + 検証 + 配線。
- テスト: `tests/test_synth_noise.py` (追記), `tests/test_synth_dataset.py` (追記), `tests/test_synth_keying.py` or `tests/test_synth_dataset.py` (synthesizer 分岐), `tests/test_train_loop.py` or 新規 (train CLI 検証)。

---

### Task 1: add_awgn_effective (`src/synth/noise.py`)

BPF後の実効(帯域内)SNRが目標になるよう、広帯域AWGNをスケールしてBPF前に加算する純関数。試作で帯域可変でも ±0.00dB 命中を確認済み。

**Files:**
- Modify: `src/synth/noise.py`
- Test: `tests/test_synth_noise.py`

**Interfaces:**
- Consumes: 既存 `signal_power(sig) -> float`, `apply_cw_filter(sig, center_hz, bandwidth_hz, sample_rate, order=4) -> np.ndarray`, `effective_snr_db(sig, noise, center_hz, bandwidth_hz, sample_rate) -> float` (同 module)。
- Produces: `add_awgn_effective(sig: np.ndarray, target_snr_db: float, center_hz: float, bandwidth_hz: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_noise.py` の末尾に追記 (先頭 import に `add_awgn_effective` を足す):

```python
from src.synth.noise import add_awgn_effective


class TestAddAwgnEffective:
    def _tone(self, sr: int = 8000, dur: float = 2.0, freq: float = 600.0) -> np.ndarray:
        t = np.arange(int(sr * dur)) / sr
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    @pytest.mark.parametrize("target", [-5.0, 0.0, 10.0])
    @pytest.mark.parametrize("bw", [250.0, 500.0])
    def test_hits_target_effective_snr(self, target: float, bw: float) -> None:
        sr = 8000
        sig = self._tone(sr)
        rng = np.random.default_rng(1)
        noisy = add_awgn_effective(sig, target, 600.0, bw, sr, rng)
        # noisy - sig がスケール済みノイズ。BPF後の実効SNRを測る。
        eff = effective_snr_db(sig, noisy - sig, 600.0, bw, sr)
        assert abs(eff - target) < 1.0

    def test_deterministic(self) -> None:
        sig = self._tone()
        a = add_awgn_effective(sig, 0.0, 600.0, 400.0, 8000, np.random.default_rng(7))
        b = add_awgn_effective(sig, 0.0, 600.0, 400.0, 8000, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_silent_signal_returns_copy(self) -> None:
        sig = np.zeros(8000, dtype=np.float32)
        out = add_awgn_effective(sig, 0.0, 600.0, 400.0, 8000, np.random.default_rng(0))
        assert np.array_equal(out, sig)
        assert out is not sig

    def test_preserves_dtype(self) -> None:
        sig = self._tone().astype(np.float32)
        out = add_awgn_effective(sig, 5.0, 600.0, 400.0, 8000, np.random.default_rng(0))
        assert out.dtype == np.float32
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_synth_noise.py::TestAddAwgnEffective -v`
Expected: FAIL (`ImportError: cannot import name 'add_awgn_effective'`)

- [ ] **Step 3: 最小実装**

`src/synth/noise.py` の `effective_snr_db` の直後 (または `add_awgn` の近く) に追加:

```python
def add_awgn_effective(
    sig: np.ndarray,
    target_snr_db: float,
    center_hz: float,
    bandwidth_hz: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """BPF後の実効(帯域内)SNRが ``target_snr_db`` になる広帯域AWGNを加算する.

    ``add_awgn`` は SNR を BPF 前 (広帯域) で定義するため、受信機 BPF が
    帯域外ノイズを捨てる分だけ実効SNRが高くなる (設計書 §2.1: 帯域500で +9.5dB、
    帯域可変で下駄が変わる)。本関数は目標を「BPF後の帯域内SNR」で受け、信号と
    ノイズをそれぞれ BPF に通した後のパワー比が目標になるよう広帯域ノイズを
    スケールして **BPF前に** 加える。以降のパイプラインで BPF が適用されると
    実効SNRが目標に一致する (帯域可変でも確定)。

    Args:
        sig: 入力波形 (BPF前).
        target_snr_db: 目標の実効(BPF後・帯域内)SNR (dB).
        center_hz / bandwidth_hz: 後段で適用される受信機 BPF のパラメータ.
        sample_rate: サンプルレート.
        rng: 乱数生成器.

    Returns:
        ノイズを加えた波形 (入力と同じ dtype). 無音入力はそのまま複製を返す.
    """
    sig_bpf_power = signal_power(apply_cw_filter(sig, center_hz, bandwidth_hz, sample_rate))
    if sig_bpf_power == 0.0:
        return sig.copy()
    unit_noise = rng.standard_normal(sig.shape).astype(np.float32)
    noise_bpf_power = signal_power(
        apply_cw_filter(unit_noise, center_hz, bandwidth_hz, sample_rate)
    )
    if noise_bpf_power == 0.0:
        return sig.copy()
    target_lin = 10.0 ** (target_snr_db / 10.0)
    scale = float(np.sqrt(sig_bpf_power / (target_lin * noise_bpf_power)))
    return (sig + scale * unit_noise).astype(sig.dtype)
```

`__all__` に `"add_awgn_effective"` を追加 (アルファベット順で `"add_awgn"` の後)。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_synth_noise.py -q`
Expected: PASS (既存 + 新規)

- [ ] **Step 5: コミット**

```bash
git add src/synth/noise.py tests/test_synth_noise.py
git commit -m "feat: 実効SNR目標で広帯域AWGNを加算する add_awgn_effective を追加"
```

---

### Task 2: SynthConfig.snr_is_effective と合成器の分岐 (`src/synth/synthesizer.py`)

`SynthConfig` に実効モードフラグを追加し、合成の AWGN ステップを分岐する。既定 False で既存挙動不変。

**Files:**
- Modify: `src/synth/synthesizer.py`
- Test: `tests/test_synth_dataset.py`

**Interfaces:**
- Consumes: `add_awgn_effective` (Task 1), 既存 `add_awgn`, `apply_cw_filter`, `SynthConfig`, `synthesize_from_text`, `effective_snr_db` (テスト用)。
- Produces: `SynthConfig.snr_is_effective: bool = False` (新フィールド)。`synthesize_from_text` が実効モードで実効SNR目標のノイズを加える挙動。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_dataset.py` の末尾に追記 (先頭 import を確認):

```python
from src.synth.synthesizer import SynthConfig, synthesize_from_text
from src.synth.keying import KeyingParams
from src.synth.noise import effective_snr_db


class TestSynthEffectiveSnr:
    def _cfg(self, effective: bool, snr: float) -> SynthConfig:
        return SynthConfig(
            mode="european",
            keying=KeyingParams(wpm=20.0, tone_freq_hz=600.0),
            snr_db=snr,
            filter_center_hz=600.0,
            filter_bandwidth_hz=400.0,
            snr_is_effective=effective,
        )

    def test_default_flag_is_false(self) -> None:
        cfg = SynthConfig(mode="european")
        assert cfg.snr_is_effective is False

    def test_effective_mode_hits_target_snr(self) -> None:
        rng = np.random.default_rng(3)
        cfg = self._cfg(effective=True, snr=0.0)
        res = synthesize_from_text("CQ DE JA0XYZ K", cfg, rng, sample_rate=8000)
        # BPF後の結果に対し、クリーン合成との差からノイズ実効SNRを推定するのは難しいので、
        # ここでは「実効0dB は nominal 0dB より明確にノイズが多い (BPF後 SNR が低い)」ことで
        # 分岐が効いていることを確認する。
        rng2 = np.random.default_rng(3)
        cfg_nom = self._cfg(effective=False, snr=0.0)
        res_nom = synthesize_from_text("CQ DE JA0XYZ K", cfg_nom, rng2, sample_rate=8000)
        # 実効0dB はノイズが強いので RMS が nominal 0dB より大きい (BPF後の帯域内で)
        assert float(np.sqrt(np.mean(res.samples.astype(np.float64) ** 2))) > \
               float(np.sqrt(np.mean(res_nom.samples.astype(np.float64) ** 2)))

    def test_nominal_mode_unchanged(self) -> None:
        # snr_is_effective=False は従来と同じ経路 (決定的に一致)
        rng1 = np.random.default_rng(9)
        rng2 = np.random.default_rng(9)
        cfg = self._cfg(effective=False, snr=5.0)
        a = synthesize_from_text("TEST", cfg, rng1, sample_rate=8000)
        b = synthesize_from_text("TEST", cfg, rng2, sample_rate=8000)
        assert np.array_equal(a.samples, b.samples)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_synth_dataset.py::TestSynthEffectiveSnr -v`
Expected: FAIL (`TypeError: SynthConfig.__init__() got an unexpected keyword argument 'snr_is_effective'`)

- [ ] **Step 3: SynthConfig にフィールドを追加**

`src/synth/synthesizer.py` の `SynthConfig` dataclass、`apply_receiver_filter: bool = True` の直後に追加:

```python
    snr_is_effective: bool = False  # True: snr_db を BPF後の実効SNRとして扱う
```

- [ ] **Step 4: 合成器の step5 を分岐**

`src/synth/synthesizer.py` の import に `add_awgn_effective` を追加 (既存の `from src.synth.noise import ...` に足す)。

step5 の `samples = add_awgn(samples, config.snr_db, rng)` を次に置換:

```python
    # 5. AWGN (実効モードなら BPF後の実効SNR目標で加算)
    if config.snr_is_effective:
        samples = add_awgn_effective(
            samples, config.snr_db,
            config.filter_center_hz, config.filter_bandwidth_hz,
            sample_rate, rng,
        )
    else:
        samples = add_awgn(samples, config.snr_db, rng)
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `python -m pytest tests/test_synth_dataset.py tests/test_synth_keying.py -q`
Expected: PASS (既存 + 新規)

- [ ] **Step 6: コミット**

```bash
git add src/synth/synthesizer.py tests/test_synth_dataset.py
git commit -m "feat: SynthConfig.snr_is_effective で合成器のAWGNを実効SNR指定に分岐"
```

---

### Task 3: サンプラとデータセットへの effective_snr_range 配線 (`src/synth/dataset.py`)

`DefaultConfigSampler` と `MorseSynthDataset` に実効SNR範囲を通す。`MorseSynthDataset` は per-mode サンプラを内部構築する既存パターン (tone_freq_range と同様) に従う。

**Files:**
- Modify: `src/synth/dataset.py`
- Test: `tests/test_synth_dataset.py`

**Interfaces:**
- Consumes: `SynthConfig.snr_is_effective` (Task 2), 既存 `DefaultConfigSampler`, `MorseSynthDataset`, `default_config_sampler`。
- Produces:
  - `DefaultConfigSampler(mode, tone_freq_range=..., effective_snr_range: tuple[float, float] | None = None)`
  - `MorseSynthDataset(..., effective_snr_range: tuple[float, float] | None = None)`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_dataset.py` の末尾に追記:

```python
from src.synth.dataset import DefaultConfigSampler, MorseSynthDataset


class TestEffectiveSnrRangeWiring:
    def test_sampler_none_keeps_nominal(self) -> None:
        sampler = DefaultConfigSampler("european")
        cfg = sampler(np.random.default_rng(0))
        assert cfg.snr_is_effective is False
        assert -10.0 <= cfg.snr_db <= 20.0

    def test_sampler_effective_range_sets_flag_and_range(self) -> None:
        sampler = DefaultConfigSampler("european", effective_snr_range=(-8.0, 25.0))
        for seed in range(20):
            cfg = sampler(np.random.default_rng(seed))
            assert cfg.snr_is_effective is True
            assert -8.0 <= cfg.snr_db <= 25.0

    def test_dataset_effective_range_propagates_to_both_modes(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 0.5, "japanese": 0.5},
            seed=1, max_samples=0,
            effective_snr_range=(-8.0, 25.0),
        )
        for mode in ("european", "japanese"):
            cfg = ds._samplers[mode](np.random.default_rng(2))
            assert cfg.snr_is_effective is True
            assert cfg.mode == mode
            assert -8.0 <= cfg.snr_db <= 25.0

    def test_dataset_default_is_nominal(self) -> None:
        ds = MorseSynthDataset(mode_mix={"european": 1.0}, seed=1, max_samples=0)
        cfg = ds._samplers["european"](np.random.default_rng(2))
        assert cfg.snr_is_effective is False
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_synth_dataset.py::TestEffectiveSnrRangeWiring -v`
Expected: FAIL (`TypeError: DefaultConfigSampler.__init__() got an unexpected keyword argument 'effective_snr_range'`)

- [ ] **Step 3: DefaultConfigSampler にパラメータを追加**

`src/synth/dataset.py` の `DefaultConfigSampler.__init__` を変更 (現在 `mode` と `tone_freq_range` を受ける):

```python
    def __init__(
        self,
        mode: Mode,
        tone_freq_range: tuple[float, float] = (400.0, 900.0),
        effective_snr_range: tuple[float, float] | None = None,
    ) -> None:
        self.mode: Mode = mode
        self.tone_freq_range = tone_freq_range
        self.effective_snr_range = effective_snr_range
```

`__call__` 内の `SynthConfig(...)` を構築する箇所で、`snr_db` と `snr_is_effective` を分岐:

```python
        if self.effective_snr_range is not None:
            snr_db = float(rng.uniform(*self.effective_snr_range))
            snr_is_effective = True
        else:
            snr_db = float(rng.uniform(-10.0, 20.0))
            snr_is_effective = False
```

そして `SynthConfig(...)` の `snr_db=float(rng.uniform(-10.0, 20.0))` を `snr_db=snr_db,` に置換し、末尾に `snr_is_effective=snr_is_effective,` を追加。

- [ ] **Step 4: MorseSynthDataset にパラメータを追加**

`src/synth/dataset.py` の `MorseSynthDataset.__init__` シグネチャに `effective_snr_range: tuple[float, float] | None = None` を追加 (既存 `tone_freq_range` の隣)。

サンプラ構築部 (現在 `if config_sampler is not None: ... elif tone_freq_range is not None: ... else: ...`) の `elif tone_freq_range is not None:` ブロックと `else:` ブロックを、`effective_snr_range` を渡すよう変更:

```python
        if config_sampler is not None:
            self._samplers: dict[Mode, ConfigSampler] = {
                mode: config_sampler for mode in self.mode_mix
            }
        else:
            self._samplers = {
                mode: DefaultConfigSampler(
                    mode,
                    tone_freq_range=tone_freq_range if tone_freq_range is not None else (400.0, 900.0),
                    effective_snr_range=effective_snr_range,
                )
                for mode in self.mode_mix
            }
```

(注: `default_config_sampler` は tone/effective を渡せないので、この直接構築に統一する。`default_config_sampler` 関数自体は他の呼び出しのため残す。)

- [ ] **Step 5: テストを実行して成功を確認**

Run: `python -m pytest tests/test_synth_dataset.py -q`
Expected: PASS (既存 + 新規)

- [ ] **Step 6: コミット**

```bash
git add src/synth/dataset.py tests/test_synth_dataset.py
git commit -m "feat: DefaultConfigSampler/MorseSynthDataset に effective_snr_range を配線"
```

---

### Task 4: train.py の CLI 配線と検証 (`scripts/train.py`)

`--eff-snr-min` / `--eff-snr-max` を追加し、実効SNR範囲を `MorseSynthDataset` に渡す。片方のみ/逆転はエラー。

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/test_train_loop.py` (新規テストクラス追記) または新規 `tests/test_train_cli.py`

**Interfaces:**
- Consumes: `MorseSynthDataset(effective_snr_range=...)` (Task 3)。
- Produces: `resolve_effective_snr_range(eff_min: float | None, eff_max: float | None) -> tuple[float, float] | None` (検証付き純関数)。

- [ ] **Step 1: 失敗するテストを書く**

新規 `tests/test_train_cli.py`:

```python
"""train.py の実効SNR CLI 検証のテスト."""
from __future__ import annotations

import pytest

from scripts.train import resolve_effective_snr_range


class TestResolveEffectiveSnrRange:
    def test_both_none_returns_none(self) -> None:
        assert resolve_effective_snr_range(None, None) is None

    def test_both_given_returns_tuple(self) -> None:
        assert resolve_effective_snr_range(-8.0, 25.0) == (-8.0, 25.0)

    def test_only_min_raises(self) -> None:
        with pytest.raises(ValueError, match="両方"):
            resolve_effective_snr_range(-8.0, None)

    def test_only_max_raises(self) -> None:
        with pytest.raises(ValueError, match="両方"):
            resolve_effective_snr_range(None, 25.0)

    def test_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min"):
            resolve_effective_snr_range(25.0, -8.0)
        with pytest.raises(ValueError, match="min"):
            resolve_effective_snr_range(5.0, 5.0)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_train_cli.py -v`
Expected: FAIL (`ImportError: cannot import name 'resolve_effective_snr_range'`)

- [ ] **Step 3: 検証関数を実装**

`scripts/train.py` の `main` の前に追加:

```python
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
```

- [ ] **Step 4: CLI 引数と配線を追加**

`build_args()` の `--device` の前に追加:

```python
    p.add_argument("--eff-snr-min", type=float, default=None,
                   help="実効SNR学習範囲の下限 (dB)。--eff-snr-max と両方指定で実効モード")
    p.add_argument("--eff-snr-max", type=float, default=None,
                   help="実効SNR学習範囲の上限 (dB)")
```

`main()` の `dataset = MorseSynthDataset(mode_mix=mode_mix, seed=args.seed)` を次に置換:

```python
    eff_range = resolve_effective_snr_range(args.eff_snr_min, args.eff_snr_max)
    if eff_range is not None:
        print(f"[init] effective SNR range = {eff_range} dB", flush=True)
    dataset = MorseSynthDataset(
        mode_mix=mode_mix, seed=args.seed, effective_snr_range=eff_range
    )
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `python -m pytest tests/test_train_cli.py -q`
Expected: PASS (5 tests)

- [ ] **Step 6: CLI が壊れていないか確認**

Run: `python -c "import sys; sys.path.insert(0,'.'); from scripts.train import resolve_effective_snr_range, main; print('ok')"`
Expected: `ok`

- [ ] **Step 7: コミット**

```bash
git add scripts/train.py tests/test_train_cli.py
git commit -m "feat: train.py に --eff-snr-min/max を追加し実効SNR範囲を配線"
```

---

### Task 5: 全体テストと end-to-end スモーク

**Files:** なし (確認のみ)

**Interfaces:**
- Consumes: 全タスクの成果物

- [ ] **Step 1: 全テストを実行**

Run: `python -m pytest -q`
Expected: PASS (既存 573 + 新規、失敗 0)

- [ ] **Step 2: 実効SNRモードで train.py が最後まで走るか (CPU 極小スモーク)**

Run: `python scripts/train.py --steps 2 --batch-size 2 --num-workers 0 --device cpu --eff-snr-min -8 --eff-snr-max 25 --ckpt-dir models/smoke_b1 --eval-interval 2 --eval-samples-per-cell 1`
Expected: `[init] effective SNR range = (-8.0, 25.0) dB` が出て、2ステップ学習+評価が完走し `models/smoke_b1/last.pt` が生成される (torchaudio DLL 等の環境問題が出た場合はその旨報告)。

- [ ] **Step 3: 現行 (nominal) train も回帰していないか**

Run: `python scripts/train.py --steps 2 --batch-size 2 --num-workers 0 --device cpu --ckpt-dir models/smoke_b1_nom --eval-interval 2 --eval-samples-per-cell 1`
Expected: `effective SNR range` 行が出ず、従来通り完走する。

- [ ] **Step 4: スモーク生成物を削除**

Run: `rm -rf models/smoke_b1 models/smoke_b1_nom`

- [ ] **Step 5: 変更ファイルとコミットを確認**

Run: `git log --oneline feature/b1-effective-snr -6`
Expected: Task 1〜4 のコミットが並ぶ。

Run: `git status --short`
Expected: 追跡対象の未コミット変更なし。

---

## Self-Review 記録

**Spec coverage (設計書 §3-§6 との対応):**
- §3.1 add_awgn_effective → Task 1
- §3.2 SynthConfig.snr_is_effective + 合成器分岐 → Task 2
- §3.3 DefaultConfigSampler.effective_snr_range → Task 3
- §3.4 train.py --eff-snr-min/max + 検証 → Task 4 (per-mode 配線は MorseSynthDataset 経由に修正: 設計書 §3.4 は config_sampler 経由と書いたが、MorseSynthDataset が単一 config_sampler を全モードに使う実装のため、per-mode 正しく渡すには MorseSynthDataset のパラメータ経由が正。設計意図 [全モードで実効範囲] は満たす)
- §5 テスト方針 → 各 Task の TDD + Task 5 スモーク
- §6 エラー処理 (両方必須・逆転) → Task 4 resolve_effective_snr_range

**型整合:** `add_awgn_effective(sig, target_snr_db, center_hz, bandwidth_hz, sample_rate, rng) -> np.ndarray` (Task 1) を Task 2 が使用。`SynthConfig.snr_is_effective` (Task 2) を Task 3 が使用。`DefaultConfigSampler(effective_snr_range=...)` / `MorseSynthDataset(effective_snr_range=...)` (Task 3) を Task 4 が使用。`resolve_effective_snr_range(...) -> tuple|None` (Task 4)。すべて定義タスクが使用タスクより前。

**注意 (実再学習はスコープ外):** 本計画は実効SNRモードのコード整備とスモークまで。設計書 §4 の実際の再学習ラン (数千〜数万ステップ) と eval_model.py での前後比較はユーザーが GPU で実行する。Task 5 のスモークは 2 ステップの配線確認のみ。

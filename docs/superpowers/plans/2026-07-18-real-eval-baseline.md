# 実信号評価基盤の整備 (Phase A 残り) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 合成器のノイズ/SNR の実効値化・実ノイズ固定評価セット・ラベル検品/オンエア切り出しツールを整備し、CW デコーダの改善前後を実信号基準で定量比較できるようにする。

**Architecture:** 評価と収集の周辺のみを触る。3つの評価セット (synth_val=主指標, keyed_val=診断, onair_val=次フェーズ枠) を分離。ノイズは受信機 BPF の前で加わるため公称 SNR が実効値より約 9.5 dB 甘い問題を、実効 SNR ユーティリティと記録で解消する。モデル・推論・特徴量・decode・符号定義は一切変更しない。

**Tech Stack:** Python 3.11, numpy (ベクトル化必須), scipy.signal, PyTorch, pytest, soundfile。

## Global Constraints

- Python 3.11+。型ヒント必須 (`mypy` 互換)。不変データは `@dataclass(frozen=True)`。
- 波形処理は numpy ベクトル化。for ループでのサンプル単位処理は禁止。
- 乱数は `np.random.Generator` を引数で受け取る。グローバル `np.random` 禁止。固定セット生成 API は決定的であること。
- 符号定義の単一ソースは `src/tokens/morse_tokens.py`。符号・語彙・変換表は変更しない。
- 既存 API 非破壊: `levenshtein_distance` / `error_rate` / `EvalRecord` / `EvalReport` / `make_fixed_eval_set` / `add_awgn` / `add_real_noise` / `apply_cw_filter` の既存シグネチャと出力を壊さない。
- ファイル UTF-8 (BOM なし)、改行 LF。コメント・docstring・コミットメッセージは日本語。
- コミット規則: `feat:` / `fix:` / `docs:` / `test:` / `refactor:` / `chore:`。
- 既存 538 テストを壊さない。符号表の e-Gov 照合テストは不変。
- 作業ブランチ: `feature/eval-token-error-analysis` (Phase A 継続)。

---

## File Structure

- `.gitignore` — 変更: `data/keying_scripts/*.wav` を除外 (原稿 TXT は追跡継続)。
- `src/synth/noise.py` — 追加: `effective_snr_db()` 純関数 (受信機 BPF 後のパワー比)。
- `src/synth/dataset.py` — 追加: `RealNoiseEvalSample` dataclass と `make_fixed_real_noise_eval_set()`。`make_fixed_eval_set` は不変。
- `src/train/metrics.py` — 変更: `EvalRecord` に `eff_snr_db` 任意フィールド追加。`EvalReport` に実効 SNR 別集計。`DetailedEvalReport.to_dict()` に実効 SNR 別内訳。
- `src/finetune/keying_scripts.py` — 追加: `normalize_label_markers()` (`[ホレ]→{HORE}` 等)。
- `scripts/finetune.py` — 変更: `--eval-dir` で固定 val ディレクトリ指定。
- `scripts/inspect_real_labels.py` — 新規: ラベル検品ツール。
- `scripts/clip_onair.py` — 新規: オンエア切り出しツール。
- `docs/phase4_data_collection.md` — 変更: `?` 運用方針の更新。
- テスト: `tests/test_synth_noise.py` (追記), `tests/test_synth_dataset.py` (追記), `tests/test_metrics.py` (追記), `tests/test_finetune_dataset.py` or 新規 `tests/test_keying_scripts.py` (追記), `tests/test_clip_onair.py` (新規)。

---

### Task 1: wav データ保護 (.gitignore)

大容量 wav (`noise_sample.wav` 87MB 等) の誤コミットを防ぐ。最初に行うことで以降の作業中の事故を防止する。

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 現状確認 — 対象 wav が未追跡であること**

Run: `git status --short data/keying_scripts/ | grep -c '\.wav'`
Expected: `21` (すべて `??` 未追跡)。もし追跡済み (`A`/`M`) の wav があれば、この計画を止めてユーザーに報告する (履歴混入の判断が要るため)。

- [ ] **Step 2: `.gitignore` に wav 除外を追記**

`.gitignore` の `data/real/` の行の直後に以下を追加:

```gitignore
# 打鍵録音・実ノイズ WAV は大容量のため除外 (原稿 TXT は追跡)
data/keying_scripts/*.wav
```

- [ ] **Step 3: 除外が効いていることを確認**

Run: `git status --short data/keying_scripts/ | grep -c '\.wav'`
Expected: `0` (wav が status に出ない)

Run: `git check-ignore data/keying_scripts/noise_sample.wav`
Expected: `data/keying_scripts/noise_sample.wav`

- [ ] **Step 4: コミット**

```bash
git add .gitignore
git commit -m "chore: 打鍵録音・実ノイズ WAV を gitignore に追加"
```

---

### Task 2: 実効 SNR ユーティリティ (`src/synth/noise.py`)

信号とノイズを受信機 BPF に通した後のパワー比から実効 (帯域内) SNR を計算する純関数。公称 SNR と実効 SNR の +9.5 dB のズレ (設計書 §2.1) を定量化する土台。

**Files:**
- Modify: `src/synth/noise.py`
- Test: `tests/test_synth_noise.py`

**Interfaces:**
- Consumes: 既存 `signal_power(sig) -> float`, `apply_cw_filter(sig, center_hz, bandwidth_hz, sample_rate, order=4) -> np.ndarray` (同 module 内)
- Produces: `effective_snr_db(sig: np.ndarray, noise: np.ndarray, center_hz: float, bandwidth_hz: float, sample_rate: int) -> float` — sig と noise をそれぞれ BPF に通した後のパワー比 `10*log10(P_bpf(sig)/P_bpf(noise))`。noise 側 BPF 後パワーが 0 なら `float('inf')`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_noise.py` の末尾に追記 (先頭の import に必要なものを足す):

```python
from src.synth.noise import effective_snr_db, add_awgn, signal_power


class TestEffectiveSnrDb:
    def test_broadband_awgn_has_positive_bpf_gain(self) -> None:
        """帯域外に広がる AWGN は BPF 後に減るので、実効 SNR は公称より高くなる."""
        rng = np.random.default_rng(0)
        sr = 8000
        # 600 Hz トーン (帯域内の信号)
        t = np.arange(sr * 2) / sr
        sig = np.sin(2 * np.pi * 600 * t).astype(np.float32)
        # 公称 0 dB になるよう広帯域ノイズを作る
        nominal_snr = 0.0
        noise_power = signal_power(sig) / (10.0 ** (nominal_snr / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=sig.size).astype(np.float32)

        eff = effective_snr_db(sig, noise, center_hz=600.0, bandwidth_hz=500.0, sample_rate=sr)
        # BPF が帯域外ノイズを捨てるので実効 SNR は公称 0 dB より数 dB 高い
        assert eff > nominal_snr + 3.0

    def test_inband_noise_has_near_zero_gain(self) -> None:
        """既に帯域内のノイズは BPF でほぼ減らない → 実効 ≒ 公称."""
        rng = np.random.default_rng(1)
        sr = 8000
        t = np.arange(sr * 2) / sr
        sig = np.sin(2 * np.pi * 600 * t).astype(np.float32)
        # 帯域内 (590-610Hz) の狭帯域ノイズ
        narrow = np.sin(2 * np.pi * 600 * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        narrow *= rng.normal(1.0, 0.05, size=sig.size).astype(np.float32)
        eff = effective_snr_db(sig, narrow, center_hz=600.0, bandwidth_hz=500.0, sample_rate=sr)
        nominal = 10.0 * np.log10(signal_power(sig) / signal_power(narrow))
        assert abs(eff - nominal) < 3.0

    def test_zero_noise_returns_inf(self) -> None:
        sr = 8000
        sig = np.ones(sr, dtype=np.float32)
        noise = np.zeros(sr, dtype=np.float32)
        eff = effective_snr_db(sig, noise, center_hz=600.0, bandwidth_hz=500.0, sample_rate=sr)
        assert eff == float("inf")
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_synth_noise.py::TestEffectiveSnrDb -v`
Expected: FAIL (`ImportError: cannot import name 'effective_snr_db'`)

- [ ] **Step 3: 最小実装**

`src/synth/noise.py` の `apply_cw_filter` の直後 (`add_real_noise` の前) に追加:

```python
def effective_snr_db(
    sig: np.ndarray,
    noise: np.ndarray,
    center_hz: float,
    bandwidth_hz: float,
    sample_rate: int,
) -> float:
    """受信機 BPF 後の帯域内パワー比から実効 SNR (dB) を求める.

    公称 SNR は BPF 前で定義されるが、AWGN は帯域外にも広がるため BPF 後の
    実効 SNR は公称より高くなる (設計書 §2.1: 約 +9.5 dB)。実録音ノイズは
    最初から帯域内なので実効 ≒ 公称。評価表の SNR 列をこの実効値で記録する。

    Args:
        sig: 信号波形 (BPF 前).
        noise: ノイズ波形 (BPF 前、``sig`` と同一サンプルレート).
        center_hz: 受信機 BPF の中心周波数.
        bandwidth_hz: 受信機 BPF の帯域幅.
        sample_rate: サンプルレート.

    Returns:
        実効 SNR (dB). BPF 後のノイズパワーが 0 なら ``inf``.
    """
    sig_f = apply_cw_filter(sig, center_hz, bandwidth_hz, sample_rate)
    noi_f = apply_cw_filter(noise, center_hz, bandwidth_hz, sample_rate)
    np_ = signal_power(noi_f)
    if np_ == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(signal_power(sig_f) / np_))
```

`__all__` に `"effective_snr_db"` を追加 (アルファベット順で `"apply_qsb"` の後、`"signal_power"` の前あたり)。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_synth_noise.py::TestEffectiveSnrDb -v`
Expected: PASS (3 tests)

- [ ] **Step 5: コミット**

```bash
git add src/synth/noise.py tests/test_synth_noise.py
git commit -m "feat: 受信機 BPF 後の実効 SNR を計算する effective_snr_db を追加"
```

---

### Task 3: EvalRecord に実効 SNR フィールドと集計 (`src/train/metrics.py`)

評価記録に実効 SNR を持たせ、`EvalReport` で実効 SNR 別に集計できるようにする。既存フィールド・既存テストは非破壊 (任意フィールド、デフォルト None)。

**Files:**
- Modify: `src/train/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: 既存 `EvalRecord`, `EvalReport`, `AggregateMetrics`, `bin_snr` (同 module)
- Produces:
  - `EvalRecord.eff_snr_db: float | None = None` (新フィールド)
  - `EvalReport.by_eff_snr: dict[float, AggregateMetrics]` (新フィールド)
  - `EvalReport.add(record, snr_bin=None, wpm_bin=None, eff_snr_bin=None)` (引数追加、後方互換)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_metrics.py` の `TestEvalReport` クラスに追記:

```python
    def test_breakdown_by_eff_snr(self) -> None:
        report = EvalReport()
        rec_good = EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[1, 2],
            ref_text="AB", pred_text="AB", eff_snr_db=9.5,
        )
        rec_bad = EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[3, 4],
            ref_text="AB", pred_text="XY", eff_snr_db=-0.5,
        )
        report.add(rec_good, eff_snr_bin=10.0)
        report.add(rec_bad, eff_snr_bin=0.0)
        assert report.by_eff_snr[10.0].ter == 0.0
        assert report.by_eff_snr[0.0].ter == 1.0
        assert report.overall.ter == 0.5

    def test_eff_snr_defaults_none_and_optional(self) -> None:
        # eff_snr を渡さなくても既存どおり動く
        report = EvalReport()
        rec = EvalRecord(ref_tokens=[1], pred_tokens=[1], ref_text="A", pred_text="A")
        assert rec.eff_snr_db is None
        report.add(rec, snr_bin=10.0)
        assert report.by_eff_snr == {}
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_metrics.py::TestEvalReport::test_breakdown_by_eff_snr -v`
Expected: FAIL (`TypeError: EvalRecord.__init__() got an unexpected keyword argument 'eff_snr_db'`)

- [ ] **Step 3: EvalRecord にフィールド追加**

`src/train/metrics.py` の `EvalRecord` (現在 `snr_db` / `wpm` フィールドがある箇所) に追記:

```python
@dataclass
class EvalRecord:
    """1 サンプル分の評価記録."""

    ref_tokens: list[int]
    pred_tokens: list[int]
    ref_text: str
    pred_text: str
    snr_db: float | None = None
    wpm: float | None = None
    eff_snr_db: float | None = None   # 受信機 BPF 後の実効 SNR (dB)
```

- [ ] **Step 4: EvalReport に集計を追加**

`EvalReport` の `by_wpm` フィールドの直後に追加:

```python
    by_eff_snr: dict[float, AggregateMetrics] = field(default_factory=dict)
```

`EvalReport.add` を差し替え (既存の snr/wpm 処理は保持):

```python
    def add(
        self,
        record: EvalRecord,
        snr_bin: float | None = None,
        wpm_bin: float | None = None,
        eff_snr_bin: float | None = None,
    ) -> None:
        self.overall.add(record)
        if snr_bin is not None:
            self.by_snr.setdefault(snr_bin, AggregateMetrics()).add(record)
        if wpm_bin is not None:
            self.by_wpm.setdefault(wpm_bin, AggregateMetrics()).add(record)
        if eff_snr_bin is not None:
            self.by_eff_snr.setdefault(eff_snr_bin, AggregateMetrics()).add(record)
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `python -m pytest tests/test_metrics.py tests/test_metrics_errors.py -q`
Expected: PASS (既存 + 新規すべて)

- [ ] **Step 6: コミット**

```bash
git add src/train/metrics.py tests/test_metrics.py
git commit -m "feat: EvalRecord に実効 SNR フィールドと EvalReport 実効 SNR 別集計を追加"
```

---

### Task 4: 実ノイズ固定評価セット生成 (`src/synth/dataset.py`)

合成キーイング + 実録音ノイズの決定的な固定評価セット (synth_val 主指標) を作る。実効 SNR を各サンプルに記録する。`make_fixed_eval_set` は返り値型を壊さないよう不変とし、新しい sibling 関数を追加する (設計書 §4.2 の「default None で既存維持」の意図を、より安全な形で満たす)。

**Files:**
- Modify: `src/synth/dataset.py`
- Test: `tests/test_synth_dataset.py`

**Interfaces:**
- Consumes: 既存 `SynthConfig`, `KeyingParams`, `synthesize_random`, `RealNoisePool`, `add_real_noise` (import 済み), `effective_snr_db` (Task 2, 追加 import 要), `Mode`
- Produces:
  - `RealNoiseEvalSample` (frozen dataclass): `samples: np.ndarray`, `token_ids: np.ndarray`, `text: str`, `mode: Mode`, `wpm: float`, `target_snr_db: float`, `eff_snr_db: float`
  - `make_fixed_real_noise_eval_set(noise_pool, snr_grid, wpm_grid, samples_per_cell, seed, mode="european", tone_center_hz=494.0, filter_bandwidth_hz=300.0, sample_rate=8000) -> list[RealNoiseEvalSample]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_synth_dataset.py` の末尾に追記 (先頭 import に `numpy as np` 済みを確認、無ければ足す):

```python
from src.synth.dataset import (
    RealNoiseEvalSample,
    make_fixed_real_noise_eval_set,
)
from src.synth.noise import RealNoisePool


def _dummy_noise_pool() -> RealNoisePool:
    rng = np.random.default_rng(0)
    # 帯域内 (約 500Hz) の合成ノイズ 8 秒分
    t = np.arange(8000 * 8) / 8000
    wave = (np.sin(2 * np.pi * 500 * t) * rng.normal(1.0, 0.1, t.size)).astype(np.float32)
    return RealNoisePool([wave], sample_rate=8000)


class TestFixedRealNoiseEvalSet:
    def test_deterministic(self) -> None:
        pool = _dummy_noise_pool()
        kw = dict(noise_pool=pool, snr_grid=[10.0, 0.0], wpm_grid=[20.0],
                  samples_per_cell=2, seed=42, mode="european")
        set1 = make_fixed_real_noise_eval_set(**kw)
        set2 = make_fixed_real_noise_eval_set(**kw)
        assert len(set1) == len(set2) == 4  # 2 snr * 1 wpm * 2
        for a, b in zip(set1, set2, strict=True):
            assert np.array_equal(a.samples, b.samples)
            assert a.token_ids.tolist() == b.token_ids.tolist()
            assert a.eff_snr_db == b.eff_snr_db

    def test_records_effective_snr_close_to_target_for_inband_noise(self) -> None:
        # 帯域内ノイズは BPF で減らないので実効 ≒ 目標
        pool = _dummy_noise_pool()
        samples = make_fixed_real_noise_eval_set(
            noise_pool=pool, snr_grid=[5.0], wpm_grid=[20.0],
            samples_per_cell=3, seed=7, mode="european",
        )
        for s in samples:
            assert s.target_snr_db == 5.0
            assert abs(s.eff_snr_db - 5.0) < 4.0
            assert s.samples.dtype == np.float32
            assert s.token_ids.size > 0

    def test_grid_covers_all_cells(self) -> None:
        pool = _dummy_noise_pool()
        samples = make_fixed_real_noise_eval_set(
            noise_pool=pool, snr_grid=[10.0, 0.0], wpm_grid=[17.0, 25.0],
            samples_per_cell=1, seed=1, mode="european",
        )
        pairs = {(s.target_snr_db, s.wpm) for s in samples}
        assert pairs == {(10.0, 17.0), (10.0, 25.0), (0.0, 17.0), (0.0, 25.0)}
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_synth_dataset.py::TestFixedRealNoiseEvalSet -v`
Expected: FAIL (`ImportError: cannot import name 'RealNoiseEvalSample'`)

- [ ] **Step 3: dataclass と関数を実装**

`src/synth/dataset.py` の import 節に追加:

```python
from src.synth.noise import RealNoisePool, add_real_noise, effective_snr_db
```
(既存の `from src.synth.noise import RealNoisePool, add_real_noise` 行を上記に置換)

`make_fixed_eval_set` の直後に追加:

```python
@dataclass(frozen=True)
class RealNoiseEvalSample:
    """実ノイズ混合の固定評価サンプル 1 件."""

    samples: np.ndarray          # float32 波形 (実ノイズ混合済み)
    token_ids: np.ndarray        # int64 正解トークン列
    text: str                    # 正解テキスト
    mode: Mode
    wpm: float
    target_snr_db: float         # add_real_noise に渡した目標 SNR
    eff_snr_db: float            # 受信機 BPF 後の実効 SNR


def make_fixed_real_noise_eval_set(
    noise_pool: RealNoisePool,
    snr_grid: list[float],
    wpm_grid: list[float],
    samples_per_cell: int,
    seed: int,
    mode: Mode = "european",
    tone_center_hz: float = 494.0,
    filter_bandwidth_hz: float = 300.0,
    sample_rate: int = 8000,
) -> list[RealNoiseEvalSample]:
    """合成キーイング + 実録音ノイズの固定評価セットを生成 (決定的).

    実運用に近い評価のため、トーン中心・BPF 帯域はユーザー実測値
    (設計書 §2.6: 494 Hz, 帯域内ノイズ 300-550 Hz) を既定にする。
    ``target_snr_db`` は ``add_real_noise`` に渡す SNR。実録音ノイズは
    既に帯域内なので実効 SNR ≒ 目標だが、``effective_snr_db`` で実測して記録する。

    手送りジッタは小さく (機械キーイング相当) 固定し、評価の再現性を保つ。
    """
    rng = np.random.default_rng(seed)
    results: list[RealNoiseEvalSample] = []
    for snr in snr_grid:
        for wpm in wpm_grid:
            for _ in range(samples_per_cell):
                config = SynthConfig(
                    mode=mode,
                    keying=KeyingParams(
                        wpm=wpm,
                        dash_dot_ratio=3.0,
                        element_jitter_sigma_ratio=0.02,
                        tone_freq_hz=tone_center_hz,
                        rise_fall_ms=5.0,
                        pre_silence_sec=0.05,
                        post_silence_sec=0.05,
                    ),
                    snr_db=40.0,  # 高 SNR でクリーン合成 → 実ノイズを後段で加算
                    qsb_depth=0.0,
                    qrm_stations=0,
                    qrn_rate_per_sec=0.0,
                    filter_center_hz=tone_center_hz,
                    filter_bandwidth_hz=filter_bandwidth_hz,
                )
                res = synthesize_random(rng, config, sample_rate=sample_rate)
                clean = res.samples
                if clean.size == 0:
                    continue
                segment = noise_pool.sample_segment(clean.size, rng)
                mixed = add_real_noise(clean, segment, snr)
                eff = effective_snr_db(
                    clean, segment, tone_center_hz, filter_bandwidth_hz, sample_rate
                )
                results.append(RealNoiseEvalSample(
                    samples=mixed.astype(np.float32, copy=False),
                    token_ids=res.token_ids,
                    text=res.text,
                    mode=mode,
                    wpm=wpm,
                    target_snr_db=snr,
                    eff_snr_db=eff,
                ))
    return results
```

`__all__` に `"RealNoiseEvalSample"` と `"make_fixed_real_noise_eval_set"` を追加。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_synth_dataset.py -q`
Expected: PASS (既存 + 新規 3 tests)

- [ ] **Step 5: コミット**

```bash
git add src/synth/dataset.py tests/test_synth_dataset.py
git commit -m "feat: 実ノイズ混合の固定評価セット make_fixed_real_noise_eval_set を追加"
```

---

### Task 5: ラベルマーカー正規化 (`src/finetune/keying_scripts.py`)

`TokenConverter` の表示形式 (`[ホレ]` 等) を `text_to_codes` が受理する形式 (`{HORE}` 等) に変換する純関数。オンエア擬似ラベルの往復に必須 (設計書 §4.7)。

**Files:**
- Modify: `src/finetune/keying_scripts.py`
- Test: `tests/test_keying_scripts.py` (新規)

**Interfaces:**
- Consumes: `src.tokens.morse_tokens.text_to_codes`, `Mode`
- Produces: `normalize_label_markers(text: str) -> str` — `[ホレ]→{HORE}`, `[ラタ]→{RATA}`, `[SK]→{SK}` に置換。合成入力マーカー未定義の `[SN]`/`[KN]`/`[HH]` を含む場合は `ValueError`。

- [ ] **Step 1: 失敗するテストを書く**

新規 `tests/test_keying_scripts.py`:

```python
"""打鍵原稿・ラベル正規化のテスト."""
from __future__ import annotations

import pytest

from src.finetune.keying_scripts import normalize_label_markers
from src.tokens.morse_tokens import text_to_codes


class TestNormalizeLabelMarkers:
    def test_converts_display_prosigns_to_input_markers(self) -> None:
        assert normalize_label_markers("[ホレ]テンキ[ラタ]") == "{HORE}テンキ{RATA}"
        assert normalize_label_markers("TU 73 [SK]") == "TU 73 {SK}"

    def test_normalized_japanese_is_tokenizable(self) -> None:
        norm = normalize_label_markers("[ホレ]テンキ ハ ハレ[ラタ]")
        # 正規化後は text_to_codes を通る (KeyError にならない)
        codes = text_to_codes(norm, "japanese")
        assert len(codes) > 0

    def test_normalized_european_is_tokenizable(self) -> None:
        norm = normalize_label_markers("CQ DE JA0XYZ [SK]")
        codes = text_to_codes(norm, "european")
        assert len(codes) > 0

    def test_no_markers_passthrough(self) -> None:
        assert normalize_label_markers("CQ DE JA0XYZ K") == "CQ DE JA0XYZ K"

    def test_question_mark_is_left_as_is(self) -> None:
        # ? は疑問符符号として扱う (設計書 §4.7)。置換しない。
        assert normalize_label_markers("QRL?") == "QRL?"

    def test_unsupported_prosign_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[SN\]|\[KN\]|\[HH\]"):
            normalize_label_markers("R [SN] TU")
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_keying_scripts.py -v`
Expected: FAIL (`ImportError: cannot import name 'normalize_label_markers'`)

- [ ] **Step 3: 最小実装**

`src/finetune/keying_scripts.py` の `read_script_text` の直前に追加:

```python
# TokenConverter の表示表記 → text_to_codes が受理する合成入力マーカー
_MARKER_NORMALIZE: dict[str, str] = {
    "[ホレ]": "{HORE}",
    "[ラタ]": "{RATA}",
    "[SK]": "{SK}",
}
# 合成入力マーカーが未定義の表示プロサイン (採用不可)
_UNSUPPORTED_MARKERS: tuple[str, ...] = ("[SN]", "[KN]", "[HH]")


def normalize_label_markers(text: str) -> str:
    """デコード表示のラベルを text_to_codes が受理する形式へ正規化する.

    ``[ホレ]``→``{HORE}`` のように表示プロサインを合成入力マーカーへ変換する。
    ``?`` は疑問符符号 (``・・--・・``) として扱い、置換しない (設計書 §4.7)。
    合成入力マーカーが未定義の ``[SN]`` / ``[KN]`` / ``[HH]`` を含む場合は
    その区間を採用できないため ``ValueError``。
    """
    for bad in _UNSUPPORTED_MARKERS:
        if bad in text:
            raise ValueError(
                f"合成入力マーカー未定義のプロサインを含みます: {bad} — この区間は採用不可"
            )
    result = text
    for display, marker in _MARKER_NORMALIZE.items():
        result = result.replace(display, marker)
    return result
```

`__all__` に `"normalize_label_markers"` を追加。

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_keying_scripts.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: コミット**

```bash
git add src/finetune/keying_scripts.py tests/test_keying_scripts.py
git commit -m "feat: ラベル表示プロサインを合成入力マーカーへ正規化する関数を追加"
```

---

### Task 6: finetune の固定 val ディレクトリ対応 (`scripts/finetune.py`)

`--eval-dir` を追加し、乱数分割の代わりに固定 val ディレクトリを使えるようにする。改善前後を同じ val で比較可能にする (設計書 §4.4)。既存 `--eval-ratio` 経路は後方互換で残す。

**Files:**
- Modify: `scripts/finetune.py`
- Test: `tests/test_finetune_dataset.py` (ヘルパ関数のテストを追記)

**Interfaces:**
- Consumes: `discover_real_samples`, `RealSignalDataset`, `split_train_validation`
- Produces: `resolve_train_eval_samples(data_dir, eval_dir, mode_filter, eval_ratio, seed) -> tuple[list[RealSignalSample], list[RealSignalSample]]` — `eval_dir` が指定されればそこを固定 val とし、`data_dir` 全件を train にする。未指定なら従来の乱数分割。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_finetune_dataset.py` の末尾に追記 (先頭 import に足す):

```python
from scripts.finetune import resolve_train_eval_samples


class TestResolveTrainEval:
    def test_fixed_eval_dir_keeps_all_data_for_train(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()
        _make_test_pair(train_dir, "t1_european", "CQ", mode="european")
        _make_test_pair(train_dir, "t2_european", "DE", mode="european")
        _make_test_pair(val_dir, "v1_european", "K", mode="european")

        train, val = resolve_train_eval_samples(
            data_dir=train_dir, eval_dir=val_dir,
            mode_filter=None, eval_ratio=0.2, seed=0,
        )
        assert len(train) == 2   # data_dir 全件が train
        assert len(val) == 1     # eval_dir が固定 val
        assert val[0].text == "K"

    def test_no_eval_dir_falls_back_to_random_split(self, tmp_path: Path) -> None:
        for i in range(5):
            _make_test_pair(tmp_path, f"s{i}_european", "CQ", mode="european")
        train, val = resolve_train_eval_samples(
            data_dir=tmp_path, eval_dir=None,
            mode_filter=None, eval_ratio=0.2, seed=0,
        )
        assert len(train) + len(val) == 5
        assert len(val) >= 1
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_finetune_dataset.py::TestResolveTrainEval -v`
Expected: FAIL (`ImportError: cannot import name 'resolve_train_eval_samples'`)

- [ ] **Step 3: ヘルパ関数を実装**

`scripts/finetune.py` の `evaluate_real` の前に追加:

```python
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
```

- [ ] **Step 4: CLI 引数と呼び出しを配線**

`build_args()` の `--eval-ratio` 引数の直後に追加:

```python
    p.add_argument(
        "--eval-dir", type=Path, default=None,
        help="固定 validation ディレクトリ (指定時 --data-dir 全件を train、ここを val)",
    )
```

`main()` の現在の分割処理:

```python
    samples = discover_real_samples(args.data_dir, mode_filter=args.mode_filter)  # type: ignore[arg-type]
    if not samples:
        print(f"[err] 実信号サンプルが見つかりません: {args.data_dir}", flush=True)
        return 2
    print(f"[scan] found {len(samples)} samples", flush=True)

    train_samples, eval_samples = split_train_validation(
        samples, validation_ratio=args.eval_ratio, seed=args.seed
    )
```

を次に置換:

```python
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
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `python -m pytest tests/test_finetune_dataset.py -q`
Expected: PASS (既存 + 新規)

- [ ] **Step 6: CLI が壊れていないか確認**

Run: `python scripts/finetune.py --help`
Expected: `--eval-dir` がヘルプに表示される (torchaudio の DLL 警告が Git Bash で出る場合があるが、PowerShell/実行環境では無害。ヘルプ全体が出れば OK)

- [ ] **Step 7: コミット**

```bash
git add scripts/finetune.py tests/test_finetune_dataset.py
git commit -m "feat: finetune に固定 validation ディレクトリ --eval-dir を追加"
```

---

### Task 7: ラベル検品ツール (`scripts/inspect_real_labels.py`)

録音を原稿/ラベルと照合し、TER 降順で「打鍵ミス疑い」を提示、符号長別・token 別 recall を出力する運用ツール。新規録音のラベル品質を聴かずに検査する (設計書 §4.5)。ロジックを再利用可能な関数に切り出してテストする。

**Files:**
- Create: `scripts/inspect_real_labels.py`
- Create: `src/finetune/label_inspection.py`
- Test: `tests/test_label_inspection.py`

**Interfaces:**
- Consumes: `RealSignalDataset`, `discover_real_samples`, `MelExtractor`, `CWModel`, `ModelConfig`, `ctc_greedy_decode`, `compute_input_lengths`, `TokenConverter`, `DetailedEvalReport`, `EvalRecord`, `describe_token`, `BLANK_TOKEN_ID`, `VOCAB_SIZE`
- Produces:
  - `recall_by_code_length(analysis: TokenErrorAnalysis) -> dict[int, tuple[int, float]]` — 符号長 → (ref 出現数, recall%)。純関数、torch 不要。
  - `scripts/inspect_real_labels.py::main(argv) -> int` — CLI。

- [ ] **Step 1: 失敗するテストを書く (純関数のみ、torch 不要)**

新規 `tests/test_label_inspection.py`:

```python
"""ラベル検品ロジックのテスト."""
from __future__ import annotations

from src.finetune.label_inspection import recall_by_code_length
from src.train.metrics import EvalRecord, TokenErrorAnalysis
from src.tokens.morse_tokens import TOKEN_TO_ID


class TestRecallByCodeLength:
    def test_groups_recall_by_code_length(self) -> None:
        e = TOKEN_TO_ID["・"]        # 長さ 1
        i = TOKEN_TO_ID["・・"]       # 長さ 2
        sk = TOKEN_TO_ID["・・・-・-"]  # 長さ 6
        analysis = TokenErrorAnalysis()
        # ・ 正解、・・ 正解、6要素符号は脱落
        analysis.add_record(EvalRecord(
            ref_tokens=[e, i, sk], pred_tokens=[e, i],
            ref_text="", pred_text="",
        ))
        result = recall_by_code_length(analysis)
        assert result[1] == (1, 100.0)
        assert result[2] == (1, 100.0)
        assert result[6] == (1, 0.0)   # 6要素符号 recall 0%

    def test_empty_analysis(self) -> None:
        assert recall_by_code_length(TokenErrorAnalysis()) == {}
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_label_inspection.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.finetune.label_inspection'`)

- [ ] **Step 3: 純関数を実装**

新規 `src/finetune/label_inspection.py`:

```python
"""ラベル検品の集計ロジック (torch 非依存の純関数)."""
from __future__ import annotations

from src.train.metrics import TokenErrorAnalysis, describe_token


def _is_code(code: str) -> bool:
    return bool(code) and all(ch in "・-" for ch in code)


def recall_by_code_length(analysis: TokenErrorAnalysis) -> dict[int, tuple[int, float]]:
    """符号長ごとの (ref 出現数, recall%) を返す.

    設計書 §2.3 の「6要素符号 (プロサイン・記号) だけ recall が崩壊する」現象を
    検出するための集計。実符号 (・ と - のみ) を対象にし、WORD_BREAK や
    未知トークンは除外する。
    """
    by_len_correct: dict[int, int] = {}
    by_len_ref: dict[int, int] = {}
    for tid, counts in analysis.counts.items():
        code = str(describe_token(tid)["code"])
        if not _is_code(code):
            continue
        n = len(code)
        by_len_correct[n] = by_len_correct.get(n, 0) + counts.correct
        by_len_ref[n] = by_len_ref.get(n, 0) + counts.ref_count
    result: dict[int, tuple[int, float]] = {}
    for n, ref in by_len_ref.items():
        recall = 100.0 * by_len_correct.get(n, 0) / ref if ref else 0.0
        result[n] = (ref, recall)
    return result


__all__ = ["recall_by_code_length"]
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_label_inspection.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: CLI スクリプトを実装**

新規 `scripts/inspect_real_labels.py`:

```python
"""実信号ラベル検品 CLI.

録音 (WAV+TXT) を既存モデルでデコードし、正解ラベルと照合して TER 降順に
「打鍵ミス / 誤ラベル疑い」を提示する。符号長別・token 別 recall も出力し、
新規録音を聴かずにラベル品質を検査する運用ツール。

使い方::

    python scripts/inspect_real_labels.py --data-dir data/keying_scripts \\
        --ckpt models/full/best_infer.pt
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

from src.finetune.dataset import RealSignalDataset, discover_real_samples  # noqa: E402
from src.finetune.label_inspection import recall_by_code_length            # noqa: E402
from src.tokens.converter import TokenConverter                            # noqa: E402
from src.tokens.morse_tokens import BLANK_TOKEN_ID, VOCAB_SIZE             # noqa: E402
from src.train.checkpoint import load_checkpoint                           # noqa: E402
from src.train.decode import ctc_greedy_decode                            # noqa: E402
from src.train.loop import compute_input_lengths                          # noqa: E402
from src.train.metrics import DetailedEvalReport, EvalRecord, token_label  # noqa: E402
from src.train.model import CWModel, ModelConfig                          # noqa: E402
from src.train.preprocessing import MelExtractor                         # noqa: E402


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="実信号ラベル検品")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--top-n", type=int, default=10, help="疑い上位の表示件数")
    return p


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    samples = discover_real_samples(args.data_dir)
    if not samples:
        print(f"[err] サンプルが見つかりません: {args.data_dir}", flush=True)
        return 2
    dataset = RealSignalDataset(samples)

    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.train(False)
    mel = MelExtractor().to(device)
    hop = mel.config.hop_length

    report = DetailedEvalReport()
    rows: list[tuple[str, float, str, str]] = []
    for i in range(len(dataset)):
        wave, target = dataset[i]
        meta = dataset.sample_at(i)
        t = wave.unsqueeze(0).to(device)
        lp = torch.nn.functional.log_softmax(model(mel(t)).float(), dim=-1)
        il = compute_input_lengths(
            torch.tensor([wave.numel()], device=device), hop, lp.size(1)
        )
        res = ctc_greedy_decode(lp, il, blank_id=BLANK_TOKEN_ID)[0]
        pred_text = TokenConverter(mode=meta.mode, confidence_threshold=0.0).convert(
            res.token_ids
        ).text
        rec = EvalRecord(
            ref_tokens=target.tolist(), pred_tokens=res.token_ids,
            ref_text=meta.text, pred_text=pred_text,
        )
        report.add(rec, name=meta.stem, mode=meta.mode)
        rows.append((meta.stem, report.samples[-1].ter, meta.text, pred_text))

    rows.sort(key=lambda r: -r[1])
    print(f"\n=== TER 降順 (上位 {args.top_n} = 打鍵ミス/誤ラベル疑い) ===")
    for stem, ter, ref, pred in rows[: args.top_n]:
        print(f"\n--- {stem}  TER={ter * 100:.1f}%")
        print(f"    ref : {ref}")
        print(f"    pred: {pred}")

    print("\n=== 符号長別 recall ===")
    for n, (ref, recall) in sorted(recall_by_code_length(report.analysis).items()):
        print(f"  長さ{n}: ref={ref:4d}  recall={recall:5.1f}%")

    print()
    for line in report.summary_lines(top_n=args.top_n):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 実データで動作確認 (存在すれば)**

Run: `python scripts/inspect_real_labels.py --data-dir data/keying_scripts --ckpt models/full/best_infer.pt --device cpu`
Expected: TER 降順リスト + 符号長別 recall (長さ6が低い) + サマリが表示される。`data/keying_scripts` に wav が無い環境では `[err] サンプルが見つかりません` (これは正常、その旨報告)。

- [ ] **Step 7: コミット**

```bash
git add scripts/inspect_real_labels.py src/finetune/label_inspection.py tests/test_label_inspection.py
git commit -m "feat: 実信号ラベル検品ツール inspect_real_labels を追加"
```

---

### Task 8: オンエア切り出しツール (`scripts/clip_onair.py`)

長い QSO 録音から時間区間を切り出し、ラベルを付けて `data/real/onair/` に実信号 TXT 形式で保存する。ラベルは正規化 + `text_to_codes` で即検証し、不能なら弾く (設計書 §4.6)。切り出しロジックを純関数に切り出してテストする。

**Files:**
- Create: `scripts/clip_onair.py`
- Create: `src/finetune/onair_clip.py`
- Test: `tests/test_clip_onair.py`

**Interfaces:**
- Consumes: `normalize_label_markers` (Task 5), `text_to_codes`, `Mode`, soundfile
- Produces:
  - `clip_segment(wave: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray` — 時間区間を切り出す純関数。範囲外は `ValueError`。
  - `validate_label(text: str, mode: Mode) -> str` — 正規化してトークン化可能か検証、正規化済みテキストを返す。不能なら `ValueError`。
  - `scripts/clip_onair.py::main(argv) -> int` — CLI。

- [ ] **Step 1: 失敗するテストを書く**

新規 `tests/test_clip_onair.py`:

```python
"""オンエア切り出しロジックのテスト."""
from __future__ import annotations

import numpy as np
import pytest

from src.finetune.onair_clip import clip_segment, validate_label


class TestClipSegment:
    def test_extracts_time_range(self) -> None:
        sr = 8000
        wave = np.arange(sr * 10, dtype=np.float32)  # 10 秒
        seg = clip_segment(wave, sr, start_s=2.0, end_s=5.0)
        assert seg.size == sr * 3
        assert seg[0] == 2.0 * sr

    def test_end_beyond_length_raises(self) -> None:
        wave = np.zeros(8000 * 3, dtype=np.float32)
        with pytest.raises(ValueError):
            clip_segment(wave, 8000, start_s=1.0, end_s=5.0)

    def test_start_after_end_raises(self) -> None:
        wave = np.zeros(8000 * 10, dtype=np.float32)
        with pytest.raises(ValueError):
            clip_segment(wave, 8000, start_s=5.0, end_s=2.0)


class TestValidateLabel:
    def test_normalizes_and_accepts_tokenizable(self) -> None:
        assert validate_label("[SK] TU", "european") == "{SK} TU"

    def test_rejects_untokenizable(self) -> None:
        with pytest.raises(ValueError):
            validate_label("漢字混入", "european")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            validate_label("   ", "european")
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_clip_onair.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.finetune.onair_clip'`)

- [ ] **Step 3: 純関数を実装**

新規 `src/finetune/onair_clip.py`:

```python
"""オンエア録音の切り出し・ラベル検証 (torch 非依存の純関数)."""
from __future__ import annotations

import numpy as np

from src.finetune.keying_scripts import normalize_label_markers
from src.tokens.morse_tokens import Mode, text_to_codes


def clip_segment(
    wave: np.ndarray, sample_rate: int, start_s: float, end_s: float
) -> np.ndarray:
    """波形から ``[start_s, end_s)`` 秒の区間を切り出す.

    範囲が波形長を超える / start >= end の場合は ``ValueError``。
    """
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"不正な区間: start={start_s}, end={end_s}")
    start = int(round(start_s * sample_rate))
    end = int(round(end_s * sample_rate))
    if end > wave.size:
        raise ValueError(
            f"区間が波形長を超えます: end={end_s}s > {wave.size / sample_rate:.2f}s"
        )
    return np.ascontiguousarray(wave[start:end], dtype=np.float32)


def validate_label(text: str, mode: Mode) -> str:
    """ラベルを正規化してトークン化可能か検証し、正規化済みテキストを返す.

    ``[ホレ]``→``{HORE}`` 等へ正規化した上で ``text_to_codes`` を通す。
    トークン化不能 (漢字混入・未対応プロサイン) や空文字なら ``ValueError``。
    """
    normalized = normalize_label_markers(text).strip()
    if not normalized:
        raise ValueError("ラベルが空です")
    try:
        codes = text_to_codes(normalized, mode)
    except KeyError as exc:
        raise ValueError(f"トークン化できない文字を含みます: {exc}") from exc
    if not codes:
        raise ValueError("符号列が空になりました")
    return normalized


__all__ = ["clip_segment", "validate_label"]
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_clip_onair.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: CLI スクリプトを実装**

新規 `scripts/clip_onair.py`:

```python
"""オンエア録音の区間切り出し CLI.

長い QSO 録音から時間区間を切り出し、正解ラベルを付けて data/real/onair/ に
実信号 TXT 形式で保存する。ラベルは正規化 + トークン化検証を通す。
自信のない区間は保存しない運用 (誤ラベルは指標を壊すため)。

使い方::

    python scripts/clip_onair.py --wav rec/qso1.wav --mode european \\
        --start 12.0 --end 20.0 --label "CQ DE JA0XYZ K"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.onair_clip import clip_segment, validate_label  # noqa: E402


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="オンエア録音の区間切り出し")
    p.add_argument("--wav", type=Path, required=True, help="長い QSO 録音 WAV")
    p.add_argument("--mode", choices=["european", "japanese"], required=True)
    p.add_argument("--start", type=float, required=True, help="開始秒")
    p.add_argument("--end", type=float, required=True, help="終了秒")
    p.add_argument("--label", type=str, required=True, help="正解テキスト")
    p.add_argument("--out-dir", type=Path, default=Path("data/real/onair"))
    p.add_argument("--stem", type=str, default=None, help="出力ファイル名 (省略時自動)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)

    # ラベル検証を先に (不能なら音声を書かずに終了)
    try:
        label = validate_label(args.label, args.mode)  # type: ignore[arg-type]
    except ValueError as exc:
        print(f"[err] ラベル不正: {exc}", flush=True)
        return 2

    wave, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave[:, 0]
    if sr != 8000:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), 8000)
        wave = resample_poly(wave, 8000 // g, int(sr) // g).astype(np.float32)
        sr = 8000

    try:
        seg = clip_segment(wave, sr, args.start, args.end)
    except ValueError as exc:
        print(f"[err] 区間不正: {exc}", flush=True)
        return 2

    peak = float(np.max(np.abs(seg))) if seg.size else 0.0
    if peak > 0:
        seg = (seg / peak * 0.9).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{args.wav.stem}_{int(args.start)}_{int(args.end)}_{args.mode}"
    wav_path = args.out_dir / f"{stem}.wav"
    txt_path = args.out_dir / f"{stem}.txt"
    sf.write(wav_path, seg, sr, subtype="PCM_16")
    txt_path.write_text(
        f"mode: {args.mode}\n"
        f"sample_rate: {sr}\n"
        f"duration_s: {seg.size / sr:.3f}\n"
        f"source: {args.wav.name} [{args.start:.1f}-{args.end:.1f}s]\n"
        "---\n"
        f"{label}\n",
        encoding="utf-8",
    )
    print(f"[done] {wav_path} ({seg.size / sr:.1f}s)  label={label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: CLI が壊れていないか確認**

Run: `python scripts/clip_onair.py --help`
Expected: 引数一覧が表示される

- [ ] **Step 7: コミット**

```bash
git add scripts/clip_onair.py src/finetune/onair_clip.py tests/test_clip_onair.py
git commit -m "feat: オンエア録音の区間切り出しツール clip_onair を追加"
```

---

### Task 9: 収集ガイドの `?` 運用更新 (`docs/phase4_data_collection.md`)

「判別不能は `?`」という誤ラベルを招く旧記述を、設計書 §4.7 の方針 (判別不能な区間は捨てる、`?` は疑問符符号) に更新する。

**Files:**
- Modify: `docs/phase4_data_collection.md`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 該当記述を確認**

Run: `grep -n "判別不能\|？\|?" docs/phase4_data_collection.md`
Expected: 「和文はカタカナで打鍵通り、漢字不可、判別不能は `?`」の行 (§③ 付近) が見つかる。

- [ ] **Step 2: 記述を更新**

`docs/phase4_data_collection.md` の該当行:

```markdown
ラベル書式は `docs/phase4_finetuning.md` と同じ:
和文はカタカナで打鍵通り、漢字不可、判別不能は `?`。
```

を次に置換:

```markdown
ラベル書式は `docs/phase4_finetuning.md` と同じ:
和文はカタカナで打鍵通り、漢字不可。

**判別不能箇所の扱い (重要):** 聞き取れない箇所を `?` で埋めてはいけない。
`?` は疑問符符号 (`・・--・・`) として学習・評価されるため、誤ラベルになる。
確信の持てない箇所を含む区間は **その区間ごと採用しない** (捨てる)。
評価セットは少数でも正確であることが最優先で、誤ラベルは 1 件でも指標を壊す。

デコード結果を修正してラベルにする場合は、表示プロサイン (`[ホレ]` / `[ラタ]` /
`[SK]`) が自動で `{HORE}` / `{RATA}` / `{SK}` に正規化される
(`normalize_label_markers` / `clip_onair.py`)。
```

- [ ] **Step 3: コミット**

```bash
git add docs/phase4_data_collection.md
git commit -m "docs: 判別不能箇所は捨てる方針に収集ガイドを更新"
```

---

### Task 10: 全体テストと最終確認

すべての変更後、全テストが通り既存機能が壊れていないことを確認する。

**Files:**
- なし (確認のみ)

**Interfaces:**
- Consumes: 全タスクの成果物
- Produces: なし

- [ ] **Step 1: 全テストを実行**

Run: `python -m pytest -q`
Expected: PASS (既存 538 + 新規タスクのテスト、失敗 0)

- [ ] **Step 2: 実効 SNR の +9.5 dB を実データで確認 (任意、noise_sample.wav があれば)**

Run:
```bash
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np, soundfile as sf
from src.synth.keying import KeyingParams, codes_to_waveform
from src.synth.noise import effective_snr_db, signal_power
from src.tokens.morse_tokens import text_to_codes
rng = np.random.default_rng(0)
codes = text_to_codes('CQ DE JA0XYZ K', 'european')
wave = codes_to_waveform(codes, KeyingParams(wpm=20, tone_freq_hz=494), rng, sample_rate=8000).samples
sp = signal_power(wave)
for nom in [0.0, -5.0, -10.0]:
    npow = sp / (10**(nom/10))
    noise = rng.normal(0, np.sqrt(npow), wave.size).astype(np.float32)
    eff = effective_snr_db(wave, noise, 494, 500, 8000)
    print(f'nominal {nom:+.0f} dB -> effective {eff:+.1f} dB (下駄 {eff-nom:+.1f})')
"
```
Expected: 各行の「下駄」が約 +9〜10 dB (設計書 §2.1 の再現)。

- [ ] **Step 3: 変更ファイル一覧を確認**

Run: `git log --oneline feature/eval-token-error-analysis -12`
Expected: Task 1〜9 のコミットが並ぶ。

- [ ] **Step 4: 最終コミット (必要なら)**

作業中に未コミットの変更が無いことを確認:

Run: `git status --short`
Expected: 追跡対象の変更なし (wav の `??` は gitignore 済みなので出ない)。

---

## Self-Review 記録

**Spec coverage (設計書 §4 との対応):**
- §4.1 実効 SNR ユーティリティ → Task 2
- §4.2 固定評価セットの実ノイズ対応 → Task 4
- §4.3 metrics 実効 SNR 別集計 → Task 3
- §4.4 `--eval-dir` 固定 val → Task 6
- §4.5 ラベル検品ツール → Task 7
- §4.6 オンエア切り出しツール → Task 8
- §4.7 ラベル書式往復修正 + `?` 方針 → Task 5 (正規化) + Task 9 (ガイド)
- §4.8 .gitignore wav 保護 → Task 1
- §7 UI 表示切替え → **スコープ外** (設計書で別件明記、計画に含めない)

**型整合:** `effective_snr_db` の引数順 (Task 2 定義 = Task 4 呼び出し)、`RealNoiseEvalSample` のフィールド (Task 4 定義)、`EvalRecord.eff_snr_db` (Task 3 定義)、`normalize_label_markers` (Task 5 定義 = Task 8 で使用)、`recall_by_code_length` (Task 7)、`clip_segment`/`validate_label` (Task 8) — すべて定義タスクが使用タスクより前。

**注意:** `DetailedEvalReport.to_dict()` への実効 SNR 別内訳追加 (§4.3 後半) は、`EvalReport.by_eff_snr` を持たせた Task 3 の範囲で最小限に留めた。JSON への実効 SNR 別出力が必要になった時点で `to_dict()` を拡張する (現状の JSON はサンプル別 `eff_snr_db` を SampleEval 経由で持てるが、今回のスコープでは EvalReport 集計までで十分)。

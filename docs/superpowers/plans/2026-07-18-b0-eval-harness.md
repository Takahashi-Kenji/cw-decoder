# B0 評価ハーネス配線 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase A の dormant な評価部品を独立評価スクリプト `eval_model.py` に配線し、任意のチェックポイントに対し synth_val (実効SNR別・モード別) と keyed_val を測定・JSON保存・baseline比較できるようにする。

**Architecture:** 再利用可能な評価ロジックを `src/eval/harness.py` (torch依存: decode/evaluate) と `src/eval/compare.py` (torch非依存: report比較) に分離。`DetailedEvalReport` を by_eff_snr / by_mode 対応に拡張。`scripts/eval_model.py` がこれらを束ねる CLI。finetune と inspect_real_labels の decode 重複を harness に集約。モデル本体は不変。

**Tech Stack:** Python 3.11, PyTorch, numpy, pytest, soundfile。

## Global Constraints

- Python 3.11+。型ヒント必須 (`mypy` 互換)。
- モデル・特徴量・decode アルゴリズム・学習ロジックは変更しない (B0 は測定の配線のみ)。
- 既存 API 非破壊: `DetailedEvalReport.to_dict()` の既存キー (`overall`/`totals`/`token_errors`/`samples`) を壊さない。`EvalReport`/`AggregateMetrics`/`bin_snr` の既存挙動不変。`evaluate_real` の CLI 呼び出し側 (finetune main) の挙動不変。
- `src/eval/compare.py` は torch 非依存 (dict 操作のみ)。
- 乱数は `np.random.Generator` / `torch.manual_seed` を明示。決定的固定セット生成を保つ。
- 符号定義の単一ソースは `src/tokens/morse_tokens.py` (不変)。
- ファイル UTF-8 (BOM なし)、改行 LF。docstring/コメント/コミットメッセージは日本語。
- コミット規則: `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`。
- 既存 562 テストを壊さない。
- 作業ブランチ: `feature/b0-eval-harness`。

---

## File Structure

- `src/train/metrics.py` — 変更: `DetailedEvalReport` に `by_mode` 集計、`to_dict()` に `by_eff_snr`/`by_mode` 追加、`EvalReport.summary_lines()` に "By EffSNR" 節。
- `src/eval/__init__.py` — 新規 (空パッケージマーカー)。
- `src/eval/compare.py` — 新規: `compare_reports()` + delta ヘルパ (torch非依存・純関数)。
- `src/eval/harness.py` — 新規: `decode_wave()` / `evaluate_real_dataset()` / `evaluate_synth_noise()` (torch依存)。
- `scripts/eval_model.py` — 新規: CLI。
- `scripts/finetune.py` — 変更: `evaluate_real` を `harness.evaluate_real_dataset` に委譲。
- `scripts/inspect_real_labels.py` — 変更: decode ループを `harness.decode_wave` に委譲。
- テスト: `tests/test_metrics.py` (追記), `tests/test_eval_compare.py` (新規), `tests/test_eval_harness.py` (新規)。

---

### Task 1: metrics の by_eff_snr / by_mode 出力対応

`DetailedEvalReport` に「実効SNR別」「モード別」の集計を追加し、JSON とサマリに出す。Phase A で dormant だった `by_eff_snr` をここで surface する。

**Files:**
- Modify: `src/train/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: 既存 `DetailedEvalReport` (fields: `report: EvalReport`, `analysis`, `samples: list[SampleEval]`; `SampleEval.mode: str | None`, `SampleEval.record: EvalRecord`), `EvalReport.by_eff_snr: dict[float, AggregateMetrics]`, `AggregateMetrics` (`.n_samples/.ter/.cer/.add(record)`)。
- Produces:
  - `DetailedEvalReport.by_mode() -> dict[str, AggregateMetrics]` (新メソッド)
  - `DetailedEvalReport.to_dict()` に新キー `by_eff_snr`, `by_mode`
  - `EvalReport.summary_lines()` に "By EffSNR" 節 (by_eff_snr が非空のとき)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_metrics.py` の末尾に追記 (先頭 import に `EvalRecord`, `DetailedEvalReport`, `bin_snr` が無ければ足す):

```python
from src.train.metrics import DetailedEvalReport, bin_snr


class TestDetailedReportByModeEffSnr:
    def _report(self) -> DetailedEvalReport:
        d = DetailedEvalReport()
        # 欧文2件 (1件 TER0, 1件 TER1), eff_snr 10.0
        d.add(EvalRecord(ref_tokens=[1, 2], pred_tokens=[1, 2], ref_text="AB", pred_text="AB",
                         eff_snr_db=10.0), name="e1", mode="european", eff_snr_bin=10.0)
        d.add(EvalRecord(ref_tokens=[1, 2], pred_tokens=[3, 4], ref_text="AB", pred_text="XY",
                         eff_snr_db=0.0), name="e2", mode="european", eff_snr_bin=0.0)
        # 和文1件 TER0, eff_snr 10.0
        d.add(EvalRecord(ref_tokens=[5], pred_tokens=[5], ref_text="C", pred_text="C",
                         eff_snr_db=10.0), name="j1", mode="japanese", eff_snr_bin=10.0)
        return d

    def test_by_mode_separates_modes(self) -> None:
        bm = self._report().by_mode()
        assert bm["european"].n_samples == 2
        assert bm["japanese"].n_samples == 1
        assert bm["european"].ter == 0.5   # (0 + 2 errors) / 4 ref tokens
        assert bm["japanese"].ter == 0.0

    def test_to_dict_has_by_eff_snr_and_by_mode(self) -> None:
        import json
        payload = self._report().to_dict()
        # 既存キーは維持
        assert "overall" in payload and "totals" in payload and "token_errors" in payload
        # 新キー
        assert payload["by_mode"]["european"]["n_samples"] == 2
        assert payload["by_mode"]["japanese"]["ter"] == 0.0
        assert payload["by_eff_snr"]["10.0"]["n_samples"] == 2
        assert payload["by_eff_snr"]["0.0"]["ter"] == 1.0
        json.dumps(payload, ensure_ascii=False)  # JSON 化できる

    def test_summary_lines_has_effsnr_section(self) -> None:
        lines = self._report().summary_lines()
        assert any("By EffSNR" in l for l in lines)

    def test_empty_report_has_empty_by_mode(self) -> None:
        d = DetailedEvalReport()
        payload = d.to_dict()
        assert payload["by_mode"] == {}
        assert payload["by_eff_snr"] == {}
        assert not any("By EffSNR" in l for l in d.summary_lines())
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_metrics.py::TestDetailedReportByModeEffSnr -v`
Expected: FAIL (`AttributeError: 'DetailedEvalReport' object has no attribute 'by_mode'`)

- [ ] **Step 3: by_mode メソッドを実装**

`src/train/metrics.py` の `DetailedEvalReport` クラス、`overall` プロパティの直後に追加:

```python
    def by_mode(self) -> dict[str, AggregateMetrics]:
        """モード別の集計 (サンプルの mode で分類)."""
        out: dict[str, AggregateMetrics] = {}
        for s in self.samples:
            key = s.mode or "unknown"
            out.setdefault(key, AggregateMetrics()).add(s.record)
        return out
```

- [ ] **Step 4: to_dict を拡張**

`DetailedEvalReport.to_dict()` の `return {...}` に、`"token_errors"` の行と `"samples"` の行の間へ新キーを追加 (既存キーは変更しない):

```python
            "by_eff_snr": {
                f"{b}": {"n_samples": m.n_samples, "ter": m.ter, "cer": m.cer}
                for b, m in sorted(self.report.by_eff_snr.items())
            },
            "by_mode": {
                k: {"n_samples": m.n_samples, "ter": m.ter, "cer": m.cer}
                for k, m in self.by_mode().items()
            },
```

- [ ] **Step 5: EvalReport.summary_lines に By EffSNR 節を追加**

`src/train/metrics.py` の `EvalReport.summary_lines()` 内、`by_wpm` を出力する `if self.by_wpm:` ブロックの直後に追加:

```python
        if self.by_eff_snr:
            lines.append("By EffSNR:")
            for eff in sorted(self.by_eff_snr):
                m = self.by_eff_snr[eff]
                lines.append(
                    f"  EffSNR={eff:+5.1f}dB n={m.n_samples:5d}  "
                    f"TER={m.ter * 100:6.2f}%  CER={m.cer * 100:6.2f}%"
                )
```

- [ ] **Step 6: テストを実行して成功を確認**

Run: `python -m pytest tests/test_metrics.py tests/test_metrics_errors.py -q`
Expected: PASS (既存 + 新規)

- [ ] **Step 7: コミット**

```bash
git add src/train/metrics.py tests/test_metrics.py
git commit -m "feat: DetailedEvalReport に by_eff_snr / by_mode 出力を追加"
```

---

### Task 2: report 比較の純関数 (`src/eval/compare.py`)

2 つの eval report dict を受けて section ごとの TER/CER delta を返す純関数。torch 非依存でテストする。

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/compare.py`
- Test: `tests/test_eval_compare.py`

**Interfaces:**
- Consumes: なし (dict 入力のみ)。入力 dict の形は Task 1 の `to_dict()` 出力 (`overall`/`by_eff_snr`/`by_mode`/`token_errors`) を section (`synth_val`/`keyed_val`) 下に持つ report。
- Produces: `compare_reports(baseline: dict, current: dict) -> dict` — section ごとに `overall`/`by_eff_snr`/`by_mode` の delta と `token_recall_improvements` を返す。

- [ ] **Step 1: 失敗するテストを書く**

新規 `tests/test_eval_compare.py`:

```python
"""report 比較ロジックのテスト."""
from __future__ import annotations

from src.eval.compare import compare_reports


def _report(overall_ter, eff, mode, tokens):
    return {
        "synth_val": {
            "overall": {"ter": overall_ter, "cer": overall_ter, "n_samples": 10},
            "by_eff_snr": eff,
            "by_mode": mode,
            "token_errors": tokens,
        }
    }


class TestCompareReports:
    def test_overall_ter_delta(self) -> None:
        base = _report(0.45, {}, {}, [])
        cur = _report(0.31, {}, {}, [])
        out = compare_reports(base, cur)
        d = out["synth_val"]["overall"]
        assert d["ter_baseline"] == 0.45
        assert d["ter_current"] == 0.31
        assert round(d["ter_delta"], 2) == -0.14   # 減少 = 改善

    def test_by_eff_snr_delta_and_missing_bin(self) -> None:
        base = _report(0.4, {"0.0": {"ter": 0.45}, "5.0": {"ter": 0.03}}, {}, [])
        cur = _report(0.3, {"0.0": {"ter": 0.22}}, {}, [])  # 5.0 が current に無い
        out = compare_reports(base, cur)["synth_val"]["by_eff_snr"]
        assert round(out["0.0"]["ter_delta"], 2) == -0.23
        assert out["5.0"]["ter_delta"] == "n/a"   # 欠損ビン

    def test_by_mode_delta(self) -> None:
        base = _report(0.4, {}, {"european": {"ter": 0.30}, "japanese": {"ter": 0.55}}, [])
        cur = _report(0.3, {}, {"european": {"ter": 0.20}, "japanese": {"ter": 0.40}}, [])
        out = compare_reports(base, cur)["synth_val"]["by_mode"]
        assert round(out["european"]["ter_delta"], 2) == -0.10
        assert round(out["japanese"]["ter_delta"], 2) == -0.15

    def test_token_recall_improvements_sorted(self) -> None:
        base = _report(0.4, {}, {}, [
            {"token_id": 10, "code": "・・・-・", "recall": 0.0},
            {"token_id": 11, "code": "・", "recall": 0.9},
        ])
        cur = _report(0.3, {}, {}, [
            {"token_id": 10, "code": "・・・-・", "recall": 0.6},
            {"token_id": 11, "code": "・", "recall": 0.92},
        ])
        imp = compare_reports(base, cur)["synth_val"]["token_recall_improvements"]
        # token 10 の改善 (+0.6) が token 11 (+0.02) より上位
        assert imp[0]["token_id"] == 10
        assert round(imp[0]["recall_delta"], 2) == 0.60

    def test_missing_section_skipped(self) -> None:
        # keyed_val が baseline に無い → keyed_val は結果に出ない
        base = _report(0.4, {}, {}, [])
        cur = {**_report(0.3, {}, {}, []), "keyed_val": {"overall": {"ter": 0.2}}}
        out = compare_reports(base, cur)
        assert "synth_val" in out
        assert "keyed_val" not in out
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_eval_compare.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.eval'`)

- [ ] **Step 3: パッケージと純関数を実装**

新規 `src/eval/__init__.py` (空):

```python
"""評価ハーネス (Phase B B0)."""
```

新規 `src/eval/compare.py`:

```python
"""eval report 同士の比較 (torch 非依存の純関数).

``scripts/eval_model.py`` が出力する report dict を 2 つ受け、改善前後の
TER/CER delta と token recall 改善を算出する。TER は減少が改善なので
``*_delta`` は負が改善を意味する。
"""
from __future__ import annotations

from typing import Any

_SECTIONS = ("synth_val", "keyed_val")


def _delta_metrics(base: dict | None, cur: dict | None) -> dict[str, Any]:
    """overall/mode/bin の TER・CER delta を作る. 片方欠損は 'n/a'."""
    if not base or not cur:
        return {"ter_delta": "n/a", "cer_delta": "n/a"}
    out: dict[str, Any] = {}
    for key in ("ter", "cer"):
        b = base.get(key)
        c = cur.get(key)
        if b is None or c is None:
            out[f"{key}_delta"] = "n/a"
        else:
            out[f"{key}_baseline"] = b
            out[f"{key}_current"] = c
            out[f"{key}_delta"] = c - b
    return out


def _delta_binned(base: dict, cur: dict) -> dict[str, Any]:
    """ビン (eff_snr / mode) ごとの delta. 欠損ビンは 'n/a'."""
    out: dict[str, Any] = {}
    for key in sorted(set(base) | set(cur)):
        out[key] = _delta_metrics(base.get(key), cur.get(key))
    return out


def _token_recall_delta(base: list[dict], cur: list[dict]) -> list[dict]:
    """token recall の改善を大きい順に返す."""
    base_by_id = {t["token_id"]: t for t in base}
    improvements: list[dict] = []
    for t in cur:
        tid = t["token_id"]
        b = base_by_id.get(tid)
        if b is None or "recall" not in b or "recall" not in t:
            continue
        improvements.append({
            "token_id": tid,
            "code": t.get("code"),
            "recall_baseline": b["recall"],
            "recall_current": t["recall"],
            "recall_delta": t["recall"] - b["recall"],
        })
    improvements.sort(key=lambda x: -x["recall_delta"])
    return improvements


def compare_reports(baseline: dict, current: dict) -> dict:
    """2 つの report dict を section ごとに比較する."""
    out: dict[str, Any] = {}
    for section in _SECTIONS:
        b = baseline.get(section)
        c = current.get(section)
        if b is None or c is None:
            continue
        out[section] = {
            "overall": _delta_metrics(b.get("overall"), c.get("overall")),
            "by_eff_snr": _delta_binned(b.get("by_eff_snr", {}), c.get("by_eff_snr", {})),
            "by_mode": _delta_binned(b.get("by_mode", {}), c.get("by_mode", {})),
            "token_recall_improvements": _token_recall_delta(
                b.get("token_errors", []), c.get("token_errors", [])
            ),
        }
    return out


__all__ = ["compare_reports"]
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_eval_compare.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: コミット**

```bash
git add src/eval/__init__.py src/eval/compare.py tests/test_eval_compare.py
git commit -m "feat: eval report 比較の純関数 compare_reports を追加"
```

---

### Task 3: 評価ハーネス (`src/eval/harness.py`)

decode の共通化と、keyed_val / synth_val 用の評価関数。既存 `evaluate_real` (finetune) と `inspect_real_labels` の decode ループをここに集約する。

**Files:**
- Create: `src/eval/harness.py`
- Test: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `CWModel`, `MelExtractor` (`.config.hop_length`), `compute_input_lengths(lengths, hop, max_frames)`, `ctc_greedy_decode(log_probs, input_lengths, blank_id) -> list[CTCDecodeResult]` (`.token_ids: list[int]`), `BLANK_TOKEN_ID`, `TokenConverter`, `EvalRecord`, `DetailedEvalReport`, `bin_snr`, `RealSignalDataset` (`.sample_at(i)`, `__getitem__` → `(wave, target)`), `RealNoiseEvalSample` (fields: `samples: np.ndarray`, `token_ids: np.ndarray`, `text`, `mode`, `eff_snr_db`).
- Produces:
  - `decode_wave(model, mel_extractor, wave: torch.Tensor, device) -> list[int]`
  - `evaluate_real_dataset(model, mel_extractor, dataset, device) -> DetailedEvalReport`
  - `evaluate_synth_noise(model, mel_extractor, samples: list[RealNoiseEvalSample], device) -> DetailedEvalReport`

- [ ] **Step 1: 失敗するテストを書く**

新規 `tests/test_eval_harness.py`:

```python
"""評価ハーネスのテスト (小さな seed 固定モデルで構造を検証)."""
from __future__ import annotations

import numpy as np
import torch

from src.eval.harness import decode_wave, evaluate_synth_noise
from src.synth.dataset import RealNoiseEvalSample
from src.train.model import CWModel, ModelConfig
from src.train.preprocessing import MelExtractor
from src.tokens.morse_tokens import VOCAB_SIZE


def _model_and_mel():
    torch.manual_seed(0)
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE))
    model.train(False)
    return model, MelExtractor()


class TestDecodeWave:
    def test_returns_token_id_list(self) -> None:
        model, mel = _model_and_mel()
        wave = torch.zeros(8000, dtype=torch.float32)  # 1 秒
        ids = decode_wave(model, mel, wave, torch.device("cpu"))
        assert isinstance(ids, list)
        assert all(isinstance(x, int) for x in ids)


class TestEvaluateSynthNoise:
    def _samples(self) -> list[RealNoiseEvalSample]:
        rng = np.random.default_rng(0)
        out = []
        for mode, eff in [("european", 10.0), ("european", 0.0), ("japanese", 10.0)]:
            out.append(RealNoiseEvalSample(
                samples=rng.standard_normal(8000).astype(np.float32) * 0.1,
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                text="ABC", mode=mode, wpm=20.0, target_snr_db=eff, eff_snr_db=eff,
            ))
        return out

    def test_structure_and_bins_populated(self) -> None:
        model, mel = _model_and_mel()
        report = evaluate_synth_noise(model, mel, self._samples(), torch.device("cpu"))
        assert report.overall.n_samples == 3
        # eff_snr ビン (10.0 が2件, 0.0 が1件) が集計される
        assert report.report.by_eff_snr[10.0].n_samples == 2
        assert report.report.by_eff_snr[0.0].n_samples == 1
        # モード別
        bm = report.by_mode()
        assert bm["european"].n_samples == 2
        assert bm["japanese"].n_samples == 1
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_eval_harness.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.eval.harness'`)

- [ ] **Step 3: harness を実装**

新規 `src/eval/harness.py`:

```python
"""再利用可能な評価ロジック (decode / keyed_val / synth_val).

``scripts/eval_model.py`` と ``scripts/finetune.py`` / ``scripts/inspect_real_labels.py``
が共有する。decode 経路を一本化して測定の一貫性を保つ。
"""
from __future__ import annotations

import torch

from src.synth.dataset import RealNoiseEvalSample
from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import BLANK_TOKEN_ID
from src.train.decode import ctc_greedy_decode
from src.train.loop import compute_input_lengths
from src.train.metrics import DetailedEvalReport, EvalRecord, bin_snr
from src.train.model import CWModel
from src.train.preprocessing import MelExtractor


@torch.no_grad()
def decode_wave(
    model: CWModel,
    mel_extractor: MelExtractor,
    wave: torch.Tensor,
    device: torch.device,
) -> list[int]:
    """波形 (1D tensor) を greedy CTC でデコードし token ID 列を返す."""
    t = wave.unsqueeze(0).to(device)
    log_probs = torch.nn.functional.log_softmax(model(mel_extractor(t)).float(), dim=-1)
    input_lengths = compute_input_lengths(
        torch.tensor([wave.numel()], device=device),
        mel_extractor.config.hop_length,
        log_probs.size(1),
    )
    return ctc_greedy_decode(log_probs, input_lengths, blank_id=BLANK_TOKEN_ID)[0].token_ids


@torch.no_grad()
def evaluate_real_dataset(
    model: CWModel,
    mel_extractor: MelExtractor,
    dataset,
    device: torch.device,
) -> DetailedEvalReport:
    """実信号データセット (keyed_val) を評価."""
    model.train(False)
    report = DetailedEvalReport()
    for i in range(len(dataset)):
        wave, target = dataset[i]
        meta = dataset.sample_at(i)
        pred_ids = decode_wave(model, mel_extractor, wave, device)
        pred_text = TokenConverter(mode=meta.mode, confidence_threshold=0.0).convert(pred_ids).text
        report.add(
            EvalRecord(
                ref_tokens=target.tolist(), pred_tokens=pred_ids,
                ref_text=meta.text, pred_text=pred_text,
            ),
            name=meta.stem, mode=meta.mode,
        )
    return report


@torch.no_grad()
def evaluate_synth_noise(
    model: CWModel,
    mel_extractor: MelExtractor,
    samples: list[RealNoiseEvalSample],
    device: torch.device,
) -> DetailedEvalReport:
    """実ノイズ混合の固定評価セット (synth_val) を評価. 実効SNR別に集計."""
    model.train(False)
    report = DetailedEvalReport()
    for s in samples:
        wave = torch.from_numpy(s.samples)
        pred_ids = decode_wave(model, mel_extractor, wave, device)
        pred_text = TokenConverter(mode=s.mode, confidence_threshold=0.0).convert(pred_ids).text
        report.add(
            EvalRecord(
                ref_tokens=s.token_ids.tolist(), pred_tokens=pred_ids,
                ref_text=s.text, pred_text=pred_text, eff_snr_db=s.eff_snr_db,
            ),
            name="", mode=s.mode, eff_snr_bin=bin_snr(s.eff_snr_db),
        )
    return report


__all__ = ["decode_wave", "evaluate_real_dataset", "evaluate_synth_noise"]
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `python -m pytest tests/test_eval_harness.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: コミット**

```bash
git add src/eval/harness.py tests/test_eval_harness.py
git commit -m "feat: 評価ハーネス harness (decode_wave / keyed / synth 評価) を追加"
```

---

### Task 4: 評価スクリプト (`scripts/eval_model.py`)

synth_val + keyed_val を評価し JSON 保存、`--baseline` で比較表示する CLI。

**Files:**
- Create: `scripts/eval_model.py`

**Interfaces:**
- Consumes: `harness.evaluate_real_dataset` / `harness.evaluate_synth_noise` (Task 3), `compare_reports` (Task 2), `make_fixed_real_noise_eval_set` (src.synth.dataset), `RealNoisePool.from_dir`, `discover_real_samples`, `RealSignalDataset`, `CWModel`/`ModelConfig`, `MelExtractor`, `load_checkpoint`, `VOCAB_SIZE`, `DetailedEvalReport.to_dict()`/`summary_lines()`。
- Produces: `scripts/eval_model.py::main(argv) -> int`。

- [ ] **Step 1: CLI を実装**

新規 `scripts/eval_model.py`:

```python
"""チェックポイント評価 CLI (Phase B B0).

任意の ckpt に対し synth_val (合成+実ノイズ、実効SNR別) と keyed_val (打鍵録音) を
評価し JSON 保存する。--baseline 指定で改善前後を比較表示する。

使い方::

    python scripts/eval_model.py --ckpt models/full/best.pt \\
        --noise-dir data/keying_scripts --keyed-dir data/keying_scripts \\
        --out models/eval/baseline.json

    python scripts/eval_model.py --ckpt models/ft/best.pt \\
        --noise-dir data/keying_scripts --keyed-dir data/real/val \\
        --out models/eval/ft.json --baseline models/eval/baseline.json
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

from src.eval.compare import compare_reports                                   # noqa: E402
from src.eval.harness import evaluate_real_dataset, evaluate_synth_noise       # noqa: E402
from src.finetune.dataset import RealSignalDataset, discover_real_samples      # noqa: E402
from src.synth.dataset import make_fixed_real_noise_eval_set                   # noqa: E402
from src.synth.noise import RealNoisePool                                      # noqa: E402
from src.tokens.morse_tokens import VOCAB_SIZE                                 # noqa: E402
from src.train.checkpoint import load_checkpoint                              # noqa: E402
from src.train.metrics import DetailedEvalReport, bin_snr                     # noqa: E402
from src.train.model import CWModel, ModelConfig                             # noqa: E402
from src.train.preprocessing import MelExtractor                            # noqa: E402


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="チェックポイント評価 (synth_val + keyed_val)")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--noise-dir", type=Path, default=None, help="synth_val 用ノイズ WAV ディレクトリ")
    p.add_argument("--keyed-dir", type=Path, default=None, help="keyed_val 用 WAV+TXT ディレクトリ")
    p.add_argument("--out", type=Path, default=Path("models/eval/eval.json"))
    p.add_argument("--baseline", type=Path, default=None, help="比較元 JSON")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--tone-center", type=float, default=494.0)
    p.add_argument("--bpf-bandwidth", type=float, default=300.0)
    p.add_argument("--wpm", type=float, nargs="+", default=[17.0, 25.0])
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0, -5.0])
    p.add_argument("--samples-per-cell", type=int, default=25)
    return p


def _section_dict(report) -> dict:
    d = report.to_dict()
    d.pop("samples", None)  # synth_val はサンプル別を含めない (件数が多い)
    d["confusion"] = report.analysis.confusion_to_dict()
    return d


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE)).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.train(False)
    mel = MelExtractor().to(device)

    report: dict = {"ckpt": str(args.ckpt), "seed": args.seed}

    # ---- synth_val ----
    if args.noise_dir is not None:
        pool = RealNoisePool.from_dir(args.noise_dir)
        synth_report = DetailedEvalReport()   # 欧文/和文を 1 レポートに統合
        for mode in ("european", "japanese"):
            samples = make_fixed_real_noise_eval_set(
                noise_pool=pool, snr_grid=args.snr, wpm_grid=args.wpm,
                samples_per_cell=args.samples_per_cell, seed=args.seed, mode=mode,
                tone_center_hz=args.tone_center, filter_bandwidth_hz=args.bpf_bandwidth,
            )
            r = evaluate_synth_noise(model, mel, samples, device)
            synth_report = _merge_reports(synth_report, r)
        section = _section_dict(synth_report)
        section["config"] = {
            "tone_center_hz": args.tone_center, "bpf_bandwidth_hz": args.bpf_bandwidth,
            "wpm_grid": args.wpm, "snr_grid": args.snr,
            "samples_per_cell": args.samples_per_cell,
        }
        report["synth_val"] = section
        for line in synth_report.summary_lines():
            print(f"[synth] {line}", flush=True)

    # ---- keyed_val ----
    if args.keyed_dir is not None:
        samples = discover_real_samples(args.keyed_dir)
        if samples:
            dataset = RealSignalDataset(samples)
            keyed_report = evaluate_real_dataset(model, mel, dataset, device)
            report["keyed_val"] = keyed_report.to_dict() | {
                "confusion": keyed_report.analysis.confusion_to_dict()
            }
            for line in keyed_report.summary_lines():
                print(f"[keyed] {line}", flush=True)
        else:
            print(f"[warn] keyed-dir にサンプルがありません: {args.keyed_dir}", flush=True)

    if "synth_val" not in report and "keyed_val" not in report:
        print("[err] synth_val も keyed_val も評価できませんでした", flush=True)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[out] {args.out}", flush=True)

    # ---- baseline 比較 ----
    if args.baseline is not None:
        try:
            base = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] baseline を読めません ({exc}) — 比較をスキップ", flush=True)
        else:
            _print_comparison(compare_reports(base, report))
    return 0


def _merge_reports(a: DetailedEvalReport, b: DetailedEvalReport) -> DetailedEvalReport:
    """2 つの DetailedEvalReport を統合 (samples を足し合わせ、eff_snr ビンを再構築)."""
    merged = DetailedEvalReport()
    for src_report in (a, b):
        for s in src_report.samples:
            eff_bin = None if s.record.eff_snr_db is None else bin_snr(s.record.eff_snr_db)
            merged.add(s.record, name=s.name, mode=s.mode, eff_snr_bin=eff_bin)
    return merged


def _print_comparison(cmp: dict) -> None:
    for section, data in cmp.items():
        print(f"\n=== {section}: baseline → current ===", flush=True)
        ov = data["overall"]
        if "ter_baseline" in ov:
            print(f"Overall  TER {ov['ter_baseline']*100:.1f}% → {ov['ter_current']*100:.1f}% "
                  f"({ov['ter_delta']*100:+.1f})", flush=True)
        if data["by_eff_snr"]:
            print("By EffSNR:", flush=True)
            for b, d in sorted(data["by_eff_snr"].items(), key=lambda kv: float(kv[0])):
                delta = d["ter_delta"]
                s = "n/a" if delta == "n/a" else f"{delta*100:+.1f}"
                print(f"  {b:>6}dB  TER delta {s}", flush=True)
        imp = data["token_recall_improvements"][:5]
        if imp:
            print("Top token recall improvements:", flush=True)
            for t in imp:
                print(f"  {t.get('code')}: recall {t['recall_baseline']*100:.0f}% "
                      f"→ {t['recall_current']*100:.0f}% ({t['recall_delta']*100:+.0f})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: CLI が import できることを確認**

Run: `python -c "import sys; sys.path.insert(0, '.'); import scripts.eval_model; print('ok')"`
Expected: `ok`

- [ ] **Step 3: 実データでスモークテスト (存在すれば)**

Run: `PYTHONIOENCODING=utf-8 python scripts/eval_model.py --ckpt models/full/best_infer.pt --noise-dir data/keying_scripts --keyed-dir data/keying_scripts --out models/eval/smoke.json --device cpu --samples-per-cell 5`
Expected: `[synth]` と `[keyed]` のサマリが出て `models/eval/smoke.json` が生成される。synth_val の By EffSNR で 0dB 付近の TER が +5dB より高い (崖が再現)。`models/full/best_infer.pt` や `data/keying_scripts` の wav が無ければその旨報告 (スモークはスキップ可、import 確認が gate)。

- [ ] **Step 4: baseline 比較のスモーク (Step 3 が成功した場合)**

Run: `PYTHONIOENCODING=utf-8 python scripts/eval_model.py --ckpt models/full/best_infer.pt --noise-dir data/keying_scripts --out models/eval/smoke2.json --device cpu --samples-per-cell 5 --baseline models/eval/smoke.json`
Expected: `=== synth_val: baseline → current ===` 比較表が出る (同 ckpt なので delta ≈ 0)。

- [ ] **Step 5: スモーク生成物を削除してコミット**

```bash
rm -f models/eval/smoke.json models/eval/smoke2.json
git add scripts/eval_model.py
git commit -m "feat: チェックポイント評価スクリプト eval_model を追加"
```

---

### Task 5: finetune / inspect_real_labels の decode 委譲

decode ループを `harness` に集約し重複を解消する。挙動は不変。

**Files:**
- Modify: `scripts/finetune.py`
- Modify: `scripts/inspect_real_labels.py`

**Interfaces:**
- Consumes: `harness.evaluate_real_dataset`, `harness.decode_wave` (Task 3)。
- Produces: なし (内部委譲のみ)。

- [ ] **Step 1: finetune の evaluate_real を委譲**

`scripts/finetune.py` の `evaluate_real` 関数全体 (`@torch.no_grad()` から `return report` まで) を次に置換:

```python
from src.eval.harness import evaluate_real_dataset  # noqa: E402  (ファイル冒頭の import 群へ移動)


def evaluate_real(
    model: CWModel,
    mel_extractor: MelExtractor,
    eval_dataset: RealSignalDataset,
    device: torch.device,
) -> DetailedEvalReport:
    """実信号評価セットに対し TER/CER と token 別エラーを集計 (harness へ委譲)."""
    return evaluate_real_dataset(model, mel_extractor, eval_dataset, device)
```

実際には `from src.eval.harness import evaluate_real_dataset` はファイル冒頭の他の `# noqa: E402` import 群に置くこと (関数内 import にしない)。置換後、finetune が未使用になった import (`ctc_greedy_decode`, `compute_input_lengths`, `TokenConverter`, `BLANK_TOKEN_ID` 等) がある場合、他で使われていなければ削除する。使われているかは `grep` で確認してから消す。

- [ ] **Step 2: finetune テストが緑のままか確認**

Run: `python -m pytest tests/test_finetune_dataset.py tests/test_train_loop.py -q`
Expected: PASS (既存のまま)

- [ ] **Step 3: inspect_real_labels の decode を委譲**

`scripts/inspect_real_labels.py` の decode 行:

```python
        t = wave.unsqueeze(0).to(device)
        lp = torch.nn.functional.log_softmax(model(mel(t)).float(), dim=-1)
        il = compute_input_lengths(
            torch.tensor([wave.numel()], device=device), hop, lp.size(1)
        )
        res = ctc_greedy_decode(lp, il, blank_id=BLANK_TOKEN_ID)[0]
        pred_text = TokenConverter(mode=meta.mode, confidence_threshold=0.0).convert(
            res.token_ids
        ).text
```

を次に置換:

```python
        pred_ids = decode_wave(model, mel, wave, device)
        pred_text = TokenConverter(mode=meta.mode, confidence_threshold=0.0).convert(
            pred_ids
        ).text
```

そして下の `EvalRecord(...)` の `pred_tokens=res.token_ids` を `pred_tokens=pred_ids` に変える。冒頭に `from src.eval.harness import decode_wave  # noqa: E402` を追加。`hop = mel.config.hop_length` と、未使用になる import (`ctc_greedy_decode`, `compute_input_lengths`, `BLANK_TOKEN_ID`, `numpy as np` — 既に未使用) を、他で使われていなければ削除する。

- [ ] **Step 4: inspect_real_labels が import できるか確認**

Run: `python -c "import sys; sys.path.insert(0, '.'); import scripts.inspect_real_labels; print('ok')"`
Expected: `ok`

- [ ] **Step 5: 委譲後のスモーク (実データがあれば)**

Run: `PYTHONIOENCODING=utf-8 python scripts/inspect_real_labels.py --data-dir data/keying_scripts --ckpt models/full/best_infer.pt --device cpu`
Expected: 従来通り TER 降順リスト + 符号長別 recall が出る。データが無ければその旨報告。

- [ ] **Step 6: コミット**

```bash
git add scripts/finetune.py scripts/inspect_real_labels.py
git commit -m "refactor: finetune / inspect_real_labels の decode を harness へ集約"
```

---

### Task 6: 全体テストと最終確認

**Files:** なし (確認のみ)

**Interfaces:**
- Consumes: 全タスクの成果物
- Produces: なし

- [ ] **Step 1: 全テストを実行**

Run: `python -m pytest -q`
Expected: PASS (既存 562 + 新規、失敗 0)

- [ ] **Step 2: eval_model と inspect が import できることを最終確認**

Run: `python -c "import sys; sys.path.insert(0, '.'); import scripts.eval_model, scripts.inspect_real_labels, scripts.finetune; print('all import ok')"`
Expected: `all import ok`

- [ ] **Step 3: 変更ファイルとコミット一覧を確認**

Run: `git log --oneline feature/b0-eval-harness -8`
Expected: Task 1〜5 のコミットが並ぶ。

Run: `git status --short`
Expected: 追跡対象の未コミット変更なし (`models/eval/` のスモーク生成物は削除済み)。

---

## Self-Review 記録

**Spec coverage (設計書 §4-§7 との対応):**
- §4.1 harness (decode_wave / evaluate_real_dataset / evaluate_synth_noise) → Task 3
- §4.2 eval_model.py CLI → Task 4
- §4.3 to_dict/summary_lines 拡張 (by_eff_snr/by_mode) → Task 1
- §4.4 finetune 委譲 + inspect_real_labels 集約 → Task 5
- §4.5 compare_reports → Task 2
- §5 出力 JSON (synth は samples 除外, keyed は含む) → Task 4 `_section_dict`
- §6 テスト方針 → 各 Task の TDD + Task 4 スモーク
- §7 エラー処理 (noise-dir 空→exit2, keyed 空→skip, baseline 壊れ→skip) → Task 4 main

**型整合:** `decode_wave -> list[int]` (Task 3 定義) を Task 4/5 が使用。`evaluate_synth_noise`/`evaluate_real_dataset` (Task 3) を Task 4 が使用。`compare_reports(baseline, current) -> dict` (Task 2) を Task 4 が使用。`DetailedEvalReport.by_mode()` / `to_dict()` 新キー (Task 1) を Task 4 の JSON が前提。すべて定義タスクが使用タスクより前。

**注意 (スモーク依存):** Task 4 Step 3/4 のスモークは `models/full/best_infer.pt` と `data/keying_scripts` の wav (gitignore 済みローカル) に依存する。CI や wav 不在環境ではスモーク不可 → import 確認 (Step 2) が gate、スモークは best-effort でその旨報告。

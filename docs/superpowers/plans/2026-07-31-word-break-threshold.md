# WORD_BREAK 抑制パラメータ掃引 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 再学習せずに、CTC デコード側の 2 パラメータ（WORD_BREAK ロジットバイアス β・確信度閾値 τ）を掃引して keyed_val TER を下げられるか判定し、下げられる場合のみ評価経路に恒久実装する。

**Architecture:** 純関数モジュール `src/infer/word_break_policy.py` に β 適用と τ フィルタを置く。掃引 CLI は推論を 1 サンプル 1 回だけ実行して `log_probs` を CPU にキャッシュし、以降のグリッド探索を純後処理として回す。採否基準を満たしたときだけ `src/eval/harness.py` の `decode_wave` にポリシー引数を追加する。

**Tech Stack:** Python 3.11+ / PyTorch / numpy / pytest

## Global Constraints

- 設計書: `docs/superpowers/specs/2026-07-31-word-break-threshold-design.md`
- 作業ブランチ: `feature/word-break-threshold`（設計書コミット 8142e4f が既に載っている）
- 言語: コメント・docstring・コミットメッセージはすべて日本語
- 文字コード: UTF-8 (BOM なし)、改行 LF
- 型ヒント必須（mypy 互換）。不変データは `@dataclass(frozen=True)`
- トークン ID は `src.tokens.morse_tokens.WORD_BREAK_TOKEN_ID` を import して使う。`72` をハードコードしない
- 乱数は `torch.Generator` / `np.random.Generator` を明示的に渡す。グローバル乱数を使わない
- baseline 実測値（`models/eval/base_valnoise.json`、`models/full/best.pt`）:
  - keyed_val TER **23.35%**（参照 471 トークン、S=34 / D=23 / I=53、うち WORD_BREAK 挿入 33）
  - synth_val overall **48.06%**
- `pytest` 全実行は全パス後に `0xC0000409` で落ちサマリ行が出ない既知の環境問題がある。合否は `FAILED` / `ERROR` の有無で判定する
- テストは既存の書き方に合わせる（`from __future__ import annotations`、クラスでグループ化、日本語 docstring、`-> None` 付き）

---

### Task 1: WordBreakPolicy と apply_logit_bias

**Files:**
- Create: `cw-decorder/src/infer/word_break_policy.py`
- Test: `cw-decorder/tests/test_word_break_policy.py`

**Interfaces:**
- Consumes: `src.tokens.morse_tokens.WORD_BREAK_TOKEN_ID` (int, 値 72), `VOCAB_SIZE` (int, 値 73)
- Produces:
  - `WordBreakPolicy(logit_bias: float = 0.0, conf_threshold: float = 0.0)` — frozen dataclass、`is_identity: bool` プロパティ付き
  - `apply_logit_bias(log_probs: Tensor, bias: float) -> Tensor`

- [ ] **Step 1: 失敗するテストを書く**

`cw-decorder/tests/test_word_break_policy.py` を新規作成:

```python
"""WORD_BREAK 抑制ポリシーのテスト."""
from __future__ import annotations

import pytest
import torch

from src.infer.word_break_policy import WordBreakPolicy, apply_logit_bias
from src.tokens.morse_tokens import VOCAB_SIZE, WORD_BREAK_TOKEN_ID


def _log_probs(t: int = 4, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    """``(1, t, vocab)`` の決定的な log-softmax を作る."""
    g = torch.Generator().manual_seed(20260731)
    return torch.log_softmax(torch.randn(1, t, vocab, generator=g), dim=-1)


class TestWordBreakPolicy:
    def test_default_is_identity(self) -> None:
        assert WordBreakPolicy().is_identity is True

    def test_nonzero_bias_is_not_identity(self) -> None:
        assert WordBreakPolicy(logit_bias=-1.0).is_identity is False

    def test_nonzero_threshold_is_not_identity(self) -> None:
        assert WordBreakPolicy(conf_threshold=0.5).is_identity is False

    def test_rejects_nan_bias(self) -> None:
        with pytest.raises(ValueError, match="logit_bias"):
            WordBreakPolicy(logit_bias=float("nan"))

    def test_rejects_inf_bias(self) -> None:
        with pytest.raises(ValueError, match="logit_bias"):
            WordBreakPolicy(logit_bias=float("-inf"))

    def test_rejects_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="conf_threshold"):
            WordBreakPolicy(conf_threshold=1.5)

    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="conf_threshold"):
            WordBreakPolicy(conf_threshold=-0.1)


class TestApplyLogitBias:
    def test_zero_bias_returns_input_unchanged(self) -> None:
        lp = _log_probs()
        assert torch.equal(apply_logit_bias(lp, 0.0), lp)

    def test_negative_bias_lowers_word_break_probability(self) -> None:
        lp = _log_probs()
        out = apply_logit_bias(lp, -3.0)
        assert (out[..., WORD_BREAK_TOKEN_ID] < lp[..., WORD_BREAK_TOKEN_ID]).all()

    def test_preserves_ratio_between_other_tokens(self) -> None:
        """WORD_BREAK 以外のトークン同士の確率比は歪まない."""
        lp = _log_probs()
        out = apply_logit_bias(lp, -3.0)
        # log 空間での差 = 確率比の log。id 1 と 2 は WORD_BREAK ではない
        before = lp[..., 1] - lp[..., 2]
        after = out[..., 1] - out[..., 2]
        assert torch.allclose(before, after, atol=1e-6)

    def test_output_is_normalized(self) -> None:
        out = apply_logit_bias(_log_probs(), -2.0)
        assert torch.allclose(out.exp().sum(dim=-1), torch.ones(1, 4), atol=1e-6)

    def test_positive_bias_is_allowed(self) -> None:
        """掃引の対称性確認に使うため β>0 も通す."""
        lp = _log_probs()
        out = apply_logit_bias(lp, 2.0)
        assert (out[..., WORD_BREAK_TOKEN_ID] > lp[..., WORD_BREAK_TOKEN_ID]).all()

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            apply_logit_bias(torch.zeros(4, VOCAB_SIZE), -1.0)

    def test_rejects_vocab_without_word_break(self) -> None:
        """WORD_BREAK 追加前の古い ckpt を黙って処理しない."""
        with pytest.raises(ValueError, match="WORD_BREAK"):
            apply_logit_bias(torch.log_softmax(torch.zeros(1, 4, 10), dim=-1), -1.0)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_word_break_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.infer.word_break_policy'`

- [ ] **Step 3: 最小実装を書く**

`cw-decorder/src/infer/word_break_policy.py` を新規作成:

```python
"""WORD_BREAK トークンの抑制ポリシー (再学習なしのデコード側レバー).

keyed_val の誤り 110 件のうち 33 件が WORD_BREAK の過剰出力であるため、
argmax **前** のロジットバイアス β と argmax **後** の確信度閾値 τ の 2 つで
過剰出力を抑える。設計は
``docs/superpowers/specs/2026-07-31-word-break-threshold-design.md`` を参照.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.tokens.morse_tokens import WORD_BREAK_TOKEN_ID


@dataclass(frozen=True)
class WordBreakPolicy:
    """WORD_BREAK 抑制のパラメータ.

    Attributes:
        logit_bias: argmax 前に WORD_BREAK の log 確率へ加算する値 (nat).
            負で抑制、0 で無効. 掃引の対称性確認に使うため正も許容する.
        conf_threshold: argmax 後、この確信度**未満**の WORD_BREAK を除去する.
            0.0 で無効.
    """

    logit_bias: float = 0.0
    conf_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.logit_bias):
            raise ValueError(f"logit_bias must be finite, got {self.logit_bias}")
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ValueError(
                f"conf_threshold must be in [0, 1], got {self.conf_threshold}"
            )

    @property
    def is_identity(self) -> bool:
        """何もしないポリシーか (現行挙動と完全一致する)."""
        return self.logit_bias == 0.0 and self.conf_threshold == 0.0


def apply_logit_bias(log_probs: Tensor, bias: float) -> Tensor:
    """``(B, T, V)`` log-softmax の WORD_BREAK 列に ``bias`` を加えて再正規化.

    WORD_BREAK 以外のトークン同士の確率比は保存される (同じ定数で割るため).

    Args:
        log_probs: log-softmax 済みの ``(B, T, V)``.
        bias: WORD_BREAK の log 確率へ加算する値 (nat). 負で抑制.

    Returns:
        再正規化済みの ``(B, T, V)``. ``bias == 0.0`` なら入力をそのまま返す.

    Raises:
        ValueError: 次元数が 3 でない / bias が非有限 /
            語彙に WORD_BREAK が含まれない (WORD_BREAK 追加前の古い ckpt).
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be 3D, got {tuple(log_probs.shape)}")
    if not math.isfinite(bias):
        raise ValueError(f"bias must be finite, got {bias}")
    if bias == 0.0:
        return log_probs
    vocab = log_probs.size(-1)
    if vocab <= WORD_BREAK_TOKEN_ID:
        raise ValueError(
            f"語彙サイズ {vocab} に WORD_BREAK (id={WORD_BREAK_TOKEN_ID}) が含まれない. "
            "WORD_BREAK 追加前の古いチェックポイントの可能性がある"
        )
    biased = log_probs.clone()
    biased[..., WORD_BREAK_TOKEN_ID] += bias
    return torch.log_softmax(biased, dim=-1)


__all__ = ["WordBreakPolicy", "apply_logit_bias"]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_word_break_policy.py -q`
Expected: PASS（14 テスト）

- [ ] **Step 5: コミット**

```bash
git add cw-decorder/src/infer/word_break_policy.py cw-decorder/tests/test_word_break_policy.py
git commit -m "feat: WORD_BREAK ロジットバイアス (apply_logit_bias) を追加"
```

---

### Task 2: filter_word_breaks

**Files:**
- Modify: `cw-decorder/src/infer/word_break_policy.py`（末尾に関数追加、`__all__` 更新）
- Test: `cw-decorder/tests/test_word_break_policy.py`（クラス追加）

**Interfaces:**
- Consumes: `src.train.decode.CTCDecodeResult`（`token_ids: list[int]`, `confidences: list[float]` を持つ dataclass）、Task 1 の `WORD_BREAK_TOKEN_ID`
- Produces: `filter_word_breaks(result: CTCDecodeResult, threshold: float) -> CTCDecodeResult`

- [ ] **Step 1: 失敗するテストを書く**

`cw-decorder/tests/test_word_break_policy.py` の import 行を差し替え、末尾にクラスを追加:

```python
from src.infer.word_break_policy import (
    WordBreakPolicy,
    apply_logit_bias,
    filter_word_breaks,
)
from src.train.decode import CTCDecodeResult
```

```python
class TestFilterWordBreaks:
    @staticmethod
    def _result() -> CTCDecodeResult:
        """WORD_BREAK 2 個 (低確信度 0.2 / 高確信度 0.9) と通常トークン 2 個."""
        return CTCDecodeResult(
            token_ids=[1, WORD_BREAK_TOKEN_ID, 2, WORD_BREAK_TOKEN_ID],
            confidences=[0.15, 0.2, 0.3, 0.9],
        )

    def test_zero_threshold_returns_input(self) -> None:
        r = self._result()
        assert filter_word_breaks(r, 0.0) is r

    def test_removes_low_confidence_word_break(self) -> None:
        out = filter_word_breaks(self._result(), 0.5)
        assert out.token_ids == [1, 2, WORD_BREAK_TOKEN_ID]

    def test_keeps_high_confidence_word_break(self) -> None:
        out = filter_word_breaks(self._result(), 0.5)
        assert out.token_ids[-1] == WORD_BREAK_TOKEN_ID

    def test_keeps_low_confidence_non_word_break(self) -> None:
        """確信度が低くても WORD_BREAK 以外は残す."""
        out = filter_word_breaks(self._result(), 0.5)
        assert 1 in out.token_ids and 2 in out.token_ids

    def test_confidences_stay_aligned(self) -> None:
        out = filter_word_breaks(self._result(), 0.5)
        assert len(out.token_ids) == len(out.confidences)
        assert out.confidences == [0.15, 0.3, 0.9]

    def test_removes_all_when_threshold_is_one(self) -> None:
        out = filter_word_breaks(self._result(), 1.0)
        assert WORD_BREAK_TOKEN_ID not in out.token_ids

    def test_does_not_mutate_input(self) -> None:
        r = self._result()
        filter_word_breaks(r, 0.5)
        assert len(r.token_ids) == 4

    def test_rejects_out_of_range_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            filter_word_breaks(self._result(), 1.5)

    def test_rejects_length_mismatch(self) -> None:
        bad = CTCDecodeResult(token_ids=[1, 2], confidences=[0.5])
        with pytest.raises(ValueError):
            filter_word_breaks(bad, 0.5)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_word_break_policy.py::TestFilterWordBreaks -q`
Expected: FAIL — `ImportError: cannot import name 'filter_word_breaks'`

- [ ] **Step 3: 最小実装を書く**

`cw-decorder/src/infer/word_break_policy.py` の import に追加:

```python
from src.train.decode import CTCDecodeResult
```

`__all__` の直前に追加:

```python
def filter_word_breaks(result: CTCDecodeResult, threshold: float) -> CTCDecodeResult:
    """確信度が ``threshold`` 未満の WORD_BREAK **だけ** を除去する.

    WORD_BREAK 以外のトークンは確信度が低くても残す. 入力は変更しない.

    Args:
        result: ``ctc_greedy_decode`` の 1 サンプル分の結果.
        threshold: [0, 1]. 0.0 なら入力をそのまま返す.

    Returns:
        除去後の新しい ``CTCDecodeResult``.

    Raises:
        ValueError: threshold が [0, 1] 外 / token_ids と confidences の長さ不一致.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if len(result.token_ids) != len(result.confidences):
        raise ValueError(
            f"token_ids length {len(result.token_ids)} != confidences length "
            f"{len(result.confidences)}"
        )
    if threshold == 0.0:
        return result
    ids: list[int] = []
    confs: list[float] = []
    for tid, conf in zip(result.token_ids, result.confidences, strict=True):
        if tid == WORD_BREAK_TOKEN_ID and conf < threshold:
            continue
        ids.append(tid)
        confs.append(conf)
    return CTCDecodeResult(token_ids=ids, confidences=confs)
```

`__all__` を更新:

```python
__all__ = ["WordBreakPolicy", "apply_logit_bias", "filter_word_breaks"]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_word_break_policy.py -q`
Expected: PASS（23 テスト）

- [ ] **Step 5: コミット**

```bash
git add cw-decorder/src/infer/word_break_policy.py cw-decorder/tests/test_word_break_policy.py
git commit -m "feat: WORD_BREAK 確信度フィルタ (filter_word_breaks) を追加"
```

---

### Task 3: 掃引スクリプト — log_probs キャッシュと Step 0 診断

**Files:**
- Create: `cw-decorder/scripts/sweep_word_break.py`

**Interfaces:**
- Consumes:
  - Task 1/2 の `WordBreakPolicy`, `apply_logit_bias`, `filter_word_breaks`
  - `src.finetune.dataset.discover_real_samples(root) -> list[RealSignalSample]`, `RealSignalDataset(samples)`（`dataset[i] -> (Tensor, Tensor)`, `dataset.sample_at(i) -> meta`（`.stem`, `.mode`, `.text`））
  - `src.train.loop.compute_input_lengths(wave_lengths, hop_length, max_frames) -> Tensor`
  - `src.infer.engine.ctc_greedy_decode_with_frames(log_probs) -> list[list[FrameToken]]`（`FrameToken` は `token_id`, `confidence`, `frame_start`, `frame_end`）
  - `src.train.metrics.align_sequences(pred, ref) -> list[EditOp]`（`EditOp.kind`, `EditOp.pred_index`）
- Produces:
  - `CachedSample` — frozen dataclass（`log_probs: Tensor` (T,V) CPU, `ref_tokens: list[int]`, `ref_text: str`, `mode: str`, `name: str`, `eff_snr_db: float | None`）
  - `cache_keyed_samples(model, mel_extractor, dataset, device) -> list[CachedSample]`
  - `diagnose(samples: list[CachedSample]) -> dict[str, Any]`

- [ ] **Step 1: スクリプトの骨格とキャッシュ層を書く**

`cw-decorder/scripts/sweep_word_break.py` を新規作成:

```python
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
from src.tokens.morse_tokens import (                                       # noqa: E402
    BLANK_TOKEN_ID,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
)
from src.train.decode import ctc_greedy_decode                              # noqa: E402
from src.train.loop import compute_input_lengths                            # noqa: E402
from src.train.metrics import align_sequences                               # noqa: E402
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


def decode_cached(log_probs: torch.Tensor, policy: WordBreakPolicy) -> list[int]:
    """キャッシュ済み ``(T, V)`` にポリシーを適用してトークン ID 列を返す."""
    lp = apply_logit_bias(log_probs.unsqueeze(0), policy.logit_bias)
    result = ctc_greedy_decode(lp, blank_id=BLANK_TOKEN_ID)[0]
    return filter_word_breaks(result, policy.conf_threshold).token_ids
```

- [ ] **Step 2: Step 0 診断を書く**

同ファイルに追加:

```python
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
```

- [ ] **Step 3: 診断のみを実行できる CLI を書く**

同ファイルに追加:

```python
def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WORD_BREAK 抑制パラメータの掃引")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--keyed-dir", type=Path, required=True, help="keyed_val の WAV+TXT ディレクトリ")
    p.add_argument("--out", type=Path, default=Path("models/eval/word_break_sweep.json"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--diagnose-only", action="store_true", help="Step 0 診断だけ実行する")
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

    print("[info] グリッド掃引は Task 4 で実装する", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 診断を実行して動作確認**

Run:
```bash
python scripts/sweep_word_break.py --ckpt models/full/best.pt \
    --keyed-dir data/keying_scripts --diagnose-only
```
Expected: `[cache] keyed_val 20 件をキャッシュ` に続いて `[diag] 正解WB n=...` と `[diag] 偽陽性WB n=...` の 2 行が出る。偽陽性WB の n は 33 前後（baseline の WORD_BREAK 挿入数）になるはず。大きく外れる場合はアラインメントの解釈が誤っているので先に調べる。

- [ ] **Step 5: コミット**

```bash
git add cw-decorder/scripts/sweep_word_break.py
git commit -m "feat: WORD_BREAK 掃引スクリプトの推論キャッシュと Step 0 診断"
```

---

### Task 4: グリッド掃引と JSON 出力

**Files:**
- Modify: `cw-decorder/scripts/sweep_word_break.py`

**Interfaces:**
- Consumes: Task 3 の `CachedSample`, `decode_cached`, `diagnose`
- Produces:
  - `evaluate_cached(samples: list[CachedSample], policy: WordBreakPolicy) -> DetailedEvalReport`
  - `sweep(samples, betas, taus) -> list[dict[str, Any]]` — 各要素は `{"beta", "tau", "ter", "substitutions", "deletions", "insertions", "word_break_insertions"}`
  - `BETA_GRID: list[float]`, `TAU_GRID: list[float]`

- [ ] **Step 1: 評価関数と掃引ループを書く**

`cw-decorder/scripts/sweep_word_break.py` の import に追加:

```python
from src.tokens.converter import TokenConverter                             # noqa: E402
from src.train.metrics import DetailedEvalReport, EvalRecord, bin_snr       # noqa: E402
```

`build_args` の直前に追加:

```python
# 設計書 §5 Step 1 のグリッド
BETA_GRID: list[float] = [0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0]
TAU_GRID: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


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
```

- [ ] **Step 2: 結果表示と JSON 出力を書く**

同ファイルに追加:

```python
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
```

`main` の `if args.diagnose_only: return 0` の後を差し替え:

```python
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
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[out] {args.out}", flush=True)
    return 0
```

- [ ] **Step 3: スモークテストを書く**

`cw-decorder/tests/test_sweep_word_break.py` を新規作成:

```python
"""掃引スクリプトのスモークテスト (推論なし、キャッシュを直接組み立てる)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from src.infer.word_break_policy import WordBreakPolicy
from src.tokens.morse_tokens import VOCAB_SIZE, WORD_BREAK_TOKEN_ID

_SPEC = importlib.util.spec_from_file_location(
    "sweep_word_break",
    Path(__file__).resolve().parent.parent / "scripts" / "sweep_word_break.py",
)
assert _SPEC is not None and _SPEC.loader is not None
sweep_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sweep_mod)


def _cached(token_seq: list[int]) -> "sweep_mod.CachedSample":
    """各時刻で 1 トークンを強く出す log_probs を組み立てる."""
    t = len(token_seq)
    logits = torch.full((t, VOCAB_SIZE), -10.0)
    for j, tok in enumerate(token_seq):
        logits[j, tok] = 0.0
    return sweep_mod.CachedSample(
        log_probs=torch.log_softmax(logits, dim=-1),
        ref_tokens=[1, 2],
        ref_text="AB",
        mode="european",
        name="smoke",
    )


class TestSweepSmoke:
    def test_decode_cached_identity_matches_plain_decode(self) -> None:
        c = _cached([1, 0, 2])
        assert sweep_mod.decode_cached(c.log_probs, WordBreakPolicy()) == [1, 2]

    def test_bias_removes_word_break(self) -> None:
        c = _cached([1, 0, WORD_BREAK_TOKEN_ID, 0, 2])
        with_wb = sweep_mod.decode_cached(c.log_probs, WordBreakPolicy())
        assert WORD_BREAK_TOKEN_ID in with_wb
        without = sweep_mod.decode_cached(
            c.log_probs, WordBreakPolicy(logit_bias=-20.0)
        )
        assert WORD_BREAK_TOKEN_ID not in without

    def test_sweep_returns_one_point_per_grid_cell(self) -> None:
        points = sweep_mod.sweep([_cached([1, 0, 2])], [0.0, -1.0], [0.0, 0.5])
        assert len(points) == 4
        assert {(p["beta"], p["tau"]) for p in points} == {
            (0.0, 0.0), (0.0, 0.5), (-1.0, 0.0), (-1.0, 0.5)
        }

    def test_neighbors_are_within_grid(self) -> None:
        points = sweep_mod.sweep([_cached([1, 0, 2])], [0.0, -1.0], [0.0, 0.5])
        corner = next(p for p in points if p["beta"] == 0.0 and p["tau"] == 0.0)
        nb = sweep_mod.neighbors(points, corner)
        assert len(nb) == 2   # 角なので隣は 2 点
```

- [ ] **Step 4: テストを実行**

Run: `python -m pytest tests/test_sweep_word_break.py -q`
Expected: PASS（4 テスト）

- [ ] **Step 5: コミット**

```bash
git add cw-decorder/scripts/sweep_word_break.py cw-decorder/tests/test_sweep_word_break.py
git commit -m "feat: WORD_BREAK 抑制パラメータの 2 次元グリッド掃引"
```

---

### Task 5: keyed_val 掃引の実行と一次判定

**Files:**
- 実行のみ（コード変更なし）。出力: `cw-decorder/models/eval/word_break_sweep.json`

**Interfaces:**
- Consumes: Task 4 の `scripts/sweep_word_break.py`
- Produces: 掃引結果 JSON と、採否基準 §6 のうち「改善幅」「プラトー」2 条件の判定結果

- [ ] **Step 1: 掃引を実行する**

Run:
```bash
python scripts/sweep_word_break.py --ckpt models/full/best.pt \
    --keyed-dir data/keying_scripts \
    --out models/eval/word_break_sweep.json
```

- [ ] **Step 2: baseline 格子点が既知の値と一致することを確認**

`[base] beta=0 tau=0 の TER = 23.35%` が出ること。ここが 23.35% から外れる場合、キャッシュ経路が `src/eval/harness.py` の `decode_wave` と等価でない（`input_lengths` の切り詰め忘れなど）。**先にこれを直す。掃引結果の解釈は一切信用できない。**

同様に、baseline 点の `word_break_insertions` が 33、`insertions` が 53、`substitutions` が 34、`deletions` が 23 であることを JSON で確認する:

```bash
python -c "import json;d=json.load(open('models/eval/word_break_sweep.json',encoding='utf-8'));print(d['baseline_point'])"
```

- [ ] **Step 3: 掃引にかかった時間を確認**

掃引が 5 分を超えた場合、設計書 §5 の指示どおり `scripts/sweep_word_break.py` の
グリッド定数を以下に書き換えて Step 1 をやり直す（120 点 → 50 点）:

```python
BETA_GRID: list[float] = [0.0, -1.0, -2.0, -4.0, -8.0]
TAU_GRID: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

5 分以内なら何もしない。

- [ ] **Step 4: 一次判定を記録する**

出力された `[best]` 行から以下を判定する。

| 条件 | 合格ライン |
|---|---|
| 改善幅 | `改善 +2.00pt` 以上 |
| プラトー | `プラトー条件 満たす` |

**どちらか一方でも満たさない場合**: Task 6・Task 7 を実施せず Task 8 へ進む（不採用として文書化）。理論上限は −7.0pt（TER 16.35%）なので、改善が 0〜1pt 台なら「このレバーは効かない」という結論であり、それ自体が価値のある結果である。

- [ ] **Step 5: コミット**

```bash
git add cw-decorder/models/eval/word_break_sweep.json
git commit -m "test: WORD_BREAK 掃引結果 (keyed_val) を記録"
```

`models/eval/` が `.gitignore` の `models/*` に含まれるため、コミット前に追跡対象か確認する:

```bash
git check-ignore -v cw-decorder/models/eval/word_break_sweep.json
```

無視される場合は `git add -f` を使わず、結果 JSON を `cw-decorder/docs/` 配下へコピーして残す（`models/` の除外方針は変えない）。

---

### Task 6: synth_val による健全性確認と採否決定

**前提: Task 5 の一次判定が両方とも合格した場合のみ実施する。**

**Files:**
- Modify: `cw-decorder/scripts/sweep_word_break.py`（synth_val キャッシュの追加）

**Interfaces:**
- Consumes:
  - `src.synth.noise.RealNoisePool.from_dir(path)`
  - `src.synth.dataset.make_fixed_real_noise_eval_set(noise_pool, snr_grid, wpm_grid, samples_per_cell, seed, mode, tone_center_hz, filter_bandwidth_hz) -> list[RealNoiseEvalSample]`（各要素は `.samples`(np.ndarray), `.token_ids`(np.ndarray), `.text`, `.mode`, `.eff_snr_db`）
- Produces: `cache_synth_samples(model, mel_extractor, samples, device) -> list[CachedSample]`

- [ ] **Step 1: synth_val キャッシュ関数を書く**

`cw-decorder/scripts/sweep_word_break.py` の import に追加:

```python
from src.synth.dataset import make_fixed_real_noise_eval_set               # noqa: E402
from src.synth.noise import RealNoisePool                                  # noqa: E402
```

`cache_keyed_samples` の直後に追加:

```python
@torch.no_grad()
def cache_synth_samples(
    model: CWModel,
    mel_extractor: MelExtractor,
    samples: list[Any],
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
```

- [ ] **Step 2: CLI に synth_val オプションを足す**

`build_args` に追加:

```python
    p.add_argument("--noise-dir", type=Path, default=None,
                   help="synth_val 用ノイズ WAV ディレクトリ (指定時のみ synth_val も評価)")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--tone-center", type=float, default=494.0)
    p.add_argument("--bpf-bandwidth", type=float, default=300.0)
    p.add_argument("--wpm", type=float, nargs="+", default=[17.0, 25.0])
    p.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0, -5.0])
    p.add_argument("--samples-per-cell", type=int, default=25)
```

`main` の JSON 出力の直前に追加（`best` 決定後）:

```python
    synth_check: dict[str, Any] | None = None
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
        best_ter = evaluate_cached(synth_cached, WordBreakPolicy(
            logit_bias=best["beta"], conf_threshold=best["tau"]
        )).analysis.totals["ter"]
        degradation_pt = (best_ter - base_ter) * 100
        synth_check = {
            "baseline_ter": base_ter,
            "best_point_ter": best_ter,
            "degradation_pt": degradation_pt,
            "passes": degradation_pt <= 1.0,
        }
        print(
            f"[synth] baseline {base_ter*100:.2f}% → best点 {best_ter*100:.2f}% "
            f"({degradation_pt:+.2f}pt) 条件 {'満たす' if synth_check['passes'] else '満たさない'}",
            flush=True,
        )
```

JSON 出力の dict に `"synth_check": synth_check,` を追加する。

- [ ] **Step 3: 実行する**

Run:
```bash
python scripts/sweep_word_break.py --ckpt models/full/best.pt \
    --keyed-dir data/keying_scripts --noise-dir data/noise/val \
    --out models/eval/word_break_sweep.json
```

`[synth] baseline` の値が **48.06%** 付近になることを確認する。大きく外れる場合はノイズディレクトリが `base_valnoise.json` を作ったときと違う（train 側を渡している等）ので先に直す。

- [ ] **Step 4: 採否を決定する**

3 条件すべて（改善幅 ≥2pt・プラトー・synth 悪化 ≤+1pt）を満たせば **採用**、Task 7 へ進む。
1 つでも満たさなければ **不採用**、Task 7 を飛ばして Task 8 へ。

- [ ] **Step 5: コミット**

```bash
git add cw-decorder/scripts/sweep_word_break.py
git commit -m "feat: 掃引スクリプトに synth_val 健全性チェックを追加"
```

---

### Task 7: harness.decode_wave へのポリシー配線（採用時のみ）

**前提: Task 6 で採用と決定した場合のみ実施する。**

**Files:**
- Modify: `cw-decorder/src/eval/harness.py:22-36`（`decode_wave`）、`:40-61`（`evaluate_real_dataset`）、`:64-85`（`evaluate_synth_noise`）
- Test: `cw-decorder/tests/test_eval_harness.py`（クラス追加）

**Interfaces:**
- Consumes: Task 1/2 の `WordBreakPolicy`, `apply_logit_bias`, `filter_word_breaks`
- Produces: `decode_wave(model, mel_extractor, wave, device, policy: WordBreakPolicy | None = None) -> list[int]`

- [ ] **Step 1: 回帰テストを書く**

`cw-decorder/tests/test_eval_harness.py` には既に `_model_and_mel()`（seed 固定の未学習モデルと
`MelExtractor` を返すヘルパ）があるので、それを使う。

まずファイル先頭の import を差し替える:

```python
from src.infer.word_break_policy import WordBreakPolicy
from src.tokens.morse_tokens import VOCAB_SIZE, WORD_BREAK_TOKEN_ID
```

ファイル末尾にクラスを追加:

```python
class TestDecodeWavePolicy:
    """policy 引数が既存挙動を変えないことの回帰テスト."""

    def test_none_and_identity_policy_agree(self) -> None:
        """policy 未指定と恒等ポリシーが完全に一致する (既存挙動を壊さない保証)."""
        model, mel = _model_and_mel()
        wave = torch.zeros(8000, dtype=torch.float32)
        device = torch.device("cpu")
        assert decode_wave(model, mel, wave, device) == decode_wave(
            model, mel, wave, device, policy=WordBreakPolicy()
        )

    def test_strong_negative_bias_removes_word_breaks(self) -> None:
        """強い負バイアスで WORD_BREAK が 1 つも出なくなる."""
        model, mel = _model_and_mel()
        rng = np.random.default_rng(0)
        wave = torch.from_numpy(rng.standard_normal(8000).astype(np.float32) * 0.1)
        out = decode_wave(
            model, mel, wave, torch.device("cpu"),
            policy=WordBreakPolicy(logit_bias=-50.0),
        )
        assert WORD_BREAK_TOKEN_ID not in out
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_eval_harness.py::TestDecodeWavePolicy -q`
Expected: FAIL — `TypeError: decode_wave() got an unexpected keyword argument 'policy'`

- [ ] **Step 3: harness を変更する**

`cw-decorder/src/eval/harness.py` の import に追加:

```python
from src.infer.word_break_policy import (
    WordBreakPolicy,
    apply_logit_bias,
    filter_word_breaks,
)
```

`decode_wave` を差し替え:

```python
@torch.no_grad()
def decode_wave(
    model: CWModel,
    mel_extractor: MelExtractor,
    wave: torch.Tensor,
    device: torch.device,
    policy: WordBreakPolicy | None = None,
) -> list[int]:
    """波形 (1D tensor) を greedy CTC でデコードし token ID 列を返す.

    ``policy`` が ``None`` または恒等 (β=0, τ=0) のときは従来と完全に同じ経路を通る.
    """
    t = wave.unsqueeze(0).to(device)
    log_probs = torch.nn.functional.log_softmax(model(mel_extractor(t)).float(), dim=-1)
    input_lengths = compute_input_lengths(
        torch.tensor([wave.numel()], device=device),
        mel_extractor.config.hop_length,
        log_probs.size(1),
    )
    if policy is None or policy.is_identity:
        return ctc_greedy_decode(
            log_probs, input_lengths, blank_id=BLANK_TOKEN_ID
        )[0].token_ids
    biased = apply_logit_bias(log_probs, policy.logit_bias)
    result = ctc_greedy_decode(biased, input_lengths, blank_id=BLANK_TOKEN_ID)[0]
    return filter_word_breaks(result, policy.conf_threshold).token_ids
```

`evaluate_real_dataset` と `evaluate_synth_noise` の両方に `policy: WordBreakPolicy | None = None` 引数を足し、それぞれの中の `decode_wave(...)` 呼び出しに `policy=policy` を渡す。

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_eval_harness.py -q`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add cw-decorder/src/eval/harness.py cw-decorder/tests/test_eval_harness.py
git commit -m "feat: 評価経路に WORD_BREAK 抑制ポリシーを配線"
```

---

### Task 8: 結果の文書化と全体検証

**Files:**
- Create: `cw-decorder/docs/word_break_threshold_result.md`
- Modify: `cw-decorder/docs/word_spacing.md`

**Interfaces:**
- Consumes: Task 5/6 の掃引結果と判定
- Produces: 採否の判断根拠となる結果文書

- [ ] **Step 1: 結果文書を書く**

`cw-decorder/docs/word_break_threshold_result.md` に以下を記載する（数値は実測値で埋める）:

- baseline 格子点（β=0, τ=0）の TER と S/D/I、WORD_BREAK 挿入数
- Step 0 診断: 正解WB と偽陽性WB の確信度中央値・四分位、フレームラン長中央値・四分位。両分布が分離しているか重なっているかの評価
- 最適点 (β, τ) と TER、baseline からの改善 pt、WORD_BREAK 挿入数の変化
- 3 つの採否条件それぞれの合否
- 採否の結論
- 不採用の場合: Step 0 のラン長分布から案C（フレームラン長による抑制）が有望かどうかの評価

- [ ] **Step 2: 陳腐化した文書を直す**

`cw-decorder/docs/word_spacing.md` は WORD_BREAK が語彙トークンとして実装される**前**の状態を記述しており、「案A: 語間トークンを語彙に追加（推奨・将来作業）」が既に実装済みである点が誤っている。冒頭に注記を追加する:

```markdown
> **注記 (2026-07-31)**: 本文書の「案A: 語間トークンを語彙に追加」は既に実装済みです
> (`src/tokens/morse_tokens.py` の `WORD_BREAK_TOKEN_ID`)。以下の「現在の実装」節は
> 語彙追加前の音声エンベロープ方式の記述であり、`src/eval/harness.py` の評価経路は
> これを使いません。最新の状況は `docs/word_break_threshold_result.md` を参照。
```

- [ ] **Step 3: 全テストを実行**

Run: `python -m pytest -q 2>&1 | grep -E "FAILED|ERROR" ; echo "done"`
Expected: `FAILED` / `ERROR` の行が 1 つも出ない（サマリ行が出ずに落ちるのは既知の環境問題）。

念のためドット数も確認する:

```bash
python -m pytest -q 2>&1 | tail -3
```

Task 1〜4 で 27 テスト（14 + 9 + 4）追加されるので **629**、Task 7 を実施した場合は
さらに 2 つ増えて **631** になる。

- [ ] **Step 4: コミット**

```bash
git add cw-decorder/docs/word_break_threshold_result.md cw-decorder/docs/word_spacing.md
git commit -m "docs: WORD_BREAK 閾値掃引の結果と word_spacing.md の注記"
```

- [ ] **Step 5: ブランチを main にマージ**

採用・不採用いずれの結論でも、掃引スクリプトと結果は残す価値があるためマージする。

```bash
git checkout main
git merge --no-ff feature/word-break-threshold -m "Merge feature/word-break-threshold: WORD_BREAK 抑制パラメータの掃引"
git branch -d feature/word-break-threshold
git push origin main
```

---

## 実装順序のまとめ

| Task | 内容 | 条件 |
|---|---|---|
| 1 | `WordBreakPolicy` + `apply_logit_bias` | 常に |
| 2 | `filter_word_breaks` | 常に |
| 3 | 掃引スクリプト: キャッシュ + Step 0 診断 | 常に |
| 4 | 掃引スクリプト: グリッド掃引 + JSON | 常に |
| 5 | keyed_val 掃引の実行と一次判定 | 常に |
| 6 | synth_val 健全性確認と採否決定 | Task 5 が両条件合格 |
| 7 | `harness.decode_wave` への配線 | Task 6 で採用 |
| 8 | 結果文書化と全体検証・マージ | 常に |

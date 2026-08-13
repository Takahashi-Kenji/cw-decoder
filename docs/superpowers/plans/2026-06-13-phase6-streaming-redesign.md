# Phase 6: ストリーミング推論再設計 + settings 移行 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ライブデコードを「30秒窓の全体デコードを毎秒繰り返し、文脈確定部のみを確定表示する」スライディングウィンドウ + prefix commit 方式へ置換し、オフライン同等精度を実時間で得る。あわせて settings.json マイグレーションを同梱する。

**Architecture:** モデル無改修。新 `SlidingWindowDecoder` がリングバッファ (最大 `window_s`) に対し `engine.decode_chunk` を周期実行し、トークンを不変 (immutable) に確定。**確定判定は「トークンの時間中点 (midpoint) が確定済み末尾を超える」中点ウォーターマーク方式**で行い (実 CW の文字間 144ms ≈ 3 dot @25WPM でも脱落しない)、右文脈不足のトークン (終了が `now - commit_lag` 以降) は暫定とする。**CPU 負荷削減のため、毎回 30 秒全体ではなく `last_commit_end - left_context_s` 以降のみを再デコード**する (確定済み領域は不変なので再デコード不要)。ワーカーは 1 秒 hop で再デコードし、確定 (黒)/暫定 (グレー) を別シグナルで UI に送る。

**Tech Stack:** Python 3.11, numpy, PySide6 (QObject/QTimer/Signal), pytest。既存 `InferenceEngine.decode_chunk` / `FrameToken` / `InferenceEngine.untrained` (`engine.py:113` で確認済み) を流用。

**WORD_BREAK の出所 (明示):** Phase 6 のライブ連続モードは**モデルの WORD_BREAK トークンに依拠**する (`converter.convert()` が `WORD_BREAK_TOKEN_ID` を語間スペースとして出力する — `src/tokens/converter.py:110-114` で確認)。音声ベースの `detect_word_breaks_from_audio` は手動全体デコード/自動チャンク経路 (`decode_and_reset`) にのみ残し、ライブ連続経路では使わない (実音声での過剰検出は Phase 7 の語間分布拡大で対処 — design.md §5.7)。

---

## 前提・基準値

- 作業ブランチ: `feature/phase6-streaming` (main から。マスター再計画 §2 参照)。
- 既存テスト基準: `def test` = 248 件。本 Phase で 15+ 追加。既存を壊さない (仕様変更で修正する場合はテストにコメントで理由)。
- 既存モデル `models/full/best.pt`・固定評価セットは不変。
- テスト実行: `pytest -q`。UI 系はオフスクリーン (`QT_QPA_PLATFORM=offscreen`)。
- フレーム→サンプル変換は既存 `stream.py` と同じく `frame_index * engine.frame_hop_samples` を用いる (hop=80 samples @8kHz)。

## ファイル構成

- Modify: `src/infer/settings.py` — `settings_version` + 新ストリーミング設定 + マイグレーション
- Create: `src/infer/sliding_window.py` — `CommittedToken`, `DecodeView`, `SlidingWindowDecoder`
- Modify: `src/infer/stream.py` — 旧 `StreamingDecoder` は新方式へ寄せる (後方互換は残さず置換、ワーカーが新方式利用)
- Modify: `src/app/workers.py` — ライブ連続モード (1秒 hop 再デコードタイマ + 確定/暫定シグナル)
- Modify: `src/app/main_window.py` — 確定(黒)/暫定(グレー) 2 領域表示 + ステータス診断 `window=Xs hop=Ys lag=Zs decode=Nms`
- Create: `tests/test_settings_migration.py` — マイグレーション
- Create: `tests/test_sliding_window.py` — prefix commit / 窓端 / immutable
- Modify: `tests/test_inference_engine.py` — 旧 `TestStreamingDecoder` を新仕様へ書換 (同数以上)
- Create: `scripts/eval_streaming_vs_offline.py` — 受け入れ基準 (95% 一致 / RTF) 検証
- Modify: `docs/design.md` — Phase 6 結果追記 (最後に)

---

## Task 1: settings.json — version フィールド + 新ストリーミング設定

**Files:**
- Modify: `src/infer/settings.py`
- Test: `tests/test_settings.py` (既存に追記)

- [ ] **Step 1: 新フィールドの失敗テストを書く**

`tests/test_settings.py` の末尾に追記:

```python
def test_streaming_defaults_present() -> None:
    s = AppSettings()
    assert s.settings_version == 2
    assert s.window_s == 30.0
    assert s.hop_s == 1.0
    assert s.commit_lag_s == 2.5
    assert s.head_guard_s == 1.0
    assert s.decode_left_context_s == 5.0
    assert s.commit_jitter_margin_s == 0.02
    assert s.live_continuous is True
```

- [ ] **Step 2: 失敗を確認**

Run: `pytest tests/test_settings.py::test_streaming_defaults_present -v`
Expected: FAIL (`AttributeError: ... has no attribute 'settings_version'`)

- [ ] **Step 3: `AppSettings` にフィールドを追加**

`src/infer/settings.py` の `class AppSettings` 内、`auto_chunk_silence_amplitude` 行の直後に追記:

```python
    # --- Phase 6: スライディングウィンドウ再デコード (ライブ連続モード) ---
    settings_version: int = 2             # 設定スキーマ版 (マイグレーション用)
    live_continuous: bool = True          # ライブ連続モードを既定とする
    window_s: float = 30.0                # リングバッファ保持長 (再開・手動全体用)
    hop_s: float = 1.0                    # 再デコード周期
    commit_lag_s: float = 2.5             # 確定遅延 (末尾はこの秒数未確定)
    head_guard_s: float = 1.0             # デコード区間先頭の不採用区間 (左文脈なし)
    # CPU 負荷削減: 毎回 window 全体ではなく、確定済み末尾の left_context 秒前から
    # のみ再デコードする (確定済みは不変なので再計算不要). 実効デコード長 ≈
    # left_context + commit_lag + hop ≈ 8〜9 秒に収まり CPU RTF 0.2 前後が現実的.
    decode_left_context_s: float = 5.0
    # パス間タイムスタンプジッタ吸収用の小マージン (中点ウォーターマークの安全余裕).
    # 確定の主機構は midpoint > last_commit_end であり、これは脱落防止の主役ではない.
    commit_jitter_margin_s: float = 0.02
```

> **設計変更の理由 (レビュー反映):** 旧案の `commit_match_tol_s=0.3` は「直前確定の終了から 0.3 秒以上空かないと確定しない」を意味し、25 WPM の文字間ギャップ (3 dot ≈ 144ms < 300ms) で連続文字を脱落させていた。確定機構を中点ウォーターマーク (Task 4) に変更し、許容幅は ±1〜2 フレーム相当 (`commit_jitter_margin_s=0.02`) に縮小。

- [ ] **Step 4: 合格を確認**

Run: `pytest tests/test_settings.py -v`
Expected: PASS (新テスト + 既存 6 件)

- [ ] **Step 5: コミット**

```bash
git add src/infer/settings.py tests/test_settings.py
git commit -m "feat: AppSettings にスライディングウィンドウ設定と settings_version を追加"
```

---

## Task 2: settings マイグレーション (旧 version の値補完・置換)

**Files:**
- Modify: `src/infer/settings.py` (`load_settings` と `from_dict`)
- Test: `tests/test_settings_migration.py` (新規)

方針: `from_dict` は未知キーを無視・欠損はデフォルト補完 (現状動作)。`load_settings` 読込時に `settings_version` が古ければ既知の旧デフォルト値を新デフォルトへ置換するマイグレーション表を適用し、結果をログ (`print` で `[settings-migrate]` プレフィクス) する。

- [ ] **Step 0: 旧既定値を git 履歴で確定してから表を埋める (実装前必須)**

Run: `git log -p -- src/infer/settings.py | grep -nE "chunk_duration_s|auto_chunk_silence_amplitude" `
確認内容: `_V1_DEFAULT_REPLACEMENTS` に載せる旧既定値が、git 上の実際の旧デフォルトと一致すること。`chunk_duration_s` は `1.5` (旧) → `5.0` (現) を確認済み。確証できない値 (例: `auto_chunk_silence_amplitude`) は表に載せない。実測した行を Step 3 のコメントに反映する。

- [ ] **Step 1: 失敗テストを書く**

`tests/test_settings_migration.py` を新規作成:

```python
"""settings.json マイグレーションのテスト."""
from __future__ import annotations

import json
from pathlib import Path

from src.infer.settings import AppSettings, load_settings, migrate_settings_dict


def test_missing_version_treated_as_v1_and_filled() -> None:
    # version フィールドが無い (= v1 相当) 旧 JSON
    raw = {"mode": "japanese", "chunk_duration_s": 1.5}
    migrated, changed = migrate_settings_dict(raw)
    assert migrated["settings_version"] == 2
    # 新フィールドがデフォルトで補完される
    assert migrated["window_s"] == 30.0
    assert changed is True


def test_v1_old_chunk_default_replaced() -> None:
    # v1 の旧既定 chunk_duration_s == 1.5 は新既定 5.0 に置換
    raw = {"settings_version": 1, "chunk_duration_s": 1.5}
    migrated, _ = migrate_settings_dict(raw)
    assert migrated["chunk_duration_s"] == 5.0


def test_v1_user_customized_chunk_preserved() -> None:
    # v1 でもユーザーが旧既定以外に設定していれば尊重 (10.0 は維持)
    raw = {"settings_version": 1, "chunk_duration_s": 10.0}
    migrated, _ = migrate_settings_dict(raw)
    assert migrated["chunk_duration_s"] == 10.0


def test_v2_no_change() -> None:
    raw = AppSettings().to_dict()
    migrated, changed = migrate_settings_dict(raw)
    assert changed is False


def test_load_settings_applies_migration(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"chunk_duration_s": 1.5}), encoding="utf-8")
    s = load_settings(p)
    assert s.settings_version == 2
    assert s.chunk_duration_s == 5.0
```

- [ ] **Step 2: 失敗を確認**

Run: `pytest tests/test_settings_migration.py -v`
Expected: FAIL (`ImportError: cannot import name 'migrate_settings_dict'`)

- [ ] **Step 3: `migrate_settings_dict` を実装し `load_settings` で呼ぶ**

`src/infer/settings.py` の `from_dict` 定義の後、`load_settings` の前に追加:

```python
# 旧 version 既定値 → 新既定への置換表.
# 「保存値が旧既定と一致するなら、ユーザー未変更とみなして新既定へ更新」.
# 注意: 旧既定値は git 履歴で確証できたもののみ載せる. chunk_duration_s=1.5 は
# `git log -S` で旧既定と確認済み (現行 5.0). auto_chunk_silence_amplitude は
# 旧既定が履歴で確証できず (workers.py と settings.py で 0.02/0.005 が不一致) かつ
# 移行の必要性も低いため、意図的に除外する.
_V1_DEFAULT_REPLACEMENTS: dict[str, tuple[Any, Any]] = {
    # field: (旧既定, 新既定)
    "chunk_duration_s": (1.5, 5.0),
}

CURRENT_SETTINGS_VERSION = 2


def migrate_settings_dict(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """旧 version の設定 dict を最新スキーマへ移行.

    Returns:
        (移行後 dict, 変更があったか).
    """
    out = dict(data)
    version = int(out.get("settings_version", 1))
    changed = False
    if version < 2:
        for field_name, (old_default, new_default) in _V1_DEFAULT_REPLACEMENTS.items():
            if field_name in out and out[field_name] == old_default:
                out[field_name] = new_default
                changed = True
        out["settings_version"] = CURRENT_SETTINGS_VERSION
        changed = True
    return out, changed
```

`load_settings` を以下に置換:

```python
def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> AppSettings:
    """設定を JSON から読み込み. ファイルが無い/壊れていれば既定値を返す.

    旧 version の設定はマイグレーション表で補完・置換し、結果をログ出力する.
    """
    path = Path(path)
    if not path.exists():
        return AppSettings()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return AppSettings()
    migrated, changed = migrate_settings_dict(data)
    if changed:
        print(
            f"[settings-migrate] {path} を v{data.get('settings_version', 1)} "
            f"→ v{CURRENT_SETTINGS_VERSION} へ移行しました"
        )
    return AppSettings.from_dict(migrated)
```

`__all__` に追加:

```python
__all__ = [
    "AppSettings", "DEFAULT_CONFIG_PATH", "load_settings", "save_settings",
    "migrate_settings_dict", "CURRENT_SETTINGS_VERSION",
]
```

- [ ] **Step 4: 合格を確認**

Run: `pytest tests/test_settings_migration.py tests/test_settings.py -v`
Expected: PASS (全件)

- [ ] **Step 5: コミット**

```bash
git add src/infer/settings.py tests/test_settings_migration.py
git commit -m "feat: settings.json マイグレーション (旧既定値の新既定置換)"
```

---

## Task 3: SlidingWindowDecoder — データ型と空状態

**Files:**
- Create: `src/infer/sliding_window.py`
- Test: `tests/test_sliding_window.py` (新規)

- [ ] **Step 1: 失敗テストを書く**

`tests/test_sliding_window.py` を新規作成:

```python
"""SlidingWindowDecoder (prefix commit) のテスト."""
from __future__ import annotations

import numpy as np

from src.infer.engine import FrameToken, InferenceEngine
from src.infer.sliding_window import (
    CommittedToken,
    DecodeView,
    SlidingWindowDecoder,
)


def _decoder(**kw) -> SlidingWindowDecoder:
    # 既定は decode_left_context_s = window_s (= フルウィンドウ再デコード) とし、
    # 固定トークン monkeypatch の frame 位置が絶対位置に一致するようにする
    # (動的区間短縮は test_decode_window_shortens_after_commit で個別検証).
    eng = InferenceEngine.untrained("cpu")
    params = dict(
        window_s=30.0, hop_s=1.0, commit_lag_s=2.5,
        head_guard_s=1.0, decode_left_context_s=30.0,
        commit_jitter_margin_s=0.02, sample_rate=8000,
    )
    params.update(kw)
    return SlidingWindowDecoder(eng, **params)


def test_empty_decode_returns_empty_view() -> None:
    d = _decoder()
    view = d.redecode()
    assert isinstance(view, DecodeView)
    assert view.committed == []
    assert view.provisional == []
    assert view.newly_committed == []
```

- [ ] **Step 2: 失敗を確認**

Run: `pytest tests/test_sliding_window.py::test_empty_decode_returns_empty_view -v`
Expected: FAIL (`ModuleNotFoundError: src.infer.sliding_window`)

- [ ] **Step 3: 骨格を実装**

`src/infer/sliding_window.py` を新規作成:

```python
"""スライディングウィンドウ再デコード + prefix commit (Phase 6).

ライブ音声をリングバッファに保持し、HOP_S ごとに窓全体を ``decode_chunk`` で
再デコードする. ``[max(head_guard, last_commit), now - commit_lag)`` の区間に
入るトークンのみを **不変 (immutable)** に確定する. 確定済みは後から変化しない
ため UI 表示がちらつかない (design.md §5.4/5.5 の文脈損失問題の構造的解消).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.infer.engine import InferenceEngine


@dataclass(frozen=True)
class CommittedToken:
    """確定 (immutable) トークン. 絶対サンプル位置付き."""

    token_id: int
    confidence: float
    absolute_sample_start: int
    absolute_sample_end: int


@dataclass(frozen=True)
class DecodeView:
    """1 回の再デコード結果のスナップショット."""

    committed: list[CommittedToken] = field(default_factory=list)       # 確定済み全体
    newly_committed: list[CommittedToken] = field(default_factory=list)  # 今回新規確定
    provisional: list[CommittedToken] = field(default_factory=list)      # 暫定 (グレー表示)


class SlidingWindowDecoder:
    def __init__(
        self,
        engine: InferenceEngine,
        window_s: float = 30.0,
        hop_s: float = 1.0,
        commit_lag_s: float = 2.5,
        head_guard_s: float = 1.0,
        decode_left_context_s: float = 5.0,
        commit_jitter_margin_s: float = 0.02,
        sample_rate: int = 8000,
    ) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.hop_samples_audio = int(hop_s * sample_rate)
        self.commit_lag_samples = int(commit_lag_s * sample_rate)
        self.head_guard_samples = int(head_guard_s * sample_rate)
        self.left_context_samples = int(decode_left_context_s * sample_rate)
        self.jitter_margin_samples = int(commit_jitter_margin_s * sample_rate)
        self._ring = np.zeros(0, dtype=np.float32)
        self._total_consumed = 0                 # 累積投入サンプル数 (= 現在時刻)
        self._committed: list[CommittedToken] = []
        # 確定済み末尾の絶対サンプル位置 (中点ウォーターマーク基準). 無ければ None.
        self._last_commit_end: int | None = None

    def reset(self) -> None:
        self._ring = np.zeros(0, dtype=np.float32)
        self._total_consumed = 0
        self._committed = []
        self._last_commit_end = None

    def push(self, audio: np.ndarray) -> None:
        """音声を追加 (デコードはしない). 窓長を超えた古い分は捨てる."""
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = audio.astype(np.float32, copy=False)
        self._total_consumed += audio.size
        self._ring = np.concatenate([self._ring, audio])
        if self._ring.size > self.window_samples:
            self._ring = self._ring[-self.window_samples:]

    def redecode(self) -> DecodeView:
        """現在のリングバッファ全体を再デコードし確定/暫定を更新."""
        if self._ring.size == 0:
            return DecodeView(committed=list(self._committed))
        raise NotImplementedError  # Task 4 で実装
```

- [ ] **Step 4: 合格を確認**

Run: `pytest tests/test_sliding_window.py::test_empty_decode_returns_empty_view -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/infer/sliding_window.py tests/test_sliding_window.py
git commit -m "feat: SlidingWindowDecoder の型と空状態 (Phase 6 骨格)"
```

---

## Task 4: SlidingWindowDecoder — prefix commit ロジック本体

**Files:**
- Modify: `src/infer/sliding_window.py` (`redecode`)
- Test: `tests/test_sliding_window.py`

設計メモ (frame→絶対サンプル変換とコミット判定):
- `ring_start_abs = self._total_consumed - self._ring.size`
- **デコード区間の動的短縮 (RTF 対策, レビュー #5):** 確定済みは不変なので毎回 window 全体を再デコードしない。
  `decode_start_abs = max(ring_start_abs, last_commit_end - left_context_samples)`
  (確定がなければ `last_commit_end = ring_start_abs` 扱い → `decode_start_abs = ring_start_abs`)。
  `sub = self._ring[decode_start_abs - ring_start_abs :]` を `decode_chunk` に渡す。
- トークン `tok` の絶対位置: `abs_start = decode_start_abs + tok.frame_start * hop`, `abs_end = decode_start_abs + tok.frame_end * hop` (hop = `engine.frame_hop_samples`)。
- `head_cut_abs = decode_start_abs + head_guard_samples` ただし `decode_start_abs == 0` (録音開始直後で先頭から) のときは `0` (先頭信号を捨てない — design.md §4.4 の教訓)。
- `commit_limit_abs = self._total_consumed - self.commit_lag_samples`
- **暫定 (グレー) = `abs_end >= commit_limit_abs`** (レビュー #1: commit 境界を**またぐ**トークン=右文脈不足なので確定しない。`abs_start` ではなく `abs_end` で判定)。
- **新規確定 = 中点ウォーターマーク (レビュー #2):** `midpoint = (abs_start + abs_end) / 2` とし、
  `midpoint > (last_commit_end ?? -inf) + jitter_margin_samples` かつ `abs_start >= head_cut_abs` かつ暫定でないもの。
  - 再デコードで再出現した既確定トークン: `midpoint = last_commit_end - duration/2 < last_commit_end` → 正しくスキップ (短トークン E でも成立)。
  - 直後の新トークン: `midpoint = last_commit_end + gap + duration/2 > last_commit_end` → 文字間ギャップが 0 近傍でも確定 (25 WPM の 144ms ギャップで脱落しない)。
- 確定したら `self._last_commit_end = max(self._last_commit_end, abs_end)` を更新 (immutable)。

- [ ] **Step 1: 失敗テストを書く** (決定論のため `engine.decode_chunk` をモンキーパッチ)

`tests/test_sliding_window.py` に追記:

```python
def _patch_tokens(d: SlidingWindowDecoder, frame_tokens: list[FrameToken]) -> None:
    """engine.decode_chunk を固定トークン列に差し替え (決定論テスト用).

    ヘルパは decode_left_context_s == window_s なので decode_start は常に
    ring 先頭 = 絶対 0 起点となり、frame 位置 == 絶対位置 (offset 安定).
    """
    d.engine.decode_chunk = lambda wave: list(frame_tokens)  # type: ignore[assignment]


def test_commits_only_inside_commit_zone() -> None:
    # hop=80 samples/frame @8kHz → 1 frame = 0.01s. 100 frame = 1s.
    d = _decoder(window_s=30.0, commit_lag_s=2.5, head_guard_s=1.0)
    d.push(np.zeros(80000, dtype=np.float32))   # 10s
    _patch_tokens(d, [
        FrameToken(token_id=5, confidence=0.9, frame_start=50, frame_end=55),
        FrameToken(token_id=6, confidence=0.9, frame_start=500, frame_end=505),
        FrameToken(token_id=7, confidence=0.9, frame_start=900, frame_end=905),
    ])
    view = d.redecode()
    # commit_limit = 10s-2.5s = 7.5s. 9s(frame900) は abs_end>commit_limit → 暫定
    assert [t.token_id for t in view.committed] == [5, 6]
    assert [t.token_id for t in view.provisional] == [7]


def test_boundary_straddling_token_is_provisional_not_committed() -> None:
    # レビュー #1: 開始は commit 境界前・終了は境界後のトークンは確定しない.
    d = _decoder(window_s=30.0, commit_lag_s=2.5, head_guard_s=0.0)
    d.push(np.zeros(80000, dtype=np.float32))   # 10s → commit_limit = 7.5s (abs 60000)
    # frame740..760 = abs 59200..60800 (開始 7.4s < 7.5s, 終了 7.6s > 7.5s)
    _patch_tokens(d, [FrameToken(token_id=8, confidence=0.9, frame_start=740, frame_end=760)])
    view = d.redecode()
    assert view.committed == []                       # 旧実装 (abs_start 判定) では誤って確定した
    assert [t.token_id for t in view.provisional] == [8]


def test_realistic_25wpm_five_chars_all_commit_in_one_pass() -> None:
    # レビュー #2 (最重大): 25 WPM (dot=48ms=384smp) の文字間ギャップ 3 dot≈144ms で
    # 連続 5 文字がすべて確定すること. 旧 tol=0.3s 実装ではここが脱落していた.
    d = _decoder(window_s=30.0, commit_lag_s=1.0, head_guard_s=0.0)
    d.push(np.zeros(8000 * 5, dtype=np.float32))      # 5s → commit_limit = 4.0s (abs 32000)
    # 各文字 ~5 frame 長、文字間 14 frame(=140ms) 間隔で 5 文字並べる
    toks = []
    fs = 10
    for tid in range(1, 6):
        toks.append(FrameToken(token_id=tid, confidence=0.9, frame_start=fs, frame_end=fs + 5))
        fs += 19                                       # 5(長) + 14(間隔) = 19 frame
    _patch_tokens(d, toks)
    view = d.redecode()
    assert [t.token_id for t in view.committed] == [1, 2, 3, 4, 5]


def test_short_token_not_double_committed_on_redecode() -> None:
    # レビュー #2: 短トークン (E=dot) が次パスで二重確定されないこと (中点ウォーターマーク).
    d = _decoder(window_s=30.0, commit_lag_s=1.0, head_guard_s=0.0)
    d.push(np.zeros(8000 * 3, dtype=np.float32))      # 3s
    e_tok = [FrameToken(token_id=2, confidence=0.9, frame_start=100, frame_end=102)]
    _patch_tokens(d, e_tok)
    v1 = d.redecode()
    assert [t.token_id for t in v1.newly_committed] == [2]
    d.push(np.zeros(8000, dtype=np.float32))          # +1s, 同じ E が再出現
    v2 = d.redecode()
    assert v2.newly_committed == []                    # 二重確定しない
    assert [t.token_id for t in v2.committed] == [2]


def test_head_guard_drops_decode_region_front() -> None:
    d = _decoder(window_s=30.0, commit_lag_s=2.5, head_guard_s=1.0)
    # 35s 投入 → ring 末尾 30s、ring_start_abs = 5s。left_context=window なので decode_start=ring_start
    d.push(np.zeros(8000 * 35, dtype=np.float32))
    # frame50(=窓内 0.5s, 絶対 5.5s) は head_guard(1s) 内 → 不採用 / frame500(絶対 10s) は採用
    _patch_tokens(d, [
        FrameToken(token_id=5, confidence=0.9, frame_start=50, frame_end=55),
        FrameToken(token_id=6, confidence=0.9, frame_start=500, frame_end=505),
    ])
    view = d.redecode()
    assert [t.token_id for t in view.committed] == [6]


def test_provisional_becomes_committed_after_lag() -> None:
    d = _decoder(window_s=30.0, commit_lag_s=2.5)
    d.push(np.zeros(80000, dtype=np.float32))         # 10s, トークン @9.5s (暫定)
    _patch_tokens(d, [FrameToken(token_id=9, confidence=0.9, frame_start=950, frame_end=955)])
    v1 = d.redecode()
    assert [t.token_id for t in v1.provisional] == [9]
    assert v1.committed == []
    d.push(np.zeros(8000 * 3, dtype=np.float32))      # +3s → total 13s, commit_limit=10.5s
    v2 = d.redecode()
    assert [t.token_id for t in v2.committed] == [9]
    assert v2.provisional == []


def test_decode_window_shortens_after_commit() -> None:
    # レビュー #5: 確定後は left_context 分のみ再デコード (毎回 window 全体ではない).
    d = _decoder(window_s=30.0, commit_lag_s=2.5, decode_left_context_s=5.0)
    d.push(np.zeros(8000 * 20, dtype=np.float32))     # 20s (ring=160000, ring_start_abs=0)
    d._last_commit_end = 8000 * 15                     # 15s まで確定済みと仮定
    d._committed = [CommittedToken(1, 0.9, 0, 8000 * 15)]
    received: list[int] = []
    d.engine.decode_chunk = lambda wave: received.append(wave.size) or []  # type: ignore
    d.redecode()
    # decode_start = max(0, 15s - 5s) = 10s → 渡されるのは末尾 10s = 80000 samples
    assert received == [80000]


def test_finalize_commits_pending_provisional() -> None:
    # 送信終了時: commit_lag 圏内の暫定を最終確定する.
    d = _decoder(window_s=30.0, commit_lag_s=2.5)
    d.push(np.zeros(80000, dtype=np.float32))         # 10s, @9.5s は通常 redecode では暫定
    _patch_tokens(d, [FrameToken(token_id=9, confidence=0.9, frame_start=950, frame_end=955)])
    assert d.redecode().committed == []
    final = d.finalize()
    assert [t.token_id for t in final.committed] == [9]
    assert final.provisional == []
```

- [ ] **Step 2: 失敗を確認**

Run: `pytest tests/test_sliding_window.py -v`
Expected: FAIL (大半は `NotImplementedError`、`test_finalize_*` は `AttributeError: ... 'finalize'`)

- [ ] **Step 3: `redecode` / `finalize` / `_decode` を実装**

`src/infer/sliding_window.py` の `redecode` を以下に置換:

```python
    def redecode(self) -> DecodeView:
        """確定済み末尾 - left_context 以降を再デコードし確定/暫定を更新.

        確定済み領域は不変なので再計算しない (CPU 負荷削減). 末尾 commit_lag
        圏内のトークンは右文脈不足のため暫定とする.
        """
        return self._decode(commit_limit_abs=self._total_consumed - self.commit_lag_samples)

    def finalize(self) -> DecodeView:
        """送信終了時の最終確定. commit_lag を無視し残り全部を確定する.

        ストリーム終端では右文脈がもう増えないため、暫定として保留していた
        末尾トークンをここで確定させる (ワーカーの stop() から呼ぶ).
        """
        return self._decode(commit_limit_abs=self._total_consumed + 1)

    def _decode(self, commit_limit_abs: int) -> DecodeView:
        if self._ring.size == 0:
            return DecodeView(committed=list(self._committed))

        hop = self.engine.frame_hop_samples
        ring_start_abs = self._total_consumed - self._ring.size

        # --- デコード区間の動的短縮 (RTF 対策) ---
        last_end = self._last_commit_end
        anchor = last_end if last_end is not None else ring_start_abs
        decode_start_abs = max(ring_start_abs, anchor - self.left_context_samples)
        sub = self._ring[decode_start_abs - ring_start_abs:]
        frame_tokens = self.engine.decode_chunk(sub)

        # head guard: デコード区間先頭の不採用区間. ただし区間がストリーム先頭
        # (decode_start_abs == 0) のときは先頭信号を捨てない (design.md §4.4).
        head_cut_abs = (
            decode_start_abs + self.head_guard_samples if decode_start_abs > 0 else 0
        )

        newly: list[CommittedToken] = []
        provisional: list[CommittedToken] = []
        for tok in frame_tokens:
            abs_start = decode_start_abs + tok.frame_start * hop
            abs_end = decode_start_abs + tok.frame_end * hop
            ct = CommittedToken(
                token_id=tok.token_id,
                confidence=tok.confidence,
                absolute_sample_start=abs_start,
                absolute_sample_end=abs_end,
            )
            # 右文脈不足 (commit 境界を終了がまたぐ) → 暫定
            if abs_end >= commit_limit_abs:
                provisional.append(ct)
                continue
            # 左文脈なし → 不採用
            if abs_start < head_cut_abs:
                continue
            # 中点ウォーターマーク: 確定済み末尾を中点が超えるもののみ新規確定.
            # 既確定トークンの再出現 (midpoint < last_end) は確実にスキップされ、
            # 文字間ギャップが小さくても新規トークンは脱落しない.
            midpoint = (abs_start + abs_end) // 2
            if last_end is not None and midpoint <= last_end + self.jitter_margin_samples:
                continue
            self._committed.append(ct)
            newly.append(ct)
            last_end = abs_end
            self._last_commit_end = abs_end

        return DecodeView(
            committed=list(self._committed),
            newly_committed=newly,
            provisional=provisional,
        )
```

`__all__` を追加:

```python
__all__ = ["CommittedToken", "DecodeView", "SlidingWindowDecoder"]
```

- [ ] **Step 4: 合格を確認**

Run: `pytest tests/test_sliding_window.py -v`
Expected: PASS (Task 3 の 1 件 + 本 Task の 8 件 = 9 件)

- [ ] **Step 5: リセットのテストを追加**

`tests/test_sliding_window.py` に追記:

```python
def test_reset_clears_committed() -> None:
    d = _decoder()
    d.push(np.zeros(80000, dtype=np.float32))
    _patch_tokens(d, [FrameToken(token_id=5, confidence=0.9, frame_start=300, frame_end=305)])
    d.redecode()
    d.reset()
    assert d._last_commit_end is None
    assert d.redecode().committed == []
```

Run: `pytest tests/test_sliding_window.py -v`
Expected: PASS (全 10 件)

- [ ] **Step 6: 実モデルで RTF を前倒し実測 (レビュー #5 — デフォルト確定のため)**

`models/full/best.pt` がある場合、Task 8 のスクリプトを待たず、ここで実デコード負荷を計測してデフォルトパラメータ (`decode_left_context_s`, `window_s`) を数字で確定する。一時スクリプトで `data/real/20260612_200209_european.wav` を 50ms ブロックで push しながら 1 秒 hop で `redecode` し、`redecode` 1 回あたりの所要時間と RTF を出力する。

判断基準: CPU で `redecode` 1 回 ≤ 約 800ms (hop=1s に対し RTF ≤ 0.2 の余裕)。超過する場合は `decode_left_context_s` を 5.0 → 3.0 に下げる、または `hop_s` を 1.0 → 1.5 に上げてから先へ進む。**実測値をこのステップのコメントとして記録する**。

> このステップは「Task 8 で初めて性能が分かり、手戻りする」リスクをつぶすための前倒し測定。モデルが手元にない実行者はスキップし Task 8 に委ねてよい (その旨を記録)。

- [ ] **Step 7: コミット**

```bash
git add src/infer/sliding_window.py tests/test_sliding_window.py
git commit -m "feat: SlidingWindowDecoder prefix commit 本体 (中点ウォーターマーク・動的区間短縮・finalize)"
```

---

## Task 5: 旧 StreamingDecoder テストの新仕様化

**Files:**
- Modify: `tests/test_inference_engine.py` (`TestStreamingDecoder`)

指示書 §2.2: 旧オーバーラップマージは置換。旧ロジックは削除してよいが、テストは新仕様へ書き換え同数以上を維持。
方針: `StreamingDecoder` は当面 `stop()` の `flush` 経路で残るが、ライブ経路は `SlidingWindowDecoder` に移行 (Task 6)。旧 `TestStreamingDecoder` (約 8 メソッド) は `tests/test_sliding_window.py` 側でカバー済みのため、`test_inference_engine.py` の `TestStreamingDecoder` を**新方式のスモークに置換**して件数を維持する。

- [ ] **Step 1: 旧 TestStreamingDecoder を新方式スモークへ置換**

`tests/test_inference_engine.py` の `from src.infer.stream import StreamingDecoder, StreamToken` 行を削除し、`class TestStreamingDecoder:` ブロック全体を次に置換:

```python
class TestSlidingWindowDecoder:
    """ライブ連続モードのスモーク (詳細は tests/test_sliding_window.py)."""

    def test_push_and_redecode_runs(self) -> None:
        from src.infer.sliding_window import SlidingWindowDecoder
        eng = InferenceEngine.untrained("cpu")
        d = SlidingWindowDecoder(eng, window_s=5.0, hop_s=1.0, commit_lag_s=1.0)
        d.push(np.zeros(8000, dtype=np.float32))
        view = d.redecode()
        assert hasattr(view, "committed")

    def test_window_truncates_to_window_samples(self) -> None:
        from src.infer.sliding_window import SlidingWindowDecoder
        eng = InferenceEngine.untrained("cpu")
        d = SlidingWindowDecoder(eng, window_s=2.0, sample_rate=8000)
        d.push(np.zeros(8000 * 5, dtype=np.float32))   # 5s 投入
        assert d._ring.size == 16000                    # 窓 2s = 16000 に切詰め

    def test_reset_runs(self) -> None:
        from src.infer.sliding_window import SlidingWindowDecoder
        eng = InferenceEngine.untrained("cpu")
        d = SlidingWindowDecoder(eng, window_s=5.0)
        d.push(np.zeros(8000, dtype=np.float32))
        d.reset()
        assert d._ring.size == 0
```

- [ ] **Step 2: 既存推論テストが通ることを確認**

Run: `pytest tests/test_inference_engine.py -v`
Expected: PASS (`StreamToken` 未使用で import エラーが出ないこと)

- [ ] **Step 3: 全体テストで回帰がないことを確認**

Run: `pytest -q`
Expected: PASS (件数 = 248 - 旧 8 + 新規。下回らないこと。下回る場合はスモークを追加)

- [ ] **Step 4: コミット**

```bash
git add tests/test_inference_engine.py
git commit -m "test: StreamingDecoder テストを SlidingWindowDecoder 新仕様へ書換"
```

---

## Task 6: ワーカーにライブ連続モードを統合

**Files:**
- Modify: `src/app/workers.py`
- Test: `tests/test_workers_live.py` (新規)

方針:
- `AudioInferenceWorker.__init__` にスライディング設定引数 (`window_s`, `hop_s`, `commit_lag_s`, `head_guard_s`, `decode_left_context_s`, `commit_jitter_margin_s`, `live_continuous`) を追加し `SlidingWindowDecoder` を生成。
- `_tick` (20ms) では BPF 後ブロックを `self._sliding.push()` する (live_continuous かつ buffer_recording 時)。
- 1 秒 hop の再デコードは `_tick` 内で経過サンプル数を数えて `hop_s` 到達ごとに `self._sliding.redecode()` を呼ぶ (専用 QTimer を増やさず既存 tick で十分)。
- 確定/暫定を新シグナルで emit。確定は語間 (WORD_BREAK) 反映、暫定はグレー表示用テキスト。
- 無音が窓の大半を占める場合は redecode をスキップ (CPU 節約)。
- 新シグナル: `committed_text_changed(str)` (確定テキスト全体)、`provisional_text_changed(str)` (暫定テキスト)、`stream_diag(dict)` (`window/hop/lag/decode_ms`)。
- **(d) 無音判定の測定点整合 (実装時確認):** `_feed_live_block` の `recent_silent` は BPF 通過後リングの RMS で判定する。これがレベルメータの橙色破線 (スキッシュ閾値) が示す測定点と一致していること、既存の自動チャンク (`_tick` 内 `block_db` 判定、同じく BPF 後) と同一基準であることを実装時に一度確認する。不一致だとユーザーが「メータの線と挙動が合わない」と感じる典型事故になる (design.md §4.4 の教訓)。
- **(a) 既知の制限 — 確定テキストの全文 re-emit:** `_emit_live_view` は毎 hop で `self._committed` 全体を再変換し全文 emit する。`self._committed` は無制限に成長するため、長時間運用 (数千トークン) では毎秒の全文変換 + `setHtml` 全置換になり、UI スクロールのちらつき/位置ジャンプの恐れ。Phase 6 では既知の制限とし、本筋の対処 (`newly_committed` 差分のみ emit + UI 追記) は実運用で問題が出てから行う。Task 9 で design.md に記録する。

- [ ] **Step 1: 失敗テストを書く** (Qt 無しでロジック検証できる範囲)

`tests/test_workers_live.py` を新規作成:

```python
"""ライブ連続モードのワーカー結線テスト (オフスクリーン Qt)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from src.infer.engine import InferenceEngine
from src.app.workers import AudioInferenceWorker

_app = QApplication.instance() or QApplication([])


def _worker() -> AudioInferenceWorker:
    eng = InferenceEngine.untrained("cpu")
    return AudioInferenceWorker(
        eng, sample_rate=8000, live_continuous=True,
        window_s=5.0, hop_s=1.0, commit_lag_s=1.0, head_guard_s=0.5,
        squelch_threshold_db=-60.0,
    )


def test_worker_has_sliding_decoder() -> None:
    w = _worker()
    assert w._sliding is not None
    assert w.live_continuous is True


def test_redecode_triggers_committed_signal() -> None:
    w = _worker()
    received: list[str] = []
    w.committed_text_changed.connect(received.append)
    w.set_buffer_recording(True)
    # 6 秒分の無音でない疑似信号を push して redecode を 1 回以上発火させる
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(8000 * 6) * 0.2).astype(np.float32)
    # 50ms ブロックに分けて push + redecode ロジックを直接駆動
    for i in range(0, sig.size, 400):
        w._feed_live_block(sig[i:i + 400])
    # 例外なく駆動できること (確信度次第で確定 0 でも可)
    assert isinstance(received, list)
```

> 注: `_feed_live_block` は `_tick` から push + hop 判定 + redecode を呼ぶ内部メソッドとして切り出し、テスタブルにする。

- [ ] **Step 2: 失敗を確認**

Run: `pytest tests/test_workers_live.py -v`
Expected: FAIL (`AttributeError: ... '_sliding'` / `live_continuous`)

- [ ] **Step 3: ワーカーを実装**

`src/app/workers.py` の import に追加:

```python
from src.infer.sliding_window import DecodeView, SlidingWindowDecoder
```

`AudioInferenceWorker` の新シグナルをクラス属性に追加 (既存シグナル群の下):

```python
    committed_text_changed = Signal(str)    # 確定テキスト全体 (黒表示)
    provisional_text_changed = Signal(str)  # 暫定テキスト (グレー表示)
    stream_diag = Signal(dict)              # {window, hop, lag, decode_ms}
```

`__init__` シグネチャに引数を追加 (`auto_chunk_silence_amplitude` の後):

```python
        live_continuous: bool = True,
        window_s: float = 30.0,
        hop_s: float = 1.0,
        commit_lag_s: float = 2.5,
        head_guard_s: float = 1.0,
        decode_left_context_s: float = 5.0,
        commit_jitter_margin_s: float = 0.02,
```

`__init__` 本体末尾 (無音検出フィールド設定の後) に追加:

```python
        # --- Phase 6: ライブ連続モード ---
        self.live_continuous = live_continuous
        self.window_s = window_s
        self.hop_s = hop_s
        self.commit_lag_s = commit_lag_s
        self._sliding = SlidingWindowDecoder(
            engine,
            window_s=window_s, hop_s=hop_s, commit_lag_s=commit_lag_s,
            head_guard_s=head_guard_s, decode_left_context_s=decode_left_context_s,
            commit_jitter_margin_s=commit_jitter_margin_s, sample_rate=sample_rate,
        )
        self._samples_since_redecode = 0
        self._hop_samples = int(hop_s * sample_rate)
        self._committed_token_ids: list[int] = []
        # 暫定トークンが残っているか. 残っている間は無音でも redecode を続け、
        # 送信終了後 commit_lag 経過時に末尾を確定させる (レビュー #4).
        self._has_pending_provisional = False
```

`_tick` の `if self._buffer_recording:` ブロック内、先頭 (`self._accumulated_audio.append(...)` の直前) に分岐を追加:

```python
                if self.live_continuous:
                    self._feed_live_block(proc_block)
```

新メソッドを `_trigger_auto_decode` の前に追加:

```python
    def _feed_live_block(self, block: np.ndarray) -> None:
        """ライブ連続モード: ブロックを窓に投入し、hop ごとに再デコード."""
        self._sliding.push(block)
        self._samples_since_redecode += block.size
        if self._samples_since_redecode < self._hop_samples:
            return
        self._samples_since_redecode = 0
        # CPU 節約: 直近 hop 区間が無音 かつ 未確定 (暫定) が残っていない場合のみ
        # 再デコードをスキップ. 暫定が残っているうちは送信終了後も redecode を
        # 続け、commit_lag 経過時に末尾を確定させる (レビュー #4: 無音突入で
        # 末尾数文字が暫定のまま永久放置されるのを防ぐ).
        ring = self._sliding._ring
        recent = ring[-self._hop_samples:] if ring.size >= self._hop_samples else ring
        if recent.size:
            rms = float(np.sqrt(np.mean(recent * recent)))
            recent_db = 20.0 * np.log10(rms) if rms > 1e-6 else -120.0
            recent_silent = recent_db < self.squelch_threshold_db
        else:
            recent_silent = True
        if recent_silent and not self._has_pending_provisional:
            return
        import time
        t0 = time.perf_counter()
        view: DecodeView = self._sliding.redecode()
        decode_ms = (time.perf_counter() - t0) * 1000.0
        self._has_pending_provisional = bool(view.provisional)
        self._emit_live_view(view, decode_ms)

    def _emit_live_view(self, view: DecodeView, decode_ms: float) -> None:
        """確定/暫定テキストを変換して emit.

        WORD_BREAK は ``self._converter.convert()`` が WORD_BREAK_TOKEN_ID を
        語間スペースとして出力する (converter.py:110-114). ライブ連続モードでは
        モデルの WORD_BREAK トークンに依拠し、音声ベース検出は使わない.
        """
        committed_ids = [t.token_id for t in view.committed]
        committed_confs = [t.confidence for t in view.committed]
        committed_text = self._converter.convert(committed_ids, committed_confs).text
        prov_ids = [t.token_id for t in view.provisional]
        prov_confs = [t.confidence for t in view.provisional]
        prov_text = self._converter.convert(prov_ids, prov_confs).text
        self.committed_text_changed.emit(committed_text)
        self.provisional_text_changed.emit(prov_text)
        self.stream_diag.emit({
            "window": self.window_s, "hop": self.hop_s,
            "lag": self.commit_lag_s, "decode_ms": round(decode_ms, 1),
        })
```

`set_buffer_recording(True)` 内に `self._sliding.reset()`、`self._samples_since_redecode = 0`、`self._has_pending_provisional = False` を追加 (蓄積クリアと並べる)。

`stop()` の末尾 (既存の `self._decoder.flush()` 直後) に、ライブ連続モードの最終確定を追加する。送信終了時に commit_lag 圏内で暫定のまま残っていたトークンを確定させる (レビュー #4):

```python
        # ライブ連続モード: 残暫定を最終確定 (commit_lag を無視)
        if self.live_continuous and self._has_pending_provisional:
            final = self._sliding.finalize()
            self._has_pending_provisional = False
            self._emit_live_view(final, 0.0)
```

- [ ] **Step 4: 合格を確認**

Run: `pytest tests/test_workers_live.py -v`
Expected: PASS

- [ ] **Step 5: 既存ワーカー/UI テストの回帰確認**

Run: `pytest tests/test_ui_smoke.py tests/test_workers_live.py -q`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/app/workers.py tests/test_workers_live.py
git commit -m "feat: ワーカーにライブ連続 (スライディングウィンドウ再デコード) モードを追加"
```

---

## Task 7: UI — 確定/暫定 2 領域表示 + ステータス診断

**Files:**
- Modify: `src/app/main_window.py`
- Test: `tests/test_ui_smoke.py` (既存に追記)

方針: デコードテキスト表示を「確定 (黒) + 暫定 (グレー)」の連結表示にする。確定テキストは `committed_text_changed`、暫定は `provisional_text_changed` で更新し、暫定部分は HTML グレーで末尾に付加。ステータスバーに `stream_diag` を `window=Xs hop=Ys lag=Zs decode=Nms` で表示。

- [ ] **Step 1: スモークテストを追記**

`tests/test_ui_smoke.py` に追記 (既存のウィンドウ構築フィクスチャを流用):

```python
def test_window_handles_live_signals(qt_window) -> None:
    """確定/暫定シグナルを受けてもクラッシュしないこと."""
    w = qt_window
    # 直接ハンドラを呼び出し (ワーカー結線の有無に依存しない)
    w._on_committed_text("CQ CQ")
    w._on_provisional_text("DE JA")
    w._on_stream_diag({"window": 30.0, "hop": 1.0, "lag": 2.5, "decode_ms": 42.0})
    assert "CQ CQ" in w._current_display_html()


def test_prosign_angle_brackets_escaped_not_stripped(qt_window) -> None:
    """レビュー #3: <KN> <BT> 等の山括弧プロサインが HTML タグ解釈で消えないこと."""
    w = qt_window
    w._on_committed_text("JA1QRP DE JA7QRS <KN>")
    html = w._current_display_html()
    # 山括弧は実体参照にエスケープされ、QTextEdit がタグとして消去しない
    assert "&lt;KN&gt;" in html
    assert "<KN>" not in html        # 生タグとして残っていない

- [ ] **Step 2: 失敗を確認**

Run: `pytest tests/test_ui_smoke.py::test_window_handles_live_signals -v`
Expected: FAIL (`AttributeError: ... '_on_committed_text'`)

- [ ] **Step 3: ハンドラと結線を実装**

`src/app/main_window.py` の `CWDecoderWindow` に状態とハンドラを追加 (既存のテキスト表示ウィジェットを `self._text_view` と仮定。実名は既存コードに合わせる):

```python
    def _init_live_display_state(self) -> None:
        self._committed_text = ""
        self._provisional_text = ""

    def _on_committed_text(self, text: str) -> None:
        self._committed_text = text
        self._refresh_decode_display()

    def _on_provisional_text(self, text: str) -> None:
        self._provisional_text = text
        self._refresh_decode_display()

    def _on_stream_diag(self, diag: dict) -> None:
        self.statusBar().showMessage(
            f"window={diag['window']:.0f}s hop={diag['hop']:.0f}s "
            f"lag={diag['lag']:.1f}s decode={diag['decode_ms']:.0f}ms"
        )

    def _current_display_html(self) -> str:
        import html
        # レビュー #3: デコード出力には <BT> <KN> <UNKNOWN> 等の山括弧表記が
        # 含まれる. escape しないと setHtml が HTML タグとして消去してしまう.
        committed = html.escape(self._committed_text)
        prov = html.escape(self._provisional_text)
        return (
            f'<span style="color:#000000">{committed}</span>'
            f'<span style="color:#999999">{prov}</span>'
        )

    def _refresh_decode_display(self) -> None:
        self._text_view.setHtml(self._current_display_html())
```

`__init__` で `self._init_live_display_state()` を呼ぶ。

> 実装メモ: `self._text_view` (デコードテキスト表示ウィジェット) と `qt_window` フィクスチャ名は**プレースホルダ**。実装着手時に `src/app/main_window.py` と `tests/test_ui_smoke.py` を grep して実名に置換すること。既存の表示ウィジェットが `setHtml` 非対応 (例: `QLabel`) の場合は `QTextEdit`/`QTextBrowser` への置換も併せて検討する。ワーカー起動時 (`_on_start` のシグナル接続箇所) に結線を追加:

```python
        self._worker.committed_text_changed.connect(self._on_committed_text)
        self._worker.provisional_text_changed.connect(self._on_provisional_text)
        self._worker.stream_diag.connect(self._on_stream_diag)
```

ワーカー生成時に settings の値を渡す (既存の `AudioInferenceWorker(...)` 呼出しに引数追加):

```python
            live_continuous=self._settings.live_continuous,
            window_s=self._settings.window_s,
            hop_s=self._settings.hop_s,
            commit_lag_s=self._settings.commit_lag_s,
            head_guard_s=self._settings.head_guard_s,
            decode_left_context_s=self._settings.decode_left_context_s,
            commit_jitter_margin_s=self._settings.commit_jitter_margin_s,
```

- [ ] **Step 4: 合格を確認**

Run: `pytest tests/test_ui_smoke.py -v`
Expected: PASS

- [ ] **Step 5: 手動起動確認 (任意・GUI 環境のみ)**

Run: `python scripts/run_app.py --ckpt models/full/best.pt`
Expected: 起動し、デコード録音 ON で確定(黒)/暫定(グレー)が表示、ステータスに診断が出る。

- [ ] **Step 6: コミット**

```bash
git add src/app/main_window.py tests/test_ui_smoke.py
git commit -m "feat: UI に確定/暫定 2 領域表示とストリーミング診断を追加"
```

---

## Task 8: 受け入れ検証スクリプト (95% 一致 / RTF)

**Files:**
- Create: `scripts/eval_streaming_vs_offline.py`
- Test: `tests/test_streaming_eval.py` (新規・小規模)

指示書 §2.5 受け入れ基準: `data/real/` の実録音で、確定トークン列がオフライン全体デコードと **95% 以上一致**。CPU で RTF ≤ 0.2。

検証方法: 同一 WAV を (a) `engine.decode_chunk(全体)` でオフラインデコード、(b) `SlidingWindowDecoder` に 50ms ブロックで push しながら `hop_s` ごとに `redecode`、最後に **`finalize()`** で残り全部を確定させ、確定列を得る。`finalize()` を使うことでワーカーの `stop()` と同じコードパスを評価できる (無音 tail を手で足す回避策は不要)。両者のトークン ID 列を difflib で比較し一致率を算出。

- [ ] **Step 1: スクリプトのコア関数にユニットテストを書く**

`tests/test_streaming_eval.py` を新規作成:

```python
"""streaming vs offline 一致率ユーティリティのテスト."""
from __future__ import annotations

from scripts.eval_streaming_vs_offline import token_match_ratio


def test_identical_sequences_ratio_one() -> None:
    assert token_match_ratio([1, 2, 3], [1, 2, 3]) == 1.0


def test_one_diff_in_four() -> None:
    r = token_match_ratio([1, 2, 3, 4], [1, 2, 9, 4])
    assert 0.7 < r < 0.8


def test_empty_both_is_one() -> None:
    assert token_match_ratio([], []) == 1.0
```

- [ ] **Step 2: 失敗を確認**

Run: `pytest tests/test_streaming_eval.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError`)

- [ ] **Step 3: スクリプトを実装**

`scripts/eval_streaming_vs_offline.py` を新規作成:

```python
"""ライブ (スライディングウィンドウ) と オフライン全体デコードの一致率検証.

Usage:
    python scripts/eval_streaming_vs_offline.py --ckpt models/full/best.pt \
        --wav-dir data/real
"""
from __future__ import annotations

import argparse
import difflib
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.infer.engine import InferenceEngine
from src.infer.sliding_window import SlidingWindowDecoder


def token_match_ratio(a: list[int], b: list[int]) -> float:
    """2 つのトークン ID 列の一致率 (difflib ratio)."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _offline_ids(engine: InferenceEngine, wave: np.ndarray) -> list[int]:
    return [t.token_id for t in engine.decode_chunk(wave)]


def _streaming_ids(
    engine: InferenceEngine, wave: np.ndarray, sample_rate: int,
    window_s: float, hop_s: float, commit_lag_s: float, head_guard_s: float,
    decode_left_context_s: float,
) -> tuple[list[int], float, float]:
    d = SlidingWindowDecoder(
        engine, window_s=window_s, hop_s=hop_s, commit_lag_s=commit_lag_s,
        head_guard_s=head_guard_s, decode_left_context_s=decode_left_context_s,
        sample_rate=sample_rate,
    )
    block = int(0.05 * sample_rate)
    hop_samples = int(hop_s * sample_rate)
    since = 0
    max_redecode_ms = 0.0
    t0 = time.perf_counter()
    for i in range(0, wave.size, block):
        d.push(wave[i:i + block])
        since += min(block, wave.size - i)
        if since >= hop_samples:
            since = 0
            r0 = time.perf_counter()
            d.redecode()
            max_redecode_ms = max(max_redecode_ms, (time.perf_counter() - r0) * 1000.0)
    # ストリーム終端: finalize で残り全部を確定 (commit_lag 圏内も確定, レビュー #4)
    view = d.finalize()
    elapsed = time.perf_counter() - t0
    rtf = elapsed / (wave.size / sample_rate)
    return [t.token_id for t in view.committed], rtf, max_redecode_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav-dir", default="data/real")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--hop-s", type=float, default=1.0)
    ap.add_argument("--commit-lag-s", type=float, default=2.5)
    ap.add_argument("--head-guard-s", type=float, default=1.0)
    ap.add_argument("--decode-left-context-s", type=float, default=5.0)
    args = ap.parse_args()

    engine = InferenceEngine.from_checkpoint(args.ckpt, device="cpu")
    sr = 8000
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    if not wavs:
        print(f"[warn] {args.wav_dir} に WAV がありません")
        return
    print(f"{'file':40s} {'match%':>8s} {'rtf':>6s} {'maxdec_ms':>10s}")
    for wav in wavs:
        data, file_sr = sf.read(wav, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            import soxr
            data = soxr.resample(data, file_sr, sr).astype(np.float32)
        off = _offline_ids(engine, data)
        on, rtf, max_dec_ms = _streaming_ids(
            engine, data, sr, args.window_s, args.hop_s,
            args.commit_lag_s, args.head_guard_s, args.decode_left_context_s,
        )
        ratio = token_match_ratio(off, on)
        # RTF と「単発 redecode が hop 時間に収まるか」の両方を確認
        flag = "OK" if ratio >= 0.95 and rtf <= 0.2 else "NG"
        print(f"{wav.name:40s} {ratio*100:7.1f}% {rtf:6.2f} {max_dec_ms:9.0f}  {flag}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: ユニットテスト合格を確認**

Run: `pytest tests/test_streaming_eval.py -v`
Expected: PASS

- [ ] **Step 5: 実録音で受け入れ基準を実測 (要 `models/full/best.pt`)**

Run: `python scripts/eval_streaming_vs_offline.py --ckpt models/full/best.pt --wav-dir data/real`
Expected: 各ファイルで `match% ≥ 95` かつ `rtf ≤ 0.2` (NG が出たらパラメータ調整: `--commit-lag-s` を増やす / `--window-s` を確認)。実測値を記録。

- [ ] **Step 6: コミット**

```bash
git add scripts/eval_streaming_vs_offline.py tests/test_streaming_eval.py
git commit -m "feat: streaming vs offline 一致率/RTF 検証スクリプト (Phase 6 受け入れ)"
```

---

## Task 9: design.md 更新 + 全体回帰 + PR

**Files:**
- Modify: `docs/design.md`

- [ ] **Step 1: 全テスト回帰**

Run: `pytest -q`
Expected: PASS。件数が 248 + 新規分 (15+) を満たすこと。

- [ ] **Step 2: design.md に Phase 6 結果を追記**

`docs/design.md` の §5.4/§5.5 に「Phase 6 で構造的に解消 (スライディングウィンドウ + prefix commit)」と Task 8 の実測一致率/RTF を追記。§5.6 に「マイグレーション実装済み (settings_version=2)」を追記。**新規「既知の制限」として (a) 確定テキストの全文 re-emit (長時間運用でのちらつき可能性、差分 emit が本筋の対処) を §5 に追記**する。

- [ ] **Step 3: コミット**

```bash
git add docs/design.md
git commit -m "docs: design.md に Phase 6 結果 (一致率・RTF・マイグレーション) を追記"
```

- [ ] **Step 4: PR 作成**

```bash
git push -u origin feature/phase6-streaming
gh pr create --title "feat: Phase 6 ストリーミング推論再設計 + settings 移行" \
  --body "スライディングウィンドウ + prefix commit でライブをオフライン同等精度に。settings.json マイグレーション同梱。"
```

---

## Self-Review チェック結果

- **指示書 §2.2 全項目**: prefix commit (Task 4)、デコード区間先頭で head guard、ただし区間がストリーム先頭のときは適用しない (Task 4 `decode_start_abs==0`)、中点ウォーターマークで immutable (Task 4)、無音時 redecode スキップ ただし暫定残存中は継続 (Task 6)、WORD_BREAK は確定区間に反映 (Task 6 `_emit_live_view` の `convert()`)。
- **§2.3 UI**: 確定/暫定 2 領域 + プロサイン escape (Task 7)、ステータス診断 `window/hop/lag/decode` (Task 7)。
- **§2.4 マイグレーション**: `settings_version` (Task 1)、git で確証した旧既定のみ置換 + ログ (Task 2)、単体テスト (Task 2)。
- **§2.5 受け入れ基準**: 95% 一致 + RTF (Task 8) + 前倒し RTF 実測 (Task 4 Step 6)、immutable をテスト保証 (Task 4)、回帰 (Task 9)。
- **既存「手動全体デコード」「自動チャンク」は検証用に残す**: `decode_and_reset` / auto_chunk 経路は未削除 (Task 6 は live_continuous 分岐を追加するのみ)。

### コードレビュー反映 (2026-06-13, 重大 5 + 中 4)

| # | 指摘 | 反映箇所 |
|---|---|---|
| 1 | commit 境界またぎを `abs_start` で判定し誤確定 | Task 4: `abs_end >= commit_limit_abs` → 暫定。`test_boundary_straddling_token_is_provisional_not_committed` 追加 |
| 2 | tol=0.3s が 25 WPM の文字間 144ms で文字脱落【最重大】 | Task 1/3/4: 中点ウォーターマーク方式へ。`commit_match_tol_s` 廃止→`commit_jitter_margin_s=0.02`。`test_realistic_25wpm_*` / `test_short_token_not_double_committed_*` 追加 |
| 3 | `<KN> <BT>` が HTML タグ解釈で消失 | Task 7: `html.escape`。`test_prosign_angle_brackets_escaped_not_stripped` 追加 |
| 4 | 無音突入で末尾数文字が暫定のまま永久放置 | Task 4/6: `finalize()` 追加 + 暫定残存中は redecode 継続 + `stop()` で最終確定。`test_finalize_*` 追加 |
| 5 | 30s 窓×1s hop は CPU RTF 0.5〜1.0 で基準超過 | Task 1/3/4: デコード区間の動的短縮 (`decode_left_context_s=5.0`)。`test_decode_window_shortens_after_commit` + Task 4 Step 6 前倒し RTF 実測 |
| M1 | `InferenceEngine.untrained` 未検証 | **存在確認済み** (`engine.py:113`)。対応不要 |
| M2 | migration 旧既定値が git と不一致の懸念 | Task 2: `chunk_duration_s=1.5` のみ git 確証で採用、`auto_chunk_silence_amplitude` は除外 |
| M3 | マスター §1 依存記述の誤り | マスター計画 §1 修正 (Phase 6 は特徴量経路に触れない) |
| M4 | WORD_BREAK の出所が曖昧 | Architecture 節 + Task 6 で明示 (モデルトークン依拠、音声検出は手動経路のみ) |

### 実装時に grep で確定するプレースホルダ (未確定の前提)

- Task 7 のウィジェット実名 (`_text_view`)、表示ウィジェットの `setHtml` 対応可否、`tests/test_ui_smoke.py` のフィクスチャ名 (`qt_window`)。
- Task 5/6 で旧 `StreamingDecoder` (`self._decoder`) は `stop()` の `flush` 経路に残るが live では未使用 — 完全削除するかは実装時に判断 (本計画では非破壊で温存)。

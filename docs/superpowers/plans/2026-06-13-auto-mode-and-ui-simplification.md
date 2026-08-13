# 自動モード切替 + UI 簡素化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** モデルが出力する `[ホレ]`/`[ラタ]` プロサインで欧文⇄和文を自動切替し、ライブ連続モードに一本化してレガシー手動デコード経路を撤去する。

**Architecture:** 「符号→文字」の自動モード切替は変換表層 (`TokenConverter`) に閉じる。トークン列を 1 回走査し、ローカルな「現在サブモード」を `[ホレ]`/`[ラタ]` で切替えながら、各トークンを現在サブモードの表で変換する。NN・`SlidingWindowDecoder` はモード非依存のまま。worker は確定→暫定のモード引き継ぎと現在モード表示のみ担う。

**Tech Stack:** Python 3.11, PySide6 (Qt), numpy, pytest。設計書: `docs/superpowers/specs/2026-06-13-auto-mode-and-ui-simplification-design.md`。

---

## ファイル構成

| ファイル | 役割 | 変更種別 |
|---|---|---|
| `src/tokens/morse_tokens.py` | ホレ/ラタ符号定数 + 表示モード型を追加 | Modify |
| `src/tokens/converter.py` | `mode="auto"` + `initial_mode`/`final_mode` 対応 | Modify |
| `src/app/workers.py` | 確定→暫定モード引き継ぎ・現在モード表示・レガシー撤去 | Modify |
| `src/app/main_window.py` | モード3択・デコードトグル・レガシー UI 撤去 | Modify |
| `src/infer/settings.py` | 自動モード許容・settings_version 3・キー撤去 | Modify |
| `scripts/run_app.py` | `--chunk`/`--overlap` 撤去 | Modify |
| `src/infer/stream.py` | StreamingDecoder (参照消滅後に削除) | Delete |
| `tests/test_converter_auto_mode.py` | 自動モード変換器テスト | Create |
| `tests/test_settings_migration.py` | v2→v3 マイグレーションテスト追加 | Modify |
| `tests/test_workers_live.py` | モード引き継ぎテスト追加・レガシーテスト除去 | Modify |

---

## Task 1: ホレ/ラタ符号定数と表示モード型を追加

符号定義の単一ソース (`morse_tokens.py`) にプロサイン符号定数と、UI 用 3 値モード型を追加する。

**Files:**
- Modify: `src/tokens/morse_tokens.py`

- [ ] **Step 1: 定数と型を追加**

`src/tokens/morse_tokens.py` の `Mode = Literal["european", "japanese"]` (21 行目付近) の直後に追加:

```python
# UI 表示モード: 固定 2 モード + 自動切替.
DisplayMode = Literal["european", "japanese", "auto"]

# 和文開始/終了プロサインの符号 (自動モード切替のトリガ).
# ラタは欧文プロサイン SN と完全に同符号 (・・・-・).
HORE_CODE: Final[str] = "-・・---"   # 和文開始 (→ japanese)
RATA_CODE: Final[str] = "・・・-・"  # 和文終了 (japanese 中のみ → european)
```

`__all__` (334 行目付近) に `"DisplayMode"`, `"HORE_CODE"`, `"RATA_CODE"` を追加する。

- [ ] **Step 2: import 確認**

`Final` と `Literal` は既に `morse_tokens.py` で import 済みであることを確認 (`from typing import Final, Literal` 等)。未 import なら追加する。

- [ ] **Step 3: 値を検証**

Run: `python -c "from src.tokens.morse_tokens import HORE_CODE, RATA_CODE, TOKEN_TO_ID, EUROPEAN_TABLE, JAPANESE_TABLE; assert JAPANESE_TABLE[HORE_CODE]=='[ホレ]'; assert JAPANESE_TABLE[RATA_CODE]=='[ラタ]'; assert EUROPEAN_TABLE[RATA_CODE]=='[SN]'; assert HORE_CODE not in EUROPEAN_TABLE; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/tokens/morse_tokens.py
git commit -m "feat: ホレ/ラタ符号定数と DisplayMode 型を追加"
```

---

## Task 2: TokenConverter を自動モード対応に拡張

`mode="auto"` を受け付け、走査中に現在サブモードを切替える。`convert()` に `initial_mode` 引数、`ConvertResult` に `final_mode` を追加する。固定モード (european/japanese) の挙動は不変。

**Files:**
- Modify: `src/tokens/converter.py`
- Test: `tests/test_converter_auto_mode.py` (Create)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_converter_auto_mode.py` を新規作成:

```python
"""自動モード変換器のテスト (ホレ/ラタによる欧文⇄和文切替)."""
from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import TOKEN_TO_ID

HORE = TOKEN_TO_ID["-・・---"]   # 和文開始
RATA = TOKEN_TO_ID["・・・-・"]  # 和文終了 / 欧文 SN (同符号, id 共通)
A_I = TOKEN_TO_ID["・-"]         # 欧文 A / 和文 イ


def _conv(ids):
    return TokenConverter(mode="auto").convert(ids)


def test_starts_in_european():
    # 初期は欧文: ・- は A
    assert _conv([A_I]).text == "A"


def test_hore_switches_to_japanese():
    # ホレ以降は和文表: ・- は イ
    res = _conv([HORE, A_I])
    assert res.text == "[ホレ]イ"
    assert res.final_mode == "japanese"


def test_rata_switches_back_to_european():
    # ホレ→和文(イ)→ラタ→欧文(A)
    res = _conv([HORE, A_I, RATA, A_I])
    assert res.text == "[ホレ]イ[ラタ]A"
    assert res.final_mode == "european"


def test_rata_code_in_european_is_sn_no_switch():
    # 欧文中の ・・・-・ は SN 表示でモード切替しない
    res = _conv([RATA, A_I])
    assert res.text == "[SN]A"
    assert res.final_mode == "european"


def test_hore_idempotent_when_already_japanese():
    res = _conv([HORE, HORE, A_I])
    assert res.text == "[ホレ][ホレ]イ"
    assert res.final_mode == "japanese"


def test_initial_mode_continues_from_japanese():
    # 暫定引き継ぎ: 和文で開始すると ・- は イ
    res = TokenConverter(mode="auto").convert([A_I], initial_mode="japanese")
    assert res.text == "イ"
    assert res.final_mode == "japanese"


def test_dakuten_composes_only_in_japanese_segment():
    # 和文セグメント内: ハ(-・・・) + 濁点(・・) → バ
    ha = TOKEN_TO_ID["-・・・"]
    dak = TOKEN_TO_ID["・・"]
    res = _conv([HORE, ha, dak])
    assert res.text == "[ホレ]バ"


def test_fixed_european_unchanged_and_final_mode():
    res = TokenConverter(mode="european").convert([A_I])
    assert res.text == "A"
    assert res.final_mode == "european"


def test_fixed_japanese_unchanged_and_final_mode():
    res = TokenConverter(mode="japanese").convert([A_I])
    assert res.text == "イ"
    assert res.final_mode == "japanese"
```

- [ ] **Step 2: 失敗を確認**

Run: `python -m pytest tests/test_converter_auto_mode.py -q`
Expected: FAIL (`ConvertResult` に `final_mode` が無い / `mode="auto"` で `ValueError` / `initial_mode` 引数無し)。

- [ ] **Step 3: ConvertResult に final_mode を追加**

`src/tokens/converter.py` の `ConvertResult` (57-60 行) を変更:

```python
@dataclass
class ConvertResult:
    text: str
    fallback_log: list[FallbackEvent] = field(default_factory=list)
    final_mode: Mode = "european"
```

import に `Mode` が含まれていることを確認 (既に 28 行で import 済み)。`DisplayMode` も import に追加:

```python
from src.tokens.morse_tokens import (
    BLANK_TOKEN_ID,
    DAKUTEN_CHAR,
    DAKUTEN_COMPOSE,
    DisplayMode,
    EUROPEAN_TABLE,
    HANDAKUTEN_CHAR,
    HANDAKUTEN_COMPOSE,
    HORE_CODE,
    ID_TO_TOKEN,
    JAPANESE_TABLE,
    Mode,
    RATA_CODE,
    WORD_BREAK_TOKEN_ID,
)
```

- [ ] **Step 4: コンストラクタを auto 対応に**

`__init__` (72-83 行) を変更:

```python
def __init__(self, mode: DisplayMode, confidence_threshold: float = 0.5) -> None:
    if mode not in ("european", "japanese", "auto"):
        raise ValueError(f"Unknown mode: {mode!r}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
        )
    self.mode: DisplayMode = mode
    self._auto: bool = mode == "auto"
    self.confidence_threshold: float = confidence_threshold
    # auto では走査中に表を切替えるため、固定表は持たない (european を既定保持).
    self._table: dict[str, str] = (
        JAPANESE_TABLE if mode == "japanese" else EUROPEAN_TABLE
    )

@staticmethod
def _table_for(active: Mode) -> dict[str, str]:
    return JAPANESE_TABLE if active == "japanese" else EUROPEAN_TABLE
```

- [ ] **Step 5: convert() を現在サブモード追従に書き換え**

`convert()` (85-166 行) を次の実装に置換する。`initial_mode` 引数を追加し、ループ内で `active`/`table` を使う。プロサイン処理は確信度判定の後に行う。

```python
def convert(
    self,
    token_ids: list[int],
    confidences: list[float] | None = None,
    initial_mode: Mode | None = None,
) -> ConvertResult:
    """トークン ID 列を表示テキストに変換.

    ``confidences`` を渡すと閾値判定を有効化. ``None`` ならすべて確信度 1.0 扱い.
    ``initial_mode`` は走査開始時のサブモード. 自動モードで暫定列を確定列の末尾
    モードから継続させる用途. ``None`` のとき auto は ``"european"`` から、
    固定モードは ``self.mode`` から開始する.
    """
    if confidences is not None and len(confidences) != len(token_ids):
        raise ValueError(
            f"confidences length {len(confidences)} != token_ids length "
            f"{len(token_ids)}"
        )

    if initial_mode is not None:
        active: Mode = initial_mode
    elif self._auto:
        active = "european"
    else:
        active = self.mode  # type: ignore[assignment]
    table = self._table_for(active)

    out_chars: list[str] = []
    log: list[FallbackEvent] = []
    composable_at: int | None = None

    for i, tid in enumerate(token_ids):
        if tid == BLANK_TOKEN_ID:
            continue

        if tid == WORD_BREAK_TOKEN_ID:
            if out_chars and out_chars[-1] != " ":
                out_chars.append(" ")
            composable_at = None
            continue

        conf = confidences[i] if confidences is not None else 1.0
        token = ID_TO_TOKEN.get(tid)

        if token is None:
            self._emit_fallback(
                out_chars, log, i, tid, "<UNKNOWN>", "TABLE_MISS", conf
            )
            composable_at = None
            continue

        if conf < self.confidence_threshold:
            self._emit_fallback(
                out_chars, log, i, tid, token.code, "LOW_CONFIDENCE", conf
            )
            composable_at = None
            continue

        # --- 自動モード: プロサインによる切替 (確信度を満たしたトークンのみ) ---
        if self._auto:
            if token.code == HORE_CODE:
                out_chars.append("[ホレ]")
                active = "japanese"
                table = self._table_for(active)
                composable_at = None
                continue
            if token.code == RATA_CODE and active == "japanese":
                out_chars.append("[ラタ]")
                active = "european"
                table = self._table_for(active)
                composable_at = None
                continue
            # それ以外 (欧文中の RATA=SN を含む) は通常変換へ落ちる.

        display = table.get(token.code)
        if display is None:
            self._emit_fallback(
                out_chars, log, i, tid, token.code, "TABLE_MISS", conf
            )
            composable_at = None
            continue

        if active == "japanese" and display in (DAKUTEN_CHAR, HANDAKUTEN_CHAR):
            compose_map = (
                DAKUTEN_COMPOSE if display == DAKUTEN_CHAR else HANDAKUTEN_COMPOSE
            )
            if composable_at is not None:
                composed = compose_map.get(out_chars[composable_at])
                if composed is not None:
                    out_chars[composable_at] = composed
                    composable_at = None
                    continue
            self._emit_fallback(
                out_chars, log, i, tid, token.code, "TABLE_MISS", conf
            )
            composable_at = None
            continue

        out_chars.append(display)
        composable_at = (
            len(out_chars) - 1
            if active == "japanese"
            and (display in DAKUTEN_COMPOSE or display in HANDAKUTEN_COMPOSE)
            else None
        )

    return ConvertResult(
        text="".join(out_chars), fallback_log=log, final_mode=active
    )
```

注意: 旧コードの `self.mode == "japanese"` 判定 2 箇所 (濁点合成・composable_at) は
`active == "japanese"` に置換済み。`self._table` 直接参照はループ内から除去 (ローカル
`table` を使用)。

- [ ] **Step 6: convert_timed の partial 呼び出しを確認**

`convert_timed` (168 行) と `_reinsert_spaces` (237 行) は内部で `self.convert(ids[:n], confs[:n])` を呼ぶ。`initial_mode` 省略時 auto は european 開始で、同一接頭辞は同一モード遷移になるため `idx_to_pos` は整合する。**変更不要**だが、念のため `_reinsert_spaces` 冒頭の partial ループがそのまま動くことを目視確認する。

- [ ] **Step 7: テストが通ることを確認**

Run: `python -m pytest tests/test_converter_auto_mode.py -q`
Expected: PASS (9 件)。

- [ ] **Step 8: 既存テスト回帰確認**

Run: `python -m pytest tests/test_converter.py tests/test_converter_timed.py -q`
Expected: PASS (固定モード挙動が不変)。失敗時は `final_mode` 既定や `active` 置換漏れを調査。

- [ ] **Step 9: Commit**

```bash
git add src/tokens/converter.py tests/test_converter_auto_mode.py
git commit -m "feat: TokenConverter に自動モード(ホレ/ラタ切替)と initial/final_mode を追加"
```

---

## Task 3: worker のレガシー経路を撤去しライブ一本化

手動バッファ蓄積・自動チャンク・旧 StreamingDecoder を削除し、`_buffer_recording` を `_decoding` にリネーム。確定→暫定のモード引き継ぎと現在モード表示を実装する。

**Files:**
- Modify: `src/app/workers.py`
- Test: `tests/test_workers_live.py`

- [ ] **Step 1: モード引き継ぎの失敗テストを追加**

`tests/test_workers_live.py` に追加 (既存の import/フィクスチャを流用。worker の `_emit_live_view` がモードを引き継ぐことを、`_converter` を auto にして検証する)。最小の単体として `_emit_live_view` のロジックを直接突く代わりに、変換の引き継ぎを worker 経由で確認する純粋関数テストを置く:

```python
def test_auto_mode_provisional_inherits_committed_mode():
    """確定列がホレで和文に入ったら、暫定列も和文で変換される."""
    from src.tokens.converter import TokenConverter
    from src.tokens.morse_tokens import TOKEN_TO_ID

    conv = TokenConverter(mode="auto")
    hore = TOKEN_TO_ID["-・・---"]
    a_i = TOKEN_TO_ID["・-"]

    res_c = conv.convert([hore], initial_mode="european")
    res_p = conv.convert([a_i], initial_mode=res_c.final_mode)
    assert res_c.text == "[ホレ]"
    assert res_c.final_mode == "japanese"
    assert res_p.text == "イ"   # european 開始なら "A" になってしまう
```

Run: `python -m pytest tests/test_workers_live.py::test_auto_mode_provisional_inherits_committed_mode -q`
Expected: PASS (Task 2 完了済みなら成立)。この時点では仕様確認テスト。

- [ ] **Step 2: worker の `_emit_live_view` をモード引き継ぎに変更**

`src/app/workers.py` の `_emit_live_view` (330-348 行) の変換 2 行を変更:

```python
committed_ids = [t.token_id for t in view.committed]
committed_confs = [t.confidence for t in view.committed]
res_c = self._converter.convert(committed_ids, committed_confs, initial_mode="european")
prov_ids = [t.token_id for t in view.provisional]
prov_confs = [t.confidence for t in view.provisional]
res_p = self._converter.convert(prov_ids, prov_confs, initial_mode=res_c.final_mode)
committed_text = res_c.text
prov_text = res_p.text
self.committed_text_changed.emit(committed_text)
self.provisional_text_changed.emit(prov_text)
self.current_mode_changed.emit(res_c.final_mode)   # ステータス表示用
self.stream_diag.emit({
    "window": self.window_s, "hop": self.hop_s,
    "lag": self.commit_lag_s, "decode_ms": round(decode_ms, 1),
})
```

- [ ] **Step 3: current_mode_changed シグナルを追加**

`AudioInferenceWorker` のシグナル定義群 (`committed_text_changed = Signal(str)` 102 行付近) に追加:

```python
current_mode_changed = Signal(str)   # auto モードの現在サブモード ("european"/"japanese")
```

`full_decode_completed = Signal(str)` (101 行) を**削除**する。

- [ ] **Step 4: コンストラクタからレガシー引数・状態を削除**

`__init__` (106-195 行) を整理:
- 削除する引数: `chunk_duration_s`, `overlap_duration_s`, `auto_chunk_enabled`,
  `auto_chunk_silence_sec`, `auto_chunk_min_buffer_sec`, `auto_chunk_silence_amplitude`,
  `live_continuous`。
- `mode: Mode = "european"` を `mode: str = "european"` に変更 (auto を受けるため)。
- 削除する本体: `self._decoder = StreamingDecoder(...)` (150-155 行)、
  `self._accumulated_audio`, `self._buffer_recording`, `self.auto_chunk_*`,
  `self._silence_run_samples`, `self._waiting_for_first_silence`, `self.live_continuous`。
- 追加: `self._decoding = False` (旧 `_buffer_recording` の置換)。
- `self._converter = TokenConverter(mode=mode, ...)` はそのまま (auto を受けられる)。

import 行 (13 行) `from src.infer.stream import StreamToken, StreamingDecoder` を削除。
`StreamToken`/`StreamingDecoder` が他で使われていないことを確認:
Run: `grep -n "StreamToken\|StreamingDecoder" src/app/workers.py`
Expected: 出力なし (削除後)。

- [ ] **Step 5: set_mode を auto 対応に**

`set_mode` (199-209 行) を変更:

```python
@Slot(str)
def set_mode(self, mode: str) -> None:
    if mode not in ("european", "japanese", "auto"):
        return
    self.mode = mode
    self._converter = TokenConverter(
        mode=mode, confidence_threshold=self.confidence_threshold
    )
    self._sliding.reset()
```

`self._decoder.reset()` 行を削除 (B2 修正の `_sliding.reset()` は維持)。

- [ ] **Step 6: レガシーメソッドを削除**

以下を `workers.py` から削除:
- `set_buffer_recording` (257-) → `set_decoding` にリネーム実装:
  ```python
  @Slot(bool)
  def set_decoding(self, on: bool) -> None:
      self._decoding = bool(on)
      self.status.emit("デコード中" if on else "デコード停止")
  ```
- `set_auto_chunk_enabled` (274-) を削除。
- `_trigger_auto_decode` (350-356) を削除。
- `decode_and_reset` (358-388) を削除。
- `_process_recording_block` (460-494) を削除 (給餌は `_tick` から直接呼ぶ)。
- `_emit_tokens` (496-501) を削除。

- [ ] **Step 7: start()/stop()/_tick の参照を整理**

- `start()` (284-301) 内の `self._buffer_recording = False` と
  `self._accumulated_audio.clear()` (295-296 行) を `self._decoding = False` のみに置換。
- `stop()` (390-412) の旧 stream flush 分岐 (403-406 行
  `if self._decoder is not None: tokens = self._decoder.flush() ...`) を削除。
  ライブ finalize 分岐 (408-411 行) は `if self._has_pending_provisional:` に簡約
  (`self.live_continuous and` を除去)。
- `_tick` (415-458) の給餌部 (438-439 行) を変更:
  ```python
  if self._decoding:
      self._feed_live_block(proc_block)
  ```
  診断ログ (449-458 行) の `_accumulated_audio` 参照と `rec_mark` を簡約:
  ```python
  self._diag_tick += 1
  if self._diag_tick >= 50:
      self._diag_tick = 0
      rec_mark = "DEC" if self._decoding else "..."
      self.status.emit(f"lvl={level_db:.1f}dB [{rec_mark}]")
  ```

- [ ] **Step 8: テストとスモークを実行**

Run: `python -m pytest tests/test_workers_live.py -q`
Expected: PASS。旧 `decode_and_reset`/auto_chunk を参照するテストがあれば次タスクで除去。
Run: `grep -rn "decode_and_reset\|auto_chunk\|buffer_recording\|live_continuous\|full_decode_completed\|StreamingDecoder" src/app/workers.py`
Expected: 出力なし。

- [ ] **Step 9: Commit**

```bash
git add src/app/workers.py tests/test_workers_live.py
git commit -m "refactor: worker をライブ連続一本化しレガシー経路を撤去 + 自動モード引き継ぎ"
```

---

## Task 4: main_window の UI を更新

モード3択・デコードトグル化・レガシー UI とシグナル接続の撤去・現在モード表示を行う。

**Files:**
- Modify: `src/app/main_window.py`
- Test: `tests/test_ui_smoke.py`

- [ ] **Step 1: モードコンボに「自動」を追加**

`mode_combo.addItems([...])` (83 行) を変更:

```python
self.mode_combo.addItems(["欧文 (european)", "和文 (japanese)", "自動 (auto)"])
```

`mode_combo.setCurrentIndex(...)` (84 行) を 3 値対応に:

```python
_mode_index = {"european": 0, "japanese": 1, "auto": 2}
self.mode_combo.setCurrentIndex(_mode_index.get(self._settings.mode, 0))
```

`_current_mode()` を 3 値返却に変更 (定義箇所を `grep -n "_current_mode" src/app/main_window.py` で特定):

```python
def _current_mode(self) -> str:
    return ("european", "japanese", "auto")[self.mode_combo.currentIndex()]
```

戻り値型 `Mode` を `str` に緩める (import 維持)。

- [ ] **Step 2: デコードボタンをトグル1つに置換**

`buffer_record_btn` / `decode_btn` 定義 (99-113 行) を次に置換:

```python
self.start_btn = QPushButton("開始")
self.stop_btn = QPushButton("停止")
self.stop_btn.setEnabled(False)
self.decode_toggle_btn = QPushButton("● デコード中")
self.decode_toggle_btn.setCheckable(True)
self.decode_toggle_btn.setEnabled(False)
self.decode_toggle_btn.setToolTip("ライブデコードの ON/OFF を切替")
top.addWidget(self.start_btn)
top.addWidget(self.stop_btn)
top.addWidget(self.decode_toggle_btn)
```

ボタン文言をトグル状態で更新するスロットを追加:

```python
def _on_decode_toggled(self, on: bool) -> None:
    self.decode_toggle_btn.setText("■ デコード停止" if on else "● デコード中")
    self.request_set_decoding.emit(on)
```

- [ ] **Step 3: 自動チャンク チェックを削除**

`auto_chunk_check` 定義 (160-165 行) と、それを使う全行を削除。

- [ ] **Step 4: シグナル定義と接続を整理**

クラスのシグナル定義群から削除:
`request_full_decode`, `request_set_buffer_recording`, `request_set_auto_chunk`。
追加: `request_set_decoding = Signal(bool)`。

`_on_start` (222 行〜) の接続を整理:
- 削除: `self._worker.full_decode_completed.connect(...)` (261 行)、
  `self.request_full_decode.connect(...)` (270 行)、
  `self.request_set_buffer_recording.connect(...)` (271 行)、
  `self.request_set_auto_chunk.connect(...)` (279 行)、
  `self.auto_chunk_check.toggled.connect(...)` (280 行)。
- 変更: worker 生成 (228-253 行) から削除した引数 (`chunk_duration_s`,
  `overlap_duration_s`, `auto_chunk_*`, `live_continuous`) を除去。
- 追加: `self.request_set_decoding.connect(self._worker.set_decoding)`、
  `self._worker.current_mode_changed.connect(self._on_current_mode)`。

イベント接続 (195-203 行) を変更:
- 削除: `self.buffer_record_btn.toggled.connect(...)`、
  `self.decode_btn.clicked.connect(...)`。
- 追加: `self.decode_toggle_btn.toggled.connect(self._on_decode_toggled)`。

- [ ] **Step 5: 現在モード表示スロットを追加**

```python
def _on_current_mode(self, mode: str) -> None:
    if self._current_mode() != "auto":
        return
    label = {"european": "欧文", "japanese": "和文"}.get(mode, mode)
    self.statusBar().showMessage(f"自動 (現在: {label})")
```

`full_decode_completed` を受けていた `_on_full_decode_result` 等のスロットが
不要になれば削除する (`grep -n "_on_full_decode" src/app/main_window.py` で確認)。

- [ ] **Step 6: ボタン enable 状態の更新**

`_on_start` 末尾 (287-289 行) の `self.buffer_record_btn.setEnabled(True)` を
`self.decode_toggle_btn.setEnabled(True)` に置換。`_on_stop` 側で
`self.decode_toggle_btn.setChecked(False)` / `setEnabled(False)` を行う
(旧 `buffer_record_btn` を扱っていた箇所を同様に置換)。

- [ ] **Step 7: スモークテスト**

Run: `python -m pytest tests/test_ui_smoke.py -q`
Expected: PASS。`buffer_record_btn`/`decode_btn`/`auto_chunk_check` を参照する
スモークがあれば `decode_toggle_btn` 参照に更新する。

- [ ] **Step 8: Commit**

```bash
git add src/app/main_window.py tests/test_ui_smoke.py
git commit -m "feat: モード3択(自動)・デコードトグル化・レガシーUI撤去"
```

---

## Task 5: 設定の自動モード許容と settings_version 3 マイグレーション

**Files:**
- Modify: `src/infer/settings.py`
- Test: `tests/test_settings_migration.py`

> 注意: 実 API は `load_settings(path)` / `save_settings(settings, path)` /
> `migrate_settings_dict(data)` / 定数 `CURRENT_SETTINGS_VERSION` (`src/infer/settings.py`)。
> `AppSettings.from_dict` は有効キーのみ採用するため、dataclass からフィールドを
> 削除すれば旧キーは自動的に脱落する。

- [ ] **Step 1: 失敗するマイグレーションテストを追加**

`tests/test_settings_migration.py` に追加:

```python
def test_v2_to_v3_drops_legacy_keys_and_keeps_working():
    from src.infer.settings import migrate_settings_dict, CURRENT_SETTINGS_VERSION

    raw = {
        "mode": "european",
        "settings_version": 2,
        "auto_chunk_enabled": True,
        "auto_chunk_silence_sec": 1.2,
        "live_continuous": True,
        "chunk_duration_s": 5.0,
        "chunk_overlap_s": 0.5,
        "bpf_enabled": True,
    }
    merged, changed = migrate_settings_dict(raw)
    assert changed is True
    assert merged["settings_version"] == CURRENT_SETTINGS_VERSION == 3
    # 削除されたフィールドは merged に含まれない
    for legacy in ("auto_chunk_enabled", "live_continuous", "chunk_duration_s",
                   "chunk_overlap_s", "auto_chunk_silence_sec"):
        assert legacy not in merged
    assert merged["bpf_enabled"] is True


def test_mode_auto_round_trips(tmp_path):
    from src.infer.settings import AppSettings, load_settings, save_settings

    p = tmp_path / "s.json"
    s = AppSettings()
    s.mode = "auto"
    save_settings(s, p)
    assert load_settings(p).mode == "auto"
```

Run: `python -m pytest tests/test_settings_migration.py -q`
Expected: FAIL (旧キーが残る / `CURRENT_SETTINGS_VERSION` が 2 / `mode="auto"` で型不一致)。

- [ ] **Step 2: AppSettings からレガシーフィールドを削除し mode 型を緩める**

`src/infer/settings.py` の import を変更: `from src.tokens.morse_tokens import DisplayMode`。
`AppSettings` の `mode: Mode = "european"` を `mode: DisplayMode = "european"` に変更。
`AppSettings` から削除:
`chunk_duration_s` (28), `chunk_overlap_s` (29), `auto_chunk_enabled` (47),
`auto_chunk_silence_sec` (48), `auto_chunk_min_buffer_sec` (49),
`auto_chunk_silence_amplitude` (50), `live_continuous` (54)。
`settings_version` 既定値 (53 行) を `3` に変更。

- [ ] **Step 3: 置換表と CURRENT_SETTINGS_VERSION を更新**

`_V1_DEFAULT_REPLACEMENTS` (82-85 行) から `chunk_duration_s` エントリを削除し、
空 dict `{}` にする (置換対象が無くなるため)。
`CURRENT_SETTINGS_VERSION = 2` (87 行) を `= 3` に変更。
`migrate_settings_dict` は `merged` を `AppSettings().to_dict()` 土台 + 有効キー上書きで
構築しているため、削除済みフィールドは自動的に `merged` から脱落する (実装変更不要)。

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_settings_migration.py -q`
Expected: PASS。既存の v1→v2 を前提とするテストが `CURRENT_SETTINGS_VERSION`/version を
ハードコードしている場合は、新値 3 に更新する
(`grep -n "settings_version\|== 2\|CURRENT_SETTINGS" tests/test_settings_migration.py`)。

- [ ] **Step 5: Commit**

```bash
git add src/infer/settings.py tests/test_settings_migration.py
git commit -m "feat: settings に auto モードを許容し v3 マイグレーション(レガシーキー撤去)"
```

---

## Task 6: run_app.py / 起動経路と StreamingDecoder の撤去

**Files:**
- Modify: `scripts/run_app.py`, `src/app/main_window.py`
- Delete: `src/infer/stream.py` (参照消滅を確認後)

- [ ] **Step 1: run_app.py の CLI 引数を削除**

`scripts/run_app.py` の `--chunk`/`--overlap` 定義と、`run_app_main(...)` への
`chunk_duration_s`/`chunk_overlap_s` 受け渡しを削除。`main()` は `--ckpt` のみ残す。

- [ ] **Step 2: main_window.main() を整理**

`src/app/main_window.py` の `main(checkpoint_path=..., chunk_duration_s=..., chunk_overlap_s=...)` (497-499 行)
から `chunk_duration_s`/`chunk_overlap_s` 引数と代入 (508-511 行) を削除。
`[init]` ログ (524-526 行) の chunk/overlap 表示を削除し、ckpt と mode のみ表示に簡約。

- [ ] **Step 3: StreamingDecoder の参照を全削除確認**

Run: `grep -rn "StreamingDecoder\|from src.infer.stream\|infer.stream" src/ scripts/ tests/ | grep -v pyc`
Expected: `src/infer/stream.py` 自身の定義のみ (テスト `test_streaming_*` が残る場合は次ステップで対応)。

- [ ] **Step 4: stream.py とその専用テストを削除/整理**

`src/infer/stream.py` を参照するテスト (`tests/test_stream*.py` 等) を確認:
Run: `grep -rln "infer.stream\|StreamingDecoder" tests/ | grep -v pyc`
- 旧 StreamingDecoder 専用テストは削除。
- `src/infer/stream.py` を削除。
- `sliding_window.py` 等が `stream.py` の型を import していないことを再確認
  (`grep -rn "infer.stream" src/`).

- [ ] **Step 5: アプリ起動スモーク**

Run: `python -c "import scripts.run_app"` および `python -m pytest tests/test_ui_smoke.py -q`
Expected: import エラーなし / PASS。

- [ ] **Step 6: Commit**

```bash
git add scripts/run_app.py src/app/main_window.py
git rm src/infer/stream.py
git commit -m "refactor: 旧 StreamingDecoder と chunk/overlap 起動引数を撤去"
```

---

## Task 7: 全体回帰と最終検証

**Files:** なし (検証のみ)

- [ ] **Step 1: 残存レガシー参照の全削除確認**

Run: `grep -rn "auto_chunk\|decode_and_reset\|buffer_recording\|live_continuous\|full_decode_completed\|StreamingDecoder\|chunk_duration_s\|chunk_overlap_s" src/ scripts/ | grep -v pyc`
Expected: 出力なし。

- [ ] **Step 2: 全テスト実行**

Run: `python -m pytest -q`
Expected: 全 PASS (失敗・エラー 0)。削除に伴い件数は減少してよい。

- [ ] **Step 3: アプリ起動確認 (手動)**

Run: `python scripts/run_app.py --ckpt models/full/best.pt`
確認: モードコンボに「自動」がある / ボタンが `[開始][停止][● デコード中]` のみ /
下段に自動チャンクが無い / 起動エラーが出ない。確認後ウィンドウを閉じる。

- [ ] **Step 4: 仕上げ — design.md 追記 (任意)**

`docs/design.md` に自動モード切替と UI 簡素化の節を追記 (実機確認後でも可)。

- [ ] **Step 5: 最終コミット (必要時)**

```bash
git add -A
git commit -m "docs: 自動モード/UI簡素化の結果を反映"
```

---

## 受け入れ基準 (design.md §9 と対応)

- 既存 + 新規テストが全緑。
- 自動モードで `[ホレ]`…`[ラタ]` を含む列がホレ以降和文表・ラタ以降欧文表で表示。
- UI に手動デコード/自動チャンク操作が残っていない。
- 旧 settings.json から起動してもエラーなく version 3 に更新。

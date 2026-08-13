# 送信専用符号の拡張 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ITU/無線局運用規則に符号がある欧文記号 `: ' " ( ) ×` と和文括弧 (`{KAKKO}` `{TOJI}` マーカー) を、受信語彙に触れず送信専用表で送れるようにする。

**Architecture:** 符号定義は `src/tokens/morse_tokens.py` の送信専用補助表 (`TX_ONLY_EUROPEAN_CHAR_TO_CODE` / `TX_ONLY_MARKERS`) に置き、`text_to_codes(..., include_tx_only=False)` のフラグで送信経路だけがマージして使う。合成器・学習・受信は既定 False のまま無変更。設計書: `docs/superpowers/specs/2026-08-13-tx-only-chars-design.md`。

**Tech Stack:** Python 3.11 / pytest。追加ライブラリなし。

## Global Constraints

- ブランチ `feature/tx-extra-chars` で作業。main へ直接コミットしない
- pytest は venv の python で: `C:/Users/gaoqi/Documents/GitHub/community-tools/cw-decorder/.venv/Scripts/python.exe -m pytest`。実行ディレクトリは `C:/Users/gaoqi/Documents/GitHub/community-tools/cw-decorder`
- **トークン集合は不変**: `VOCAB_SIZE == 73`、`WORD_BREAK_TOKEN_ID == 72`、`sorted(TOKEN_TO_ID.items())` の sha256 先頭 16 桁 == `d5b369163e2c0881`。これが変わる変更はすべて誤り
- `SPECIAL_INPUT_MARKERS` (合成器から見える) に {KAKKO}/{TOJI} を**入れない**
- `EUROPEAN_TABLE` / `JAPANESE_TABLE` / `_build_unified_tokens` / `converter.py` / `web/src/generated/tokens.ts` に触れない
- 符号表記: ドット `・` (U+30FB)、ダッシュ `-` (U+002D)。UTF-8 (BOM なし)・LF
- テスト名は既存に倣い日本語可 (`test_括弧を送れる` 等)
- コミットメッセージ末尾: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: 原典照合と参照表の拡張

**Files:**
- Modify: `tests/data/eGov_morse_reference.py` (EGOV_EUROPEAN_PUNCT の直後に新セクション追加)

**Interfaces:**
- Produces: `EGOV_EUROPEAN_PUNCT_TX_ONLY: Final[dict[str, str]]` — Task 2 の照合テストが参照する。和文括弧は既存の `EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB` (`「` = `-・--・-`、`」` = `・-・・-・`) をそのまま使う

- [ ] **Step 1: 原典を照合する** — WebFetch ツールで次を順に試す

1. `https://laws.e-gov.go.jp/law/325M50080000017` — 別表第一号の欧文記号 (コロン・アポストロフィ・引用符・括弧・乗算) の符号を質問する
2. 1 が取れない (別表が画像等) 場合: `https://www.itu.int/rec/R-REC-M.1677-1-200910-I/en` から ITU-R M.1677-1 の記号符号を質問する

**取得できた符号を Step 2 の値と突き合わせ、食い違いがあれば原典側を正として Step 2 の値を直すこと。** どちらも取得できない場合は Step 2 の値をそのまま使い、docstring に「オンライン照合未実施 (2026-08-13)。運用者による原典確認が必要」と明記する。照合の結果 (どの URL で何を確認したか) を必ず報告に書く。

- [ ] **Step 2: 参照表に新セクションを追加** — `EGOV_EUROPEAN_PUNCT` の閉じ括弧の直後に:

```python
# ============================================================
# 欧文記号のうち語彙 (トークン集合) に無いもの
# ============================================================
# **実在することを記録する。** 受信語彙に足すには再学習が要るため、
# 送信専用表 (morse_tokens.TX_ONLY_EUROPEAN_CHAR_TO_CODE) とだけ照合する。
# 「(」はプロサイン KN と、「"」は和文の上向き括弧と同符号 (同じ電波)。
EGOV_EUROPEAN_PUNCT_TX_ONLY: Final[dict[str, str]] = {
    ":": "---・・・",
    "'": "・----・",
    '"': "・-・・-・",
    "(": "-・--・",
    ")": "-・--・-",
    "×": "-・・-",      # 乗算記号 (X と同符号)
}
```

- [ ] **Step 3: 参照ファイルが単体で壊れていないことを確認**

Run: `.venv/Scripts/python.exe -c "import tests.data.eGov_morse_reference as r; print(len(r.EGOV_EUROPEAN_PUNCT_TX_ONLY))"`
Expected: `6`

- [ ] **Step 4: Commit**

```bash
git add tests/data/eGov_morse_reference.py
git commit -m "test: e-Gov 参照表に送信専用の欧文記号セクションを追加

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 2: morse_tokens.py の送信専用表と include_tx_only (TDD)

**Files:**
- Create: `tests/test_tx_only_chars.py`
- Modify: `src/tokens/morse_tokens.py` (逆引きセクションの末尾に表を追加、`text_to_codes` にフラグ追加、`__all__` に 3 名を追加)

**Interfaces:**
- Consumes: Task 1 の `EGOV_EUROPEAN_PUNCT_TX_ONLY`
- Produces: `TX_ONLY_EUROPEAN_CHAR_TO_CODE: Final[dict[str, str]]`、`TX_ONLY_MARKERS: Final[dict[str, str]]`、`TX_INPUT_MARKERS: Final[dict[str, str]]` (= SPECIAL | TX_ONLY)、`text_to_codes(text, mode, emit_word_breaks=True, include_tx_only=False)`

- [ ] **Step 1: 失敗するテストを書く** — `tests/test_tx_only_chars.py` を新規作成:

```python
"""送信専用符号 (TX_ONLY) のテスト.

受信語彙 (トークン集合) に触れずに送信だけ拡張していることを検証する。
設計書: docs/superpowers/specs/2026-08-13-tx-only-chars-design.md
"""
from __future__ import annotations

import hashlib

import pytest

from src.tokens.morse_tokens import (
    SPECIAL_INPUT_MARKERS,
    TOKEN_TO_ID,
    TX_INPUT_MARKERS,
    TX_ONLY_EUROPEAN_CHAR_TO_CODE,
    TX_ONLY_MARKERS,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
    text_to_codes,
)
from tests.data.eGov_morse_reference import (
    EGOV_EUROPEAN_PUNCT_TX_ONLY,
    EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB,
)


class TestReferenceMatch:
    """参照表 (実装と独立) との照合."""

    def test_欧文送信専用は参照表と一致(self) -> None:
        assert TX_ONLY_EUROPEAN_CHAR_TO_CODE == EGOV_EUROPEAN_PUNCT_TX_ONLY

    def test_和文括弧マーカーは参照表と一致(self) -> None:
        assert TX_ONLY_MARKERS == {
            "{KAKKO}": EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB["「"],
            "{TOJI}": EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB["」"],
        }


class TestTokenSetUnchanged:
    """受信語彙の不変 (これが崩れたら学習済みモデルが無意味になる)."""

    def test_語彙サイズ(self) -> None:
        assert VOCAB_SIZE == 73

    def test_wordbreak_id(self) -> None:
        assert WORD_BREAK_TOKEN_ID == 72

    def test_全トークンIDのハッシュ(self) -> None:
        digest = hashlib.sha256(repr(sorted(TOKEN_TO_ID.items())).encode()).hexdigest()
        assert digest[:16] == "d5b369163e2c0881"

    def test_合成器用マーカーは3個のまま(self) -> None:
        assert set(SPECIAL_INPUT_MARKERS) == {"{HORE}", "{RATA}", "{SK}"}


class TestTextToCodes:
    """text_to_codes のフラグ挙動."""

    def test_欧文の送信専用文字(self) -> None:
        assert text_to_codes("(A)", "european", include_tx_only=True) == [
            "-・--・", "・-", "-・--・-",
        ]

    def test_欧文の記号ぜんぶ(self) -> None:
        got = text_to_codes(":'\"×", "european", include_tx_only=True)
        assert got == ["---・・・", "・----・", "・-・・-・", "-・・-"]

    def test_和文括弧マーカー(self) -> None:
        assert text_to_codes("{KAKKO}アイ{TOJI}", "japanese", include_tx_only=True) == [
            "-・--・-", "--・--", "・-", "・-・・-・",
        ]

    def test_既定では欧文送信専用文字は従来どおりKeyError(self) -> None:
        with pytest.raises(KeyError):
            text_to_codes("(", "european")

    def test_既定では括弧マーカーは従来どおりKeyError(self) -> None:
        with pytest.raises(KeyError):
            text_to_codes("{KAKKO}", "japanese")

    def test_TX_INPUT_MARKERSは合成器マーカーを含む(self) -> None:
        assert set(TX_INPUT_MARKERS) == set(SPECIAL_INPUT_MARKERS) | {"{KAKKO}", "{TOJI}"}
```

- [ ] **Step 2: 失敗を確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_only_chars.py -q`
Expected: FAIL — `ImportError: cannot import name 'TX_INPUT_MARKERS'`

- [ ] **Step 3: 実装** — `src/tokens/morse_tokens.py`:

(a) `JAPANESE_CHAR_TO_CODES` の定義の直後に追加:

```python
# ============================================================
# 送信専用の補助表 (受信語彙に無い符号。送信は NN を通らないので送れる)
# ============================================================
# **トークン集合には入れない。** 入れると ID がずれて学習済みモデルが無意味になる
# (tests/test_tx_only_chars.py の TestTokenSetUnchanged が歯止め)。
# 出典は tests/data/eGov_morse_reference.py の EGOV_EUROPEAN_PUNCT_TX_ONLY と
# EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB を参照 (照合テストあり)。
# 受信側は従来どおり: これらの符号を受けたときは ? (TABLE_MISS) になる。
TX_ONLY_EUROPEAN_CHAR_TO_CODE: Final[dict[str, str]] = {
    ":": "---・・・",
    "'": "・----・",
    '"': "・-・・-・",   # 和文の上向き括弧と同符号
    "(": "-・--・",      # プロサイン KN と同符号
    ")": "-・--・-",     # 和文の下向き括弧と同符号
    "×": "-・・-",       # 乗算記号。X と同符号
}

# 和文の本物の括弧。文字 「」 は欧文区間マーカー・段落の別名として既に
# 使われているため、{HORE} と同じ中括弧マーカーで入力する。
TX_ONLY_MARKERS: Final[dict[str, str]] = {
    "{KAKKO}": "-・--・-",   # 下向き括弧 「
    "{TOJI}": "・-・・-・",   # 上向き括弧 」
}
```

(b) `SPECIAL_INPUT_MARKERS` の定義の直後に追加:

```python
# 送信側が使うマーカー全集合。**合成器は SPECIAL_INPUT_MARKERS だけを見る**
# ({KAKKO}/{TOJI} の符号は語彙に無く、ラベル生成に流れると TOKEN_TO_ID で落ちる)。
TX_INPUT_MARKERS: Final[dict[str, str]] = {**SPECIAL_INPUT_MARKERS, **TX_ONLY_MARKERS}
```

(c) `text_to_codes` の変更 — シグネチャと表の選択:

```python
def text_to_codes(
    text: str, mode: Mode, emit_word_breaks: bool = True, include_tx_only: bool = False
) -> list[str]:
```

docstring に 1 行追加: `- ``include_tx_only=True`` (送信側専用) なら送信専用表とマーカーもマージして解釈`

本文の表の選択を差し替え:

```python
    markers = TX_INPUT_MARKERS if include_tx_only else SPECIAL_INPUT_MARKERS
    char_to_code: dict[str, str] | None = None
    if mode == "european":
        char_to_code = (
            {**EUROPEAN_CHAR_TO_CODE, **TX_ONLY_EUROPEAN_CHAR_TO_CODE}
            if include_tx_only
            else EUROPEAN_CHAR_TO_CODE
        )
    char_to_codes_ja = JAPANESE_CHAR_TO_CODES if mode == "japanese" else None
```

マーカーのループは `SPECIAL_INPUT_MARKERS.items()` → `markers.items()` に変更。

(d) `__all__` に `"TX_INPUT_MARKERS"`, `"TX_ONLY_EUROPEAN_CHAR_TO_CODE"`, `"TX_ONLY_MARKERS"` を追加 (アルファベット順の位置に)。

- [ ] **Step 4: テスト通過を確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_only_chars.py tests/test_tokens.py tests/test_export_tokens.py -q`
Expected: 全 PASS (トークン集合と web 生成物が不変であることも同時に確認される)

- [ ] **Step 5: Commit**

```bash
git add tests/test_tx_only_chars.py src/tokens/morse_tokens.py
git commit -m "feat: 送信専用の符号表を morse_tokens に追加する

受信語彙 (トークン集合) は不変。欧文 : ' \" ( ) × と和文括弧マーカー
{KAKKO}/{TOJI} を text_to_codes(include_tx_only=True) でだけ解釈する。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: 送信経路の統合 — encoder / reading / key_server (TDD)

**Files:**
- Modify: `src/tx/encoder.py` (import、`_MARKER_RE`、`_JAPANESE_TX_CHARS` はそのまま、欧文判定表、`text_to_codes` 呼び出し 2 箇所)
- Modify: `src/tx/reading.py` (`_MARKER_RE` の元を `TX_INPUT_MARKERS` に)
- Test: `tests/test_tx_encoder.py`、`tests/test_tx_reading.py` に追記

**Interfaces:**
- Consumes: Task 2 の `TX_ONLY_EUROPEAN_CHAR_TO_CODE` / `TX_INPUT_MARKERS` / `text_to_codes(include_tx_only=True)`
- Produces: `find_unsendable` / `encode` / `to_sendable_kana` が新文字を受理する (シグネチャ不変)

- [ ] **Step 1: 失敗するテストを書く** — `tests/test_tx_encoder.py` の末尾にクラスを追加:

```python
class TestTxOnlyChars:
    """送信専用符号 (設計書 2026-08-13-tx-only-chars-design.md)."""

    def test_欧文の記号が送信可能判定(self) -> None:
        assert find_unsendable('CQ (TEST) "73" RIG:IC7610 5 × 9') == ()

    def test_欧文の記号が符号化される(self) -> None:
        assert encode("(A)") == ["-・--・", "・-", "-・--・-"]

    def test_和文括弧マーカーが送信可能判定(self) -> None:
        assert find_unsendable("{HORE}アイ {KAKKO}アイ{TOJI} ウエ") == ()

    def test_和文括弧マーカーが符号化される(self) -> None:
        got = encode("{HORE}{KAKKO}ア{TOJI}")
        assert got[-3:] == ["-・--・-", "--・--", "・-・・-・"]

    def test_欧文区間の意味は不変(self) -> None:
        # 「…」 は今までどおり「中身を欧文で送り、閉じは段落」であり、
        # 本物の括弧符号 (-・--・- / ・-・・-・) は出ない
        got = encode("{HORE}ア「A」イ")
        assert got.count("・-・-・・") == 1   # 段落 (区間の閉じ) がちょうど 1 回
        assert "・-・・-・" not in got        # 上向き括弧の符号は出ない

    def test_和文モードで生の括弧文字は従来どおり送れない(self) -> None:
        bad = find_unsendable("{HORE}ア(イ)")
        assert [b.char for b in bad] == ["(", ")"]
```

`tests/test_tx_reading.py` の末尾に追記:

```python
class TestTxOnlyMarkers:
    def test_括弧マーカーは変換を素通りする(self) -> None:
        profile = OperatorProfile()
        result = to_sendable_kana("{KAKKO}アイ{TOJI}", profile)
        assert "{KAKKO}" in result.text and "{TOJI}" in result.text
        assert result.unsendable == ()
```

(既存の import に合わせて `OperatorProfile` / `to_sendable_kana` の import を確認。既にファイル冒頭にあるはず)

- [ ] **Step 2: 失敗を確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_encoder.py::TestTxOnlyChars tests/test_tx_reading.py::TestTxOnlyMarkers -q`
Expected: FAIL (`(` 等が BadChar として返る / マーカーが unsendable 扱い)

注意: `test_欧文区間の意味は不変` の 1 本目の assert が書いた形のまま意味を成さない場合は、実際の `encode` 出力を観察して「区間の閉じが段落符号 1 回だけであること」を主張する形に直してよい (意図: 「A」 の入力から括弧符号 `-・--・-` が**閉じとして**出ないこと)。

- [ ] **Step 3: 実装**

`src/tx/encoder.py`:
1. import に `TX_INPUT_MARKERS`, `TX_ONLY_EUROPEAN_CHAR_TO_CODE` を追加 (`from src.tokens.morse_tokens import ...` の既存行へ)
2. `_MARKER_RE = re.compile("|".join(re.escape(m) for m in SPECIAL_INPUT_MARKERS))` → `TX_INPUT_MARKERS` に変更 (コメント: 送信側は {KAKKO}/{TOJI} も通す)
3. `find_unsendable` の欧文表を差し替え: モジュールレベルに `_EUROPEAN_TX_CHARS: frozenset[str] = frozenset(EUROPEAN_CHAR_TO_CODE) | frozenset(TX_ONLY_EUROPEAN_CHAR_TO_CODE)` を `_JAPANESE_TX_CHARS` の隣に定義し、`table = _EUROPEAN_TX_CHARS if segment.mode == "european" else _JAPANESE_TX_CHARS`
4. `encode` 内の `text_to_codes(...)` 呼び出し (2 箇所あれば両方) に `include_tx_only=True` を追加

`src/tx/reading.py`:
5. import の `SPECIAL_INPUT_MARKERS` を `TX_INPUT_MARKERS` に変え、`_MARKER_RE` の元も差し替え (SPECIAL... が他で使われていないか確認してから)

- [ ] **Step 4: テスト通過と既存不変の確認**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_encoder.py tests/test_tx_reading.py tests/test_tx_key_server.py tests/test_tx_templates.py tests/test_tx_lan_end_to_end.py -q`
Expected: 全 PASS (既存の欧文区間・段落・テンプレートのテストが通る = 意味の不変)

- [ ] **Step 5: key_server の受理テストを 1 本追加** — `tests/test_tx_key_server.py` の既存の prepare 系テストの書き方に倣い (ファイル内の既存 fixture/ヘルパーを使うこと)、本文 `'CQ (TEST) "73" K'` の prepare が拒否されないことを確認するテストを追加。既存テストで prepare の成功/拒否をどう表明しているかを読み、同じ形式で書く。

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_key_server.py -q`
Expected: 全 PASS

- [ ] **Step 6: Commit**

```bash
git add src/tx/encoder.py src/tx/reading.py tests/test_tx_encoder.py tests/test_tx_reading.py tests/test_tx_key_server.py
git commit -m "feat: 送信経路で送信専用符号を受理する

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 4: ドキュメントとフルスイート

**Files:**
- Modify: `docs/USAGE.md` (§12 の送信できる文字の説明)
- Modify: `README.md` (既知の制限「鍵括弧の符号」行)

**Interfaces:**
- Consumes: Task 3 までの完成した挙動

- [ ] **Step 1: USAGE.md §12 に追記** — 送信できる文字の説明がある箇所 (`grep -n "送信できない文字\|送れる文字" docs/USAGE.md` で探す) に以下の内容を、周囲の文体 (です・ます調) に合わせて追記:

- 欧文では `: ' " ( ) ×` も送れます (国際符号にある記号)
- 和文の本物の括弧は `{KAKKO}` (「) と `{TOJI}` (」) と書きます。`「…」` と書いた場合は従来どおり「中身を欧文として送る」印です
- これらの記号は**受信側の辞書には無い**ため、相手が送ってきたときは `?` と表示されます (自分が送る分には正しい符号が出ます)

- [ ] **Step 2: README.md の既知の制限を更新** — 「鍵括弧の符号」行を次に置き換え:

```
| **鍵括弧などの受信** | `「` `」` `(` `)` `:` `'` `"` などの記号符号は**送信は可能**ですが、受信語彙に無いため受けたときは `?` になります。受信対応はトークン追加 (再学習) の機会に行います |
```

- [ ] **Step 3: フルスイートを 1 回実行**

Run: `.venv/Scripts/python.exe -m pytest -q` (timeout 長め)
Expected: 全 PASS (既知: 終了時セグフォルトは結果出力後・無害、test_tx_dialog は稀にフレーキーで単体再実行で判定)

- [ ] **Step 4: Commit**

```bash
git add docs/USAGE.md README.md
git commit -m "docs: 送信専用符号の使い方と受信の非対称性を記載する

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 完了後 (計画外の申し送り)

- push + PR 作成 URL の提示 (gh 無し)。PR 本文に Task 1 の原典照合の結果 (どの URL で確認できたか) を明記
- **無線機 PC への再配布が必要** (怠ると「手元は通るが打鍵側が拒否」が再発する)
- 公開リポジトリへは次回同期で反映

## Self-Review 済みの注意

- 設計書 §5 の「合成器不変」は Task 2 の `test_既定では〜KeyError` 2 本が担う
- 設計書 §5 の「トークン不変」は TestTokenSetUnchanged + 既存 `test_export_tokens` (Task 2 Step 4 で実行) が担う
- `_JAPANESE_TX_CHARS` は変更しない (和文モードの文字集合は不変。`」`=段落の別名も不変)
- reading.py の変更は `_MARKER_RE` の元だけ。`SPECIAL_INPUT_MARKERS` が同ファイル内の別用途で使われていたら、その用途の意味を確認してから置き換えること (Task 3 Step 3-5 に明記)

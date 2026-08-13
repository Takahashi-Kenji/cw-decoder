# 和文の中の欧文区間 `「…」` 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 和文の中に `「FT991」` と書けば、その中身を欧文の符号で送れるようにする。

**Architecture:** `split_segments` が `{HORE}`/`{RATA}` に加えて `「…」` でもモードを切り替える。和文の中では `」` を和文の符号として出し、欧文の中では括弧を両方落とす。`「` はどちらの符号表にも無いので出さない。カナ変換が `「` を消していたのをやめる。

**Tech Stack:** Python 3.11+ / pytest。**外部依存を増やさない。**

**設計書:** `docs/superpowers/specs/2026-08-12-european-span-design.md`

## Global Constraints

- 言語: コメント・docstring・コミットメッセージはすべて**日本語**
- 文字コード **UTF-8 (BOM なし)**、改行 **LF**
- 型ヒント必須。**mypy の新規エラーを増やさないこと**
- ruff: `line-length = 120`、`target-version = "py311"`
- **`src/tokens/morse_tokens.py` を変更しない** (符号定義の唯一の真正ソース)。**符号表そのものは何も変わらない**
- **符号表を書き写さない。** 判定は `EUROPEAN_CHAR_TO_CODE` / `JAPANESE_CHAR_TO_CODES` から導出する
- **テストは必ず venv の python で走らせること**: `.venv/Scripts/python.exe -m pytest ...`
  素の `python` には `pykakasi` / `PySide6` が入っておらず**嘘の失敗が出る**
- **既存のテスト (1508 件) を 1 件も壊さないこと**
- ブランチ: `feature/template-editor` (作成済み。ここに積む)

## File Structure

| ファイル | 責務 |
|---|---|
| `src/tx/encoder.py` (改) | `「…」` を欧文区間として刻む |
| `src/tx/reading.py` (改) | `「` を落とすのをやめる。`「` `」` を送れない文字として報告しない |
| `docs/reply_templates_example.json` (改) | 「設備の紹介」を `「…」` で書き直す |
| `docs/USAGE.md` (改) | §13 に `「…」` の説明を足す |

**`encoder.py` と `reading.py` は 1 つのタスクにする。** 片方だけでは木が壊れた状態になり、
レビュアーが単独で承認できないため。`reading.py` が `「` を通すようになっても
`encoder.py` が区間として解釈しなければ、`「` はただの「送信できない文字」になる。

---

## Task 1: `「…」` を欧文区間として実装する

**`reading.py` と `encoder.py` を一緒に直す。** 片方だけでは木が壊れる。

### 1-A. カナ変換が `「` を消すのをやめる

**Files:**
- Modify: `src/tx/reading.py` (`PUNCTUATION_MAP`、`_SENDABLE`)
- Test: `tests/test_tx_reading.py`

**Interfaces:**
- Consumes: なし
- Produces: `to_sendable_kana(text).text` が `「` を保つ。`find_bad_chars` が `「` `」` を報告しない

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_reading.py` に追記:

```python
class TestEuropeanSpanBrackets:
    """`「…」` は欧文区間の印なので、カナ変換で消してはいけない.

    **`「` は ``PUNCTUATION_MAP`` で空文字に落とされていた** (2026-08-12 に判明)。
    そのため `「FT991」` が `FT991」` になり、印が届く前に消えていた。
    区間として解釈するのは `encoder.split_segments` の仕事なので、
    ここでは**そのまま通す**。
    """

    def test_開き括弧が消えない(self) -> None:
        assert "「" in to_sendable_kana("「FT991」").text

    def test_閉じ括弧も残る(self) -> None:
        assert "」" in to_sendable_kana("「FT991」").text

    def test_括弧の中身が変わらない(self) -> None:
        assert "FT991" in to_sendable_kana("リグ ハ 「FT991」 デス").text

    def test_括弧を送れない文字として報告しない(self) -> None:
        """判定は encoder 側がモードを見て行う。ここでは通す."""
        result = to_sendable_kana("「FT991」")
        assert "「" not in "".join(b.char for b in result.bad_chars)

    def test_単独の閉じ括弧は従来どおり(self) -> None:
        """`」` は和文の終わりとして既に使われている. 壊さないこと."""
        assert to_sendable_kana("こんにちは」").text.endswith("」")
```

`tests/test_tx_reading.py` の import に `to_sendable_kana` があることを確かめる (既にあるはず)。

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_reading.py::TestEuropeanSpanBrackets -v`
Expected: `test_開き括弧が消えない` と `test_括弧を送れない文字として報告しない` が FAIL

- [ ] **Step 3: 実装する**

`src/tx/reading.py` の `PUNCTUATION_MAP` から `"「": ""` の行を消す。
`"『": ""` と `"』": "」"` はそのまま残す (別の記号であり、今回の対象外)。

```python
PUNCTUATION_MAP: dict[str, str] = {
    "。": "、", "．": "、", "，": "、", "！": "、", "!": "、",
    "・": "、", "；": "、", ";": "、", "：": "、", ":": "、",
    "？": "?", "『": "", "』": "」",
    "（": "", "）": "", "(": "", ")": "",
    "　": " ",          # 全角スペースは語間へ
    "〜": "-", "～": "-", "―": "-", "—": "-",
}
```

**`「` を落とすのをやめた理由をコメントで残すこと:**

```python
# **`「` は落とさない。** `「…」` は「ここは欧文で打つ」という印であり
# (docs/superpowers/specs/2026-08-12-european-span-design.md)、区間として
# 解釈するのは encoder.split_segments の仕事である。ここで消すと印が
# 届く前に無くなる。`」` は元から残している (和文の終わりでもあるため)。
```

`_SENDABLE` に `「` を足す。**符号表を書き写さない**ので、集合に明示的に加える形にする:

```python
# 送信可能な文字の集合。**符号表を書き写さず、そこから作る** (原則 2)。
# `「` は符号表に無いが、`「…」` の印として通す (判定は encoder が行う)。
_SENDABLE: frozenset[str] = frozenset(JAPANESE_CHAR_TO_CODES) | frozenset(" 「")
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_reading.py -v`
Expected: 追記した 5 件を含めて全 passed

- [ ] **Step 5: 既存を壊していないか確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/`
Expected: `FAILED` も `ERROR` も出ない

**ここで既存のテストが落ちる可能性がある。** `「` が残るようになったことで、
`「` を「送信できない文字」として扱うテストがあれば落ちる。
**落ちたら 1-B で直るので、この時点では落ちたテスト名を記録して次へ進むこと。**
**コミットは 1-B の後にまとめて行う** (途中で壊れた状態を残さない)。

- [ ] **Step 6: ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/tx/reading.py tests/test_tx_reading.py`
Run: `.venv/Scripts/python.exe -m mypy src/tx/reading.py` (`pykakasi` 由来の既存エラーは無関係)

---

### 1-B. `「…」` を欧文区間として刻む

**Files:**
- Modify: `src/tx/encoder.py` (`split_segments`)
- Test: `tests/test_tx_encoder.py`

**Interfaces:**
- Consumes: 1-A の「`「` が消えない」振る舞い
- Produces: `split_segments(text)` が `「…」` を `european` の `Segment` として刻む。`find_unsendable` / `encode` / `build_sequence` はそのまま恩恵を受ける

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_encoder.py` に追記:

```python
class TestEuropeanSpan:
    """`「…」` は「ここは欧文で打つ」という印.

    運用者は実際の交信で `「FT991」` と書き、**欧文で短い単語を打つときは
    ホレ・ラタを使わない** (2026-08-12 の聞き取り)。

    * `「` は**出さない** (どちらの符号表にも無い)
    * 中身は**欧文の符号表**で送る
    * `」` は**和文の符号として出す** (区切りとして相手に届く)
    * 欧文の中では**括弧を両方落とす** (`」` は和文表にしかないため)
    """

    def test_和文の中の括弧が欧文区間になる(self) -> None:
        segments = [(s.text, s.mode) for s in split_segments(f"{HORE}リグ ハ 「FT991」 デス{RATA}")]
        assert ("FT991", "european") in segments

    def test_和文の中では閉じ括弧を和文で出す(self) -> None:
        """`」` は和文の符号 (・-・-・・) として送る."""
        codes = encode(f"{HORE}リグ ハ 「FT991」{RATA}")
        assert JAPANESE_CHAR_TO_CODES["」"][0] in codes

    def test_開き括弧は出さない(self) -> None:
        """どちらの符号表にも無いので出しようがない."""
        assert find_unsendable(f"{HORE}リグ ハ 「FT991」 デス{RATA}") == ()

    def test_欧文の中では括弧を両方落とす(self) -> None:
        """`」` は和文表にしかない。欧文の文に残すと送れなくなる."""
        assert find_unsendable("RIG 「FT991」 ANT 「DP」 K") == ()

    def test_欧文の中の括弧は符号を増やさない(self) -> None:
        assert encode("RIG 「FT991」 K") == encode("RIG FT991 K")

    def test_単独の閉じ括弧は和文の終わり(self) -> None:
        """従来の使い方を壊さない."""
        assert find_unsendable(f"{HORE}コンニチハ」{RATA}") == ()

    def test_閉じていない括弧は印にしない(self) -> None:
        """**書き間違いを黙って通さない。** `「` が送れない文字として見える."""
        bad = find_unsendable(f"{HORE}リグ ハ 「FT991 デス{RATA}")
        assert "「" in "".join(b.char for b in bad)

    def test_中身が和文なら送れない(self) -> None:
        """欧文区間なのでカタカナが送れない. 黙って通さないこと."""
        bad = find_unsendable(f"{HORE}「コンニチハ」{RATA}")
        assert bad != ()

    def test_空の括弧(self) -> None:
        """中身が空なら `」` だけが出る.

        `inner` が空文字になるが、``split_segments`` の末尾で
        ``s.text.strip()`` により捨てられる。**落ちないことが要件**である。
        """
        assert find_unsendable(f"{HORE}アイ「」{RATA}") == ()
        assert encode(f"{HORE}アイ「」{RATA}") == encode(f"{HORE}アイ」{RATA}")

    def test_入れ子は考えない(self) -> None:
        """`「` から**次の** `」` までを 1 区間とする (設計書 §3.4).

        `「A「B」C」` は `A「B` が欧文区間になり、`C」` が続く。
        **区間の中に残った `「` は欧文表に無いので「送信できない文字」として見える。**
        入れ子を書くと気づけるということであり、これでよい。
        """
        segments = [(s.text, s.mode) for s in split_segments(f"{HORE}「A「B」C」{RATA}")]
        assert ("A「B", "european") in segments
        assert "「" in "".join(b.char for b in find_unsendable(f"{HORE}「A「B」C」{RATA}"))

    def test_括弧が複数あってもよい(self) -> None:
        assert find_unsendable(f"{HORE}リグ ハ 「FT991」 アンテナ ハ 「DP」{RATA}") == ()

    def test_ホレラタの振る舞いは変わらない(self) -> None:
        segments = [(s.text, s.mode) for s in split_segments(f"{HORE}ア{RATA}")]
        assert segments == [(HORE, "european"), (f"ア{RATA}", "japanese")]
```

テストファイルの import に `JAPANESE_CHAR_TO_CODES` を足す:

```python
from src.tokens.morse_tokens import JAPANESE_CHAR_TO_CODES
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_encoder.py::TestEuropeanSpan -v`
Expected: 多くが FAIL

- [ ] **Step 3: 実装する**

`src/tx/encoder.py` に定数を足す (`HORE` / `RATA` の隣):

```python
# 欧文区間の印。**和文の中に短い欧文を打つときに使う。**
# `「` はどちらの符号表にも無いので出さない。`」` は和文の符号として出す
# (docs/superpowers/specs/2026-08-12-european-span-design.md)。
SPAN_OPEN = "「"
SPAN_CLOSE = "」"
```

`split_segments` を差し替える:

```python
def split_segments(text: str) -> list[Segment]:
    """``{HORE}`` … ``{RATA}`` と ``「…」`` を境にモードを切り替えて刻む.

    マーカーは**それが属する側**の segment に入れる。``{HORE}`` は欧文側の
    終わりに置く (欧文の符号として送られ、受信側が和文へ切り替える)。

    ``「…」`` は**欧文区間**である。運用者は和文の中に短い欧文 (リグ名など) を
    打つときこう書き、**ホレ・ラタは使わない** (2026-08-12 の聞き取り)。

    * ``「`` は出さない (どちらの符号表にも無い)
    * 中身は欧文の符号表で送る
    * ``」`` は**周りが和文なら和文の符号として出す** (区切りとして届く)。
      **周りが欧文なら落とす** (``」`` は和文表にしかないため)
    * **対になっているときだけ印として扱う。** 単独の ``」`` は従来どおり
      和文の終わりであり、閉じていない ``「`` は印にしない (書き間違いが
      「送信できない文字」として見える)

    始まりのモードは中身から決める (:func:`_initial_mode`)。
    """
    segments: list[Segment] = []
    mode = _initial_mode(text)
    buffer = ""
    index = 0
    while index < len(text):
        if text.startswith(HORE, index):
            buffer += HORE
            segments.append(Segment(buffer, mode))
            buffer, mode = "", "japanese"
            index += len(HORE)
        elif text.startswith(RATA, index):
            buffer += RATA
            segments.append(Segment(buffer, mode))
            buffer, mode = "", "european"
            index += len(RATA)
        elif text.startswith(SPAN_OPEN, index):
            close = text.find(SPAN_CLOSE, index + len(SPAN_OPEN))
            if close < 0:
                # 閉じていない。**印として扱わない。** `「` をそのまま buffer へ
                # 積むので、符号化のときに「送信できない文字」として見える
                buffer += text[index]
                index += len(SPAN_OPEN)
                continue
            inner = text[index + len(SPAN_OPEN) : close]
            # 区間の前までを今のモードで確定させる
            if buffer:
                segments.append(Segment(buffer, mode))
                buffer = ""
            segments.append(Segment(inner, "european"))
            # **`」` は周りが和文のときだけ出す。** 欧文の中では落とす
            if mode == "japanese":
                segments.append(Segment(SPAN_CLOSE, "japanese"))
            index = close + len(SPAN_CLOSE)
        else:
            buffer += text[index]
            index += 1
    if buffer:
        segments.append(Segment(buffer, mode))
    return [s for s in segments if s.text.strip()]
```

`__all__` に `"SPAN_CLOSE"`, `"SPAN_OPEN"` を足す。

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_encoder.py -v`
Expected: 追記した 12 件を含めて全 passed

- [ ] **Step 5: `find_unsendable` の位置がずれていないか確かめる**

`find_unsendable` は `text.index(segment.text, offset)` で元テキスト上の位置を求めている。
**`「…」` を刻むと segment のテキストが元テキストと一致しなくなる** (`「` を除いているため)。

Run: `.venv/Scripts/python.exe -X utf8 -c "from src.tx.encoder import find_unsendable; print(find_unsendable('{HORE}リグ ハ 「コンニチハ」 デス{RATA}'))"`

**制御側で追跡済み**: segment は元テキストの順序どおりに並ぶので、`offset` の前進で
正しく解決される (`「` が飛ばされても `index()` は前方検索なので破綻しない)。
`{HORE}リグ ハ 「FT991」 デス{RATA}` で手で追った結果、5 つの segment すべてが
正しい位置に解決された。

**それでも実際に確かめること。** ずれていたら**赤く出る文字が実際と違う**という
分かりにくい壊れ方をする。上のコマンドで出た `index` が、元テキストのその位置の
文字と一致しているかを見ること。

- [ ] **Step 6: 全体を走らせる**

Run: `.venv/Scripts/python.exe -m pytest tests/`
Expected: `FAILED` も `ERROR` も出ない

**1-A の Step 5 で記録した落ちたテストがここで通るはず。** まだ落ちていれば、
そのテストが**古い前提 (`「` は送れない文字である) を固定している**ので、
テストのほうを直すこと。**理由をコミットメッセージに書くこと。**

- [ ] **Step 7: ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/tx/encoder.py src/tx/reading.py tests/`
Run: `.venv/Scripts/python.exe -m mypy src/tx/encoder.py src/tx/reading.py`

- [ ] **Step 8: コミット (1-A と 1-B をまとめて)**

```bash
git add src/tx/encoder.py src/tx/reading.py tests/test_tx_encoder.py tests/test_tx_reading.py
git commit -m "feat: 和文の中の欧文区間 「…」 を追加する

和文の交信でもリグ名は欧文で送る。運用者は実際に 「FT991」 と書き、
欧文で短い単語を打つときはホレ・ラタを使わない。

「 は出さない (どちらの符号表にも無い)。中身は欧文の符号表で送る。
」 は周りが和文なら和文の符号として出し、欧文なら落とす (」 は和文表に
しかないため)。対になっているときだけ印として扱い、閉じていない 「 は
送信できない文字として見せる。

カナ変換が 「 を消していたのをやめた。区間として解釈するのは
encoder.split_segments の仕事である。"
```

---

## Task 2: 例文と説明書を直す

**Files:**
- Modify: `docs/reply_templates_example.json` (「設備の紹介」)
- Modify: `docs/USAGE.md` (§13)
- Test: `tests/test_tx_templates.py` (例文の検証は既存のものがそのまま効く)

**Interfaces:**
- Consumes: Task 1 の `「…」`
- Produces: なし (文書とデータのみ)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_templates.py` の `TestExampleFile` に追記:

```python
    def test_設備の紹介がホレの中で完結する(self, 例文) -> None:
        """`「…」` があるので、欧文をホレの外に出す必要がなくなった.

        以前は `RIG {リグ} ANT {アンテナ}` をホレの外に置いていた
        (和文モードの中では R I G が送れないため)。`「…」` で中に戻せる。
        """
        設備 = next(t for t in 例文 if t.name == "設備の紹介")
        # ホレの前に RIG / ANT / PWR が出ていないこと
        ホレの前 = 設備.text.split("{HORE}")[0]
        assert "RIG" not in ホレの前
        assert "ANT" not in ホレの前
        assert "PWR" not in ホレの前
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py::TestExampleFile -v`
Expected: `test_設備の紹介がホレの中で完結する` が FAIL

- [ ] **Step 3: 例文を直す**

`docs/reply_templates_example.json` の「設備の紹介」を書き換える:

```json
    {
      "name": "設備の紹介",
      "mode": "japanese",
      "text": "{相手コール} DE {自局コール} {HORE}リグ ハ {リグ}、アンテナ ハ {アンテナ}、シュツリョク ハ {出力} デス{RATA} K"
    },
```

**経歴の和文側に `「FT991」` のように書く前提である。** 型の側には `「…」` を書かない
(設計書 §3: 値に書いても型に書いても同じように効くが、**リグが欧文かどうかは値の性質**なので
値の側に書くほうが素直)。

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py -v`
Expected: 全 passed

**`test_全部送れる` が落ちる場合**、`unsendable_in_template` の仮の値 (`"0"`) が
`「…」` を持たないため。**仮の値は数字のままでよい** — 数字は両方の表にあるので
和文の中でも送れる。落ちたら実装を疑うこと。

- [ ] **Step 5: 実際の経歴の値で確かめる**

Run:

```bash
.venv/Scripts/python.exe -X utf8 -c "
from src.tx.templates import load_templates, fill
from src.tx.encoder import find_unsendable
from src.tx.reading import to_sendable_kana
値 = {'相手コール':'JA1ABC','自局コール':'JH0ILL','RST':'599','名前':'タロウ',
      'QTH':'ヨコハマシ','リグ':'「FT991」','アンテナ':'「DP」','出力':'「50W」'}
t = next(x for x in load_templates('docs/reply_templates_example.json') if x.name=='設備の紹介')
conv = to_sendable_kana(fill(t.text, 値)).text
bad = ''.join(b.char for b in find_unsendable(conv))
print(repr(conv)); print('送れる' if not bad else 'NG ' + bad)
"
```

Expected: `送れる`

- [ ] **Step 6: 説明書に書く**

`docs/USAGE.md` の §13 に「和文の中で欧文を打つ」という項目を足す。含める内容:

- **和文の中でリグ名などの欧文を打つときは `「…」` で囲む**こと
- `「` は電波に出ないこと、`」` は和文の符号として出ること
- **ホレ・ラタは付かない**こと
- **経歴の和文側に `「FT991」` のように書く**のが素直であること (型には書かない)
- **欧文の型で同じ値を使うと括弧が両方落ちる**ので、1 つの値が両方で使えること
- **閉じ忘れると `「` が「送信できない文字」として出る**こと
- 長音は使わない (`FT-991` ではなく `FT991` と書く) という運用者の書き方

- [ ] **Step 7: 全体を走らせる**

Run: `.venv/Scripts/python.exe -m pytest tests/`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **Step 8: コミット**

```bash
git add docs/reply_templates_example.json docs/USAGE.md tests/test_tx_templates.py
git commit -m "docs: 例文と説明書を 「…」 に合わせる

設備の紹介の RIG / ANT / PWR をホレの中に戻せるようになった。
経歴の和文側に 「FT991」 と書く前提。"
```

---

## 完了の確認

- [ ] **全テストが通る**

Run: `.venv/Scripts/python.exe -m pytest tests/`
Expected: 1508 件 + 新規分。`FAILED` も `ERROR` も出ない

- [ ] **符号表を変更していない**

Run: `git diff main --stat -- src/tokens/morse_tokens.py`
Expected: 出力なし

- [ ] **ruff と mypy の新規エラーが無い**

Run: `.venv/Scripts/python.exe -m ruff check src/ tests/`
Run: `.venv/Scripts/python.exe -m mypy src/tx/encoder.py src/tx/reading.py src/tx/templates.py`

- [ ] **画面を通しても動く**

`QT_QPA_PLATFORM=offscreen` で送信ダイアログを作り、和文の型に `「FT991」` を含む
経歴の値を差し込んで、`wire_text()` が送れることを確かめる。

## 次の計画に持ち越すもの

経歴の 2 値化 (`2026-08-12-profile-editor-design.md`) は**この計画に含まれない**。
本計画が終わった時点では、経歴の和文側 (`reading`) に `「FT991」` と書けば動く。

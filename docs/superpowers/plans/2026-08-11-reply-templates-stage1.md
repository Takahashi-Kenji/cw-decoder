# 返信の型 実装計画 (第 1 段)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 返信の型を前もって登録しておき、送信ダイアログで選んで欄を差し込めるようにする。

**Architecture:** 型は `~/.cw-decorder/templates.json` に置く (経歴と同じ流儀、設定とは分ける)。型は `{相手コール}` のような欄を持ち、**差し込んでから**日本語ボックスへ入れる。受信テキストから相手のコールを拾って欄に入れるが、**拾えなければ空のままにする**。LLM は使わない。

**Tech Stack:** Python 3.11+ / PySide6 / pytest。**外部依存を増やさない。**

**設計書:** `docs/superpowers/specs/2026-08-11-reply-templates-design.md`

**この計画は第 1 段だけを対象とする。** 第 2 段 (LLM 返信案、設計書 §7) は別計画。

## Global Constraints

- 言語: コメント・docstring・コミットメッセージはすべて**日本語**
- 文字コード **UTF-8 (BOM なし)**、改行 **LF**
- 型ヒント必須。**mypy の新規エラーを増やさないこと** (`serial_key.py` / `keyer.py` / `reading.py` 由来の既存エラーは無関係)
- 不変データは `@dataclass(frozen=True)`。パス操作は `pathlib.Path`
- ruff: `line-length = 120`、`target-version = "py311"`
- **`src/tokens/morse_tokens.py` を変更しない** (符号定義の唯一の真正ソース)
- **符号表を書き写さない。** 判定は `encoder.find_unsendable` 等の既存 API 経由で行う
- **テストは利用者の実ファイルを壊さないこと。** 型の保存先は必ず一時パスを渡す (`profile.py` の `DEFAULT_PROFILE_PATH` と同じ約束)
- **テストは venv の python で走らせること**: `.venv/Scripts/python.exe -m pytest ...`
  素の `python` には `pykakasi` / `PySide6` が入っておらず、嘘の失敗が出る
- Qt は `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` で GUI 無し環境でも動く
- ブランチ: `feature/reply-templates` (作成済み。ここに積む)
- **既存のテスト (1407 件) を 1 件も壊さないこと**

## File Structure

| ファイル | 責務 |
|---|---|
| `src/tx/templates.py` (新) | 型の定義・保存・読み込み・欄の差し込み・検証 |
| `src/tx/qso_fields.py` (新) | 受信テキストから相手のコール・名前を拾う |
| `src/app/tx_dialog.py` (改) | 赤字判定の修正、型の選択、拾った欄、差し込み |
| `src/app/main_window.py` (改) | 受信テキストをダイアログへ渡す |

`templates.py` と `qso_fields.py` を分けるのは、**片方が LLM 無しで完結し、もう片方が
デコード結果に依存する**ため。テストの前提もデータの出どころも違う。

---

## Task 1: 画面の赤字判定を直す

**設計書 §3。これは今出荷されている画面の欠陥であり、型を作る前に直す。**

**Files:**
- Modify: `src/app/tx_dialog.py` (`refresh_kana`)
- Test: `tests/test_tx_dialog.py`

**Interfaces:**
- Consumes: `src.tx.encoder.find_unsendable(text) -> tuple[BadChar, ...]` (`BadChar` は `index` / `char`)
- Produces: なし (既存メソッドの振る舞いの修正)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_dialog.py` に追記 (既存の `build` ヘルパを使う):

```python
class TestSendableCheck:
    """送信できない文字の判定.

    **画面は打鍵側と同じ規則で判定しなければならない。** 以前は
    ``reading.find_bad_chars`` (和文表だけで照合) を使っており、
    **コールサインを含む文が必ず赤くなっていた** (2026-08-11 に発覚)。
    定型交換は必ずコールサインを含むので、これでは使えない。
    """

    def test_コールサインを含む和文が赤くならない(self, qapp) -> None:
        dialog, _ = build(qapp)
        dialog.wrap_check.setChecked(False)
        dialog.japanese_edit.setPlainText("JA1ABC DE JH0ILL {HORE}コンニチハ{RATA} K")
        dialog.refresh_kana()
        assert "送信できない" not in dialog.status_label.text()

    def test_欧文だけの文が赤くならない(self, qapp) -> None:
        dialog, _ = build(qapp)
        dialog.wrap_check.setChecked(False)
        dialog.japanese_edit.setPlainText("CQ CQ DE JH0ILL K")
        dialog.refresh_kana()
        assert "送信できない" not in dialog.status_label.text()

    def test_本当に送れない文字は赤くなる(self, qapp) -> None:
        """**歯止めを外さない。** 符号表に無い文字は今までどおり弾く."""
        dialog, _ = build(qapp)
        dialog.japanese_edit.setPlainText("コンニチハ+")
        dialog.refresh_kana()
        assert "送信できない" in dialog.status_label.text()
        assert "+" in dialog.status_label.text()
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_dialog.py::TestSendableCheck -v`
Expected: 最初の 2 件が FAIL (`送信できない文字があります: JAABCDEJQCWK` が出る)

- [ ] **Step 3: 実装する**

`src/app/tx_dialog.py` の import に足す:

```python
from src.tx.encoder import find_unsendable, wrap_japanese
```

`refresh_kana` を差し替える:

```python
    def refresh_kana(self) -> None:
        """日本語をカタカナに直し、**関門を閉じ直す**.

        送信できるかの判定は **``encoder.find_unsendable`` を使う**。
        ``reading`` 側の ``bad_chars`` は和文表だけで照合しており、
        **コールサインを含む文が必ず赤くなっていた** (設計書 §3)。
        打鍵側 (``key_server.prepare``) と同じ規則で判定しないと、
        画面と実際の可否が食い違う。
        """
        source = self.japanese_edit.toPlainText()
        result = to_sendable_kana(source, self._profile)
        text = wrap_japanese(result.text) if self.wrap_check.isChecked() else result.text
        self.kana_view.setPlainText(text)
        self._confirmed_text = None            # 編集したら確認をやり直す
        bad = find_unsendable(text)
        if bad:
            chars = "".join(b.char for b in bad)
            self.status_label.setText(f"送信できない文字があります: {chars}")
        self._update_buttons()
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_dialog.py -v`
Expected: 既存分と合わせて全 passed

- [ ] **Step 5: ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/app/tx_dialog.py tests/test_tx_dialog.py`
Run: `.venv/Scripts/python.exe -m mypy src/app/tx_dialog.py`

- [ ] **Step 6: コミット**

```bash
git add src/app/tx_dialog.py tests/test_tx_dialog.py
git commit -m "fix: 画面の送信可否判定を打鍵側と同じ規則に揃える

reading 側の判定は和文表だけで照合しており、コールサインを含む文が
必ず「送信できない」と出ていた。定型交換は必ずコールサインを含むので
使い物にならない。encoder.find_unsendable (モードを見る) に寄せる。"
```

---

## Task 2: 型の器と差し込み

**Files:**
- Create: `src/tx/templates.py`
- Test: `tests/test_tx_templates.py`

**Interfaces:**
- Consumes: `src.tx.profile.OperatorProfile` (`callsign` / `name` / `qth` / `rig` / `antenna` / `power` はいずれも `ProfileField`、`display` と `reading` を持ち `sendable()` で送信用の読みを返す)
- Produces:
  - `ReplyTemplate` (`@dataclass(frozen=True)`): `name: str` / `mode: str` / `text: str`
  - `PLACEHOLDERS: frozenset[str]` — 差し込める欄の名前
  - `fill(text: str, values: dict[str, str]) -> str`
  - `missing_placeholders(text: str) -> tuple[str, ...]`
  - `profile_values(profile: OperatorProfile, mode: str) -> dict[str, str]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_templates.py`:

```python
"""返信の型のテスト."""
from __future__ import annotations

from src.tx.profile import OperatorProfile, ProfileField
from src.tx.templates import (
    PLACEHOLDERS,
    ReplyTemplate,
    fill,
    missing_placeholders,
    profile_values,
)


class TestFill:
    def test_欄を差し込む(self) -> None:
        assert fill("{相手コール} DE {自局コール}", {"相手コール": "JA1ABC", "自局コール": "JH0ILL"}) == (
            "JA1ABC DE JH0ILL"
        )

    def test_知らない欄はそのまま残す(self) -> None:
        """**これが効くのは {HORE} と {RATA} である。**

        型の中に書いたマーカーが差し込みで壊れてはいけない。
        """
        assert fill("{HORE}コンニチハ{RATA}", {"相手コール": "JA1ABC"}) == "{HORE}コンニチハ{RATA}"

    def test_マーカーと欄が混ざっていても壊れない(self) -> None:
        text = "{相手コール} DE {自局コール} {HORE}コンニチハ{RATA} K"
        got = fill(text, {"相手コール": "JA1ABC", "自局コール": "JH0ILL"})
        assert got == "JA1ABC DE JH0ILL {HORE}コンニチハ{RATA} K"

    def test_空の値は差し込まない(self) -> None:
        """空で埋めると「埋まった」ことになり、漏れに気づけなくなる."""
        assert fill("{相手コール} K", {"相手コール": ""}) == "{相手コール} K"

    def test_同じ欄が二度出てもよい(self) -> None:
        assert fill("{相手コール} {相手コール}", {"相手コール": "JA1ABC"}) == "JA1ABC JA1ABC"


class TestMissingPlaceholders:
    def test_埋まらなかった欄を名指しで返す(self) -> None:
        assert missing_placeholders("{相手コール} DE JH0ILL") == ("相手コール",)

    def test_マーカーは欄として数えない(self) -> None:
        assert missing_placeholders("{HORE}コンニチハ{RATA}") == ()

    def test_全部埋まっていれば空(self) -> None:
        assert missing_placeholders("JA1ABC DE JH0ILL K") == ()

    def test_知らない語も欄として数えない(self) -> None:
        """``PLACEHOLDERS`` に無いものは型の書き間違いではなく、ただの文字."""
        assert missing_placeholders("{SK}") == ()

    def test_重複は一度だけ数える(self) -> None:
        assert missing_placeholders("{相手コール} {相手コール}") == ("相手コール",)


class TestProfileValues:
    """**差し込む値は型のモードで変わる。**

    欧文の型に読み (カタカナ) を差し込むと、そのカタカナのせいでモードが和文に
    倒れ、**文中の欧文がまるごと送れなくなる** (実測: ``NAME タロウ`` を入れた
    欧文の型が ``JAABCDEJQCWGMURRST…`` と全滅した)。和文の型のときだけ読みを使う。
    """

    def test_和文の型では読みを使う(self) -> None:
        profile = OperatorProfile(
            callsign=ProfileField(display="JH0ILL"),
            name=ProfileField(display="TARO", reading="タロウ"),
        )
        values = profile_values(profile, "japanese")
        assert values["名前"] == "タロウ"

    def test_欧文の型では表示形を使う(self) -> None:
        profile = OperatorProfile(name=ProfileField(display="TARO", reading="タロウ"))
        assert profile_values(profile, "european")["名前"] == "TARO"

    def test_どちらでもの型では表示形を使う(self) -> None:
        """``any`` の型は欧文の略語が主なので表示形に寄せる."""
        profile = OperatorProfile(name=ProfileField(display="TARO", reading="タロウ"))
        assert profile_values(profile, "any")["名前"] == "TARO"

    def test_読みが無ければ表示形を使う(self) -> None:
        profile = OperatorProfile(rig=ProfileField(display="FT-991"))
        assert profile_values(profile, "japanese")["リグ"] == "FT-991"

    def test_空の欄は入れない(self) -> None:
        """空で埋めると漏れに気づけなくなる (fill と同じ理由)."""
        assert "リグ" not in profile_values(OperatorProfile(), "japanese")


class TestReplyTemplate:
    def test_三つの欄を持つ(self) -> None:
        t = ReplyTemplate(name="応答", mode="japanese", text="{相手コール} K")
        assert (t.name, t.mode, t.text) == ("応答", "japanese", "{相手コール} K")

    def test_差し込める欄の一覧がある(self) -> None:
        assert "相手コール" in PLACEHOLDERS
        assert "自局コール" in PLACEHOLDERS
        assert "RST" in PLACEHOLDERS
        assert "HORE" not in PLACEHOLDERS       # マーカーは欄ではない
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tx.templates'`

- [ ] **Step 3: 実装する**

`src/tx/templates.py`:

```python
"""返信の型 — 前もって書いた文に欄を差し込む.

なぜ型が要るか
--------------
**こちらが主導する場面には受信内容が無い** (CQ、こちらから呼ぶ、締め)。
LLM に作らせる材料が無いので、前もって書いた文を呼ぶのが確実である。
相手に応答する場面でも、良い案が出たら型として取っておけば次から即座に使える。

欄の差し込みは符号化より前
--------------------------
``{相手コール}`` のような欄は**日本語ボックスへ入れる前に**差し込む。
逆にすると欄がカナ変換器 (pykakasi) を通って壊れる。

知らない ``{…}`` は触らない
---------------------------
**これが効くのは ``{HORE}`` と ``{RATA}`` である。** 型の中に書いたマーカーが
差し込みで壊れてはいけない。``PLACEHOLDERS`` に載っている名前だけを置き換える。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.tx.profile import OperatorProfile

# 差し込める欄。**増やすほど型が書きにくくなるので絞る** (設計書 §4.1)。
PLACEHOLDERS: frozenset[str] = frozenset(
    {
        # 経歴から
        "自局コール", "名前", "QTH", "リグ", "アンテナ", "出力",
        # 拾った値、または運用者が打った値
        "相手コール", "相手名前",
        # **こちらが相手に与える RST。** 相手からもらった値ではない
        "RST",
    }
)

# 経歴の欄名 → 差し込む欄名
_PROFILE_FIELDS: dict[str, str] = {
    "callsign": "自局コール",
    "name": "名前",
    "qth": "QTH",
    "rig": "リグ",
    "antenna": "アンテナ",
    "power": "出力",
}

_FIELD_RE = re.compile(r"\{([^}]*)\}")


@dataclass(frozen=True)
class ReplyTemplate:
    """返信の型.

    Args:
        name: 画面の一覧に出す名前。
        mode: ``"european"`` / ``"japanese"`` / ``"any"``。
            **デコーダのモードに合う型だけを一覧に出す**ために使う。
        text: 本文。漢字かな交じりで書いてよい (差し込みの後にカナ変換を通る)。
    """

    name: str
    mode: str = "any"
    text: str = ""


def fill(text: str, values: dict[str, str]) -> str:
    """欄を差し込む. **知らない ``{…}`` と空の値には触らない。**

    空の値を差し込まないのは、埋まったことにすると**漏れに気づけなくなる**ため
    (:func:`missing_placeholders` が見つけられなくなる)。
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in PLACEHOLDERS:
            return match.group(0)          # {HORE} などはそのまま
        value = values.get(name, "")
        return value or match.group(0)     # 空なら埋めない

    return _FIELD_RE.sub(replace, text)


def missing_placeholders(text: str) -> tuple[str, ...]:
    """まだ埋まっていない欄の名前を、出てきた順に返す (重複は 1 度だけ).

    **符号化でも弾かれるが、そのメッセージは読めない。** 埋め忘れた欄がカナに
    変換されてモードが和文に倒れ、関係のない欧文まで巻き添えで並ぶ (実測:
    ``送信できない文字があります: {}DEJQCWRST``)。ここで名指しするほうが早い。
    """
    found: list[str] = []
    for match in _FIELD_RE.finditer(text):
        name = match.group(1)
        if name in PLACEHOLDERS and name not in found:
            found.append(name)
    return tuple(found)


def profile_values(profile: OperatorProfile, mode: str) -> dict[str, str]:
    """経歴を差し込む値にする. **空の欄は入れない** (:func:`fill` と同じ理由).

    **和文の型のときだけ読み (カタカナ) を使う。** 欧文の型に読みを差し込むと、
    そのカタカナのせいでモードが和文に倒れ、**文中の欧文がまるごと送れなくなる**
    (実測: ``NAME タロウ`` を入れた欧文の型が ``JAABCDEJQCWGMURRST…`` と全滅した)。

    欧文の型では表示形を使う。運用者は欧文交信用にローマ字を入れておくこと
    (``display="TARO"`` / ``reading="タロウ"``)。漢字のままにしてあれば
    「送信できない文字」として見えるので、そこで気づける。
    """
    values: dict[str, str] = {}
    for attr, placeholder in _PROFILE_FIELDS.items():
        field = getattr(profile, attr)
        value = field.sendable() if mode == "japanese" else field.display
        if value:
            values[placeholder] = value
    return values


__all__ = [
    "PLACEHOLDERS",
    "ReplyTemplate",
    "fill",
    "missing_placeholders",
    "profile_values",
]
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py -v`
Expected: 15 passed

- [ ] **Step 5: ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/tx/templates.py tests/test_tx_templates.py`
Run: `.venv/Scripts/python.exe -m mypy src/tx/templates.py`

- [ ] **Step 6: コミット**

```bash
git add src/tx/templates.py tests/test_tx_templates.py
git commit -m "feat: 返信の型の器と欄の差し込みを追加する

知らない {…} には触らない。型の中に書いた {HORE} {RATA} が
差し込みで壊れないようにするため。空の値も差し込まない
(埋まったことにすると漏れに気づけなくなる)。"
```

---

## Task 3: 型の保存と読み込み、検証

**Files:**
- Modify: `src/tx/templates.py` (追記)
- Test: `tests/test_tx_templates.py` (追記)

**Interfaces:**
- Consumes: Task 2 の `ReplyTemplate` / `fill` / `PLACEHOLDERS`、`src.tx.encoder.find_unsendable`
- Produces:
  - `DEFAULT_TEMPLATES_PATH: Path`
  - `load_templates(path=DEFAULT_TEMPLATES_PATH) -> list[ReplyTemplate]`
  - `save_templates(templates: list[ReplyTemplate], path=DEFAULT_TEMPLATES_PATH) -> None`
  - `unsendable_in_template(template: ReplyTemplate) -> str` (**`to_sendable_kana` を通してから調べる**)
  - `templates_for_mode(templates: list[ReplyTemplate], mode: str) -> list[ReplyTemplate]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_templates.py` に追記:

```python
import json

import pytest

from src.tx.templates import (
    load_templates,
    save_templates,
    templates_for_mode,
    unsendable_in_template,
)


class TestStore:
    def test_書いて読み戻せる(self, tmp_path) -> None:
        path = tmp_path / "templates.json"
        original = [
            ReplyTemplate(name="CQ", mode="european", text="CQ CQ DE {自局コール} K"),
            ReplyTemplate(name="応答", mode="japanese", text="{相手コール} DE {自局コール} K"),
        ]
        save_templates(original, path)
        assert load_templates(path) == original

    def test_ファイルが無ければ空(self, tmp_path) -> None:
        assert load_templates(tmp_path / "ない.json") == []

    def test_壊れたファイルは空として扱う(self, tmp_path) -> None:
        path = tmp_path / "templates.json"
        path.write_text("{壊れている", encoding="utf-8")
        assert load_templates(path) == []

    def test_壊れたファイルを上書きしない(self, tmp_path) -> None:
        """**読んだだけで消してはいけない。** 手で直せる余地を残す."""
        path = tmp_path / "templates.json"
        path.write_text("{壊れている", encoding="utf-8")
        load_templates(path)
        assert path.read_text(encoding="utf-8") == "{壊れている"

    def test_知らない欄があっても落ちない(self, tmp_path) -> None:
        path = tmp_path / "templates.json"
        path.write_text(
            json.dumps({"templates": [{"name": "A", "text": "K", "未知": 1}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_templates(path) == [ReplyTemplate(name="A", mode="any", text="K")]

    def test_日本語をそのまま書く(self, tmp_path) -> None:
        """ログと同じで、人が開いて読めることに価値がある."""
        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="応答", text="コンニチハ")], path)
        assert "コンニチハ" in path.read_text(encoding="utf-8")


class TestValidation:
    def test_送れる型は空を返す(self) -> None:
        t = ReplyTemplate(
            name="応答",
            mode="japanese",
            text="{相手コール} DE {自局コール} RST {RST} {HORE}コンニチハ{RATA} K",
        )
        assert unsendable_in_template(t) == ""

    def test_小書きカナは送れる扱いになる(self) -> None:
        """**実際の経路では小書きが自動で倒される** (reading.SMALL_KANA_MAP)。

        生の型だけを見ると ``キョウテン`` の ``ョ`` が符号表に無いので
        「送れない」と誤判定する。検証は実際に通る道と同じ道を通ること。
        """
        t = ReplyTemplate(name="小書き", mode="japanese", text="{HORE}キョウテン{RATA}")
        assert unsendable_in_template(t) == ""

    def test_ホレの中の欧文を見つける(self) -> None:
        """**型を書くときに一番間違えやすいところ** (設計書 §4.3).

        RST は欧文で送るものなので、ホレの中に書くと送れない。
        """
        t = ReplyTemplate(name="悪い例", mode="japanese", text="{HORE}コンニチハ RST 599{RATA}")
        assert "R" in unsendable_in_template(t)

    def test_符号表に無い文字を見つける(self) -> None:
        t = ReplyTemplate(name="悪い例", mode="european", text="CQ DE JH0ILL +++")
        assert "+" in unsendable_in_template(t)

    def test_欄は仮の値で埋めてから調べる(self) -> None:
        """**欄が空のせいで「送れない」と言ってはいけない。**

        埋まっていないことは :func:`missing_placeholders` の仕事である。
        """
        t = ReplyTemplate(name="CQ", mode="european", text="CQ DE {自局コール} K")
        assert unsendable_in_template(t) == ""


class TestModeFilter:
    @pytest.fixture
    def 型たち(self) -> list[ReplyTemplate]:
        return [
            ReplyTemplate(name="欧文", mode="european", text="K"),
            ReplyTemplate(name="和文", mode="japanese", text="K"),
            ReplyTemplate(name="どちらでも", mode="any", text="K"),
        ]

    def test_欧文モードでは和文の型を出さない(self, 型たち) -> None:
        names = [t.name for t in templates_for_mode(型たち, "european")]
        assert names == ["欧文", "どちらでも"]

    def test_和文モードでは欧文の型を出さない(self, 型たち) -> None:
        names = [t.name for t in templates_for_mode(型たち, "japanese")]
        assert names == ["和文", "どちらでも"]
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_templates'`

- [ ] **Step 3: 実装する**

`src/tx/templates.py` の import に追記:

```python
import json
from pathlib import Path

from src.tx.encoder import find_unsendable
from src.tx.reading import to_sendable_kana
```

`PLACEHOLDERS` の下に追記:

```python
# 型の保存先。**テストは必ず一時パスを渡すこと** (利用者の実ファイルを壊さない)。
# 設定 (settings.json) ではなく内容なので、経歴と同じく別ファイルにする。
DEFAULT_TEMPLATES_PATH = Path.home() / ".cw-decorder" / "templates.json"

# 型を検証するときに欄へ入れる仮の値。
#
# **数字にするのには理由がある。** 数字は欧文表にも和文表にもあるので、
# どちらのモードの型に差し込んでもモードを倒さない。カタカナを入れると
# 欧文の型が和文と判定され、文中の欧文がまるごと「送れない」になる。
#
# 欄が空のせいで「送れない」と言わないようにするため、必ず何かを入れる
# (埋まっていないことは missing_placeholders の仕事)。
_PROBE_VALUES: dict[str, str] = dict.fromkeys(PLACEHOLDERS, "0")
```

末尾 (`__all__` の前) に追記:

```python
def load_templates(path: Path | str = DEFAULT_TEMPLATES_PATH) -> list[ReplyTemplate]:
    """型を読み込む. 無い/壊れていれば空を返す.

    **壊れていても上書きしない。** 手で直せる余地を残す。
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("templates")
    if not isinstance(raw, list):
        return []
    out: list[ReplyTemplate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            ReplyTemplate(
                name=str(item.get("name", "")),
                mode=str(item.get("mode", "any")),
                text=str(item.get("text", "")),
            )
        )
    return out


def save_templates(
    templates: list[ReplyTemplate], path: Path | str = DEFAULT_TEMPLATES_PATH
) -> None:
    """型を保存する. **日本語はそのまま書く** (人が開いて読めるように)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "templates": [
            {"name": t.name, "mode": t.mode, "text": t.text} for t in templates
        ]
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def unsendable_in_template(template: ReplyTemplate) -> str:
    """型に送れない文字があれば、その文字を並べて返す。無ければ空文字.

    **欄はすべて仮の値で埋めてから調べる。** 埋まっていないことは
    :func:`missing_placeholders` の仕事であり、ここの関心ではない。

    一番よく起きるのは**ホレの中に欧文を書いてしまう**ことである
    (``{HORE}コンニチハ RST 599{RATA}``)。RST は欧文で送るものなので
    和文モードの中では送れない。交信中に気づくのでは遅いので保存時に見せる。
    """
    filled = fill(template.text, _PROBE_VALUES)
    # **実際に通る道と同じ道を通す。** 小書きカナ (ャュョッ) は
    # ``reading.SMALL_KANA_MAP`` で大書きに倒されてから符号化される。
    # 生の型だけを見ると ``キョウテン`` を「送れない」と誤判定する。
    converted = to_sendable_kana(filled).text
    return "".join(bad.char for bad in find_unsendable(converted))


def templates_for_mode(templates: list[ReplyTemplate], mode: str) -> list[ReplyTemplate]:
    """そのモードで使える型だけを、元の順のまま返す."""
    return [t for t in templates if t.mode == mode or t.mode == "any"]
```

`__all__` に `"DEFAULT_TEMPLATES_PATH"`, `"load_templates"`, `"save_templates"`,
`"templates_for_mode"`, `"unsendable_in_template"` を足す。

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py -v`
Expected: 28 passed

- [ ] **Step 5: ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/tx/templates.py tests/test_tx_templates.py`
Run: `.venv/Scripts/python.exe -m mypy src/tx/templates.py`

- [ ] **Step 6: コミット**

```bash
git add src/tx/templates.py tests/test_tx_templates.py
git commit -m "feat: 型の保存・読み込みと検証を追加する

壊れたファイルは空として扱い、上書きしない (手で直せる余地を残す)。
検証は欄を仮の値で埋めてから行う。一番よく起きるのはホレの中に
欧文を書くことで、交信中に気づくのでは遅いので保存時に見せる。"
```

---

## Task 4: 受信テキストから欄を拾う

**Files:**
- Create: `src/tx/qso_fields.py`
- Test: `tests/test_tx_qso_fields.py`

**Interfaces:**
- Consumes: なし (標準ライブラリのみ)
- Produces:
  - `QsoFields` (`@dataclass(frozen=True)`): `their_call: str = ""` / `their_name: str = ""`
  - `strip_guess_marks(text: str) -> str`
  - `extract_fields(text: str, my_call: str = "") -> QsoFields`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_qso_fields.py`:

```python
"""受信テキストから欄を拾うテスト.

**拾えなかったら空のままにする。** 当てにならないものを黙って埋めると、
運用者が気づかないまま誤った電波が出る。
"""
from __future__ import annotations

from src.tx.qso_fields import QsoFields, extract_fields, strip_guess_marks


class TestStripGuessMarks:
    def test_清書の推測マーカーを外す(self) -> None:
        """清書結果には ⟦…⟧ が入っている (src/llm/markup.py)."""
        assert strip_guess_marks("CQ DE ⟦JA1ABC⟧ K") == "CQ DE JA1ABC K"

    def test_マーカーが無ければそのまま(self) -> None:
        assert strip_guess_marks("CQ DE JA1ABC K") == "CQ DE JA1ABC K"


class TestTheirCall:
    def test_DEの次を採る(self) -> None:
        """CW は <相手> DE <自分> の順で送る。**DE の次が送信者**."""
        assert extract_fields("JH0ILL DE JA1ABC K").their_call == "JA1ABC"

    def test_小文字でも拾う(self) -> None:
        assert extract_fields("jh0ill de ja1abc k").their_call == "JA1ABC"

    def test_最後のDEを採る(self) -> None:
        """一度の送信に DE が複数出ることがある。**最後が今の送信者**."""
        text = "JH0ILL DE JA1ABC JA1ABC K JH0ILL DE JH2XYZ K"
        assert extract_fields(text).their_call == "JH2XYZ"

    def test_DEが無ければ自局でないコールを採る(self) -> None:
        assert extract_fields("JH0ILL JA1ABC K", my_call="JH0ILL").their_call == "JA1ABC"

    def test_DEが無く自局も分からなければ空(self) -> None:
        """**どちらが相手か決められないなら埋めない。**"""
        assert extract_fields("JH0ILL JA1ABC K").their_call == ""

    def test_コールが無ければ空(self) -> None:
        assert extract_fields("{HORE}コンニチハ{RATA}").their_call == ""

    def test_DEの次がコールの形でなければ空(self) -> None:
        assert extract_fields("CQ DE CQ").their_call == ""

    def test_数字を含む局も拾う(self) -> None:
        assert extract_fields("JH0ILL DE 7K1ABC K").their_call == "7K1ABC"


class TestTheirName:
    def test_NAMEの次を採る(self) -> None:
        assert extract_fields("UR RST 599 NAME TARO K").their_name == "TARO"

    def test_OPの次を採る(self) -> None:
        assert extract_fields("OP JOHN QTH TOKYO").their_name == "JOHN"

    def test_和文のナマエハの次を採る(self) -> None:
        assert extract_fields("{HORE}ナマエ ハ タロウ、ヨロシク{RATA}").their_name == "タロウ"

    def test_見つからなければ空(self) -> None:
        """**当てにならないので黙って埋めない。**"""
        assert extract_fields("JH0ILL DE JA1ABC K").their_name == ""


class TestQsoFields:
    def test_何も無ければ全部空(self) -> None:
        assert extract_fields("") == QsoFields()

    def test_清書マーカー入りでも拾える(self) -> None:
        assert extract_fields("JH0ILL DE ⟦JA1ABC⟧ K").their_call == "JA1ABC"
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_qso_fields.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tx.qso_fields'`

- [ ] **Step 3: 実装する**

`src/tx/qso_fields.py`:

```python
"""受信テキストから返信に使う欄を拾う.

拾えなかったら空のままにする
----------------------------
**当てにならないものを黙って埋めない。** 運用者が気づかないまま誤った電波が
出るほうが、欄が空で手で打つより悪い。

相手の RST は拾わない
---------------------
返信に書くのは「**こちらが相手に与える RST**」であって、相手からもらった値では
ない。混同すると事故になるので、そもそも拾わない (設計書 §4.1)。

入力は清書済みテキスト
----------------------
生のデコードより誤りが減っている。清書結果には推測箇所の ``⟦…⟧`` が入っている
ので、:func:`strip_guess_marks` で外してから拾う。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 清書が付ける推測箇所のマーカー (src/llm/markup.py の OPEN_MARK / CLOSE_MARK)。
_GUESS_MARKS = str.maketrans("", "", "⟦⟧")

# アマチュア無線のコールサインのおおまかな形 (JA1ABC / JH0ILL / 7K1ABC など)。
# **厳密な判定はしない。** 拾えなければ空にするので、緩くても害が小さい。
_CALL_RE = re.compile(r"\b[A-Z0-9]{1,3}[0-9][A-Z]{1,4}\b")

# 名前の手掛かり。**当てにならないので、手掛かりが無ければ拾わない。**
_NAME_RES = (
    re.compile(r"\bNAME\s+([A-Z0-9]+)\b"),
    re.compile(r"\bOP\s+([A-Z0-9]+)\b"),
    re.compile(r"ナマエ\s*ハ\s*([ァ-ヴー]+)"),
)


@dataclass(frozen=True)
class QsoFields:
    """受信テキストから拾えた欄. **拾えなかったものは空文字。**"""

    their_call: str = ""
    their_name: str = ""


def strip_guess_marks(text: str) -> str:
    """清書が付けた推測箇所のマーカー ``⟦…⟧`` を外す."""
    return text.translate(_GUESS_MARKS)


def _their_call(text: str, my_call: str) -> str:
    """相手のコールを拾う.

    **``DE`` を手掛かりにする。** CW は ``<相手> DE <自分>`` の順で送るので、
    ``DE`` の次に来るのが送信者である。一度の送信に ``DE`` が複数出ることが
    あるので**最後のものを採る** (今の送信者がそれ)。

    ``DE`` が無いときは、コールの形をした語のうち**自局でないほう**を採る。
    自局が分からなければ**どちらが相手か決められないので空にする**。
    """
    tokens = text.split()
    for index in range(len(tokens) - 1, 0, -1):
        if tokens[index - 1] == "DE":
            candidate = tokens[index]
            if _CALL_RE.fullmatch(candidate):
                return candidate
            return ""
    if not my_call:
        return ""
    for token in tokens:
        if _CALL_RE.fullmatch(token) and token != my_call:
            return token
    return ""


def _their_name(text: str) -> str:
    for pattern in _NAME_RES:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return ""


def extract_fields(text: str, my_call: str = "") -> QsoFields:
    """受信テキストから欄を拾う.

    Args:
        text: 清書済みテキスト (``⟦…⟧`` が入っていてよい)。
        my_call: 自局のコール。``DE`` が無いときの手掛かりに使う。
    """
    cleaned = strip_guess_marks(text).upper()
    return QsoFields(
        their_call=_their_call(cleaned, my_call.upper()),
        their_name=_their_name(cleaned),
    )


__all__ = ["QsoFields", "extract_fields", "strip_guess_marks"]
```

**注意**: `extract_fields` は `.upper()` を掛けるので、和文の名前は
カタカナのまま残る (`upper()` はカタカナを変えない)。`_NAME_RES` の
和文パターンはそれを前提にしている。

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_qso_fields.py -v`
Expected: 16 passed

- [ ] **Step 5: ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/tx/qso_fields.py tests/test_tx_qso_fields.py`
Run: `.venv/Scripts/python.exe -m mypy src/tx/qso_fields.py`

- [ ] **Step 6: コミット**

```bash
git add src/tx/qso_fields.py tests/test_tx_qso_fields.py
git commit -m "feat: 受信テキストから相手のコールと名前を拾う

DE の次が送信者。複数あれば最後を採る。DE が無ければ自局でないコールを
採り、自局が分からなければ空にする (どちらが相手か決められないため)。
拾えなかったら空のままにする。当てにならないものを黙って埋めない。"
```

---

## Task 5: ダイアログに型と欄を足す

**Files:**
- Modify: `src/app/tx_dialog.py`
- Modify: `src/app/main_window.py` (受信テキストを渡す)
- Test: `tests/test_tx_dialog.py` (追記)

**Interfaces:**
- Consumes: Task 2〜4 の `ReplyTemplate` / `fill` / `missing_placeholders` / `profile_values` / `load_templates` / `templates_for_mode` / `unsendable_in_template` / `extract_fields`
- Produces:
  - `TxDialog(settings, profile=None, client_factory=NetKeyClient, parent=None, *, received_text: str = "", mode: str = "european", templates_path=DEFAULT_TEMPLATES_PATH)`
  - 属性 `their_call_edit` / `their_name_edit` / `rst_edit` / `template_combo` / `use_template_btn`
  - メソッド `apply_template() -> None` / `field_values(mode: str) -> dict[str, str]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_dialog.py` に追記:

```python
from src.tx.templates import ReplyTemplate, save_templates


def build_with_templates(qapp, tmp_path, templates, **overrides):
    """型を書いたファイルを用意してダイアログを作る."""
    from src.app.tx_dialog import TxDialog

    path = tmp_path / "templates.json"
    save_templates(templates, path)
    settings = AppSettings(**{"tx_endpoint": "127.0.0.1:45679", "tx_wpm": 20.0, **overrides})
    dialog = TxDialog(
        settings,
        profile=None,
        client_factory=lambda host, port=45679, **kw: FakeClient(host, port, **kw),
        templates_path=path,
        **{k: v for k, v in overrides.items() if k in {"received_text", "mode"}},
    )
    return dialog


class TestTemplates:
    def test_モードに合う型だけ一覧に出る(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(
            [
                ReplyTemplate(name="欧文", mode="european", text="CQ DE {自局コール} K"),
                ReplyTemplate(name="和文", mode="japanese", text="{HORE}コンニチハ{RATA}"),
                ReplyTemplate(name="どちらでも", mode="any", text="K"),
            ],
            path,
        )
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=None, templates_path=path, mode="european")
        names = [dialog.template_combo.itemText(i) for i in range(dialog.template_combo.count())]
        assert names == ["欧文", "どちらでも"]

    def test_型を使うと日本語ボックスに入る(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="CQ", mode="european", text="CQ DE {自局コール} K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        profile = OperatorProfile(callsign=ProfileField(display="JH0ILL"))
        dialog = TxDialog(settings, profile=profile, templates_path=path, mode="european")
        dialog.apply_template()
        assert dialog.japanese_edit.toPlainText() == "CQ DE JH0ILL K"

    def test_相手コールの欄が差し込まれる(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="応答", mode="european", text="{相手コール} DE {自局コール} K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        profile = OperatorProfile(callsign=ProfileField(display="JH0ILL"))
        dialog = TxDialog(
            settings, profile=profile, templates_path=path, mode="european",
            received_text="JH0ILL DE JA1ABC K",
        )
        assert dialog.their_call_edit.text() == "JA1ABC"
        dialog.apply_template()
        assert dialog.japanese_edit.toPlainText() == "JA1ABC DE JH0ILL K"

    def test_埋まらない欄を名指しで出す(self, qapp, tmp_path) -> None:
        """**符号化でも弾かれるが、そのメッセージは読めない** (設計書 §4.2)."""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="応答", mode="european", text="{相手コール} DE {自局コール} K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=None, templates_path=path, mode="european")
        dialog.apply_template()
        assert "埋まっていない" in dialog.status_label.text()
        assert "相手コール" in dialog.status_label.text()

    def test_RSTの既定は599(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=None, templates_path=tmp_path / "なし.json")
        assert dialog.rst_edit.text() == "599"

    def test_型を使うと関門が閉じ直る(self, qapp, tmp_path) -> None:
        """**確認していない文字列は送れない。**"""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="CQ", mode="european", text="CQ DE JH0ILL K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        clients: list[FakeClient] = []

        def factory(host, port=45679, **kw):
            client = FakeClient(host, port, **kw)
            clients.append(client)
            return client

        dialog = TxDialog(
            settings, profile=None, client_factory=factory, templates_path=path, mode="european"
        )
        dialog.connect_to_keyer()
        dialog.japanese_edit.setPlainText("CQ TEST")
        dialog.refresh_kana()
        dialog.run_check()
        assert dialog.can_send() is True
        dialog.apply_template()
        assert dialog.can_send() is False

    def test_型が無くても落ちない(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=None, templates_path=tmp_path / "なし.json")
        assert dialog.template_combo.count() == 0
        dialog.apply_template()                      # 落ちないこと
        assert dialog.japanese_edit.toPlainText() == ""
```

テストファイル冒頭の import に足す:

```python
from src.tx.profile import OperatorProfile, ProfileField
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_dialog.py::TestTemplates -v`
Expected: FAIL — `TypeError: TxDialog.__init__() got an unexpected keyword argument 'templates_path'`

- [ ] **Step 3: ダイアログを直す**

`src/app/tx_dialog.py` の import に追記:

```python
from PySide6.QtWidgets import QComboBox
from src.tx.qso_fields import extract_fields
from src.tx.templates import (
    DEFAULT_TEMPLATES_PATH,
    fill,
    load_templates,
    missing_placeholders,
    profile_values,
    templates_for_mode,
)
```

`__init__` の signature と本体を差し替える:

```python
    def __init__(
        self,
        settings: AppSettings,
        profile: OperatorProfile | None = None,
        client_factory: Callable[..., NetKeyClient] = NetKeyClient,
        parent=None,
        *,
        received_text: str = "",
        mode: str = "european",
        templates_path=DEFAULT_TEMPLATES_PATH,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("送信")
        self._settings = settings
        self._profile = profile if profile is not None else load_profile()
        self._client_factory = client_factory
        self._client: NetKeyClient | None = None
        self._worker: _SendWorker | None = None
        # **確認が通った文字列。** これと今の文字列が一致するときだけ送れる
        self._confirmed_text: str | None = None
        # そのモードで使える型だけを持つ (一覧の並びと同じ順)
        self._templates = templates_for_mode(load_templates(templates_path), mode)

        self._build_ui()
        self._fill_from_received(received_text)
        self._update_buttons()

        # **待機中は自動で繋ぎ直す** (設計書 §8.3)
        self._retry_timer = QTimer(self)
        self._retry_timer.setInterval(int(RETRY_INTERVAL_S * 1000))
        self._retry_timer.timeout.connect(self.retry_tick)
        self._retry_timer.start()
```

`_build_ui` の `top` レイアウトを足した直後 (`layout.addLayout(top)` の後) に挿入:

```python
        fields = QHBoxLayout()
        fields.addWidget(QLabel("相手:"))
        self.their_call_edit = QLineEdit()
        self.their_call_edit.setPlaceholderText("JA1ABC")
        fields.addWidget(self.their_call_edit, 1)
        fields.addWidget(QLabel("相手名前:"))
        self.their_name_edit = QLineEdit()
        fields.addWidget(self.their_name_edit, 1)
        fields.addWidget(QLabel("送る RST:"))
        self.rst_edit = QLineEdit("599")
        self.rst_edit.setМaxLength(3)
        fields.addWidget(self.rst_edit)
        layout.addLayout(fields)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("型:"))
        self.template_combo = QComboBox()
        for template in self._templates:
            self.template_combo.addItem(template.name)
        picker.addWidget(self.template_combo, 1)
        self.use_template_btn = QPushButton("型を使う")
        # **``clicked`` は ``checked: bool`` を渡す。** 引数の食い違いは
        # 過去にこのリポジトリで実際に踏んでいる
        self.use_template_btn.clicked.connect(lambda: self.apply_template())
        picker.addWidget(self.use_template_btn)
        layout.addLayout(picker)
```

**注意**: 上の `setМaxLength` は全角の М が混入した書き損じである。
`self.rst_edit.setMaxLength(3)` と半角で書くこと。

メソッドを足す (`refresh_kana` の隣):

```python
    def _fill_from_received(self, received_text: str) -> None:
        """受信テキストから拾えた欄を入れる. **拾えなければ空のまま。**"""
        if not received_text:
            return
        found = extract_fields(received_text, self._profile.callsign.sendable())
        self.their_call_edit.setText(found.their_call)
        self.their_name_edit.setText(found.their_name)

    def field_values(self, mode: str) -> dict[str, str]:
        """型に差し込む値 (経歴 + 画面の欄).

        **経歴の値は型のモードで変わる。** 欧文の型に読み (カタカナ) を
        差し込むと、そのカタカナのせいでモードが和文に倒れ、文中の欧文が
        まるごと送れなくなる (:func:`profile_values` 参照)。
        """
        values = profile_values(self._profile, mode)
        values["相手コール"] = self.their_call_edit.text().strip()
        values["相手名前"] = self.their_name_edit.text().strip()
        values["RST"] = self.rst_edit.text().strip()
        return {k: v for k, v in values.items() if v}

    def apply_template(self) -> None:
        """選んだ型に欄を差し込み、日本語ボックスへ入れる.

        **欄を差し込んでから入れる。** 逆にすると ``{相手コール}`` が
        カナ変換器を通って壊れる (設計書 §6.1)。
        """
        index = self.template_combo.currentIndex()
        if index < 0 or index >= len(self._templates):
            return
        template = self._templates[index]
        filled = fill(template.text, self.field_values(template.mode))
        # setPlainText が textChanged を起こし refresh_kana が走る
        # (関門もそこで閉じ直る)
        self.japanese_edit.setPlainText(filled)
        missing = missing_placeholders(filled)
        if missing:
            self.status_label.setText("埋まっていない欄があります: " + "、".join(missing))
```

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_dialog.py -v`
Expected: 既存分と合わせて全 passed

- [ ] **Step 5: メインウィンドウから受信テキストとモードを渡す**

`src/app/main_window.py` の `_open_tx_dialog` を差し替える:

```python
    def _open_tx_dialog(self) -> None:
        """送信ダイアログを開く.

        **打鍵はこの PC ではしない。** 無線機を繋いだ PC の CLI が行う
        (この PC には COM ポートが無い)。

        受信テキストは**清書済みがあればそちらを渡す**。生のデコードより
        誤りが減っており、拾える欄の精度が上がる (設計書 §5.1)。
        """
        from src.app.tx_dialog import TxDialog

        received = "\n".join(self._llm_refined_html) or self._committed_text
        dialog = TxDialog(
            self._settings,
            parent=self,
            received_text=received,
            mode=self._settings.mode,
        )
        try:
            dialog.exec()
        finally:
            dialog.shutdown()
            dialog.deleteLater()
        self._save_settings()          # ダイアログが tx_endpoint と tx_wpm を書き換えている
```

**注意**: `_open_tx_dialog` の既存の中身 (`try`/`finally` での `shutdown`/`deleteLater`)
を消さないこと。上はそれを保った形である。実物と食い違う場合は**実物に合わせ、
`received_text` と `mode` を足すだけにすること**。

- [ ] **Step 6: 画面が壊れていないことを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ui_smoke.py tests/test_tx_dialog.py -v`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **Step 7: ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/app/ src/tx/ tests/`
Run: `.venv/Scripts/python.exe -m mypy src/app/tx_dialog.py src/tx/templates.py src/tx/qso_fields.py`

- [ ] **Step 8: 全部走らせる**

Run: `.venv/Scripts/python.exe -m pytest tests/`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **Step 9: コミット**

```bash
git add src/app/tx_dialog.py src/app/main_window.py tests/test_tx_dialog.py
git commit -m "feat: 送信ダイアログに型と拾った欄を足す

欄を差し込んでから日本語ボックスへ入れる (逆にすると欄がカナ変換器を
通って壊れる)。埋まらない欄は名指しで出す。型を使うと関門が閉じ直る。
受信テキストは清書済みがあればそちらを使う。"
```

---

## Task 6: 例文を 10 個添える

**Files:**
- Create: `docs/reply_templates_example.json`
- Modify: `docs/USAGE.md` (§13 として追記)

**Interfaces:**
- Consumes: Task 3 の `load_templates` が読める形式
- Produces: なし (文書とデータのみ)

- [ ] **Step 1: 例文を書き、送れることを機械で確かめるテストを書く**

`tests/test_tx_templates.py` に追記:

```python
from pathlib import Path


class TestExampleFile:
    """添える例文が**実際に送れる**こと.

    運用者が最初に触るのがこれなので、送れない型が混ざっていてはいけない。
    """

    @pytest.fixture
    def 例文(self) -> list[ReplyTemplate]:
        path = Path(__file__).resolve().parent.parent / "docs" / "reply_templates_example.json"
        return load_templates(path)

    def test_十個ある(self, 例文) -> None:
        assert len(例文) == 10

    def test_全部送れる(self, 例文) -> None:
        for t in 例文:
            assert unsendable_in_template(t) == "", f"{t.name} が送れない"

    def test_名前が重複していない(self, 例文) -> None:
        names = [t.name for t in 例文]
        assert len(names) == len(set(names))

    def test_モードが正しい値(self, 例文) -> None:
        for t in 例文:
            assert t.mode in {"european", "japanese", "any"}, f"{t.name} の mode が不正"
```

- [ ] **Step 2: 失敗を確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py::TestExampleFile -v`
Expected: FAIL — ファイルが無いので `len(例文) == 10` が落ちる

- [ ] **Step 3: 例文を書く**

`docs/reply_templates_example.json`:

```json
{
  "templates": [
    {
      "name": "CQ (欧文)",
      "mode": "european",
      "text": "CQ CQ CQ DE {自局コール} {自局コール} {自局コール} PSE K"
    },
    {
      "name": "CQ (和文)",
      "mode": "japanese",
      "text": "CQ CQ CQ DE {自局コール} {自局コール} {HORE}ワブン ドウゾ{RATA} K"
    },
    {
      "name": "応答 (欧文)",
      "mode": "european",
      "text": "{相手コール} DE {自局コール} GM UR RST {RST} {RST} NAME {名前} QTH {QTH} HW? {相手コール} DE {自局コール} K"
    },
    {
      "name": "応答 (和文)",
      "mode": "japanese",
      "text": "{相手コール} DE {自局コール} RST {RST} {HORE}コンニチハ、ナマエ ハ {名前}、キョウテン ハ {QTH} デス、ヨロシク オネガイシマス{RATA} K"
    },
    {
      "name": "レポート交換",
      "mode": "european",
      "text": "{相手コール} DE {自局コール} R R TNX FER RPT UR RST {RST} {RST} K"
    },
    {
      "name": "設備の紹介",
      "mode": "japanese",
      "text": "{相手コール} DE {自局コール} RIG {リグ} ANT {アンテナ} PWR {出力} {HORE}デス、ヨロシク{RATA} K"
    },
    {
      "name": "天気の話",
      "mode": "japanese",
      "text": "{相手コール} DE {自局コール} {HORE}コチラ ノ テンキ ハ ハレ デス、ソチラ ハ イカガ デスカ{RATA} K"
    },
    {
      "name": "もう一度お願い",
      "mode": "any",
      "text": "{相手コール} DE {自局コール} PSE AGN AGN K"
    },
    {
      "name": "締め (欧文)",
      "mode": "european",
      "text": "{相手コール} DE {自局コール} TNX FER NICE QSO CUAGN 73 73 {相手コール} DE {自局コール} SK"
    },
    {
      "name": "締め (和文)",
      "mode": "japanese",
      "text": "{相手コール} DE {自局コール} {HORE}コウシン アリガトウ ゴザイマシタ、マタ ヨロシク オネガイシマス{RATA} 73 SK"
    }
  ]
}
```

**欧文は必ずホレの外に置くこと。** 和文モードの中では A〜Z が送れない (設計書 §4.3)。

とくに**経歴の値は英数字になりがち**である。`{リグ}` に `FT-991`、`{出力}` に `50W` を
入れると、ホレの中では `FT` と `W` が送れない。**実測でこれを踏んだ**ので、
上の「設備の紹介」は `RIG {リグ}` をホレの外に出してある。

上の 10 個は**次の値で実際に符号化まで通ることを確認済み**である:
`{自局コール}=JH0ILL` `{相手コール}=JA1ABC` `{RST}=599` `{名前}=TARO/タロウ`
`{QTH}=YOKOHAMA/ヨコハマシ` `{リグ}=FT-991` `{アンテナ}=DP` `{出力}=50W`

- [ ] **Step 4: 通ることを確かめる**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py -v`
Expected: 全 passed

**落ちたら例文を直すこと。** テストが正しく、例文が間違っている。

- [ ] **Step 5: 取扱説明書に書く**

`docs/USAGE.md` の **§12 の直後**に §13 として追記する。含める内容:

- 型は `~/.cw-decorder/templates.json` に置くこと
- **`docs/reply_templates_example.json` をそこへ複製すれば 10 個の例文から始められる**こと
- 差し込める欄の一覧 (`{自局コール}` `{名前}` `{QTH}` `{リグ}` `{アンテナ}` `{出力}` `{相手コール}` `{相手名前}` `{RST}`)
- **`{RST}` は「こちらが相手に与える RST」**であること
- **欧文 (コールサイン・RST・CQ・73 など) はホレの外に置く**こと。中に入れると送れない
- 経歴 (`~/.cw-decorder/operator.json`) を書いておくと `{名前}` などが埋まること
- 相手のコールは受信内容から自動で拾うが、**拾えなければ空のままなので手で打つ**こと
- 相手の名前は当てにならないこと

- [ ] **Step 6: 全部走らせる**

Run: `.venv/Scripts/python.exe -m pytest tests/`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **Step 7: コミット**

```bash
git add docs/reply_templates_example.json docs/USAGE.md tests/test_tx_templates.py
git commit -m "docs: 返信の型の例文 10 個と取扱説明書を追加する

例文が実際に送れることをテストで機械的に確かめる。運用者が最初に
触るものなので、送れない型が混ざっていてはいけない。"
```

---

## 完了の確認

- [ ] **返信の型まわりのテストが全部通る**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tx_templates.py tests/test_tx_qso_fields.py tests/test_tx_dialog.py -v`

- [ ] **既存のテストを壊していない**

Run: `.venv/Scripts/python.exe -m pytest tests/`
Expected: 1407 件 + 新規分。`FAILED` も `ERROR` も出ない

- [ ] **ruff と mypy**

Run: `.venv/Scripts/python.exe -m ruff check src/ scripts/ tests/`
Run: `.venv/Scripts/python.exe -m mypy src/tx/templates.py src/tx/qso_fields.py src/app/tx_dialog.py`

- [ ] **単一ソースを変更していない**

Run: `git diff main --stat -- src/tokens/morse_tokens.py`
Expected: 出力なし

- [ ] **赤字判定の修正が効いている (設計書 §3)**

`QT_QPA_PLATFORM=offscreen` で、日本語ボックスに
`JA1ABC DE JH0ILL {HORE}コンニチハ{RATA} K` を入れて
「送信できない」が出ないことを確かめる。

## 第 2 段に持ち越すもの

設計書 §7 (LLM の入口、返信案の生成、型として保存) は**この計画に含まれない**。
第 1 段を実際に使ってもらってから着手する。

# LLM テキスト清書機能 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CW デコード結果を LLM (Claude/OpenAI/Ollama) に渡し、誤り訂正・日本語清書・和文モード時の欧文化を行い、推測箇所を赤文字で表示する機能を追加する。

**Architecture:** 「音→符号→文字」三層分離を侵さない第 4 の独立層 `src/llm/` を新設。抽象 `LLMProvider` の背後に 3 実装を置き、共通 httpx クライアントで呼び出す。カナ⟷欧文対応表は `morse_tokens.py` から実行時生成。API 呼び出しは Qt ワーカーで非ブロッキング実行し、結果は下部の清書パネルに表示する (推測箇所 `⟦…⟧` → 赤 span)。

**Tech Stack:** Python 3.11+, httpx, python-dotenv, PySide6, pytest

設計書: `docs/superpowers/specs/2026-06-19-llm-text-refinement-design.md`

---

## ファイル構成

| ファイル | 責務 |
|---|---|
| `src/llm/__init__.py` | パッケージ公開 API |
| `src/llm/base.py` | `LLMResult`, `LLMProvider` (Protocol), `LLMError` |
| `src/llm/markup.py` | `⟦…⟧` マーカー入りテキスト → 赤 span 付き HTML (純粋関数, Qt 非依存) |
| `src/llm/prompt.py` | カナ→欧文対応表生成 + システム/ユーザープロンプト構築 |
| `src/llm/client.py` | 共通 httpx POST + 例外正規化 (`LLMError`) |
| `src/llm/providers/__init__.py` | プロバイダ公開 |
| `src/llm/providers/ollama.py` | Ollama `/api/chat` 実装 |
| `src/llm/providers/openai.py` | OpenAI Chat Completions 実装 |
| `src/llm/providers/claude.py` | Anthropic Messages 実装 |
| `src/llm/config.py` | `.env` 読込 + 設定 → プロバイダ生成ファクトリ |
| `src/llm/auto.py` | 自動清書デバウンス判定 (純粋関数, Qt 非依存) |
| `src/app/llm_worker.py` | Qt ワーカー (`QObject`) |
| `src/infer/settings.py` | LLM 設定フィールド追加 + v3→v4 マイグレーション (Modify) |
| `src/app/main_window.py` | 清書パネル・操作 UI・Signal 接続 (Modify) |
| `pyproject.toml` | `httpx`, `python-dotenv` 追加 (Modify) |
| `.gitignore` | `.env` 追加 (Modify) |
| `.env.example` | キー名テンプレート (Create) |

---

## Task 1: 依存追加と .env 雛形

**Files:**
- Modify: `pyproject.toml:14-24`
- Modify: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: pyproject.toml に依存を追加**

`dependencies` リストの末尾 (`"pyqtgraph>=0.13",` の後) に追加:

```toml
    "httpx>=0.27",
    "python-dotenv>=1.0",
```

- [ ] **Step 2: .gitignore に .env を追加**

`.gitignore` の末尾に追記 (既に存在する場合はスキップ):

```
# LLM API キー (機密)
.env
```

- [ ] **Step 3: .env.example を作成**

```
# LLM プロバイダの API キー。このファイルを .env にコピーして値を設定する。
# Ollama はローカル実行のためキー不要。
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

- [ ] **Step 4: 依存をインストール**

Run: `pip install -e .`
Expected: httpx, python-dotenv がインストールされ、エラーなく完了

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .env.example
git commit -m "chore: LLM 清書機能の依存 (httpx, python-dotenv) と .env 雛形を追加"
```

---

## Task 2: 基底型 (base.py)

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/base.py`
- Test: `tests/test_llm_base.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm_base.py`:

```python
"""LLM 基底型のテスト."""
from src.llm.base import LLMError, LLMResult


def test_llm_result_is_frozen_dataclass():
    r = LLMResult(text="本日は晴天", provider="ollama", model="llama3.1")
    assert r.text == "本日は晴天"
    assert r.provider == "ollama"
    assert r.model == "llama3.1"


def test_llm_error_is_exception():
    err = LLMError("接続失敗")
    assert isinstance(err, Exception)
    assert str(err) == "接続失敗"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/__init__.py`:

```python
"""LLM テキスト清書層 (「音→符号→文字」の後段に位置する独立層)."""
```

`src/llm/base.py`:

```python
"""LLM プロバイダ共通の基底型."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.tokens.morse_tokens import Mode


@dataclass(frozen=True)
class LLMResult:
    """LLM 清書結果. ``text`` は推測箇所 ⟦…⟧ マーカー入り."""

    text: str
    provider: str
    model: str


class LLMError(Exception):
    """全プロバイダ共通エラー型 (キー未設定/オフライン/HTTP/タイムアウト等を正規化)."""


class LLMProvider(Protocol):
    """LLM プロバイダの抽象インターフェース."""

    name: str
    model: str

    def transform(self, raw_text: str, mode: Mode, *, timeout: float) -> LLMResult:
        """生デコードテキストを清書して返す. 失敗時 ``LLMError``."""
        ...


__all__ = ["LLMResult", "LLMError", "LLMProvider"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_base.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/__init__.py src/llm/base.py tests/test_llm_base.py
git commit -m "feat: LLM 基底型 (LLMResult/LLMProvider/LLMError) を追加"
```

---

## Task 3: マーカー → 赤 HTML 変換 (markup.py)

**Files:**
- Create: `src/llm/markup.py`
- Test: `tests/test_llm_markup.py`

マーカーは `⟦` (U+27E6) と `⟧` (U+27E7)。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_markup.py`:

```python
"""マーカー → 赤 HTML 変換のテスト."""
from src.llm.markup import OPEN_MARK, CLOSE_MARK, to_html


def test_plain_text_is_escaped_black():
    html = to_html("本日は晴天")
    assert "本日は晴天" in html
    assert "color:#cc0000" not in html


def test_marked_span_becomes_red():
    html = to_html(f"こちら {OPEN_MARK}JH0ILL{CLOSE_MARK} です")
    assert '<span style="color:#cc0000;">JH0ILL</span>' in html
    assert OPEN_MARK not in html and CLOSE_MARK not in html


def test_prosign_angle_brackets_are_escaped_not_tags():
    html = to_html("<KN>")
    assert "&lt;KN&gt;" in html
    assert "<KN>" not in html


def test_marked_content_is_also_escaped():
    html = to_html(f"{OPEN_MARK}<KN>{CLOSE_MARK}")
    assert '<span style="color:#cc0000;">&lt;KN&gt;</span>' in html


def test_unbalanced_open_marker_is_stripped_safely():
    # 閉じ忘れは残りを赤として扱い、生マーカーを残さない
    html = to_html(f"abc {OPEN_MARK}def")
    assert OPEN_MARK not in html
    assert "def" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_markup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.markup'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/markup.py`:

```python
"""LLM 出力のマーカー ⟦…⟧ を赤文字 HTML に変換する純粋関数 (Qt 非依存).

推測箇所を ⟦…⟧ で囲った LLM 出力を、html.escape した上で
赤 span に変換する. プロサイン <KN> 等の角括弧はタグとして
解釈されないよう必ずエスケープする.
"""
from __future__ import annotations

import html

OPEN_MARK = "⟦"   # ⟦
CLOSE_MARK = "⟧"  # ⟧
_RED = "#cc0000"


def to_html(text: str) -> str:
    """⟦…⟧ マーカー入りテキストを赤 span 付き HTML に変換する.

    マーカー外は黒 (エスケープのみ)、マーカー内は赤 span.
    閉じ忘れ (アンバランス) の場合は開きマーカー以降を赤として扱う.
    """
    parts: list[str] = []
    in_mark = False
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        escaped = html.escape("".join(buf))
        if in_mark:
            parts.append(f'<span style="color:{_RED};">{escaped}</span>')
        else:
            parts.append(escaped)
        buf.clear()

    for ch in text:
        if ch == OPEN_MARK:
            flush()
            in_mark = True
        elif ch == CLOSE_MARK:
            flush()
            in_mark = False
        else:
            buf.append(ch)
    flush()
    return "".join(parts)


__all__ = ["OPEN_MARK", "CLOSE_MARK", "to_html"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_markup.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/markup.py tests/test_llm_markup.py
git commit -m "feat: LLM 推測箇所マーカー ⟦…⟧ → 赤 HTML 変換を追加"
```

---

## Task 4: カナ→欧文対応表の生成 (prompt.py その 1)

**Files:**
- Create: `src/llm/prompt.py`
- Test: `tests/test_llm_prompt.py`

`morse_tokens.py` の `JAPANESE_TABLE` / `EUROPEAN_TABLE` から、同一符号 `code` を
共有するエントリのみを抽出してカナ→欧文の対応 dict を生成する (二重定義しない)。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_prompt.py`:

```python
"""LLM プロンプト構築のテスト."""
from src.llm.prompt import build_kana_to_european


def test_kana_to_european_derives_from_morse_tables():
    table = build_kana_to_european()
    # ・- は欧文 A / 和文 イ (同符号) → イ→A
    assert table["イ"] == "A"
    # ・ は欧文 E / 和文 ヘ → ヘ→E
    assert table["ヘ"] == "E"
    # -・ は欧文 N / 和文 タ → タ→N
    assert table["タ"] == "N"


def test_kana_to_european_is_consistent_with_source_tables():
    from src.tokens.morse_tokens import EUROPEAN_TABLE, JAPANESE_TABLE
    table = build_kana_to_european()
    for code, kana in JAPANESE_TABLE.items():
        if code in EUROPEAN_TABLE and kana not in ("゛", "゜"):
            # 各エントリは元テーブルと矛盾しない
            assert table.get(kana) == EUROPEAN_TABLE[code]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_prompt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.prompt'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/prompt.py`:

```python
"""LLM 清書プロンプトの構築と、カナ→欧文対応表の実行時生成.

カナ→欧文対応表は morse_tokens.py を唯一の真正ソースとして生成し、
二重定義しない (アーキテクチャ原則 2).
"""
from __future__ import annotations

from src.tokens.morse_tokens import (
    DAKUTEN_CHAR,
    EUROPEAN_TABLE,
    HANDAKUTEN_CHAR,
    JAPANESE_TABLE,
)


def build_kana_to_european() -> dict[str, str]:
    """同一符号を共有する和文カナ → 欧文文字の対応表を生成する.

    濁点・半濁点は合成用記号のため除外する.
    """
    table: dict[str, str] = {}
    for code, kana in JAPANESE_TABLE.items():
        if kana in (DAKUTEN_CHAR, HANDAKUTEN_CHAR):
            continue
        eu = EUROPEAN_TABLE.get(code)
        if eu is not None:
            table[kana] = eu
    return table


__all__ = ["build_kana_to_european"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_prompt.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/prompt.py tests/test_llm_prompt.py
git commit -m "feat: カナ→欧文対応表を morse_tokens から実行時生成"
```

---

## Task 5: プロンプト本文の構築 (prompt.py その 2)

**Files:**
- Modify: `src/llm/prompt.py`
- Test: `tests/test_llm_prompt.py`

`build_messages(raw_text, mode)` が `[{"role": ..., "content": ...}]` 形式
(全プロバイダ共通の中立形式) を返す。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_prompt.py` に追記:

```python
from src.llm.prompt import build_messages


def test_build_messages_returns_system_and_user():
    msgs = build_messages("CQ CQ DE JH0ILL", mode="european")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert "CQ CQ DE JH0ILL" in msgs[1]["content"]


def test_system_prompt_mentions_marker_and_correction():
    msgs = build_messages("ABC", mode="european")
    system = msgs[0]["content"]
    assert "⟦" in system and "⟧" in system   # マーカー説明あり
    assert "訂正" in system                    # 誤り訂正の指示あり


def test_japanese_mode_includes_kana_european_table():
    msgs = build_messages("イロハ", mode="japanese")
    system = msgs[0]["content"]
    assert "欧文化" in system
    assert "イ" in system and "A" in system    # 対応表が埋め込まれている


def test_european_mode_excludes_european_conversion():
    msgs = build_messages("ABC", mode="european")
    system = msgs[0]["content"]
    assert "欧文化" not in system              # 欧文モードでは欧文化指示なし
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_prompt.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_messages'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/prompt.py` に追記 (import に `Mode` を追加):

```python
from src.tokens.morse_tokens import Mode  # 既存 import 群に追加
from src.llm.markup import OPEN_MARK, CLOSE_MARK

_BASE_SYSTEM = f"""あなたはアマチュア無線 CW (モールス信号) のデコード結果を校正する専門家です。
入力は AI デコーダの生出力で、誤り・脱落 ({{}}) が含まれます。次を行ってください。

1. デコード誤りの訂正: 文脈から D↔B, K↔T, Y↔A, 9→O, I→E 等の系統誤りや
   ? (脱落) を推測して補正する。
2. 読みやすい日本語への清書: 略語 (CQ, RST, QTH, OM, TNX, 73 等) を適度に展開し、
   自然な日本語にする。
3. あなたが推測・補正・展開した箇所は必ず {OPEN_MARK} と {CLOSE_MARK} で囲む。
   直接読めた確実な箇所は囲まない。マーカー以外の記号で囲ってはいけない。

出力は清書後テキストのみ。前置き・解説は不要。"""

_JP_EXTRA = """

4. 和文 (カタカナ) として出力されているが、欧文 (コールサイン・RST・数字・Q コード等)
   として意味が通る箇所は、次のカナ→欧文対応表に従って欧文へ変換する。
   変換した箇所も {open}…{close} で囲む。表に無い変換は行わない。

カナ→欧文対応表:
{table}"""


def _format_kana_table() -> str:
    items = sorted(build_kana_to_european().items())
    return " ".join(f"{kana}={eu}" for kana, eu in items)


def build_messages(raw_text: str, mode: Mode) -> list[dict[str, str]]:
    """全プロバイダ共通の中立メッセージ形式を構築する."""
    system = _BASE_SYSTEM
    if mode == "japanese":
        system += _JP_EXTRA.format(
            open=OPEN_MARK, close=CLOSE_MARK, table=_format_kana_table()
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": raw_text},
    ]
```

`__all__` に `"build_messages"` を追加。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_prompt.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/prompt.py tests/test_llm_prompt.py
git commit -m "feat: 清書プロンプト本文 (誤り訂正/清書/欧文化/マーカー指示) を構築"
```

---

## Task 6: 共通 HTTP クライアント (client.py)

**Files:**
- Create: `src/llm/client.py`
- Test: `tests/test_llm_client.py`

`post_json(url, json, headers, timeout)` が httpx で POST し、ネットワーク/HTTP
エラーを `LLMError` に正規化して返す。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_client.py`:

```python
"""共通 HTTP クライアントのテスト (httpx をモック)."""
import httpx
import pytest

from src.llm.base import LLMError
from src.llm.client import post_json


def _transport(handler):
    return httpx.MockTransport(handler)


def test_post_json_returns_parsed_body(monkeypatch):
    def handler(request):
        assert request.headers["x-test"] == "1"
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        "src.llm.client._build_client",
        lambda timeout: httpx.Client(transport=_transport(handler), timeout=timeout),
    )
    body = post_json("http://x/api", {"a": 1}, {"x-test": "1"}, timeout=5.0)
    assert body == {"ok": True}


def test_http_error_status_becomes_llmerror(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    monkeypatch.setattr(
        "src.llm.client._build_client",
        lambda timeout: httpx.Client(transport=_transport(handler), timeout=timeout),
    )
    with pytest.raises(LLMError) as exc:
        post_json("http://x/api", {}, {}, timeout=5.0)
    assert "401" in str(exc.value)


def test_connection_error_becomes_llmerror(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(
        "src.llm.client._build_client",
        lambda timeout: httpx.Client(transport=_transport(handler), timeout=timeout),
    )
    with pytest.raises(LLMError) as exc:
        post_json("http://x/api", {}, {}, timeout=5.0)
    assert "接続" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.client'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/client.py`:

```python
"""全プロバイダ共通の httpx POST と例外正規化."""
from __future__ import annotations

from typing import Any

import httpx

from src.llm.base import LLMError


def _build_client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout)


def post_json(
    url: str,
    json: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
) -> dict[str, Any]:
    """JSON を POST し、レスポンス body (dict) を返す.

    すべての失敗を LLMError に正規化する.
    """
    try:
        with _build_client(timeout) as client:
            resp = client.post(url, json=json, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMError(f"タイムアウトしました ({timeout:.0f}秒)") from exc
    except httpx.ConnectError as exc:
        raise LLMError(f"接続に失敗しました: {url}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"通信エラー: {exc!r}") from exc

    if resp.status_code >= 400:
        raise LLMError(f"API エラー (HTTP {resp.status_code}): {resp.text[:200]}")

    try:
        return resp.json()
    except ValueError as exc:
        raise LLMError("応答の JSON 解析に失敗しました") from exc


__all__ = ["post_json"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py tests/test_llm_client.py
git commit -m "feat: 共通 httpx クライアント (例外を LLMError に正規化) を追加"
```

---

## Task 7: Ollama プロバイダ

**Files:**
- Create: `src/llm/providers/__init__.py`
- Create: `src/llm/providers/ollama.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm_providers.py`:

```python
"""LLM プロバイダ実装のテスト (post_json をモック)."""
from src.llm.providers.ollama import OllamaProvider


def test_ollama_transform_builds_request_and_extracts_text(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, *, timeout):
        captured["url"] = url
        captured["json"] = json
        return {"message": {"content": "清書結果 ⟦推測⟧"}}

    monkeypatch.setattr("src.llm.providers.ollama.post_json", fake_post)
    p = OllamaProvider(model="llama3.1", endpoint="http://localhost:11434")
    result = p.transform("CQ DE JH0ILL", mode="european", timeout=10.0)

    assert result.text == "清書結果 ⟦推測⟧"
    assert result.provider == "ollama"
    assert result.model == "llama3.1"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "llama3.1"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["role"] == "system"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.providers'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/providers/__init__.py`:

```python
"""LLM プロバイダ実装群."""
```

`src/llm/providers/ollama.py`:

```python
"""Ollama (ローカル) プロバイダ — /api/chat. API キー不要."""
from __future__ import annotations

from src.llm.base import LLMError, LLMResult
from src.llm.client import post_json
from src.llm.prompt import build_messages
from src.tokens.morse_tokens import Mode


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, endpoint: str = "http://localhost:11434") -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def transform(self, raw_text: str, mode: Mode, *, timeout: float) -> LLMResult:
        body = post_json(
            f"{self.endpoint}/api/chat",
            {
                "model": self.model,
                "messages": build_messages(raw_text, mode),
                "stream": False,
            },
            {},
            timeout=timeout,
        )
        try:
            text = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama 応答の解析に失敗しました") from exc
        return LLMResult(text=text, provider=self.name, model=self.model)


__all__ = ["OllamaProvider"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_providers.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/providers/__init__.py src/llm/providers/ollama.py tests/test_llm_providers.py
git commit -m "feat: Ollama プロバイダを追加"
```

---

## Task 8: OpenAI プロバイダ

**Files:**
- Create: `src/llm/providers/openai.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm_providers.py` に追記:

```python
from src.llm.providers.openai import OpenAIProvider


def test_openai_transform_sets_auth_and_extracts_choice(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, *, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return {"choices": [{"message": {"content": "清書"}}]}

    monkeypatch.setattr("src.llm.providers.openai.post_json", fake_post)
    p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    result = p.transform("ABC", mode="european", timeout=10.0)

    assert result.text == "清書"
    assert result.provider == "openai"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "gpt-4o-mini"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_providers.py::test_openai_transform_sets_auth_and_extracts_choice -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.providers.openai'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/providers/openai.py`:

```python
"""OpenAI プロバイダ — Chat Completions API."""
from __future__ import annotations

from src.llm.base import LLMError, LLMResult
from src.llm.client import post_json
from src.llm.prompt import build_messages
from src.tokens.morse_tokens import Mode


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._api_key = api_key

    def transform(self, raw_text: str, mode: Mode, *, timeout: float) -> LLMResult:
        body = post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": self.model,
                "messages": build_messages(raw_text, mode),
            },
            {"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout,
        )
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI 応答の解析に失敗しました") from exc
        return LLMResult(text=text, provider=self.name, model=self.model)


__all__ = ["OpenAIProvider"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_providers.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/providers/openai.py tests/test_llm_providers.py
git commit -m "feat: OpenAI プロバイダを追加"
```

---

## Task 9: Claude プロバイダ

**Files:**
- Create: `src/llm/providers/claude.py`
- Test: `tests/test_llm_providers.py`

Anthropic Messages API は system を別フィールドに分離し、`x-api-key` /
`anthropic-version` ヘッダを使う。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_providers.py` に追記:

```python
from src.llm.providers.claude import ClaudeProvider


def test_claude_transform_splits_system_and_sets_headers(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, *, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return {"content": [{"type": "text", "text": "清書済み"}]}

    monkeypatch.setattr("src.llm.providers.claude.post_json", fake_post)
    p = ClaudeProvider(model="claude-haiku-4-5-20251001", api_key="ak-test")
    result = p.transform("ABC", mode="european", timeout=10.0)

    assert result.text == "清書済み"
    assert result.provider == "claude"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "ak-test"
    assert "anthropic-version" in captured["headers"]
    # system は messages から分離されている
    assert isinstance(captured["json"]["system"], str)
    assert all(m["role"] != "system" for m in captured["json"]["messages"])
    assert captured["json"]["max_tokens"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_providers.py::test_claude_transform_splits_system_and_sets_headers -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.providers.claude'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/providers/claude.py`:

```python
"""Claude (Anthropic) プロバイダ — Messages API."""
from __future__ import annotations

from src.llm.base import LLMError, LLMResult
from src.llm.client import post_json
from src.llm.prompt import build_messages
from src.tokens.morse_tokens import Mode

_ANTHROPIC_VERSION = "2023-06-01"
_MAX_TOKENS = 2048


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._api_key = api_key

    def transform(self, raw_text: str, mode: Mode, *, timeout: float) -> LLMResult:
        msgs = build_messages(raw_text, mode)
        system = next(m["content"] for m in msgs if m["role"] == "system")
        user_msgs = [m for m in msgs if m["role"] != "system"]
        body = post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                "max_tokens": _MAX_TOKENS,
                "system": system,
                "messages": user_msgs,
            },
            {
                "x-api-key": self._api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
            },
            timeout=timeout,
        )
        try:
            text = body["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Claude 応答の解析に失敗しました") from exc
        return LLMResult(text=text, provider=self.name, model=self.model)


__all__ = ["ClaudeProvider"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_providers.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/providers/claude.py tests/test_llm_providers.py
git commit -m "feat: Claude プロバイダを追加"
```

---

## Task 10: 設定フィールド追加と v3→v4 マイグレーション

**Files:**
- Modify: `src/infer/settings.py:42-55` (AppSettings フィールド), `:74` (CURRENT_SETTINGS_VERSION)
- Test: `tests/test_settings_migration.py`

- [ ] **Step 1: Write the failing test**

`tests/test_settings_migration.py` に追記:

```python
def test_v3_settings_migrate_to_v4_with_llm_defaults():
    from src.infer.settings import migrate_settings_dict, CURRENT_SETTINGS_VERSION
    old = {"settings_version": 3, "mode": "auto"}
    migrated, changed = migrate_settings_dict(old)
    assert changed is True
    assert migrated["settings_version"] == CURRENT_SETTINGS_VERSION == 4
    assert migrated["llm_enabled"] is False
    assert migrated["llm_provider"] == "ollama"
    assert migrated["llm_auto_interval_s"] == 20.0
    assert migrated["mode"] == "auto"   # 既存値は保持
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings_migration.py::test_v3_settings_migrate_to_v4_with_llm_defaults -v`
Expected: FAIL — `assert 3 == 4` (CURRENT_SETTINGS_VERSION がまだ 3)

- [ ] **Step 3: Write minimal implementation**

`src/infer/settings.py` の `AppSettings` で `settings_version` を更新し、LLM フィールドを追加。
`settings_version: int = 3` の行を次に変更:

```python
    settings_version: int = 4             # 設定スキーマ版 (マイグレーション用)
```

`commit_jitter_margin_s` フィールド定義の直後 (行 55 付近) に追加:

```python
    # --- LLM テキスト清書 ---
    llm_enabled: bool = False
    llm_provider: str = "ollama"          # "claude" | "openai" | "ollama"
    llm_model: str = "llama3.1"
    ollama_endpoint: str = "http://localhost:11434"
    llm_auto: bool = False                # 自動清書トグル
    llm_auto_interval_s: float = 20.0     # 自動清書の最短間隔 (秒)
    llm_timeout_s: float = 30.0
```

`CURRENT_SETTINGS_VERSION = 3` を次に変更:

```python
CURRENT_SETTINGS_VERSION = 4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_settings_migration.py tests/test_settings.py -v`
Expected: PASS (既存テスト含めすべて緑)

- [ ] **Step 5: Commit**

```bash
git add src/infer/settings.py tests/test_settings_migration.py
git commit -m "feat: LLM 設定フィールド追加と settings v3→v4 マイグレーション"
```

---

## Task 11: .env 読込とプロバイダ生成ファクトリ (config.py)

**Files:**
- Create: `src/llm/config.py`
- Test: `tests/test_llm_config.py`

`create_provider(settings)` が `AppSettings` からプロバイダを生成する。
キー未設定時は `LLMError`。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_config.py`:

```python
"""プロバイダ生成ファクトリのテスト."""
import pytest

from src.infer.settings import AppSettings
from src.llm.base import LLMError
from src.llm.config import create_provider
from src.llm.providers.claude import ClaudeProvider
from src.llm.providers.ollama import OllamaProvider


def test_create_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = AppSettings(llm_provider="ollama", llm_model="llama3.1")
    p = create_provider(s)
    assert isinstance(p, OllamaProvider)
    assert p.endpoint == "http://localhost:11434"


def test_create_claude_reads_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-xyz")
    s = AppSettings(llm_provider="claude", llm_model="claude-haiku-4-5-20251001")
    p = create_provider(s)
    assert isinstance(p, ClaudeProvider)


def test_create_openai_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = AppSettings(llm_provider="openai", llm_model="gpt-4o-mini")
    with pytest.raises(LLMError) as exc:
        create_provider(s)
    assert "OPENAI_API_KEY" in str(exc.value)


def test_unknown_provider_raises():
    s = AppSettings(llm_provider="bogus")
    with pytest.raises(LLMError):
        create_provider(s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.config'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/config.py`:

```python
"""`.env` 読込と AppSettings → LLMProvider 生成ファクトリ."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from src.infer.settings import AppSettings
from src.llm.base import LLMError, LLMProvider
from src.llm.providers.claude import ClaudeProvider
from src.llm.providers.ollama import OllamaProvider
from src.llm.providers.openai import OpenAIProvider

_loaded = False


def _ensure_env_loaded() -> None:
    global _loaded
    if not _loaded:
        load_dotenv()   # プロジェクトルートの .env を読む (無ければ無視)
        _loaded = True


def _require_key(name: str) -> str:
    _ensure_env_loaded()
    key = os.environ.get(name, "").strip()
    if not key:
        raise LLMError(
            f"{name} が設定されていません。.env または環境変数に設定してください。"
        )
    return key


def create_provider(settings: AppSettings) -> LLMProvider:
    """設定からプロバイダを生成する. キー未設定や未知プロバイダは LLMError."""
    provider = settings.llm_provider
    model = settings.llm_model
    if provider == "ollama":
        return OllamaProvider(model=model, endpoint=settings.ollama_endpoint)
    if provider == "openai":
        return OpenAIProvider(model=model, api_key=_require_key("OPENAI_API_KEY"))
    if provider == "claude":
        return ClaudeProvider(model=model, api_key=_require_key("ANTHROPIC_API_KEY"))
    raise LLMError(f"未知の LLM プロバイダ: {provider!r}")


__all__ = ["create_provider"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/config.py tests/test_llm_config.py
git commit -m "feat: .env 読込とプロバイダ生成ファクトリを追加"
```

---

## Task 12: 自動清書デバウンス判定 (auto.py)

**Files:**
- Create: `src/llm/auto.py`
- Test: `tests/test_llm_auto.py`

「前回送信時より確定テキストが伸びており、かつ最短間隔を経過していれば送信」を
判定する純粋関数。時刻は引数で受け取り、テスト可能にする。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_auto.py`:

```python
"""自動清書デバウンス判定のテスト."""
from src.llm.auto import AutoRefineState, should_refine


def test_no_refine_when_text_unchanged():
    st = AutoRefineState(last_text="ABC", last_time=0.0)
    assert should_refine("ABC", now=100.0, interval_s=20.0, state=st) is False


def test_no_refine_before_interval_elapsed():
    st = AutoRefineState(last_text="ABC", last_time=10.0)
    assert should_refine("ABCDEF", now=20.0, interval_s=20.0, state=st) is False


def test_refine_when_grown_and_interval_elapsed():
    st = AutoRefineState(last_text="ABC", last_time=10.0)
    assert should_refine("ABCDEF", now=31.0, interval_s=20.0, state=st) is True


def test_first_refine_when_no_previous():
    st = AutoRefineState(last_text="", last_time=0.0)
    assert should_refine("ABC", now=5.0, interval_s=20.0, state=st) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_auto.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.auto'`

- [ ] **Step 3: Write minimal implementation**

`src/llm/auto.py`:

```python
"""自動清書のデバウンス判定 (純粋関数, Qt 非依存)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoRefineState:
    """最後に LLM へ送ったテキストと時刻."""

    last_text: str = ""
    last_time: float = 0.0


def should_refine(
    current_text: str,
    *,
    now: float,
    interval_s: float,
    state: AutoRefineState,
) -> bool:
    """自動清書を実行すべきか判定する.

    条件: 確定テキストが前回送信時より伸びており、かつ前回送信から
    interval_s 秒以上経過している.
    """
    if current_text == state.last_text:
        return False
    if len(current_text) <= len(state.last_text):
        return False
    if now - state.last_time < interval_s:
        return False
    return True


__all__ = ["AutoRefineState", "should_refine"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_auto.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/llm/auto.py tests/test_llm_auto.py
git commit -m "feat: 自動清書デバウンス判定を追加"
```

---

## Task 13: Qt ワーカー (llm_worker.py)

**Files:**
- Create: `src/app/llm_worker.py`
- Test: `tests/test_llm_worker.py`

ワーカーは `request_transform(text, mode)` Slot を持ち、結果を
`result_ready(str)` / `error(str)` / `busy_changed(bool)` Signal で返す。
プロバイダ生成は遅延 (設定変更で再生成)。

- [ ] **Step 1: Write the failing test**

`tests/test_llm_worker.py`:

```python
"""LLM ワーカーのテスト (プロバイダをモックし、Signal を直接検証)."""
from src.app.llm_worker import LLMWorker
from src.llm.base import LLMError, LLMResult


class _FakeProvider:
    name = "fake"
    model = "m"

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def transform(self, raw_text, mode, *, timeout):
        if self._error:
            raise self._error
        return self._result


def test_worker_emits_result(qtbot_unused=None):
    worker = LLMWorker(timeout_s=5.0)
    worker.set_provider(_FakeProvider(result=LLMResult("清書 ⟦x⟧", "fake", "m")))
    received = []
    worker.result_ready.connect(received.append)
    worker.request_transform("ABC", "european")
    assert received == ["清書 ⟦x⟧"]


def test_worker_emits_error_message():
    worker = LLMWorker(timeout_s=5.0)
    worker.set_provider(_FakeProvider(error=LLMError("接続に失敗しました")))
    errors = []
    worker.error.connect(errors.append)
    worker.request_transform("ABC", "european")
    assert errors == ["接続に失敗しました"]


def test_worker_without_provider_emits_error():
    worker = LLMWorker(timeout_s=5.0)
    errors = []
    worker.error.connect(errors.append)
    worker.request_transform("ABC", "european")
    assert len(errors) == 1
    assert "プロバイダ" in errors[0]


def test_worker_toggles_busy():
    worker = LLMWorker(timeout_s=5.0)
    worker.set_provider(_FakeProvider(result=LLMResult("x", "fake", "m")))
    busy = []
    worker.busy_changed.connect(busy.append)
    worker.request_transform("ABC", "european")
    assert busy == [True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.llm_worker'`

- [ ] **Step 3: Write minimal implementation**

`src/app/llm_worker.py`:

```python
"""LLM 清書を別スレッドで実行する Qt ワーカー."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from src.llm.base import LLMError, LLMProvider
from src.tokens.morse_tokens import Mode


class LLMWorker(QObject):
    """確定テキストを LLM で清書するワーカー.

    UI → ワーカーは request_transform Slot (QueuedConnection 経由) で呼ぶ.
    """

    result_ready = Signal(str)      # 清書結果 (⟦…⟧ マーカー入り)
    error = Signal(str)             # 日本語エラーメッセージ
    busy_changed = Signal(bool)

    def __init__(self, timeout_s: float = 30.0) -> None:
        super().__init__()
        self._timeout_s = timeout_s
        self._provider: LLMProvider | None = None

    def set_provider(self, provider: LLMProvider | None) -> None:
        self._provider = provider

    def set_timeout(self, timeout_s: float) -> None:
        self._timeout_s = timeout_s

    @Slot(str, str)
    def request_transform(self, raw_text: str, mode: str) -> None:
        if self._provider is None:
            self.error.emit("LLM プロバイダが設定されていません。")
            return
        self.busy_changed.emit(True)
        try:
            result = self._provider.transform(
                raw_text, mode, timeout=self._timeout_s  # type: ignore[arg-type]
            )
            self.result_ready.emit(result.text)
        except LLMError as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - 予期せぬ失敗もUIに通知して継続
            self.error.emit(f"LLM 清書中に予期せぬエラー: {exc!r}")
        finally:
            self.busy_changed.emit(False)


__all__ = ["LLMWorker"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_worker.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/app/llm_worker.py tests/test_llm_worker.py
git commit -m "feat: LLM 清書 Qt ワーカーを追加"
```

---

## Task 14: メインウィンドウ統合 (UI パネル・操作・配線)

**Files:**
- Modify: `src/app/main_window.py`
- Test: `tests/test_ui_smoke.py`

清書パネル・プロバイダ/モデル選択・「清書」ボタン・自動清書チェックを追加し、
`LLMWorker` を別スレッドで起動して配線する。

- [ ] **Step 1: Write the failing test**

`tests/test_ui_smoke.py` に追記 (既存のフィクスチャ/QApplication 準備パターンに合わせる):

```python
def test_llm_panel_and_controls_exist(qapp):
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine
    import torch

    engine = InferenceEngine.untrained(device=torch.device("cpu"))
    win = CWDecoderWindow(engine)
    # 清書パネルと操作 UI が存在する
    assert win.llm_text_view is not None
    assert win.llm_provider_combo.count() == 3   # claude/openai/ollama
    assert win.llm_refine_btn is not None
    assert win.llm_auto_check is not None
    win.close()


def test_llm_result_renders_red_for_marked_spans(qapp):
    from src.app.main_window import CWDecoderWindow
    from src.app.llm_worker import LLMWorker  # noqa: F401
    from src.infer.engine import InferenceEngine
    from src.llm.markup import OPEN_MARK, CLOSE_MARK
    import torch

    engine = InferenceEngine.untrained(device=torch.device("cpu"))
    win = CWDecoderWindow(engine)
    win._on_llm_result(f"晴天 {OPEN_MARK}JH0ILL{CLOSE_MARK}")
    html = win.llm_text_view.toHtml()
    assert "cc0000" in html        # 赤 span が入っている
    win.close()
```

`qapp` フィクスチャが無い場合は既存 `tests/test_ui_smoke.py` の QApplication 準備方法に合わせること。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ui_smoke.py -v`
Expected: FAIL — `AttributeError: 'CWDecoderWindow' object has no attribute 'llm_text_view'`

- [ ] **Step 3: Write minimal implementation**

`src/app/main_window.py` を次のとおり変更する。

(a) import 追加 (ファイル冒頭の import 群へ):

```python
from src.app.llm_worker import LLMWorker
from src.llm.config import create_provider
from src.llm import markup
from src.llm.auto import AutoRefineState, should_refine
from src.llm.base import LLMError
import time
```

(b) クラス属性 Signal に追加 (`request_set_bpf_params` の下):

```python
    request_llm_transform = Signal(str, str)
```

(c) `__init__` のスペクトログラム追加 (`root.addWidget(self.spectrogram, 3)`) の直後に
清書 UI を構築:

```python
        # ---- LLM 清書パネル ----
        self.llm_text_view = QTextEdit()
        self.llm_text_view.setReadOnly(True)
        self.llm_text_view.setFont(QFont("Yu Gothic UI", 13))
        self.llm_text_view.setPlaceholderText("LLM 清書結果 (確定=黒 / 推測=赤)")
        root.addWidget(self.llm_text_view, 3)

        llm_bar = QHBoxLayout()
        llm_bar.addWidget(QLabel("LLM:"))
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(["ollama", "openai", "claude"])
        _p_index = {"ollama": 0, "openai": 1, "claude": 2}
        self.llm_provider_combo.setCurrentIndex(
            _p_index.get(self._settings.llm_provider, 0)
        )
        llm_bar.addWidget(self.llm_provider_combo)

        llm_bar.addWidget(QLabel("モデル:"))
        self.llm_model_edit = QComboBox()
        self.llm_model_edit.setEditable(True)
        self.llm_model_edit.setEditText(self._settings.llm_model)
        self.llm_model_edit.setMinimumWidth(180)
        llm_bar.addWidget(self.llm_model_edit)

        self.llm_refine_btn = QPushButton("清書")
        llm_bar.addWidget(self.llm_refine_btn)
        self.llm_auto_check = QCheckBox("自動清書")
        self.llm_auto_check.setChecked(self._settings.llm_auto)
        llm_bar.addWidget(self.llm_auto_check)
        llm_bar.addStretch(1)
        root.addLayout(llm_bar)
```

(d) `__init__` の `self._init_live_display_state()` 呼び出し付近で LLM 状態を初期化
(self._auto_submode 等の初期化と並べる):

```python
        self._auto_refine_state = AutoRefineState()
        self._llm_busy = False
        self._init_llm_worker()
```

(e) イベント接続 (`self.show_spectrogram_check.toggled.connect(...)` の下) に追加:

```python
        self.llm_refine_btn.clicked.connect(self._on_refine_clicked)
        self.llm_provider_combo.currentTextChanged.connect(self._on_llm_provider_changed)
        self.llm_model_edit.editTextChanged.connect(self._on_llm_model_changed)
```

(f) メソッド群を追加 (クラス内、`_on_worker_error` の前あたり):

```python
    # ---- LLM 清書 ----
    def _init_llm_worker(self) -> None:
        """LLM ワーカーを別スレッドで起動し、プロバイダを設定する."""
        self._llm_worker = LLMWorker(timeout_s=self._settings.llm_timeout_s)
        self._llm_thread = QThread()
        self._llm_worker.moveToThread(self._llm_thread)
        self._llm_worker.result_ready.connect(self._on_llm_result)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_worker.busy_changed.connect(self._on_llm_busy)
        self.request_llm_transform.connect(self._llm_worker.request_transform)
        self._llm_thread.start()
        self._refresh_llm_provider()

    def _refresh_llm_provider(self) -> None:
        """現在の設定からプロバイダを生成しワーカーへ渡す. 失敗はステータス表示."""
        self._settings.llm_provider = self.llm_provider_combo.currentText()
        self._settings.llm_model = self.llm_model_edit.currentText()
        try:
            provider = create_provider(self._settings)
            self._llm_worker.set_provider(provider)
        except LLMError as exc:
            self._llm_worker.set_provider(None)
            self.statusBar().showMessage(f"LLM 設定: {exc}")

    def _on_llm_provider_changed(self, _text: str) -> None:
        self._refresh_llm_provider()

    def _on_llm_model_changed(self, _text: str) -> None:
        self._refresh_llm_provider()

    def _on_refine_clicked(self) -> None:
        if self._llm_busy:
            return
        text = self._committed_text.strip()
        if not text:
            self.statusBar().showMessage("清書対象の確定テキストがありません")
            return
        self.request_llm_transform.emit(text, self._current_mode())

    def _on_llm_result(self, text: str) -> None:
        self.llm_text_view.setHtml(markup.to_html(text))
        # 自動清書の基準を更新
        self._auto_refine_state.last_text = self._committed_text
        self._auto_refine_state.last_time = time.monotonic()

    def _on_llm_error(self, message: str) -> None:
        self.statusBar().showMessage(f"LLM: {message}")

    def _on_llm_busy(self, busy: bool) -> None:
        self._llm_busy = busy
        self.llm_refine_btn.setEnabled(not busy)
        self.llm_refine_btn.setText("清書中…" if busy else "清書")

    def _maybe_auto_refine(self) -> None:
        """確定テキスト更新時に呼ばれ、条件を満たせば自動清書する."""
        if not self.llm_auto_check.isChecked() or self._llm_busy:
            return
        if should_refine(
            self._committed_text,
            now=time.monotonic(),
            interval_s=self._settings.llm_auto_interval_s,
            state=self._auto_refine_state,
        ):
            self.request_llm_transform.emit(
                self._committed_text.strip(), self._current_mode()
            )
```

(g) `_on_committed_text` の末尾に自動清書フックを追加:

```python
    def _on_committed_text(self, text: str) -> None:
        """確定テキスト全体を受信して表示を更新する."""
        self._committed_text = text
        self._refresh_decode_display()
        self._maybe_auto_refine()
```

(h) `_save_settings` に LLM 設定の保存を追加 (既存の代入群の後):

```python
        self._settings.llm_provider = self.llm_provider_combo.currentText()
        self._settings.llm_model = self.llm_model_edit.currentText()
        self._settings.llm_auto = self.llm_auto_check.isChecked()
```

(i) `closeEvent` でスレッドを停止する。`self._on_stop()` の後、`self._save_settings()`
の前に追加:

```python
        if self._llm_thread is not None:
            self._llm_thread.quit()
            self._llm_thread.wait(2000)
```

`__init__` で `self._llm_thread` を `_init_llm_worker` が設定するため、`closeEvent`
到達時には必ず存在する。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ui_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Run full suite to confirm no regression**

Run: `pytest -q`
Expected: 既存 410 + 新規テストすべて PASS

- [ ] **Step 6: Commit**

```bash
git add src/app/main_window.py tests/test_ui_smoke.py
git commit -m "feat: メインウィンドウに LLM 清書パネル・操作・自動清書を統合"
```

---

## Task 15: 利用者ドキュメント更新

**Files:**
- Modify: `docs/USAGE.md`
- Modify: `docs/INSTALL.md`

- [ ] **Step 1: USAGE.md に LLM 清書セクションを追加**

`docs/USAGE.md` の適切な位置 (オフラインデコードや設定の節の近く) に、次の内容を
日本語・です/ます調・テーブル併用で追記する:

- LLM 清書パネルの説明 (確定=黒 / 推測=赤)
- プロバイダ選択 (ollama / openai / claude) とモデル名入力
- 「清書」ボタンと「自動清書」チェックの使い方
- Ollama はローカルで無料・キー不要 (`ollama serve` 起動が必要)
- Claude/OpenAI は `.env` に API キーが必要 (次の INSTALL 節参照)
- 推測箇所が赤で表示される意味 (確実でない補正・欧文化箇所)

- [ ] **Step 2: INSTALL.md に .env 設定手順を追加**

`docs/INSTALL.md` に次を追記:

- `.env.example` を `.env` にコピーする手順
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` の取得先と記入例
- `.env` は他人に渡さない (機密) 旨の注意
- Ollama を使う場合はキー不要・別途 Ollama のインストールが必要な旨

- [ ] **Step 3: Commit**

```bash
git add docs/USAGE.md docs/INSTALL.md
git commit -m "docs: LLM 清書機能の使い方と .env 設定手順を追加"
```

---

## Task 16: 統合確認とブランチ完了

- [ ] **Step 1: 全テスト実行**

Run: `pytest -q`
Expected: すべて PASS

- [ ] **Step 2: ruff チェック**

Run: `ruff check src/llm src/app/llm_worker.py`
Expected: エラーなし (あれば修正)

- [ ] **Step 3: 手動スモーク (任意, 受信機 PC で)**

Ollama を起動 (`ollama serve` + `ollama pull llama3.1`) し、アプリを起動して
デコード → 「清書」ボタンで日本語清書・推測箇所の赤表示を確認する。

- [ ] **Step 4: finishing-a-development-branch スキルで完了処理**

実装完了後、superpowers:finishing-a-development-branch スキルを使い、PR 作成 or
マージを選択する。

---

## Self-Review チェック結果

- **Spec coverage:** 設計書 §3〜§13 の各要素を Task 1〜16 で網羅
  (基底型=T2, markup=T3, prompt=T4-5, client=T6, providers=T7-9, settings=T10,
  config/.env=T11, auto=T12, worker=T13, UI=T14, docs=T15)。
- **Placeholder scan:** コードステップはすべて実コードを記載。UI 統合 (T14) と
  docs (T15) のみ既存パターン参照を指示しているが、追加すべき具体項目は列挙済み。
- **Type consistency:** `LLMResult(text, provider, model)`、`transform(raw_text, mode, *, timeout)`、
  `build_messages`、`post_json`、`create_provider`、`should_refine`/`AutoRefineState`、
  Signal 名 (`result_ready`/`error`/`busy_changed`/`request_llm_transform`) を全 Task で統一。
- **Marker:** `OPEN_MARK`/`CLOSE_MARK` を markup.py で定義し prompt.py / テストで再利用。

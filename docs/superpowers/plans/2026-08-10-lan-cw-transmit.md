# LAN 越し CW 送信 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** アプリ (GPU PC) から確定テキストを LAN で送り、無線機 PC の CLI が符号化して打鍵する。接続が切れたら打鍵を止める。

**Architecture:** 1 行 1 メッセージの JSON を TCP で流す。無線機 PC が待ち受け (45679)、GPU PC が繋ぎに行く。打鍵のタイミングは**ポートと同じ機械で**刻むので LAN のジッタが鍵に乗らない。既存の `src/tx/` 5 モジュールは変更せず、実行される PC が分かれるだけ。

**Tech Stack:** Python 3.11+ / socket (標準) / json (標準) / numpy / pyserial (無線機 PC のみ) / PySide6 (アプリのみ) / pytest

**設計書:** `docs/superpowers/specs/2026-08-10-lan-cw-transmit-design.md`

## Global Constraints

- 言語: コメント・docstring・コミットメッセージはすべて**日本語**
- 文字コード **UTF-8 (BOM なし)**、改行 **LF**
- 型ヒント必須 (`mypy` 互換)。不変データは `@dataclass(frozen=True)`
- パス操作は `pathlib.Path`
- ruff: `line-length = 120`、`target-version = "py311"`
- **`src/tokens/morse_tokens.py` を変更しない。** 変更すると `scripts/export_tokens.py` の再実行が要る (CLAUDE.md)
- **`src/tx/keyer.py` `serial_key.py` `encoder.py` `reading.py` `profile.py` を変更しない。** 実行される PC が分かれるだけ
- **符号表を書き写さない。** `EUROPEAN_CHAR_TO_CODE` / `JAPANESE_CHAR_TO_CODES` / `SPECIAL_INPUT_MARKERS` から導出する (アーキテクチャ原則 2)
- **実機を使うテストは書かない。** 電波が出るため。シリアルは偽物か `--dry-run`
- テスト実行: `python -m pytest tests/<file> -v`
- **既知の環境問題**: `python -m pytest` の全実行は全パス後に `0xC0000409` で落ちサマリ行が出ない。**`FAILED` / `ERROR` の有無で判定すること**
- ブランチ: `feature/cw-transmit` (既存。ここに積む)
- 打鍵側の既定ポートは **45679** (音声の 45678 と分離)

## File Structure

| ファイル | 動く PC | 責務 |
|---|---|---|
| `src/tx/fingerprint.py` (新) | 両方 | 符号表の指紋。**読むだけ** |
| `src/tx/protocol.py` (新) | 両方 | メッセージの組み立てと解釈、行の切り出し |
| `src/tx/key_server.py` (新) | 無線機 | 要求の検証、セッション、打鍵、中止 |
| `scripts/cw_key_server.py` (新) | 無線機 | 引数の解釈と起動。**薄く保つ** |
| `src/tx/net_key.py` (新) | GPU | アプリ側クライアント。拍を打つ |
| `src/app/tx_dialog.py` (新) | GPU | 送信ダイアログ |
| `src/infer/settings.py` (改) | GPU | `tx_endpoint` / `tx_wpm`、スキーマ版 15 |
| `src/app/main_window.py` (改) | GPU | `[送信…]` ボタン 1 つと配線 |

`scripts/cw_key_server.py` を薄くし中身を `src/tx/key_server.py` に置くのは、`audio_send.py` (薄い) と `src/infer/net_audio.py` (中身) と同じ流儀にするため。テストが書ける。

---

## Task 1: 符号表の指紋

**Files:**
- Create: `src/tx/fingerprint.py`
- Test: `tests/test_tx_fingerprint.py`

**Interfaces:**
- Consumes: `src.tokens.morse_tokens` の 3 つの表 (読むだけ)
- Produces: `tokens_fingerprint() -> str` (16 進 8 桁)、`FINGERPRINT_LENGTH: int`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_fingerprint.py`:

```python
"""符号表の指紋のテスト.

指紋は**両 PC のリポジトリのずれを見つけるため**にある。決定的であること、
表が変われば変わることの 2 つが要件。
"""
from __future__ import annotations

import re

from src.tokens.morse_tokens import EUROPEAN_CHAR_TO_CODE, JAPANESE_CHAR_TO_CODES
from src.tx.fingerprint import FINGERPRINT_LENGTH, tokens_fingerprint


def test_決定的である() -> None:
    assert tokens_fingerprint() == tokens_fingerprint()


def test_16進の固定長である() -> None:
    value = tokens_fingerprint()
    assert len(value) == FINGERPRINT_LENGTH
    assert re.fullmatch(r"[0-9a-f]+", value)


def test_欧文表が変わると指紋が変わる(monkeypatch) -> None:
    before = tokens_fingerprint()
    monkeypatch.setitem(EUROPEAN_CHAR_TO_CODE, "A", "----")
    assert tokens_fingerprint() != before


def test_和文表が変わると指紋が変わる(monkeypatch) -> None:
    before = tokens_fingerprint()
    key = next(iter(JAPANESE_CHAR_TO_CODES))
    monkeypatch.setitem(JAPANESE_CHAR_TO_CODES, key, ("----",))
    assert tokens_fingerprint() != before
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_fingerprint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tx.fingerprint'`

- [ ] **Step 3: 実装する**

`src/tx/fingerprint.py`:

```python
"""符号表の指紋.

なぜ要るか
----------
**アプリ (GPU PC) と打鍵 CLI (無線機 PC) は別のリポジトリの複製を読む。**
ずれると、画面が「送れる」と言った文字を打鍵側が撥ねる (逆はない — 打鍵側が
必ず再検証するため)。接続時にこの指紋を突き合わせ、違えば警告する。

``morse_tokens.py`` は符号定義の唯一の真正ソースであり、変更すると
``scripts/export_tokens.py`` の再実行が要る (CLAUDE.md)。
**このモジュールは読むだけで、あちらを一切変更しない。**
"""
from __future__ import annotations

import hashlib

from src.tokens.morse_tokens import (
    EUROPEAN_CHAR_TO_CODE,
    JAPANESE_CHAR_TO_CODES,
    SPECIAL_INPUT_MARKERS,
)

# 指紋の桁数。人が目で読んで違いに気づける程度で足りる。
FINGERPRINT_LENGTH = 8


def tokens_fingerprint() -> str:
    """符号表から決定的な短い指紋を作る.

    **並び順に依存させない。** 辞書の反復順は保証されるが、表の作られ方が
    変わったときに中身が同じなのに指紋が変わると偽の警告になる。
    """
    digest = hashlib.sha256()
    for char, code in sorted(EUROPEAN_CHAR_TO_CODE.items()):
        digest.update(f"E{char}={code}\n".encode())
    for char, codes in sorted(JAPANESE_CHAR_TO_CODES.items()):
        digest.update(f"J{char}={'|'.join(codes)}\n".encode())
    for marker, code in sorted(SPECIAL_INPUT_MARKERS.items()):
        digest.update(f"M{marker}={code}\n".encode())
    return digest.hexdigest()[:FINGERPRINT_LENGTH]


__all__ = ["FINGERPRINT_LENGTH", "tokens_fingerprint"]
```

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_fingerprint.py -v`
Expected: 4 passed

- [ ] **Step 5: コミット**

```bash
git add src/tx/fingerprint.py tests/test_tx_fingerprint.py
git commit -m "feat: 符号表の指紋を追加する

両 PC のリポジトリのずれを接続時に見つけるため。morse_tokens.py は
読むだけで変更しない。"
```

---

## Task 2: プロトコル

**Files:**
- Create: `src/tx/protocol.py`
- Test: `tests/test_tx_protocol.py`

**Interfaces:**
- Consumes: なし (標準ライブラリのみ)
- Produces:
  - `PROTOCOL_VERSION: int = 1`、`DEFAULT_KEY_PORT: int = 45679`、`MAX_LINE_BYTES: int`
  - `CODE_UNSENDABLE` / `CODE_SERIAL` / `CODE_BUSY` / `CODE_BAD_REQUEST` / `CODE_LIFELINE` / `CODE_TOO_LONG` (すべて `str`)
  - `ProtocolError(ValueError)`
  - `encode_message(dict) -> bytes` / `decode_message(bytes | str) -> dict`
  - `hello(fingerprint: str, wiring: dict[str, str], dry_run: bool) -> dict`
  - `check(text: str, wpm: float) -> dict` / `send(text: str, wpm: float) -> dict`
  - `stop() -> dict` / `ping() -> dict`
  - `checked(chars: int, elements: int, seconds: float) -> dict`
  - `done(elements_sent: int, aborted: bool, watchdog_tripped: bool, seconds: float, max_error_ms: float, mean_error_ms: float) -> dict`
  - `error(code: str, message: str, **extra) -> dict`
  - `LineReader` (メソッド `feed(bytes) -> list[bytes]`)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_protocol.py`:

```python
"""LAN 越し打鍵のプロトコルのテスト."""
from __future__ import annotations

import pytest

from src.tx import protocol
from src.tx.protocol import LineReader, ProtocolError, decode_message, encode_message


def test_往復する() -> None:
    original = protocol.check("CQ DE JA1ABC K", 20.0)
    assert decode_message(encode_message(original)) == original


def test_日本語をそのまま載せる() -> None:
    """``ensure_ascii=False``。ログを人が読めることに価値があるため。"""
    raw = encode_message(protocol.send("{HORE}コンニチハ{RATA}", 20.0))
    assert "コンニチハ".encode() in raw
    assert raw.endswith(b"\n")


def test_1メッセージが1行である() -> None:
    raw = encode_message(protocol.error("busy", "打鍵中です\n2 行目"))
    assert raw.count(b"\n") == 1


def test_JSONでない行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b"{壊れている")


def test_辞書でない行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b'["check"]')


def test_typeの無い行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b'{"text": "CQ"}')


def test_UTF8でない行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b"\xff\xfe")


def test_errorが追加の欄を載せる() -> None:
    message = protocol.error(
        protocol.CODE_UNSENDABLE, "だめ", unsendable=[{"index": 3, "char": "髙"}]
    )
    assert message["code"] == "unsendable"
    assert message["unsendable"] == [{"index": 3, "char": "髙"}]


def test_LineReaderが行に分ける() -> None:
    reader = LineReader()
    assert reader.feed(b'{"type":"ping"}\n{"type":"stop"}\n') == [
        b'{"type":"ping"}',
        b'{"type":"stop"}',
    ]


def test_LineReaderが途中で切れた行を溜める() -> None:
    """**TCP は境界を保存しない。** 1 メッセージが 2 回の recv に割れる。"""
    reader = LineReader()
    assert reader.feed(b'{"type":"pi') == []
    assert reader.feed(b'ng"}\n') == [b'{"type":"ping"}']


def test_LineReaderが空行を捨てる() -> None:
    reader = LineReader()
    assert reader.feed(b"\n\n") == []


def test_LineReaderが長すぎる行を撥ねる() -> None:
    """壊れた相手や悪意ある相手に無限に溜めさせない."""
    reader = LineReader()
    with pytest.raises(ProtocolError):
        reader.feed(b"x" * (protocol.MAX_LINE_BYTES + 1))
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tx.protocol'`

- [ ] **Step 3: 実装する**

`src/tx/protocol.py`:

```python
"""LAN 越し打鍵のプロトコル (1 行 1 メッセージの JSON).

なぜ JSON の行なのか
--------------------
LAN に流すのは**確定したテキスト**である (設計書 §3)。テキストを流すと決めた以上、
ログがそのまま読めて ``netcat`` で叩けることに価値がある。

向きは音声 (``net_audio.py``) と同じ。**無線機 PC が待ち受け、GPU PC が繋ぎに行く。**
運用の癖が音声と揃う。ポートは音声の 45678 と分けて 45679。

暗号化しない。家庭内 LAN を前提とし、外部へ公開するポートには割り当てないこと。

メッセージ
----------
==================  ========  ==========================================
向き                type      意味
==================  ========  ==========================================
CLI → アプリ        hello     版・符号表の指紋・結線の要約
アプリ → CLI        check     **打鍵しない検査**
CLI → アプリ        checked   文字数・要素数・所要秒数
アプリ → CLI        send      打鍵する
CLI → アプリ        done      打鍵の実測
アプリ → CLI        stop      直ちに中止
アプリ → CLI        ping      心綱の拍
CLI → アプリ        error     撥ねた理由
==================  ========  ==========================================
"""
from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1

# 音声 (45678) と分ける。両方を同時に動かすため。
DEFAULT_KEY_PORT = 45679

# 1 行の上限。壊れた相手や悪意ある相手に無限に溜めさせないための歯止め。
# 和文の長文でも数 KB に収まる。
MAX_LINE_BYTES = 64 * 1024

# error の code
CODE_UNSENDABLE = "unsendable"      # 符号表に無い文字がある
CODE_SERIAL = "serial"              # ポートを開けない・書けない
CODE_BUSY = "busy"                  # 打鍵中
CODE_BAD_REQUEST = "bad_request"    # 解釈できない要求
CODE_LIFELINE = "lifeline"          # 心綱が切れた
CODE_TOO_LONG = "too_long"          # 長さの上限を超えた


class ProtocolError(ValueError):
    """受け取った行が解釈できない."""


def encode_message(message: dict[str, Any]) -> bytes:
    """1 メッセージを 1 行のバイト列にする.

    ``ensure_ascii=False`` はログを人が読めるようにするため。
    """
    return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")


def decode_message(line: bytes | str) -> dict[str, Any]:
    """1 行を解釈する.

    Raises:
        ProtocolError: UTF-8 でない、JSON でない、辞書でない、``type`` が無い.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"UTF-8 として読めません: {exc}") from exc
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"JSON として読めません: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"辞書ではありません: {type(obj).__name__}")
    if not isinstance(obj.get("type"), str):
        raise ProtocolError("type がありません")
    return obj


# ---- 組み立て ----
def hello(fingerprint: str, wiring: dict[str, str], dry_run: bool) -> dict[str, Any]:
    return {
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "tokens_fingerprint": fingerprint,
        "wiring": wiring,
        "dry_run": dry_run,
    }


def check(text: str, wpm: float) -> dict[str, Any]:
    return {"type": "check", "text": text, "wpm": float(wpm)}


def send(text: str, wpm: float) -> dict[str, Any]:
    return {"type": "send", "text": text, "wpm": float(wpm)}


def stop() -> dict[str, Any]:
    return {"type": "stop"}


def ping() -> dict[str, Any]:
    return {"type": "ping"}


def checked(chars: int, elements: int, seconds: float) -> dict[str, Any]:
    return {
        "type": "checked",
        "chars": int(chars),
        "elements": int(elements),
        "seconds": round(float(seconds), 3),
    }


def done(
    elements_sent: int,
    aborted: bool,
    watchdog_tripped: bool,
    seconds: float,
    max_error_ms: float,
    mean_error_ms: float,
) -> dict[str, Any]:
    return {
        "type": "done",
        "elements_sent": int(elements_sent),
        "aborted": bool(aborted),
        "watchdog_tripped": bool(watchdog_tripped),
        "seconds": round(float(seconds), 3),
        "max_error_ms": round(float(max_error_ms), 3),
        "mean_error_ms": round(float(mean_error_ms), 3),
    }


def error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message, **extra}


class LineReader:
    """バイト列から 1 行ずつ取り出す. 端数は次まで溜める.

    **TCP は境界を保存しない。** 1 メッセージが 2 回の ``recv`` に割れることも、
    2 メッセージが 1 回で届くこともある。
    """

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, data: bytes) -> list[bytes]:
        """受け取ったバイト列を足し、揃った行を返す.

        Raises:
            ProtocolError: 1 行が ``MAX_LINE_BYTES`` を超えた.
        """
        self._buffer += data
        if b"\n" not in self._buffer:
            if len(self._buffer) > MAX_LINE_BYTES:
                self._buffer = b""
                raise ProtocolError(f"1 行が長すぎます (> {MAX_LINE_BYTES} バイト)")
            return []
        *lines, self._buffer = self._buffer.split(b"\n")
        return [line for line in lines if line.strip()]


__all__ = [
    "CODE_BAD_REQUEST",
    "CODE_BUSY",
    "CODE_LIFELINE",
    "CODE_SERIAL",
    "CODE_TOO_LONG",
    "CODE_UNSENDABLE",
    "DEFAULT_KEY_PORT",
    "MAX_LINE_BYTES",
    "PROTOCOL_VERSION",
    "LineReader",
    "ProtocolError",
    "check",
    "checked",
    "decode_message",
    "done",
    "encode_message",
    "error",
    "hello",
    "ping",
    "send",
    "stop",
]
```

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_protocol.py -v`
Expected: 12 passed

- [ ] **Step 5: コミット**

```bash
git add src/tx/protocol.py tests/test_tx_protocol.py
git commit -m "feat: LAN 越し打鍵のプロトコルを追加する

1 行 1 メッセージの JSON。ログが読めて netcat で叩けることを優先した。
TCP が境界を保存しないので LineReader で行に組み直す。"
```

---

## Task 3: 要求の検証と要素列

**Files:**
- Create: `src/tx/key_server.py`
- Test: `tests/test_tx_key_server.py`

**Interfaces:**
- Consumes: `src.tx.encoder.find_unsendable` / `build_sequence` / `ElementSequence`、`src.tx.protocol` の code 群と `error`、`src.tx.serial_key.SerialKeyConfig`
- Produces:
  - `ServerConfig` (`@dataclass(frozen=True)`): `serial: SerialKeyConfig | None = None`、`ptt_lead_s: float = 0.05`、`ptt_tail_s: float = 0.10`、`key_watchdog_s: float = 5.0`、`max_duration_s: float = 120.0`、`lifeline_timeout_s: float = 1.0`。プロパティ `dry_run: bool`、メソッド `wiring() -> dict[str, str]`
  - `RequestRejected(Exception)`: 属性 `code: str` / `message: str` / `extra: dict`、メソッド `to_message() -> dict`
  - `visible_length(text: str) -> int`
  - `prepare(text: object, wpm: object, max_duration_s: float) -> ElementSequence`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_key_server.py`:

```python
"""打鍵サーバのテスト.

**実機は使わない。** 電波が出るため。シリアルは ``--dry-run`` 相当の偽物で代える。
"""
from __future__ import annotations

import pytest

from src.tx import protocol
from src.tx.key_server import (
    RequestRejected,
    ServerConfig,
    prepare,
    visible_length,
)
from src.tx.serial_key import SerialKeyConfig


def test_マーカーを1文字と数える() -> None:
    assert visible_length("A{HORE}B") == 3


def test_マーカーの無い文はそのまま数える() -> None:
    assert visible_length("CQ DE JA1ABC") == 12


def test_送れる文が要素列になる() -> None:
    sequence = prepare("CQ DE JA1ABC K", 20.0, max_duration_s=120.0)
    assert sequence.durations.size > 0
    assert sequence.total_seconds > 0.0


def test_送れない文字を位置つきで撥ねる() -> None:
    with pytest.raises(RequestRejected) as caught:
        prepare("CQ 髙 K", 20.0, max_duration_s=120.0)
    rejected = caught.value
    assert rejected.code == protocol.CODE_UNSENDABLE
    assert rejected.extra["unsendable"] == [{"index": 3, "char": "髙"}]


def test_撥ねた理由がメッセージになる() -> None:
    with pytest.raises(RequestRejected) as caught:
        prepare("CQ 髙 K", 20.0, max_duration_s=120.0)
    message = caught.value.to_message()
    assert message["type"] == "error"
    assert message["code"] == protocol.CODE_UNSENDABLE
    assert message["unsendable"] == [{"index": 3, "char": "髙"}]


def test_長すぎる文を撥ねる() -> None:
    with pytest.raises(RequestRejected) as caught:
        prepare("CQ DE JA1ABC K", 20.0, max_duration_s=0.5)
    assert caught.value.code == protocol.CODE_TOO_LONG


@pytest.mark.parametrize("text", ["", "   ", None, 123])
def test_textが不正なら撥ねる(text: object) -> None:
    with pytest.raises(RequestRejected) as caught:
        prepare(text, 20.0, max_duration_s=120.0)
    assert caught.value.code == protocol.CODE_BAD_REQUEST


@pytest.mark.parametrize("wpm", [0, -5, "はやい", None])
def test_wpmが不正なら撥ねる(wpm: object) -> None:
    with pytest.raises(RequestRejected) as caught:
        prepare("CQ K", wpm, max_duration_s=120.0)
    assert caught.value.code == protocol.CODE_BAD_REQUEST


def test_dry_runの設定は結線を晒さない() -> None:
    config = ServerConfig()
    assert config.dry_run is True
    assert config.wiring() == {"port": "", "key": "", "ptt": ""}


def test_実結線の設定は結線を返す() -> None:
    config = ServerConfig(serial=SerialKeyConfig(port="COM3"))
    assert config.dry_run is False
    assert config.wiring() == {"port": "COM3", "key": "DTR", "ptt": "RTS"}
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tx.key_server'`

- [ ] **Step 3: 実装する**

`src/tx/key_server.py` (このタスクの範囲だけ。Task 4〜6 で追記する):

```python
"""LAN 越し打鍵サーバの中身.

ソケットの受け口は :file:`scripts/cw_key_server.py`。**あちらは薄く保つ。**
``audio_send.py`` (薄い) と ``net_audio.py`` (中身) と同じ流儀で、
中身にテストが書けるようにするため。

打鍵はここで行う
----------------
**LAN のジッタを鍵の線に乗せない。** テキストを受け取り、符号化も要素列の生成も
時間の刻みも、すべて**ポートと同じ機械で**やる。25 WPM の短点は 48 ms であり、
線の上げ下げを LAN 越しに飛ばすと揺れが直接乗る。

受け取った文字は必ず再検証する
------------------------------
アプリ側も送れない文字を調べているが (画面に赤く出すため)、**両 PC の
リポジトリがずれることがある。** ここで撥ねれば、ずれても出るのは誤った電波
ではなくエラーである。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.tokens.morse_tokens import SPECIAL_INPUT_MARKERS
from src.tx import protocol
from src.tx.encoder import ElementSequence, build_sequence, find_unsendable
from src.tx.serial_key import SerialKeyConfig

_MARKER_RE = re.compile("|".join(re.escape(m) for m in SPECIAL_INPUT_MARKERS))


@dataclass(frozen=True)
class ServerConfig:
    """打鍵側の設定.

    **結線は LAN で流さない。** 無線機に属する情報であり、アプリが知る必要がない。

    Args:
        serial: 結線。``None`` なら dry-run (**シリアルに一切触らない**)。
    """

    serial: SerialKeyConfig | None = None
    ptt_lead_s: float = 0.05
    ptt_tail_s: float = 0.10
    key_watchdog_s: float = 5.0
    max_duration_s: float = 120.0
    lifeline_timeout_s: float = 1.0

    @property
    def dry_run(self) -> bool:
        return self.serial is None

    def wiring(self) -> dict[str, str]:
        """``hello`` に載せる結線の要約 (人が確認するためだけの情報)."""
        if self.serial is None:
            return {"port": "", "key": "", "ptt": ""}
        return {
            "port": self.serial.port,
            "key": self.serial.key_line,
            "ptt": self.serial.ptt_line,
        }


class RequestRejected(Exception):
    """要求を撥ねた. **打鍵していない。**"""

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra

    def to_message(self) -> dict[str, Any]:
        return protocol.error(self.code, self.message, **self.extra)


def visible_length(text: str) -> int:
    """マーカーを 1 文字と数えた長さ.

    ``{HORE}`` は 6 文字だが送られるのは 1 符号なので、画面に出す文字数としては
    1 と数えるほうが運用者の感覚に合う。
    """
    return len(_MARKER_RE.sub("*", text))


def prepare(text: object, wpm: object, max_duration_s: float) -> ElementSequence:
    """受け取った要求を検証し、要素列にする. **シリアルには触らない。**

    Raises:
        RequestRejected: 要求が不正、送れない文字がある、長すぎる.
    """
    if not isinstance(text, str) or not text.strip():
        raise RequestRejected(protocol.CODE_BAD_REQUEST, "text が空か文字列ではありません")
    if isinstance(wpm, bool) or not isinstance(wpm, (int, float)) or wpm <= 0:
        raise RequestRejected(protocol.CODE_BAD_REQUEST, f"wpm が不正です: {wpm!r}")

    bad = find_unsendable(text)
    if bad:
        raise RequestRejected(
            protocol.CODE_UNSENDABLE,
            "送信できない文字が含まれています: " + "".join(b.char for b in bad),
            unsendable=[{"index": b.index, "char": b.char} for b in bad],
        )

    sequence = build_sequence(text, float(wpm))
    if sequence.total_seconds > max_duration_s:
        raise RequestRejected(
            protocol.CODE_TOO_LONG,
            f"長すぎます ({sequence.total_seconds:.1f} 秒 > {max_duration_s:.1f} 秒)",
        )
    return sequence


__all__ = [
    "RequestRejected",
    "ServerConfig",
    "prepare",
    "visible_length",
]
```

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: 14 passed

- [ ] **Step 5: コミット**

```bash
git add src/tx/key_server.py tests/test_tx_key_server.py
git commit -m "feat: 打鍵側の要求検証を追加する

受け取った文字は必ず再検証する。両 PC のリポジトリがずれても、
出るのは誤った電波ではなくエラーになる。"
```

---

## Task 4: セッション — hello と check

**Files:**
- Modify: `src/tx/key_server.py` (追記)
- Test: `tests/test_tx_key_server.py` (追記)

**Interfaces:**
- Consumes: Task 3 の `ServerConfig` / `prepare` / `visible_length` / `RequestRejected`、Task 2 の `protocol`、Task 1 の `tokens_fingerprint`
- Produces: `KeySession`
  - `KeySession(config: ServerConfig, conn: socket.socket, *, sink_opener=open_sink, clock=time.monotonic)`
  - `run() -> None` — `hello` を送り、切れるまで捌く
  - `open_sink(config: ServerConfig) -> ContextManager[KeySink]`
  - `POLL_INTERVAL_S: float = 0.1`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_key_server.py` に追記:

```python
import socket
import threading

from src.tx.key_server import KeySession
from src.tx.protocol import LineReader, decode_message, encode_message


class Peer:
    """アプリ側のふり. ``socketpair`` の片側を持つ.

    **本物のソケットを使う。** 偽物にすると「TCP が境界を保存しない」ことや
    切断の扱いを取り逃がす。
    """

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._reader = LineReader()
        self._pending: list[bytes] = []

    def send(self, message: dict) -> None:
        self.sock.sendall(encode_message(message))

    def recv(self, timeout: float = 5.0) -> dict:
        """次の 1 メッセージを待つ."""
        self.sock.settimeout(timeout)
        while not self._pending:
            data = self.sock.recv(4096)
            if not data:
                raise AssertionError("打鍵側が切断しました")
            self._pending.extend(self._reader.feed(data))
        return decode_message(self._pending.pop(0))

    def close(self) -> None:
        self.sock.close()


def run_session(config: ServerConfig) -> tuple[Peer, threading.Thread]:
    """セッションを別スレッドで起こし、アプリ側の口を返す."""
    app_sock, server_sock = socket.socketpair()
    session = KeySession(config, server_sock)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    return Peer(app_sock), thread


def test_接続すると挨拶が来る() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        hello = peer.recv()
        assert hello["type"] == "hello"
        assert hello["protocol"] == protocol.PROTOCOL_VERSION
        assert len(hello["tokens_fingerprint"]) == 8
        assert hello["dry_run"] is True
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_checkが秒数を返す() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()                                   # hello
        peer.send(protocol.check("CQ DE JA1ABC K", 20.0))
        checked = peer.recv()
        assert checked["type"] == "checked"
        assert checked["chars"] == 14
        assert checked["elements"] > 0
        assert checked["seconds"] > 0.0
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_checkが送れない文字を撥ねる() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.check("CQ 髙 K", 20.0))
        rejected = peer.recv()
        assert rejected["type"] == "error"
        assert rejected["code"] == protocol.CODE_UNSENDABLE
        assert rejected["unsendable"] == [{"index": 3, "char": "髙"}]
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_checkはシリアルに触らない() -> None:
    """**``check`` でポートを開かない。** 開いた瞬間に線が上がる変換器があるため。"""
    opened: list[str] = []

    def spy_opener(config: ServerConfig):
        opened.append("開いた")
        raise AssertionError("check がシリアルを開いた")

    app_sock, server_sock = socket.socketpair()
    session = KeySession(ServerConfig(), server_sock, sink_opener=spy_opener)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    try:
        peer.recv()
        peer.send(protocol.check("CQ K", 20.0))
        assert peer.recv()["type"] == "checked"
        assert opened == []
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_知らないtypeを撥ねる() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send({"type": "おどる"})
        rejected = peer.recv()
        assert rejected["code"] == protocol.CODE_BAD_REQUEST
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_壊れた行を撥ねても切らない() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.sock.sendall(b"{壊れている\n")
        assert peer.recv()["code"] == protocol.CODE_BAD_REQUEST
        peer.send(protocol.check("CQ K", 20.0))       # まだ話せる
        assert peer.recv()["type"] == "checked"
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_pingには何も返さない() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.ping())
        peer.send(protocol.check("CQ K", 20.0))
        assert peer.recv()["type"] == "checked"       # ping の返事が挟まらない
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_切断でセッションが終わる() -> None:
    peer, thread = run_session(ServerConfig())
    peer.recv()
    peer.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: FAIL — `ImportError: cannot import name 'KeySession'`

- [ ] **Step 3: 実装する**

`src/tx/key_server.py` の import に追記:

```python
import contextlib
import socket
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager

from src.tx.fingerprint import tokens_fingerprint
from src.tx.keyer import KeySink, RecordingKeySink
from src.tx.serial_key import SerialKeySink
```

末尾 (`__all__` の前) に追記:

```python
# 読み待ちの刻み。心綱の期限を見張るため、待ちっぱなしにしない。
POLL_INTERVAL_S = 0.1

# 待機中を表す番人。切断 (None) と区別する。
_IDLE = object()

SinkOpener = Callable[["ServerConfig"], AbstractContextManager[KeySink]]


def open_sink(config: ServerConfig) -> AbstractContextManager[KeySink]:
    """送信 1 回分の窓口を開く.

    **dry-run ではシリアルに一切触らない。** COM ポートの無い PC でも
    LAN の経路を端から端まで試験できるようにするため。

    **起動時ではなく送信の直前に開く。** 多くの USB シリアル変換器はポートを
    開いた瞬間に DTR/RTS を H にする (``serial_key.py`` 参照)。
    """
    if config.serial is None:
        return contextlib.nullcontext(RecordingKeySink())
    return SerialKeySink(config.serial)


class KeySession:
    """1 接続分の処理.

    **同時に扱う接続は 1 つ。** 二重打鍵しないため。
    """

    def __init__(
        self,
        config: ServerConfig,
        conn: socket.socket,
        *,
        sink_opener: SinkOpener = open_sink,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.conn = conn
        self._sink_opener = sink_opener
        self._clock = clock
        self._reader = protocol.LineReader()
        self._pending: list[bytes] = []

    # ---- 本体 ----
    def run(self) -> None:
        """``hello`` を送り、切れるまで捌く."""
        self._send(
            protocol.hello(tokens_fingerprint(), self.config.wiring(), self.config.dry_run)
        )
        self.conn.settimeout(POLL_INTERVAL_S)
        while True:
            message = self._next_message()
            if message is None:
                return                       # 切断
            if message is _IDLE:
                continue                     # 待機中は期限を掛けない (設計書 §5)
            self._dispatch(message)          # type: ignore[arg-type]

    # ---- 受け取り ----
    def _next_message(self) -> dict[str, Any] | object | None:
        """次の 1 メッセージ / ``_IDLE`` (何も来ていない) / ``None`` (切断)."""
        while not self._pending:
            try:
                data = self.conn.recv(4096)
            except TimeoutError:
                return _IDLE
            except OSError:
                return None
            if not data:
                return None
            try:
                self._pending.extend(self._reader.feed(data))
            except protocol.ProtocolError as exc:
                self._send(protocol.error(protocol.CODE_BAD_REQUEST, str(exc)))
                return _IDLE
        line = self._pending.pop(0)
        try:
            return protocol.decode_message(line)
        except protocol.ProtocolError as exc:
            self._send(protocol.error(protocol.CODE_BAD_REQUEST, str(exc)))
            return _IDLE

    def _send(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):
            self.conn.sendall(protocol.encode_message(message))

    # ---- 割り振り ----
    def _dispatch(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        if kind == "ping":
            return                            # 拍。返事は要らない
        if kind == "stop":
            return                            # 打鍵していないときは何もしない
        if kind == "check":
            self._handle_check(message)
            return
        self._send(
            protocol.error(protocol.CODE_BAD_REQUEST, f"知らない type です: {kind!r}")
        )

    def _handle_check(self, message: dict[str, Any]) -> None:
        """**打鍵しない検査。** シリアルには触らない."""
        try:
            sequence = prepare(
                message.get("text"), message.get("wpm"), self.config.max_duration_s
            )
        except RequestRejected as exc:
            self._send(exc.to_message())
            return
        self._send(
            protocol.checked(
                chars=visible_length(message["text"]),
                elements=int(sequence.durations.size),
                seconds=sequence.total_seconds,
            )
        )
```

`__all__` に `"KeySession"`, `"POLL_INTERVAL_S"`, `"SinkOpener"`, `"open_sink"` を足す。

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: 22 passed

- [ ] **Step 5: コミット**

```bash
git add src/tx/key_server.py tests/test_tx_key_server.py
git commit -m "feat: 打鍵サーバのセッションと check を追加する

check は打鍵しない検査で、シリアルを開かない。開いた瞬間に線が上がる
変換器があるため。壊れた行を撥ねても接続は切らない。"
```

---

## Task 5: セッション — 打鍵と busy

**Files:**
- Modify: `src/tx/key_server.py` (追記)
- Test: `tests/test_tx_key_server.py` (追記)

**Interfaces:**
- Consumes: Task 4 の `KeySession` / `open_sink`、`src.tx.keyer.Keyer`
- Produces: `KeySession._handle_send` の振る舞い (`send` → `done` / `busy` / `serial`)

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_key_server.py` に追記:

```python
from src.tx.keyer import RecordingKeySink
from src.tx.serial_key import SerialKeyError


def test_sendが打鍵して実測を返す() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("E E", 40.0))          # 短い文で速く終わらせる
        done = peer.recv(timeout=10.0)
        assert done["type"] == "done"
        assert done["aborted"] is False
        assert done["watchdog_tripped"] is False
        assert done["elements_sent"] > 0
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_打鍵の最後に両線が落ちる() -> None:
    """**上げっぱなしにしない。** これが何より大事."""
    sink = RecordingKeySink()

    def opener(config: ServerConfig):
        return contextlib.nullcontext(sink)

    app_sock, server_sock = socket.socketpair()
    session = KeySession(ServerConfig(), server_sock, sink_opener=opener)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    try:
        peer.recv()
        peer.send(protocol.send("E E", 40.0))
        assert peer.recv(timeout=10.0)["type"] == "done"
        assert sink.ended_safely()
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_打鍵中のsendをbusyで撥ねる() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("CQ CQ CQ DE JA1ABC JA1ABC K", 12.0))   # 長め
        peer.send(protocol.send("E", 40.0))
        first = peer.recv(timeout=10.0)
        assert first["type"] == "error"
        assert first["code"] == protocol.CODE_BUSY
        assert peer.recv(timeout=60.0)["type"] == "done"                # 1 本目は完走する
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_打鍵中のcheckもbusyで撥ねる() -> None:
    """符号化が打鍵の時間刻みを邪魔しないようにするため."""
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("CQ CQ CQ DE JA1ABC JA1ABC K", 12.0))
        peer.send(protocol.check("E", 40.0))
        first = peer.recv(timeout=10.0)
        assert first["code"] == protocol.CODE_BUSY
        assert peer.recv(timeout=60.0)["type"] == "done"
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_sendが送れない文字を撥ねる() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("CQ 髙 K", 20.0))
        assert peer.recv()["code"] == protocol.CODE_UNSENDABLE
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_ポートを開けなければserialを返す() -> None:
    def broken_opener(config: ServerConfig):
        raise SerialKeyError("COM3 を開けませんでした")

    app_sock, server_sock = socket.socketpair()
    session = KeySession(ServerConfig(), server_sock, sink_opener=broken_opener)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    try:
        peer.recv()
        peer.send(protocol.send("E", 40.0))
        rejected = peer.recv(timeout=10.0)
        assert rejected["code"] == protocol.CODE_SERIAL
        peer.send(protocol.check("E", 40.0))          # 失敗しても話し続けられる
        assert peer.recv()["type"] == "checked"
    finally:
        peer.close()
        thread.join(timeout=5.0)
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: FAIL — `send` が `bad_request` を返すため `assert done["type"] == "done"` で落ちる

- [ ] **Step 3: 実装する**

`src/tx/key_server.py` の import に追記:

```python
from src.tx.keyer import Keyer, KeyingReport
from src.tx.serial_key import SerialKeyError
```

`KeySession.__init__` に追記:

```python
        self._keying = False
        self._abort = threading.Event()
```

(`import threading` を足す)

`_dispatch` を差し替える:

```python
    def _dispatch(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        if kind == "ping":
            return                            # 拍。返事は要らない
        if kind == "stop":
            self._abort.set()                 # 打鍵していなければ何も起きない
            return
        if self._keying:
            # **打鍵中は check も撥ねる。** 符号化が時間の刻みを邪魔しないため
            self._send(protocol.error(protocol.CODE_BUSY, "打鍵中です"))
            return
        if kind == "check":
            self._handle_check(message)
            return
        if kind == "send":
            self._handle_send(message)
            return
        self._send(
            protocol.error(protocol.CODE_BAD_REQUEST, f"知らない type です: {kind!r}")
        )
```

`__init__` に `self._result: dict[str, Any] | None = None` も足す。

`_handle_send` と `_key` を足す。**打鍵は別スレッドで回す** — 読み取りループを
止めると `stop` と拍を受け取れなくなり、`busy` も成立しないため:

```python
    def _handle_send(self, message: dict[str, Any]) -> None:
        """打鍵する.

        **打鍵は別スレッドで回す。** 読み取りループは止められない —
        ``stop`` と拍を受け取り続ける必要があるため (設計書 §5)。
        """
        try:
            sequence = prepare(
                message.get("text"), message.get("wpm"), self.config.max_duration_s
            )
        except RequestRejected as exc:
            self._send(exc.to_message())
            return

        self._abort.clear()
        self._keying = True
        self._result = None

        def worker() -> None:
            try:
                report = self._key(sequence)
            except SerialKeyError as exc:
                self._result = protocol.error(protocol.CODE_SERIAL, str(exc))
            except Exception as exc:                       # noqa: BLE001
                self._result = protocol.error(protocol.CODE_SERIAL, str(exc))
            else:
                self._result = protocol.done(
                    elements_sent=report.elements_sent,
                    aborted=report.aborted,
                    watchdog_tripped=report.watchdog_tripped,
                    seconds=sequence.total_seconds,
                    max_error_ms=report.max_error_ms,
                    mean_error_ms=report.mean_error_ms,
                )

        thread = threading.Thread(target=worker, name="cw-keying", daemon=True)
        thread.start()
        self._pump_while_keying(thread)
        self._keying = False
        if self._result is not None:
            self._send(self._result)

    def _key(self, sequence: ElementSequence) -> KeyingReport:
        """要素列を叩く. **どの経路で抜けても両線を落とす** (``Keyer`` が担保)."""
        keyer = Keyer(key_watchdog_s=self.config.key_watchdog_s)
        with self._sink_opener(self.config) as sink:
            return keyer.send(
                sequence.durations,
                sequence.is_on,
                sink,
                abort=self._abort,
                ptt_lead_s=self.config.ptt_lead_s,
                ptt_tail_s=self.config.ptt_tail_s,
                use_ptt=self.config.serial is None or self.config.serial.uses_ptt,
            )
```

`_pump_while_keying` はこのタスクでは最小限にする (Task 6 で心綱を足す):

```python
    def _pump_while_keying(self, thread: threading.Thread) -> None:
        """打鍵が終わるまで、届くメッセージを捌き続ける."""
        while thread.is_alive():
            message = self._next_message()
            if message is None:                # 切断
                self._abort.set()
                break
            if message is _IDLE:
                continue
            self._dispatch(message)            # type: ignore[arg-type]
        thread.join(timeout=5.0)
```

`__all__` は変更なし。

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: 28 passed

- [ ] **Step 5: コミット**

```bash
git add src/tx/key_server.py tests/test_tx_key_server.py
git commit -m "feat: 打鍵側の send を追加する

打鍵は別スレッドで回す。読み取りループを止めると stop と拍を
受け取れなくなるため。打鍵中の send も check も busy で撥ねる。"
```

---

## Task 6: セッション — 中止の 3 経路

**Files:**
- Modify: `src/tx/key_server.py` (`_pump_while_keying` を差し替え)
- Test: `tests/test_tx_key_server.py` (追記)

**Interfaces:**
- Consumes: Task 5 の `_pump_while_keying`
- Produces: `stop` / 切断 / 心綱の 3 経路がすべて `self._abort` に合流する振る舞い

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_key_server.py` に追記:

```python
def _長い文() -> str:
    return "CQ CQ CQ DE JA1ABC JA1ABC JA1ABC PSE K"


def test_stopで中止し両線が落ちる() -> None:
    sink = RecordingKeySink()

    def opener(config: ServerConfig):
        return contextlib.nullcontext(sink)

    app_sock, server_sock = socket.socketpair()
    session = KeySession(ServerConfig(), server_sock, sink_opener=opener)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    try:
        peer.recv()
        peer.send(protocol.send(_長い文(), 12.0))
        time.sleep(0.3)                       # 少し打たせてから止める
        peer.send(protocol.stop())
        done = peer.recv(timeout=10.0)
        assert done["type"] == "done"
        assert done["aborted"] is True
        assert sink.ended_safely()
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_切断で中止し両線が落ちる() -> None:
    sink = RecordingKeySink()

    def opener(config: ServerConfig):
        return contextlib.nullcontext(sink)

    app_sock, server_sock = socket.socketpair()
    session = KeySession(ServerConfig(), server_sock, sink_opener=opener)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    peer.recv()
    peer.send(protocol.send(_長い文(), 12.0))
    time.sleep(0.3)
    peer.close()                              # ここで切る
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert sink.ended_safely()


def test_拍が途絶えると中止し両線が落ちる() -> None:
    """**心綱。** LAN が止まったら打鍵を止める (設計書 §5)."""
    sink = RecordingKeySink()

    def opener(config: ServerConfig):
        return contextlib.nullcontext(sink)

    app_sock, server_sock = socket.socketpair()
    session = KeySession(
        ServerConfig(lifeline_timeout_s=0.3), server_sock, sink_opener=opener
    )
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    try:
        peer.recv()
        peer.send(protocol.send(_長い文(), 12.0))
        # 拍を送らない。切断もしない (ケーブルが抜けた状態の模擬)
        done = peer.recv(timeout=10.0)
        assert done["type"] == "done"
        assert done["aborted"] is True
        assert sink.ended_safely()
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_拍を打ち続ければ中止されない() -> None:
    peer, thread = run_session(ServerConfig(lifeline_timeout_s=0.3))
    try:
        peer.recv()
        peer.send(protocol.send("CQ DE JA1ABC K", 20.0))
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            peer.send(protocol.ping())
            try:
                done = peer.recv(timeout=0.1)
            except (TimeoutError, socket.timeout):
                continue
            assert done["type"] == "done"
            assert done["aborted"] is False
            return
        raise AssertionError("打鍵が終わりませんでした")
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_待機中は拍が無くても切られない() -> None:
    """待機中に期限を掛けると接続が常に落ちる (設計書 §5)."""
    peer, thread = run_session(ServerConfig(lifeline_timeout_s=0.2))
    try:
        peer.recv()
        time.sleep(0.6)                       # 期限の 3 倍だまる
        peer.send(protocol.check("CQ K", 20.0))
        assert peer.recv()["type"] == "checked"
    finally:
        peer.close()
        thread.join(timeout=5.0)
```

冒頭の import に `import time` を足す。

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: FAIL — `test_拍が途絶えると中止し両線が落ちる` が `aborted is True` で落ちる (心綱が未実装)

- [ ] **Step 3: 実装する**

`_pump_while_keying` を差し替える:

```python
    def _pump_while_keying(self, thread: threading.Thread) -> None:
        """打鍵が終わるまで、届くメッセージを捌きつつ**心綱を見張る**.

        中止の経路は 3 つあり、**すべて ``self._abort`` に合流する**:

        1. ``stop`` — 運用者が止めた
        2. 切断 — アプリが落ちた
        3. **拍の途絶** — LAN が止まった、アプリが固まった

        3 が要るのは、**ケーブルが抜けても TCP は黙ったままだから**である。
        ソケットが閉じたことはアプリが落ちれば分かるが、線が抜けたときには
        分からない。「止まったら止める」を字義どおり満たすには拍が要る。

        待機中には期限を掛けない (掛けると接続が常に落ちる)。ここは打鍵中だけ。
        """
        deadline = self._clock() + self.config.lifeline_timeout_s
        while thread.is_alive():
            message = self._next_message()
            if message is None:                          # 経路 2: 切断
                self._abort.set()
                break
            if message is not _IDLE:
                deadline = self._clock() + self.config.lifeline_timeout_s
                self._dispatch(message)                  # type: ignore[arg-type]
                continue
            if self._clock() >= deadline:                # 経路 3: 拍の途絶
                self._abort.set()
                self._send(
                    protocol.error(
                        protocol.CODE_LIFELINE,
                        f"{self.config.lifeline_timeout_s:.1f} 秒 何も届かないので中止しました",
                    )
                )
                break
        thread.join(timeout=5.0)
```

**`_dispatch` は打鍵中に `stop` を受けると `self._abort.set()` する** (Task 5 で実装済み。経路 1)。

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_key_server.py -v`
Expected: 33 passed

- [ ] **Step 5: コミット**

```bash
git add src/tx/key_server.py tests/test_tx_key_server.py
git commit -m "feat: 中止の 3 経路を打鍵側に入れる

stop・切断・拍の途絶をすべて同じ中止に合流させる。拍が要るのは
ケーブルが抜けても TCP が黙ったままだから。待機中は期限を掛けない。"
```

---

## Task 7: 打鍵 CLI

**Files:**
- Create: `scripts/cw_key_server.py`
- Test: `tests/test_cw_key_server_cli.py`

**Interfaces:**
- Consumes: Task 3〜6 の `ServerConfig` / `KeySession` / `open_sink` / `prepare`、`src.tx.protocol.DEFAULT_KEY_PORT`
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `config_from_args(args) -> ServerConfig`
  - `send_once(config: ServerConfig, text: str, wpm: float) -> int` (終了コード)
  - `check_lines(config: ServerConfig, cycles: int | None = None) -> int`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_cw_key_server_cli.py`:

```python
"""打鍵 CLI のテスト.

**実機は使わない。** ``--dry-run`` と偽の窓口で代える。
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path

import pytest

from src.tx.key_server import ServerConfig
from src.tx.keyer import RecordingKeySink

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cw_key_server.py"
_spec = importlib.util.spec_from_file_location("cw_key_server", _SCRIPT)
assert _spec and _spec.loader
cli = importlib.util.module_from_spec(_spec)
sys.modules["cw_key_server"] = cli
_spec.loader.exec_module(cli)


def test_dry_runは結線を要らない() -> None:
    args = cli.build_parser().parse_args(["--dry-run"])
    config = cli.config_from_args(args)
    assert config.dry_run is True
    assert config.serial is None


def test_ポートを渡すと結線ができる() -> None:
    args = cli.build_parser().parse_args(
        ["--port", "COM3", "--key-line", "RTS", "--ptt-line", "NONE", "--key-invert"]
    )
    config = cli.config_from_args(args)
    assert config.serial is not None
    assert config.serial.port == "COM3"
    assert config.serial.key_line == "RTS"
    assert config.serial.ptt_line == "NONE"
    assert config.serial.key_invert is True


def test_ポートもdry_runも無ければ止める() -> None:
    assert cli.main(["--listen", "0"]) == 1


def test_ミリ秒を秒に直す() -> None:
    args = cli.build_parser().parse_args(["--dry-run", "--ptt-lead-ms", "80", "--ptt-tail-ms", "200"])
    config = cli.config_from_args(args)
    assert config.ptt_lead_s == pytest.approx(0.08)
    assert config.ptt_tail_s == pytest.approx(0.20)


def test_単発送信が打鍵して0で終わる(monkeypatch) -> None:
    sink = RecordingKeySink()
    monkeypatch.setattr(cli, "open_sink", lambda config: contextlib.nullcontext(sink))
    assert cli.send_once(ServerConfig(), "E E", 40.0) == 0
    assert sink.ended_safely()
    assert sink.key_states                                   # 実際に叩いた


def test_単発送信は送れない文字で打鍵しない(monkeypatch, capsys) -> None:
    """**打たずに止める。** 落として送ると相手に意味の通らない符号が届く."""
    opened: list[str] = []

    def spy(config):
        opened.append("開いた")
        raise AssertionError("送れない文字なのにポートを開いた")

    monkeypatch.setattr(cli, "open_sink", spy)
    assert cli.send_once(ServerConfig(), "CQ 髙 K", 20.0) == 1
    assert opened == []
    assert "髙" in capsys.readouterr().out


def test_結線確認が両線を上下させて最後に落とす(monkeypatch) -> None:
    sink = RecordingKeySink()
    monkeypatch.setattr(cli, "open_sink", lambda config: contextlib.nullcontext(sink))
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    assert cli.check_lines(ServerConfig(), cycles=2) == 0
    assert True in sink.key_states                           # 上げた
    assert sink.ended_safely()                               # 落として終わった
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_cw_key_server_cli.py -v`
Expected: FAIL — `FileNotFoundError` (`scripts/cw_key_server.py` が無い)

- [ ] **Step 3: 実装する**

`scripts/cw_key_server.py`:

```python
"""無線機を繋いだ PC で動かし、LAN 越しに受け取ったテキストを打鍵する.

アプリ (GPU PC) と無線機が別の PC にあるときに使う。**この PC には COM ポートが
あり、GPU PC には無い。**

使い方 (無線機を繋いだ PC で)::

    python scripts/cw_key_server.py --port COM3
    python scripts/cw_key_server.py --port COM3 --ptt-line NONE --key-invert
    python scripts/cw_key_server.py --dry-run                    # シリアルに触らない
    python scripts/cw_key_server.py --port COM3 --check-lines    # 結線の確認
    python scripts/cw_key_server.py --port COM3 --text "CQ DE JA1ABC K"

GPU 側のアプリでは、設定の「打鍵側」にこの PC の ``host:port`` を入れる。

**初めて実機に繋ぐときは、必ず無線機の電源を切るかダミーロードで。**
多くの USB シリアル変換器はポートを開いた瞬間に DTR/RTS を H にする。
``--check-lines`` でテスターか LED を当てて確かめること。

暗号化しない。家庭内 LAN を前提とし、外部へ公開するポートには割り当てないこと
(``--bind`` で待ち受けるアドレスを絞れる)。
"""
from __future__ import annotations

import argparse
import contextlib
import socket
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tx.key_server import (  # noqa: E402
    KeySession,
    RequestRejected,
    ServerConfig,
    open_sink,
    prepare,
)
from src.tx.keyer import Keyer  # noqa: E402
from src.tx.protocol import (  # noqa: E402
    CODE_BUSY,
    DEFAULT_KEY_PORT,
    encode_message,
    error,
)
from src.tx.serial_key import SerialKeyConfig, SerialKeyError  # noqa: E402

# 結線確認モードで線を上げている時間 (秒)。テスターの針が振れる程度に長く取る。
CHECK_LINE_HOLD_S = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LAN 越しに受け取ったテキストを打鍵する (無線機を繋いだ PC で動かす)"
    )
    parser.add_argument("--port", default=None, help="COM ポート名 (--dry-run 時は不要)")
    parser.add_argument("--key-line", default="DTR", choices=["DTR", "RTS"], help="電鍵の線")
    parser.add_argument("--ptt-line", default="RTS", choices=["DTR", "RTS", "NONE"], help="PTT の線")
    parser.add_argument("--key-invert", action="store_true", help="電鍵の極性を反転する")
    parser.add_argument("--ptt-invert", action="store_true", help="PTT の極性を反転する")
    parser.add_argument("--ptt-lead-ms", type=float, default=50.0, help="PTT 先行 (ミリ秒)")
    parser.add_argument("--ptt-tail-ms", type=float, default=100.0, help="PTT 後追い (ミリ秒)")
    parser.add_argument("--key-watchdog-s", type=float, default=5.0, help="キー ON が続いてよい上限 (秒)")
    parser.add_argument("--max-duration-s", type=float, default=120.0, help="1 回の送信の長さの上限 (秒)")
    parser.add_argument("--lifeline-timeout", type=float, default=1.0, help="拍が途絶えたとみなす時間 (秒)")
    parser.add_argument("--listen", type=int, default=DEFAULT_KEY_PORT, help=f"待ち受けポート (既定 {DEFAULT_KEY_PORT})")
    parser.add_argument("--bind", default="0.0.0.0", help="待ち受けアドレス (既定 0.0.0.0)")
    parser.add_argument("--dry-run", action="store_true", help="シリアルに一切触らない")
    parser.add_argument("--check-lines", action="store_true", help="結線確認モード (無線機の電源を切って使う)")
    parser.add_argument("--text", default=None, help="単発送信モード。LAN を待たない")
    parser.add_argument("--wpm", type=float, default=20.0, help="--text 使用時の速度")
    return parser


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    """引数を設定に直す. **結線は LAN で流さないのでここだけで決まる。**"""
    serial = None
    if not args.dry_run:
        serial = SerialKeyConfig(
            port=args.port,
            key_line=args.key_line,
            ptt_line=args.ptt_line,
            key_invert=args.key_invert,
            ptt_invert=args.ptt_invert,
        )
        serial.validate()
    return ServerConfig(
        serial=serial,
        ptt_lead_s=args.ptt_lead_ms / 1000.0,
        ptt_tail_s=args.ptt_tail_ms / 1000.0,
        key_watchdog_s=args.key_watchdog_s,
        max_duration_s=args.max_duration_s,
        lifeline_timeout_s=args.lifeline_timeout,
    )


def send_once(config: ServerConfig, text: str, wpm: float) -> int:
    """単発送信モード. LAN を待たず 1 回打って終わる.

    **カナ変換は無い。** カタカナと欧文で書くこと (``{HORE}コンニチハ{RATA}``)。
    中止は Ctrl+C。``Keyer`` の ``finally`` が両線を落とす。
    """
    try:
        sequence = prepare(text, wpm, config.max_duration_s)
    except RequestRejected as exc:
        print(f"エラー: {exc.message}", file=sys.stderr)
        for bad in exc.extra.get("unsendable", []):
            print(f"  {bad['index']} 文字目: {bad['char']!r}")
        return 1

    print(f"送信 : {text}")
    print(f"速度 : {wpm} WPM   所要 {sequence.total_seconds:.1f} 秒")
    keyer = Keyer(key_watchdog_s=config.key_watchdog_s)
    try:
        with open_sink(config) as sink:
            report = keyer.send(
                sequence.durations,
                sequence.is_on,
                sink,
                ptt_lead_s=config.ptt_lead_s,
                ptt_tail_s=config.ptt_tail_s,
                use_ptt=config.serial is None or config.serial.uses_ptt,
            )
    except SerialKeyError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中止しました。")
        return 1
    print(f"完了 : {report.elements_sent} 要素   ずれ 最大 {report.max_error_ms:.1f} ms / 平均 {report.mean_error_ms:.1f} ms")
    return 0


def check_lines(config: ServerConfig, cycles: int | None = None) -> int:
    """結線確認モード. 電鍵線と PTT 線をゆっくり交互に上下させる.

    **必ず無線機の電源を切るかダミーロードで使うこと。**
    テスターか LED を当てて、線の割り当てと極性を確かめる。

    Args:
        cycles: 繰り返す回数。``None`` なら Ctrl+C まで。
    """
    print("結線確認モード — **無線機の電源が切れていることを確かめてください。**")
    print("電鍵線と PTT 線を 1 秒ずつ交互に上げます。Ctrl+C で終了。\n")
    count = 0
    try:
        with open_sink(config) as sink:
            while cycles is None or count < cycles:
                count += 1
                print(f"[{count}] 電鍵線 ON")
                sink.key(True)
                time.sleep(CHECK_LINE_HOLD_S)
                print(f"[{count}] 電鍵線 OFF")
                sink.key(False)
                time.sleep(CHECK_LINE_HOLD_S)
                print(f"[{count}] PTT 線 ON")
                sink.ptt(True)
                time.sleep(CHECK_LINE_HOLD_S)
                print(f"[{count}] PTT 線 OFF")
                sink.ptt(False)
                time.sleep(CHECK_LINE_HOLD_S)
    except SerialKeyError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n終了します。")
    return 0


def local_addresses() -> list[str]:
    """アプリ側に入れてもらう IP の候補を集める."""
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        return []
    return [a for a in addrs if not a.startswith("127.")]


def serve(config: ServerConfig, server: socket.socket) -> None:
    """接続を受け付ける.

    **同時に扱うのは 1 つ。** 二重打鍵しないため、先に繋いだ側を優先し、
    2 つ目には ``busy`` を返して閉じる。**待たせない** — アプリ側は繋がったか
    どうかをすぐ知る必要がある (画面のボタンの有効・無効がそれで決まる)。
    """
    server.settimeout(1.0)
    active: threading.Thread | None = None
    while True:
        try:
            conn, peer = server.accept()
        except TimeoutError:
            continue

        if active is not None and active.is_alive():
            print(f"拒否: {peer[0]}:{peer[1]} (既に別のアプリが繋がっています)")
            with contextlib.suppress(OSError):
                conn.sendall(
                    encode_message(error(CODE_BUSY, "別のアプリが繋がっています"))
                )
                conn.close()
            continue

        print(f"\n接続: {peer[0]}:{peer[1]}")
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        def run_session(conn: socket.socket = conn) -> None:
            try:
                KeySession(config, conn).run()
            except Exception as exc:                      # noqa: BLE001
                print(f"セッションが落ちました: {exc}", file=sys.stderr)
            finally:
                with contextlib.suppress(OSError):
                    conn.close()
                print("切断されました。次の接続を待ちます。")

        active = threading.Thread(target=run_session, name="cw-session", daemon=True)
        active.start()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dry_run and not args.port:
        print(
            "エラー: --port が要ります (COM ポート名)。\n"
            "シリアルに触らずに試すなら --dry-run を付けてください。",
            file=sys.stderr,
        )
        return 1

    try:
        config = config_from_args(args)
    except SerialKeyError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.check_lines:
        if config.dry_run:
            print("エラー: --check-lines には --port が要ります。", file=sys.stderr)
            return 1
        return check_lines(config)

    if args.text is not None:
        return send_once(config, args.text, args.wpm)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((args.bind, args.listen))
        server.listen(1)
    except OSError as exc:
        print(f"エラー: ポートを開けません ({args.bind}:{args.listen}): {exc}", file=sys.stderr)
        return 1

    wiring = config.wiring()
    print(f"結線     : {wiring['port'] or '(dry-run)'}  電鍵 {wiring['key'] or '-'} / PTT {wiring['ptt'] or '-'}")
    print(f"待ち受け : {args.bind}:{args.listen}")
    for addr in local_addresses():
        print(f"  アプリの設定に:  {addr}:{args.listen}")
    print("\n接続を待っています… (Ctrl+C で終了)")

    try:
        serve(config, server)
    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_cw_key_server_cli.py -v`
Expected: 7 passed

- [ ] **Step 5: ruff を通す**

Run: `python -m ruff check scripts/cw_key_server.py src/tx/`
Expected: `All checks passed!` (未使用 import が残っていれば消す)

- [ ] **Step 6: コミット**

```bash
git add scripts/cw_key_server.py tests/test_cw_key_server_cli.py
git commit -m "feat: 打鍵 CLI を追加する

--dry-run で COM ポート無しでも全経路を通せる。--check-lines は
テスターで結線と極性を確かめる用。--text は GPU PC が落ちていても
無線機 PC だけで打てる経路。"
```

---

## Task 8: アプリ側クライアント

**Files:**
- Create: `src/tx/net_key.py`
- Test: `tests/test_tx_net_key.py`

**Interfaces:**
- Consumes: Task 1 の `tokens_fingerprint`、Task 2 の `protocol` 一式、`src.infer.net_audio.parse_endpoint`
- Produces:
  - `NetKeyError(RuntimeError)` / `NetKeyRejected(NetKeyError)` (属性 `code: str` / `unsendable: list[dict]`)
  - `Hello` (`@dataclass(frozen=True)`): `protocol_version: int` / `tokens_fingerprint: str` / `wiring: dict[str, str]` / `dry_run: bool` / プロパティ `fingerprint_matches: bool`
  - `CheckResult` (`@dataclass(frozen=True)`): `chars: int` / `elements: int` / `seconds: float`
  - `SendResult` (`@dataclass(frozen=True)`): `elements_sent: int` / `aborted: bool` / `watchdog_tripped: bool` / `seconds: float` / `max_error_ms: float` / `mean_error_ms: float`
  - `NetKeyClient`: `connect() -> Hello` / `close()` / `check(text, wpm) -> CheckResult` / `send(text, wpm) -> SendResult` / `stop()` / プロパティ `is_connected: bool`
  - `PING_INTERVAL_S: float = 0.25`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_net_key.py`:

```python
"""アプリ側クライアントのテスト.

**本物の打鍵サーバを ``--dry-run`` 相当で起こして繋ぐ。** 偽物にすると
拍や切断の扱いを取り逃がす。
"""
from __future__ import annotations

import socket
import threading

import pytest

from src.tx.key_server import KeySession, ServerConfig
from src.tx.net_key import NetKeyClient, NetKeyError, NetKeyRejected


def start_server(config: ServerConfig | None = None) -> tuple[str, int, socket.socket]:
    """待ち受けを起こし、(host, port, server) を返す."""
    config = config or ServerConfig()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def accept_loop() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            threading.Thread(
                target=KeySession(config, conn).run, daemon=True
            ).start()

    threading.Thread(target=accept_loop, daemon=True).start()
    return host, port, server


def test_繋ぐと挨拶が読める() -> None:
    host, port, server = start_server()
    client = NetKeyClient(host, port)
    try:
        hello = client.connect()
        assert hello.dry_run is True
        assert hello.fingerprint_matches is True
        assert client.is_connected is True
    finally:
        client.close()
        server.close()


def test_打鍵側がいなければ繋がらない() -> None:
    client = NetKeyClient("127.0.0.1", 1)          # 誰もいないポート
    with pytest.raises(NetKeyError):
        client.connect()


def test_checkが秒数を返す() -> None:
    host, port, server = start_server()
    client = NetKeyClient(host, port)
    try:
        client.connect()
        result = client.check("CQ DE JA1ABC K", 20.0)
        assert result.chars == 14
        assert result.seconds > 0.0
    finally:
        client.close()
        server.close()


def test_checkが送れない文字を位置つきで知らせる() -> None:
    host, port, server = start_server()
    client = NetKeyClient(host, port)
    try:
        client.connect()
        with pytest.raises(NetKeyRejected) as caught:
            client.check("CQ 髙 K", 20.0)
        assert caught.value.code == "unsendable"
        assert caught.value.unsendable == [{"index": 3, "char": "髙"}]
    finally:
        client.close()
        server.close()


def test_sendが完了を返す() -> None:
    host, port, server = start_server()
    client = NetKeyClient(host, port)
    try:
        client.connect()
        result = client.send("E E", 40.0)
        assert result.aborted is False
        assert result.elements_sent > 0
    finally:
        client.close()
        server.close()


def test_長い送信でも拍で切られない() -> None:
    """**拍を打ち続けること。** 打たないと打鍵側が心綱で止める."""
    host, port, server = start_server(ServerConfig(lifeline_timeout_s=0.3))
    client = NetKeyClient(host, port)
    try:
        client.connect()
        result = client.send("CQ CQ DE JA1ABC K", 15.0)
        assert result.aborted is False
    finally:
        client.close()
        server.close()


def test_stopで途中で止まる() -> None:
    host, port, server = start_server()
    client = NetKeyClient(host, port)
    try:
        client.connect()
        stopper = threading.Timer(0.4, client.stop)
        stopper.start()
        result = client.send("CQ CQ CQ DE JA1ABC JA1ABC PSE K", 12.0)
        stopper.cancel()
        assert result.aborted is True
    finally:
        client.close()
        server.close()


def test_指紋が違えば分かる(monkeypatch) -> None:
    from src.tokens.morse_tokens import EUROPEAN_CHAR_TO_CODE

    host, port, server = start_server()
    client = NetKeyClient(host, port)
    try:
        hello = client.connect()
        assert hello.fingerprint_matches is True
        # 打鍵側と話したあとで自分の表を変える = リポジトリがずれた状態
        monkeypatch.setitem(EUROPEAN_CHAR_TO_CODE, "A", "----")
        assert hello.fingerprint_matches is False
    finally:
        client.close()
        server.close()
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_net_key.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tx.net_key'`

- [ ] **Step 3: 実装する**

`src/tx/net_key.py`:

```python
"""アプリ側 (GPU PC) から打鍵側 (無線機 PC) を叩くクライアント.

**この PC には COM ポートが無い。** 打鍵は無線機を繋いだ PC で行う。
ここは確定したテキストを渡し、拍を打ち、``stop`` を送るだけ。

打鍵側は :file:`scripts/cw_key_server.py`。

繋ぎ直しの方針
--------------
**待機中は呼び出し側が繋ぎ直してよい。** 打鍵側を後から起こしても繋がる。

**送信中は繋ぎ直さない。** 切れたら中止として扱う。繋ぎ直して続きから打つような
ことはしない — 何がどこまで出たのか分からない状態で電波を出さないため。
"""
from __future__ import annotations

import contextlib
import socket
import threading
from dataclasses import dataclass, field
from typing import Any

from src.tx import protocol
from src.tx.fingerprint import tokens_fingerprint

# 心綱の拍を打つ間隔。打鍵側の既定の期限 (1.0 秒) の 1/4。
PING_INTERVAL_S = 0.25


class NetKeyError(RuntimeError):
    """打鍵側に繋がらない、応答が不正、途中で切れた."""


class NetKeyRejected(NetKeyError):
    """打鍵側が要求を撥ねた. **打鍵されていない。**"""

    def __init__(self, code: str, message: str, unsendable: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.unsendable = unsendable or []


@dataclass(frozen=True)
class Hello:
    """打鍵側の名乗り."""

    protocol_version: int
    tokens_fingerprint: str
    wiring: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def fingerprint_matches(self) -> bool:
        """**両 PC の符号表が一致しているか。** 違えば画面で警告する."""
        return self.tokens_fingerprint == tokens_fingerprint()

    def describe_wiring(self) -> str:
        if self.dry_run:
            return "dry-run (シリアルに触らない)"
        port = self.wiring.get("port") or "?"
        return f"{port}  電鍵 {self.wiring.get('key') or '-'} / PTT {self.wiring.get('ptt') or '-'}"


@dataclass(frozen=True)
class CheckResult:
    """``check`` の答え. **まだ何も送っていない。**"""

    chars: int
    elements: int
    seconds: float


@dataclass(frozen=True)
class SendResult:
    """打鍵の実測."""

    elements_sent: int
    aborted: bool
    watchdog_tripped: bool
    seconds: float
    max_error_ms: float
    mean_error_ms: float


class NetKeyClient:
    """打鍵側との 1 本の接続."""

    def __init__(
        self,
        host: str,
        port: int = protocol.DEFAULT_KEY_PORT,
        *,
        connect_timeout_s: float = 5.0,
        ping_interval_s: float = PING_INTERVAL_S,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.ping_interval_s = ping_interval_s
        self._sock: socket.socket | None = None
        self._reader = protocol.LineReader()
        self._pending: list[bytes] = []
        self._write_lock = threading.Lock()

    # ---- ライフサイクル ----
    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> Hello:
        """繋いで名乗りを読む.

        Raises:
            NetKeyError: 繋がらない、名乗りが読めない.
        """
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
        except OSError as exc:
            raise NetKeyError(
                f"打鍵側に繋がりません ({self.host}:{self.port}): {exc}\n"
                "無線機を繋いだ PC で scripts/cw_key_server.py を動かしているか、\n"
                "ファイアウォールがそのポートを通しているかを確認してください。"
            ) from exc
        self._sock = sock
        self._reader = protocol.LineReader()
        self._pending = []
        message = self._recv(timeout=self.connect_timeout_s)
        if message.get("type") != "hello":
            self.close()
            raise NetKeyError(f"名乗りが来ませんでした: {message.get('type')!r}")
        return Hello(
            protocol_version=int(message.get("protocol", 0)),
            tokens_fingerprint=str(message.get("tokens_fingerprint", "")),
            wiring=dict(message.get("wiring") or {}),
            dry_run=bool(message.get("dry_run")),
        )

    def close(self) -> None:
        """閉じる. 二重呼び出しは無害."""
        sock, self._sock = self._sock, None
        if sock is None:
            return
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            sock.close()

    # ---- 操作 ----
    def check(self, text: str, wpm: float) -> CheckResult:
        """**打鍵しない検査。** これが通って初めて送ってよい.

        Raises:
            NetKeyRejected: 打鍵側が撥ねた.
            NetKeyError: 切れた.
        """
        self._send(protocol.check(text, wpm))
        message = self._recv(timeout=self.connect_timeout_s)
        self._raise_if_error(message)
        if message.get("type") != "checked":
            raise NetKeyError(f"想定外の応答です: {message.get('type')!r}")
        return CheckResult(
            chars=int(message["chars"]),
            elements=int(message["elements"]),
            seconds=float(message["seconds"]),
        )

    def send(self, text: str, wpm: float) -> SendResult:
        """打鍵させ、終わるまで待つ. **待つあいだ拍を打ち続ける。**

        Raises:
            NetKeyRejected: 打鍵側が撥ねた (打鍵されていない).
            NetKeyError: 途中で切れた.
        """
        self._send(protocol.send(text, wpm))
        while True:
            message = self._recv(timeout=self.ping_interval_s, on_timeout_ping=True)
            if message is None:
                continue                              # まだ打っている
            self._raise_if_error(message)
            if message.get("type") == "done":
                return SendResult(
                    elements_sent=int(message["elements_sent"]),
                    aborted=bool(message["aborted"]),
                    watchdog_tripped=bool(message["watchdog_tripped"]),
                    seconds=float(message["seconds"]),
                    max_error_ms=float(message["max_error_ms"]),
                    mean_error_ms=float(message["mean_error_ms"]),
                )
            raise NetKeyError(f"想定外の応答です: {message.get('type')!r}")

    def stop(self) -> None:
        """中止させる. **別のスレッドから呼んでよい** (画面の [中止] ボタン)."""
        with contextlib.suppress(NetKeyError):
            self._send(protocol.stop())

    # ---- 下回り ----
    def _send(self, message: dict[str, Any]) -> None:
        sock = self._sock
        if sock is None:
            raise NetKeyError("繋がっていません")
        try:
            with self._write_lock:
                sock.sendall(protocol.encode_message(message))
        except OSError as exc:
            self.close()
            raise NetKeyError(f"打鍵側へ送れません: {exc}") from exc

    def _recv(self, timeout: float, on_timeout_ping: bool = False) -> Any:
        """次の 1 メッセージを読む.

        Args:
            on_timeout_ping: 真なら、待ち時間が尽きたときに拍を打って ``None`` を返す。
                偽なら ``NetKeyError``。

        Raises:
            NetKeyError: 切れた、時間切れ (``on_timeout_ping`` が偽のとき).
        """
        sock = self._sock
        if sock is None:
            raise NetKeyError("繋がっていません")
        while not self._pending:
            sock.settimeout(timeout)
            try:
                data = sock.recv(4096)
            except TimeoutError:
                if on_timeout_ping:
                    self._send(protocol.ping())
                    return None
                raise NetKeyError("打鍵側が応答しません") from None
            except OSError as exc:
                self.close()
                raise NetKeyError(f"打鍵側との接続が切れました: {exc}") from exc
            if not data:
                self.close()
                raise NetKeyError("打鍵側が切断しました")
            try:
                self._pending.extend(self._reader.feed(data))
            except protocol.ProtocolError as exc:
                raise NetKeyError(str(exc)) from exc
        return protocol.decode_message(self._pending.pop(0))

    @staticmethod
    def _raise_if_error(message: dict[str, Any]) -> None:
        if message.get("type") != "error":
            return
        raise NetKeyRejected(
            code=str(message.get("code", "")),
            message=str(message.get("message", "")),
            unsendable=list(message.get("unsendable") or []),
        )


__all__ = [
    "PING_INTERVAL_S",
    "CheckResult",
    "Hello",
    "NetKeyClient",
    "NetKeyError",
    "NetKeyRejected",
    "SendResult",
]
```

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_net_key.py -v`
Expected: 8 passed

- [ ] **Step 5: コミット**

```bash
git add src/tx/net_key.py tests/test_tx_net_key.py
git commit -m "feat: アプリ側の打鍵クライアントを追加する

送信を待つあいだ 0.25 秒ごとに拍を打つ。打たないと打鍵側が心綱で
止める。stop は別スレッドから呼べる (画面の [中止] ボタン用)。"
```

---

## Task 9: 端から端まで

**Files:**
- Create: `tests/test_tx_lan_end_to_end.py`

**Interfaces:**
- Consumes: Task 7 の CLI (`main`)、Task 8 の `NetKeyClient`
- Produces: なし (テストのみ)

- [ ] **Step 1: テストを書く**

`tests/test_tx_lan_end_to_end.py`:

```python
"""端から端まで — 本物の CLI を別プロセスで起こし、本物の TCP で叩く.

**COM ポートは要らない。** ``--dry-run`` があるため、この PC で全経路を通せる。
実機でしか確かめられないのは DTR/RTS の実際の挙動・極性・PTT・実測のずれだけ。
"""
from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.tx.net_key import NetKeyClient, NetKeyError, NetKeyRejected

_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def 打鍵側():
    """CLI を別プロセスで起こす."""
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable, "scripts/cw_key_server.py",
            "--dry-run", "--bind", "127.0.0.1", "--listen", str(port),
            "--lifeline-timeout", "0.5",
        ],
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"CLI が落ちました: {process.stderr.read().decode('utf-8', 'replace')}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        process.kill()
        raise AssertionError("CLI が待ち受けを開きませんでした")
    yield "127.0.0.1", port
    process.terminate()
    process.wait(timeout=10.0)


def test_確認してから送るまで通る(打鍵側) -> None:
    host, port = 打鍵側
    client = NetKeyClient(host, port)
    try:
        hello = client.connect()
        assert hello.dry_run is True
        assert hello.fingerprint_matches is True

        checked = client.check("CQ DE JA1ABC K", 25.0)
        assert checked.seconds > 0.0

        result = client.send("CQ DE JA1ABC K", 25.0)
        assert result.aborted is False
        assert result.watchdog_tripped is False
        assert result.elements_sent > 0
    finally:
        client.close()


def test_和文が通る(打鍵側) -> None:
    """**和文でもコールサインと RST は欧文。** ホレの外は欧文表で符号化される."""
    host, port = 打鍵側
    client = NetKeyClient(host, port)
    try:
        client.connect()
        checked = client.check("JH0ILL DE JA1ABC {HORE}コンニチハ{RATA} K", 25.0)
        assert checked.seconds > 0.0
    finally:
        client.close()


def test_送れない文字が撥ねられる(打鍵側) -> None:
    host, port = 打鍵側
    client = NetKeyClient(host, port)
    try:
        client.connect()
        with pytest.raises(NetKeyRejected) as caught:
            client.check("CQ 髙 K", 25.0)
        assert caught.value.code == "unsendable"
    finally:
        client.close()


def test_拍を止めると打鍵側が中止する(打鍵側) -> None:
    """**心綱。** 拍が来なければ打鍵側が止める (設計書 §5).

    拍の間隔を打鍵側の期限 (0.5 秒) よりずっと長くして、**拍が届かない状態**を
    作る。ケーブルが抜けた状況の模擬。切断はしない。
    """
    host, port = 打鍵側
    client = NetKeyClient(host, port, ping_interval_s=60.0)
    try:
        client.connect()
        result = client.send("CQ CQ CQ DE JA1ABC PSE K", 12.0)
        assert result.aborted is True
    finally:
        client.close()


def test_2つ目の接続はbusyで断られる(打鍵側) -> None:
    """**同時に扱うのは 1 つ。** 二重打鍵しないため (設計書 §7.5)."""
    host, port = 打鍵側
    first = NetKeyClient(host, port)
    first.connect()
    try:
        second = NetKeyClient(host, port)
        with pytest.raises(NetKeyError) as caught:
            second.connect()                  # busy が来て hello ではない
        assert "busy" in str(caught.value) or "繋がって" in str(caught.value)
        second.close()
        assert first.check("E", 40.0).seconds > 0.0     # 先の接続は無事
    finally:
        first.close()


def test_送信中に切ると打鍵側が次の接続を受けられる(打鍵側) -> None:
    """切断で中止し、**サーバは生き残る**."""
    import threading

    host, port = 打鍵側
    first = NetKeyClient(host, port)
    first.connect()

    def 送る() -> None:
        # 途中で切られるので NetKeyError で終わる。それが期待される姿
        with contextlib.suppress(NetKeyError):
            first.send("CQ CQ CQ DE JA1ABC PSE K", 12.0)

    sender = threading.Thread(target=送る, daemon=True)
    sender.start()
    time.sleep(0.5)                       # 少し打たせる
    first.close()                         # ここで切る
    sender.join(timeout=10.0)

    time.sleep(1.5)                       # 中止と後片付けを待つ
    second = NetKeyClient(host, port)
    try:
        hello = second.connect()          # 次の接続を受け付ける
        assert hello.dry_run is True
        assert second.check("E", 40.0).seconds > 0.0
    finally:
        second.close()
```

- [ ] **Step 2: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_lan_end_to_end.py -v`
Expected: 6 passed

**このファイルは別プロセスと実時間を使うので遅い** (30〜60 秒)。落ちたときは
CLI の標準エラーを見ること (`打鍵側` フィクスチャが握っている)。

- [ ] **Step 3: コミット**

```bash
git add tests/test_tx_lan_end_to_end.py
git commit -m "test: LAN 送信を端から端まで通す

CLI を別プロセスで起こし本物の TCP で叩く。--dry-run があるので
COM ポートの無い PC でも全経路を試験できる。"
```

---

## Task 10: 設定

**Files:**
- Modify: `src/infer/settings.py`
- Test: `tests/test_settings.py` (既存に追記。無ければ作る)

**Interfaces:**
- Consumes: `src.tx.protocol.DEFAULT_KEY_PORT`、`src.infer.net_audio.parse_endpoint`
- Produces: `AppSettings.tx_endpoint: str = ""`、`AppSettings.tx_wpm: float = 20.0`、`CURRENT_SETTINGS_VERSION = 15`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_settings.py` に追記 (既存のテストの流儀に合わせる):

```python
def test_送信の設定に既定がある() -> None:
    settings = AppSettings()
    assert settings.tx_endpoint == ""
    assert settings.tx_wpm == 20.0


def test_旧版の設定に送信の既定が入る() -> None:
    """**既にある設定ファイルを壊さない。** 足りない欄は既定で埋まる."""
    old = {"settings_version": 14, "mode": "european"}
    merged, changed = migrate_settings_dict(old)
    assert changed is True
    assert merged["settings_version"] == CURRENT_SETTINGS_VERSION
    assert merged["tx_endpoint"] == ""
    assert merged["tx_wpm"] == 20.0


def test_送信先を書いて読み戻せる(tmp_path) -> None:
    path = tmp_path / "settings.json"
    save_settings(AppSettings(tx_endpoint="192.168.0.10:45679", tx_wpm=22.0), path)
    restored = load_settings(path)
    assert restored.tx_endpoint == "192.168.0.10:45679"
    assert restored.tx_wpm == 22.0
```

`tests/test_settings.py` の import に `CURRENT_SETTINGS_VERSION`, `migrate_settings_dict`,
`save_settings`, `load_settings` が無ければ足す。

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_settings.py -v`
Expected: FAIL — `AttributeError: 'AppSettings' object has no attribute 'tx_endpoint'`

- [ ] **Step 3: 実装する**

`src/infer/settings.py`:

版のコメントに 1 行足し、定数を上げる:

```python
# v15: tx_endpoint / tx_wpm を追加 (LAN 越し送信)。
#      結線 (COM ポート・DTR/RTS の役割・極性・PTT) は**アプリに持たせない**。
#      無線機に属する情報であり、打鍵は無線機 PC の CLI が行うため
#      (docs/superpowers/specs/2026-08-10-lan-cw-transmit-design.md §8.1)
CURRENT_SETTINGS_VERSION = 15
```

`AppSettings` の末尾付近 (`settings_version` の前) に追記:

```python
    # 打鍵側 (無線機を繋いだ PC) の `host:port`。空なら送信機能を出さない。
    # **この PC には COM ポートが無い。** 打鍵は向こうでやる。
    # 解析は net_audio.parse_endpoint を使い回す (既定ポートだけ 45679 を渡す)。
    tx_endpoint: str = ""
    # 送信速度。運用者が手で決める (自動追従はしない)。
    tx_wpm: float = 20.0
```

移行の置換表 (`_V1_DEFAULT_REPLACEMENTS`) は**変更しない**。新しい欄は既定で埋まるため。

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_settings.py -v`
Expected: 追記した 3 件を含めて全 passed

- [ ] **Step 5: 既存のテストを壊していないか確かめる**

Run: `python -m pytest tests/test_settings.py tests/test_ui_smoke.py -v`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **Step 6: コミット**

```bash
git add src/infer/settings.py tests/test_settings.py
git commit -m "feat: 送信先と速度の設定を追加する (スキーマ v15)

結線はアプリに持たせない。無線機に属する情報で、打鍵は無線機 PC の
CLI が行うため。"
```

---

## Task 11: 送信ダイアログ

**Files:**
- Create: `src/app/tx_dialog.py`
- Modify: `src/app/main_window.py` (ボタン 1 つと配線)
- Test: `tests/test_tx_dialog.py`

**Interfaces:**
- Consumes: Task 8 の `NetKeyClient` / `Hello` / `CheckResult` / `SendResult` / `NetKeyError` / `NetKeyRejected`、`src.tx.reading.to_sendable_kana`、`src.tx.encoder.wrap_japanese`、`src.tx.profile.load_profile`、Task 10 の `AppSettings`
- Produces:
  - `TxDialog(QDialog)`: `TxDialog(settings: AppSettings, profile: OperatorProfile | None = None, client_factory=NetKeyClient, parent=None)`
  - 属性 `japanese_edit: QPlainTextEdit` / `kana_view: QPlainTextEdit` / `wrap_check: QCheckBox` / `connect_btn` / `check_btn` / `send_btn` / `stop_btn` / `status_label` / `endpoint_edit` / `wpm_spin`
  - メソッド `wire_text() -> str` / `refresh_kana() -> None` / `can_send() -> bool` /
    `connect_to_keyer(quiet: bool = False) -> None` / `retry_tick() -> None` /
    `run_check() -> None` / `run_send() -> None` / `run_stop() -> None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_tx_dialog.py`:

```python
"""送信ダイアログのテスト.

**関門を機械で確かめる。** 「確認するまで [送信] を押せない」は運用者が
決めた安全の要であり、人手の確認に任せない。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from src.infer.settings import AppSettings
from src.tx.net_key import CheckResult, Hello, NetKeyRejected, SendResult


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeClient:
    """打鍵側のふり. **本物のソケットを使わない** (画面の関門だけを見る)."""

    def __init__(self, host: str, port: int = 45679, **kwargs: object) -> None:
        self.host = host
        self.port = port
        self.checked: list[tuple[str, float]] = []
        self.sent: list[tuple[str, float]] = []
        self.stopped = False
        self.reject_with: NetKeyRejected | None = None

    def connect(self) -> Hello:
        from src.tx.fingerprint import tokens_fingerprint

        return Hello(1, tokens_fingerprint(), {"port": "COM3", "key": "DTR", "ptt": "RTS"}, False)

    def close(self) -> None:
        pass

    def check(self, text: str, wpm: float) -> CheckResult:
        if self.reject_with is not None:
            raise self.reject_with
        self.checked.append((text, wpm))
        return CheckResult(chars=len(text), elements=100, seconds=12.3)

    def send(self, text: str, wpm: float) -> SendResult:
        self.sent.append((text, wpm))
        return SendResult(100, False, False, 12.3, 1.0, 0.3)

    def stop(self) -> None:
        self.stopped = True


def build(qapp, **overrides):
    from src.app.tx_dialog import TxDialog

    # **`**overrides` を後に置く。** 先に置くと tx_endpoint が二重に渡り
    # TypeError になる
    settings = AppSettings(**{"tx_endpoint": "127.0.0.1:45679", "tx_wpm": 20.0, **overrides})
    clients: list[FakeClient] = []

    def factory(host, port=45679, **kwargs):
        client = FakeClient(host, port, **kwargs)
        clients.append(client)
        return client

    dialog = TxDialog(settings, profile=None, client_factory=factory)
    return dialog, clients


def test_日本語がカタカナに直る(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    assert "コンニチハ" in dialog.kana_view.toPlainText()


def test_和文がホレとラタで囲まれる(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.wrap_check.setChecked(True)
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    text = dialog.wire_text()
    assert text.startswith("{HORE}")
    assert text.endswith("{RATA}")


def test_確認する前は送信を押せない(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    assert dialog.can_send() is False
    assert dialog.send_btn.isEnabled() is False


def test_確認すると送信を押せる(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    assert clients[0].checked
    assert dialog.can_send() is True
    assert dialog.send_btn.isEnabled() is True


def test_編集すると送信が無効に戻る(qapp) -> None:
    """**確認していない文字列は送れない。**"""
    dialog, _ = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    assert dialog.can_send() is True

    dialog.japanese_edit.setPlainText("こんばんは")
    dialog.refresh_kana()
    assert dialog.can_send() is False
    assert dialog.send_btn.isEnabled() is False


def test_ホレラタの切替も確認を無効にする(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    dialog.wrap_check.setChecked(not dialog.wrap_check.isChecked())
    dialog.refresh_kana()
    assert dialog.can_send() is False


def test_送れない文字があると送信できない(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    clients[0].reject_with = NetKeyRejected(
        "unsendable", "送信できない文字", [{"index": 0, "char": "髙"}]
    )
    dialog.japanese_edit.setPlainText("髙")
    dialog.refresh_kana()
    dialog.run_check()
    assert dialog.can_send() is False
    assert "髙" in dialog.status_label.text()


def test_未接続なら確認も送信も押せない(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    assert dialog.check_btn.isEnabled() is False
    assert dialog.send_btn.isEnabled() is False


def test_送信先が空なら繋ぎに行かない(qapp) -> None:
    dialog, clients = build(qapp, tx_endpoint="")
    dialog.connect_to_keyer()
    assert clients == []
    assert dialog.check_btn.isEnabled() is False


def test_待機中は自動で繋ぎ直す(qapp) -> None:
    """打鍵側を後から起こしても繋がる (設計書 §8.3)."""
    dialog, clients = build(qapp)
    assert clients == []
    dialog.retry_tick()
    assert len(clients) == 1


def test_繋がっていれば繋ぎ直さない(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    assert len(clients) == 1
    dialog.retry_tick()
    assert len(clients) == 1


def test_送信先が空なら自動でも繋ぎに行かない(qapp) -> None:
    dialog, clients = build(qapp, tx_endpoint="")
    dialog.retry_tick()
    assert clients == []
```

- [ ] **Step 2: 失敗を確かめる**

Run: `python -m pytest tests/test_tx_dialog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app.tx_dialog'`

- [ ] **Step 3: 実装する**

`src/app/tx_dialog.py`:

```python
"""送信ダイアログ.

**この PC には COM ポートが無い。** 打鍵は無線機を繋いだ PC の CLI が行い、
ここは確定したテキストを渡すだけ (``src/tx/net_key.py``)。

関門
----
**確認するまで [送信] は押せない。** ``check`` を打鍵側へ投げ、打鍵せずに
符号化だけさせる。それが返って初めて次の 3 つが確定する:

1. 打鍵側が生きていて繋がっている
2. そのテキストが**打鍵側の符号表**で通る
3. 何秒間 電波が出るのか

**編集したら [送信] は無効に戻る。** 確認していない文字列は送れない。

試聴は作らない (設計書 §6)。運用者は無線機の前に座っており、実際の送信は
無線機自身のサイドトーンで聞こえる。
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from src.infer.net_audio import parse_endpoint
from src.infer.settings import AppSettings
from src.tx.encoder import wrap_japanese
from src.tx.net_key import (
    NetKeyClient,
    NetKeyError,
    NetKeyRejected,
    SendResult,
)
from src.tx.profile import OperatorProfile, load_profile
from src.tx.protocol import DEFAULT_KEY_PORT
from src.tx.reading import to_sendable_kana

# 待機中に繋ぎ直す間隔 (秒)。打鍵側を後から起こしても繋がるようにするため。
RETRY_INTERVAL_S = 3.0


class _SendWorker(QThread):
    """打鍵を待つあいだ画面を固めないためのスレッド.

    ``NetKeyClient.send`` は完了まで戻らず、そのあいだ拍を打ち続ける。
    """

    finished_ok = Signal(object)      # SendResult
    failed = Signal(str)

    def __init__(self, client: NetKeyClient, text: str, wpm: float) -> None:
        super().__init__()
        self._client = client
        self._text = text
        self._wpm = wpm

    def run(self) -> None:
        try:
            result = self._client.send(self._text, self._wpm)
        except NetKeyError as exc:
            self.failed.emit(str(exc))
        else:
            self.finished_ok.emit(result)


class TxDialog(QDialog):
    """送信ダイアログ."""

    def __init__(
        self,
        settings: AppSettings,
        profile: OperatorProfile | None = None,
        client_factory: Callable[..., NetKeyClient] = NetKeyClient,
        parent=None,
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

        self._build_ui()
        self._update_buttons()

        # **待機中は自動で繋ぎ直す** (設計書 §8.3)
        self._retry_timer = QTimer(self)
        self._retry_timer.setInterval(int(RETRY_INTERVAL_S * 1000))
        self._retry_timer.timeout.connect(self.retry_tick)
        self._retry_timer.start()

    # ---- 画面 ----
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("打鍵側:"))
        self.endpoint_edit = QLineEdit(self._settings.tx_endpoint)
        self.endpoint_edit.setPlaceholderText("192.168.0.10:45679")
        top.addWidget(self.endpoint_edit, 1)
        self.connect_btn = QPushButton("接続")
        # **``clicked`` は ``checked: bool`` を渡す。** そのまま繋ぐと第 1 引数の
        # ``quiet`` に入る。引数の食い違いは過去にこのリポジトリで実際に踏んでいる
        self.connect_btn.clicked.connect(lambda: self.connect_to_keyer())
        top.addWidget(self.connect_btn)
        top.addWidget(QLabel("速度:"))
        self.wpm_spin = QDoubleSpinBox()
        self.wpm_spin.setRange(5.0, 40.0)
        self.wpm_spin.setSingleStep(1.0)
        self.wpm_spin.setValue(self._settings.tx_wpm)
        self.wpm_spin.setSuffix(" WPM")
        top.addWidget(self.wpm_spin)
        layout.addLayout(top)

        layout.addWidget(QLabel("日本語 (漢字かな交じりで可):"))
        self.japanese_edit = QPlainTextEdit()
        self.japanese_edit.textChanged.connect(self.refresh_kana)
        layout.addWidget(self.japanese_edit)

        self.wrap_check = QCheckBox("和文をホレ/ラタで囲む")
        self.wrap_check.setChecked(True)
        self.wrap_check.toggled.connect(self.refresh_kana)
        layout.addWidget(self.wrap_check)

        layout.addWidget(QLabel("送信される文字:"))
        self.kana_view = QPlainTextEdit()
        self.kana_view.setReadOnly(True)
        layout.addWidget(self.kana_view)

        self.status_label = QLabel("未接続")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.check_btn = QPushButton("確認")
        self.check_btn.clicked.connect(self.run_check)
        buttons.addWidget(self.check_btn)
        buttons.addStretch(1)
        self.send_btn = QPushButton("送信")
        self.send_btn.clicked.connect(self.run_send)
        buttons.addWidget(self.send_btn)
        self.stop_btn = QPushButton("中止")
        self.stop_btn.clicked.connect(self.run_stop)
        buttons.addWidget(self.stop_btn)
        layout.addLayout(buttons)

    # ---- 文字 ----
    def wire_text(self) -> str:
        """LAN に流す確定テキスト."""
        return self.kana_view.toPlainText().strip()

    def refresh_kana(self) -> None:
        """日本語をカタカナに直し、**関門を閉じ直す**."""
        source = self.japanese_edit.toPlainText()
        result = to_sendable_kana(source, self._profile)
        text = wrap_japanese(result.text) if self.wrap_check.isChecked() else result.text
        self.kana_view.setPlainText(text)
        self._confirmed_text = None            # 編集したら確認をやり直す
        if result.bad_chars:
            chars = "".join(b.char for b in result.bad_chars)
            self.status_label.setText(f"送信できない文字があります: {chars}")
        self._update_buttons()

    def can_send(self) -> bool:
        """**確認が通った文字列と今の文字列が一致しているか。**"""
        text = self.wire_text()
        return bool(text) and self._confirmed_text == text and self._client is not None

    # ---- 操作 ----
    def retry_tick(self) -> None:
        """**待機中は自動で繋ぎ直す** (設計書 §8.3).

        打鍵側を後から起こしても繋がる。**送信中は繋ぎ直さない** — 何がどこまで
        出たのか分からない状態で電波を出さないため。
        """
        if self._client is not None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        if not self.endpoint_edit.text().strip():
            return
        self.connect_to_keyer(quiet=True)

    def connect_to_keyer(self, quiet: bool = False) -> None:
        """打鍵側へ繋ぐ.

        Args:
            quiet: 真なら失敗しても画面に書かない (自動の繋ぎ直しから呼ぶため。
                3 秒おきに赤い文字が書き換わると読めない)。
        """
        endpoint = self.endpoint_edit.text().strip()
        if not endpoint:
            if not quiet:
                self.status_label.setText("打鍵側の host:port を入れてください。")
            self._update_buttons()
            return
        try:
            host, port = parse_endpoint(endpoint, default_port=DEFAULT_KEY_PORT)
        except ValueError as exc:
            if not quiet:
                self.status_label.setText(f"打鍵側の指定が読めません: {exc}")
            self._update_buttons()
            return

        client = self._client_factory(host, port)
        try:
            hello = client.connect()
        except NetKeyError as exc:
            self._client = None
            if not quiet:
                self.status_label.setText(str(exc))
            self._update_buttons()
            return

        self._client = client
        self._settings.tx_endpoint = endpoint
        message = f"接続しました — {hello.describe_wiring()}"
        if not hello.fingerprint_matches:
            # **静かな食い違いを見える警告にする** (設計書 §2.1)
            message += "\n**警告: 符号表が両 PC で違います。** リポジトリを揃えてください。"
        self.status_label.setText(message)
        self._confirmed_text = None
        self._update_buttons()

    def run_check(self) -> None:
        """**打鍵しない検査。** これが通って初めて送れる."""
        if self._client is None:
            return
        text = self.wire_text()
        if not text:
            return
        try:
            result = self._client.check(text, self.wpm_spin.value())
        except NetKeyRejected as exc:
            self._confirmed_text = None
            detail = "".join(bad["char"] for bad in exc.unsendable)
            self.status_label.setText(f"{exc}: {detail}" if detail else str(exc))
        except NetKeyError as exc:
            self._client = None
            self._confirmed_text = None
            self.status_label.setText(str(exc))
        else:
            self._confirmed_text = text
            self.status_label.setText(
                f"確認しました — {result.chars} 文字 / {result.elements} 要素 / {result.seconds:.1f} 秒"
            )
        self._update_buttons()

    def run_send(self) -> None:
        if not self.can_send() or self._client is None:
            return
        self.status_label.setText("送信中…")
        self._worker = _SendWorker(self._client, self.wire_text(), self.wpm_spin.value())
        self._worker.finished_ok.connect(self._on_sent)
        self._worker.failed.connect(self._on_send_failed)
        self._worker.start()
        self._update_buttons()

    def run_stop(self) -> None:
        if self._client is not None:
            self._client.stop()

    def _on_sent(self, result: SendResult) -> None:
        self._worker = None
        # **送ったものは確認済みでなくなる。** 同じ文を続けて送るときも確認から
        self._confirmed_text = None
        if result.aborted:
            self.status_label.setText(f"中止しました ({result.elements_sent} 要素まで送信)")
        else:
            self.status_label.setText(
                f"送信しました — {result.elements_sent} 要素 / {result.seconds:.1f} 秒 / "
                f"ずれ 最大 {result.max_error_ms:.1f} ms"
            )
        self._update_buttons()

    def _on_send_failed(self, message: str) -> None:
        self._worker = None
        self._client = None
        self._confirmed_text = None
        self.status_label.setText(message)
        self._update_buttons()

    # ---- 有効・無効 ----
    def _update_buttons(self) -> None:
        sending = self._worker is not None and self._worker.isRunning()
        connected = self._client is not None
        self.connect_btn.setEnabled(not sending)
        self.check_btn.setEnabled(connected and not sending and bool(self.wire_text()))
        self.send_btn.setEnabled(self.can_send() and not sending)
        self.stop_btn.setEnabled(sending)

    def closeEvent(self, event) -> None:                       # noqa: N802
        self._retry_timer.stop()
        if self._client is not None:
            self._client.close()
            self._client = None
        super().closeEvent(event)


__all__ = ["TxDialog"]
```

- [ ] **Step 4: 通ることを確かめる**

Run: `python -m pytest tests/test_tx_dialog.py -v`
Expected: 13 passed

- [ ] **Step 5: メインウィンドウにボタンを足す**

`src/app/main_window.py:167` (`second.addWidget(self.record_btn)`) の直後に足す:

```python
        self.tx_btn = QPushButton("送信…")
        self.tx_btn.setToolTip("無線機を繋いだ PC に打鍵させます")
        self.tx_btn.clicked.connect(self._open_tx_dialog)
        second.addWidget(self.tx_btn)
```

メソッドを足す (`_save_settings` の隣、`closeEvent` の前あたり):

```python
    def _open_tx_dialog(self) -> None:
        """送信ダイアログを開く.

        **打鍵はこの PC ではしない。** 無線機を繋いだ PC の CLI が行う
        (この PC には COM ポートが無い)。

        ``TxDialog`` の import を関数の中に置くのは、送信を使わない人に
        ``pykakasi`` の読み込みを負わせないため (``reading.py`` の遅延読み込みと
        同じ理由)。
        """
        from src.app.tx_dialog import TxDialog

        dialog = TxDialog(self._settings, parent=self)
        dialog.exec()
        self._save_settings()          # ダイアログが tx_endpoint を書き換えている
```

**名前は実物に合わせてある**: 設定は `self._settings` (`self.settings` ではない)、
保存は `self._save_settings()` (`main_window.py:780`)、ボタンの並びは `second`。

- [ ] **Step 6: 画面が壊れていないことを確かめる**

Run: `python -m pytest tests/test_ui_smoke.py tests/test_tx_dialog.py -v`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **Step 7: ruff を通す**

Run: `python -m ruff check src/ scripts/ tests/`
Expected: `All checks passed!`

- [ ] **Step 8: 送信まわりを全部走らせる**

Run: `python -m pytest tests/test_tx_fingerprint.py tests/test_tx_protocol.py tests/test_tx_key_server.py tests/test_cw_key_server_cli.py tests/test_tx_net_key.py tests/test_tx_lan_end_to_end.py tests/test_tx_dialog.py tests/test_settings.py -v`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **Step 9: コミット**

```bash
git add src/app/tx_dialog.py src/app/main_window.py tests/test_tx_dialog.py
git commit -m "feat: 送信ダイアログを追加する

確認するまで [送信] を押せない。編集したら無効に戻る。試聴は作らない
(運用者は無線機の前に座っており、実際の送信はサイドトーンで聞こえる)。"
```

---

## Task 12: 取扱説明書

**Files:**
- Modify: `docs/USAGE.md` (「取扱説明書 — cw-decorder」)

**Interfaces:**
- Consumes: Task 7 の CLI、Task 11 のダイアログ
- Produces: なし (文書のみ)

- [ ] **Step 1: 手順を書く**

`docs/USAGE.md` の **§11「受信機が別の PC にあるとき（LAN 経由）」の直後**に
§12 として足す (受信の LAN の話のすぐ次に送信の LAN の話が来る並びにする)。
以降の「補足：起動時のコマンドを…」はそのまま最後に残す。

**実機で最初に確かめることを最初に置く。**

```markdown
## 12. 無線機が別の PC にあるとき（LAN 経由の送信）

無線機を繋いだ PC が打鍵し、GPU PC のアプリから文字を渡す。

### 初めて繋ぐとき (必ず読むこと)

**無線機の電源を切るか、ダミーロードを付けてから始めること。**
多くの USB シリアル変換器は、ポートを開いた瞬間に DTR/RTS を H にする。
上がると、開いただけでキャリアが出る。

1. 無線機 PC で結線を確かめる (テスターか LED を当てる)

       python scripts/cw_key_server.py --port COM3 --check-lines

   電鍵線と PTT 線が 1 秒ずつ交互に上がる。上がらない・逆に動くなら
   `--key-line` `--ptt-line` `--key-invert` `--ptt-invert` で直す。

2. 無線機の電源を入れず、`--dry-run` でアプリから通してみる

       python scripts/cw_key_server.py --dry-run

3. ここまで通ってから、初めて無線機を繋ぐ。

### ふだんの使い方

無線機 PC:

    python scripts/cw_key_server.py --port COM3

GPU PC のアプリ: `[送信…]` → 打鍵側に無線機 PC の `host:port` を入れて `[接続]`
→ 日本語を書く → カタカナを目で確認 → `[確認]` → `[送信]`。

**`[確認]` を押すまで `[送信]` は押せない。** 編集したら無効に戻る。

### LAN が切れたら

**打鍵は止まる。** 拍が 1 秒途絶えると打鍵側が中止し、鍵と PTT を落とす。
途中で切れた電文が出ることになるが、制御を失ったまま電波を出し続けるよりよい。

### どうにもならない失敗

* **送信中に USB を抜かれると、ソフトからは線を落とせない。** ポートが消えるため
  番犬も届かない。**無線機側で止めるしかない。**
* **打鍵 CLI 自体が落ちると、線はドライバの既定に戻る。落ちる保証はない。**

### 無線機 PC に要るもの

`pyserial` と `numpy`。`torch` も `pykakasi` も要らない (カナ変換はアプリ側)。

    pip install pyserial numpy

**両 PC のリポジトリを揃えること。** ずれていると接続時に警告が出る。
```

- [ ] **Step 2: コミット**

```bash
git add docs/
git commit -m "docs: LAN 越し送信の手順を取扱説明書に足す

実機に初めて繋ぐ手順を最初に置く。電源を切って結線を確かめてから
dry-run、それから無線機を繋ぐ。"
```

---

## 完了の確認

- [ ] **送信まわりのテストが全部通る**

Run: `python -m pytest tests/ -k "tx or key_server or settings" -v`
Expected: `FAILED` も `ERROR` も出ない

- [ ] **既存のテストを壊していない**

Run: `python -m pytest tests/ -q`
Expected: `FAILED` も `ERROR` も出ない
(**サマリ行は出ないことがある。** 全パス後に `0xC0000409` で落ちる既知の環境問題)

- [ ] **ruff が通る**

Run: `python -m ruff check src/ scripts/ tests/`

- [ ] **既存の 5 モジュールを変更していない**

Run: `git diff main --stat -- src/tx/keyer.py src/tx/serial_key.py src/tx/encoder.py src/tx/reading.py src/tx/profile.py src/tokens/morse_tokens.py`
Expected: 出力なし

## 実機でしか確かめられないこと (人がやる)

**自動化しない。電波が出るため。** 取扱説明書の手順に従う。

1. **ポートを開いた瞬間に DTR/RTS が H にならないか** (`--check-lines`、無線機の電源オフ)
2. 電鍵線・PTT 線の割り当てと極性 (テスターか LED)
3. タイミングの実測ずれ (`[送信]` 後に画面へ出る「ずれ 最大 ○ ms」が短点長に対し十分小さいか。
   25 WPM の短点は 48 ms)
4. PTT の前後余裕が足りているか
5. **心綱の 1.0 秒が妥当か** (有線側で送信が途中で切れることがないか)

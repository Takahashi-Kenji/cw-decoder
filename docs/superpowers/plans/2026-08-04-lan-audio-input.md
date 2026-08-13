# LAN 経由の音声入力 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 無線機を繋いだ PC から 16 kHz モノラルの非圧縮 PCM を LAN 越しに送り、GPU 側の cw-decorder がローカル入力デバイスと同じようにライブデコードできるようにする。

**Architecture:** 送信側 `scripts/audio_send.py` が TCP で待ち受け、12 バイトヘッダ (`V2SA`) の後に int16 LE を流し続ける。受信側 `NetworkAudioCapture` は既存 `AudioCapture` と同じ 5 メンバ (`start` / `stop` / `source_sample_rate` / `drain` / `level_db_rms`) を提供するので、`AudioInferenceWorker` は実デバイスか LAN 経由かを意識しない。16 → 8 kHz の変換は既存の `soxr.ResampleStream` 経路を使う。

**Tech Stack:** Python 3.11 / numpy / soxr / socket + threading / PySide6 (Qt) / pytest

## Global Constraints

- 設計書は `docs/superpowers/specs/2026-08-04-lan-audio-input-design.md`。判断に迷ったらこれが正。
- プロトコルのマジックは `b"V2SA"` から**変更しない**。`voice_to_string` の送信側と相互運用するため。
- ヘッダ形式は `struct` の `"<4sIHH"` (magic 4bytes, sample_rate uint32, channels uint16, bits uint16)。既定ポートは `45678`。
- 送信は 16000 Hz / 1ch / 16bit のみ。受信側はそれ以外を**明示エラーで弾く**（無言で通すとリサンプル比が狂い符号長が全部ずれる）。
- 内部の目標サンプルレートは 8000 Hz。
- リサンプルは `soxr.ResampleStream` を使う。`scipy.signal.resample_poly` はステートレスでブロック境界に歪みが出て CW の dot/dash を破壊するため、ストリーミング経路では使わない（`soxr` 不在環境のフォールバックとしてのみ許容）。
- 言語は日本語（コメント・docstring・コミットメッセージ）。文字コード UTF-8 (BOM なし)、改行 LF。
- 型ヒント必須。不変データは `@dataclass(frozen=True)`。パス操作は `pathlib.Path`。
- テスト実行は `PYTHONIOENCODING=utf-8 python -m pytest`。**この環境では全テストがパスした後に `0xC0000409` (exit 127) で落ちてサマリ行が出ない**。合否は `FAILED` / `ERROR` の有無で判定すること。
- 実ネットワークには出ない。テストは `127.0.0.1` にダミーサーバを立てる。

---

### Task 1: プロトコル補助関数

**Files:**
- Create: `src/infer/net_audio.py`
- Test: `tests/test_net_audio.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `MAGIC: bytes` = `b"V2SA"`
  - `HEADER_FORMAT: str` = `"<4sIHH"`
  - `HEADER_SIZE: int`
  - `DEFAULT_PORT: int` = `45678`
  - `SOURCE_SAMPLE_RATE: int` = `16000`
  - `class NetworkCaptureError(RuntimeError)`
  - `parse_endpoint(text: str, *, default_port: int = DEFAULT_PORT) -> tuple[str, int]`
  - `encode_header(sample_rate: int = SOURCE_SAMPLE_RATE) -> bytes`
  - `encode_samples(block: np.ndarray) -> bytes`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_net_audio.py` を新規作成:

```python
"""LAN 経由の音声入力のテスト.

実ネットワークには出ない。127.0.0.1 にダミーの送信サーバを立てて検証する。
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from src.infer.net_audio import (
    DEFAULT_PORT,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    SOURCE_SAMPLE_RATE,
    encode_header,
    encode_samples,
    parse_endpoint,
)


def test_parse_endpoint_host_only() -> None:
    assert parse_endpoint("192.168.1.20") == ("192.168.1.20", DEFAULT_PORT)


def test_parse_endpoint_host_and_port() -> None:
    assert parse_endpoint("192.168.1.20:45000") == ("192.168.1.20", 45000)


def test_parse_endpoint_hostname() -> None:
    assert parse_endpoint("shack-pc") == ("shack-pc", DEFAULT_PORT)


def test_parse_endpoint_ipv6() -> None:
    """IPv6 は [::1]:45678 形式で書く (: がアドレス内に出るため)."""
    assert parse_endpoint("[::1]:45678") == ("::1", 45678)


def test_parse_endpoint_ipv6_without_port() -> None:
    assert parse_endpoint("[::1]") == ("::1", DEFAULT_PORT)


def test_parse_endpoint_strips_whitespace() -> None:
    assert parse_endpoint("  192.168.1.20  ") == ("192.168.1.20", DEFAULT_PORT)


def test_parse_endpoint_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_endpoint("   ")


def test_parse_endpoint_rejects_non_numeric_port() -> None:
    with pytest.raises(ValueError):
        parse_endpoint("192.168.1.20:abc")


def test_encode_header_roundtrip() -> None:
    magic, rate, channels, bits = struct.unpack(HEADER_FORMAT, encode_header())
    assert magic == MAGIC
    assert rate == SOURCE_SAMPLE_RATE
    assert channels == 1
    assert bits == 16
    assert HEADER_SIZE == 12


def test_encode_samples_to_int16_le() -> None:
    block = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    decoded = np.frombuffer(encode_samples(block), dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == pytest.approx(16383, abs=2)
    assert decoded[2] == pytest.approx(-16383, abs=2)


def test_encode_samples_clips_out_of_range() -> None:
    """範囲外はクリップする (int16 の折り返しを起こさない)."""
    block = np.array([2.0, -2.0], dtype=np.float32)
    decoded = np.frombuffer(encode_samples(block), dtype="<i2")
    assert decoded[0] == 32767
    assert decoded[1] == -32767
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.infer.net_audio'`

- [ ] **Step 3: 最小実装**

`src/infer/net_audio.py` を新規作成:

```python
"""LAN 越しに音声を受け取る擬似入力.

受信機を繋いだ PC と、GPU を積んだ PC が別なときに使う。
:class:`~src.infer.audio.AudioCapture` と同じインターフェースを持つので、
ワーカー側は本物のデバイスか転送されてきた音かを意識しない。

**なぜ必要か**: リモートデスクトップのマイク転送は音声を狭帯域コーデックで
圧縮する。同じリポジトリの ``voice_to_string`` が 2026-08-01 に実測しており、
エネルギーの 99% が 1000 Hz 以下になる。CW のトーンは 486〜496 Hz なので一見
通りそうだが、キーイングの立ち上がり・立ち下がりが鈍って符号長の判別が壊れる
懸念がある。RDP を迂回して 16 kHz モノラルの非圧縮 PCM をそのまま流す。

プロトコル (どちらもリトルエンディアン)::

    接続時に送信側が 12 バイトのヘッダを 1 回だけ送る
        magic        4 bytes  b"V2SA"
        sample_rate  uint32   16000
        channels     uint16   1
        bits         uint16   16
    以降は int16 のサンプル列が途切れなく続く

帯域は 16000 × 2 = 32 kB/s (256 kbps)。LAN なら無視できる量。

マジックが ``V2SA`` なのは ``voice_to_string`` 由来。**変更しないこと**。
揃えておけば 1 つの送信プロセスをどちらのアプリからも掴める。

送信側は :file:`scripts/audio_send.py`。
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = b"V2SA"
HEADER_FORMAT = "<4sIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DEFAULT_PORT = 45678

# 送信側が流すサンプルレート。これ以外は受け付けない。
SOURCE_SAMPLE_RATE = 16000


class NetworkCaptureError(RuntimeError):
    """転送元に接続できない、または応答が不正."""


def parse_endpoint(text: str, *, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """``"host:port"`` / ``"host"`` を (host, port) へ.

    Raises:
        ValueError: 空文字、またはポートが数値でない場合.
    """
    text = text.strip()
    if not text:
        raise ValueError("転送元が空です")
    # IPv6 の [::1]:45678 形式にも対応する (: がアドレス内に出るため)
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port_text = rest.lstrip(":")
    else:
        host, _, port_text = text.rpartition(":")
        if not host:  # ":" が無い = ホスト名だけ
            host, port_text = text, ""
    if not port_text:
        return host, default_port
    try:
        return host, int(port_text)
    except ValueError as exc:
        raise ValueError(f"ポート番号が数値ではありません: {port_text!r}") from exc


def encode_header(sample_rate: int = SOURCE_SAMPLE_RATE) -> bytes:
    """送信側が最初に送るヘッダを作る."""
    return struct.pack(HEADER_FORMAT, MAGIC, sample_rate, 1, 16)


def encode_samples(block: np.ndarray) -> bytes:
    """float32 (-1..1) を int16 リトルエンディアンのバイト列へ."""
    clipped = np.clip(block, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


__all__ = [
    "DEFAULT_PORT",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "MAGIC",
    "SOURCE_SAMPLE_RATE",
    "NetworkCaptureError",
    "encode_header",
    "encode_samples",
    "parse_endpoint",
]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: コミット**

```bash
git add src/infer/net_audio.py tests/test_net_audio.py
git commit -m "feat: LAN 音声転送のプロトコル補助関数を追加した"
```

---

### Task 2: 接続とヘッダ検証

**Files:**
- Modify: `src/infer/net_audio.py`
- Test: `tests/test_net_audio.py`

**Interfaces:**
- Consumes: Task 1 の `MAGIC` / `HEADER_FORMAT` / `HEADER_SIZE` / `DEFAULT_PORT` / `SOURCE_SAMPLE_RATE` / `NetworkCaptureError` / `encode_header` / `encode_samples`
- Produces:
  - `class NetworkAudioCapture` — `__init__(host: str, port: int = DEFAULT_PORT, *, target_sample_rate: int = 8000, expected_source_rate: int = SOURCE_SAMPLE_RATE, connect_timeout_s: float = 5.0)`
  - `.start() -> None` / `.stop() -> None`
  - `.source_sample_rate -> int | None`（`start()` 前は `None`）
  - `.is_connected -> bool` / `.last_error -> str | None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_net_audio.py` の末尾に追記（先頭の import 文に `NetworkAudioCapture` と `NetworkCaptureError` を足す）:

```python
import contextlib
import socket
import threading
import time


class _FakeSender:
    """127.0.0.1 で待ち受けるダミー送信側.

    ``header_override`` を与えるとヘッダを差し替えられる (異常系テスト用)。
    """

    def __init__(self, header: bytes | None = None) -> None:
        self._header = encode_header() if header is None else header
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self._server.settimeout(5.0)
        self.port: int = self._server.getsockname()[1]
        self._conn: socket.socket | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._server.accept()
            except (TimeoutError, OSError):
                continue
            with contextlib.suppress(OSError):
                conn.sendall(self._header)
            with self._lock:
                self._conn = conn

    def send(self, block: np.ndarray) -> None:
        """接続が確立するまで待ってから送る."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                conn = self._conn
            if conn is not None:
                with contextlib.suppress(OSError):
                    conn.sendall(encode_samples(block))
                return
            time.sleep(0.01)
        raise AssertionError("ダミー送信側に接続が来ませんでした")

    def drop_client(self) -> None:
        """接続だけ切る (待ち受けは続ける)."""
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()

    def close(self) -> None:
        self._running = False
        self.drop_client()
        with contextlib.suppress(OSError):
            self._server.close()
        self._thread.join(timeout=2.0)


@pytest.fixture()
def sender():
    s = _FakeSender()
    yield s
    s.close()


def test_start_connects_and_reads_header(sender) -> None:
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    assert cap.source_sample_rate is None
    cap.start()
    try:
        assert cap.source_sample_rate == SOURCE_SAMPLE_RATE
        assert cap.is_connected
    finally:
        cap.stop()


def test_stop_is_idempotent(sender) -> None:
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    cap.stop()
    cap.stop()  # 二度目も例外にならない
    assert not cap.is_connected


def test_start_raises_when_nothing_is_listening() -> None:
    # 誰も listen していないポートを掴む
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()

    cap = NetworkAudioCapture("127.0.0.1", free_port, connect_timeout_s=1.0)
    with pytest.raises(NetworkCaptureError) as exc:
        cap.start()
    assert "audio_send.py" in str(exc.value)


def test_start_rejects_wrong_magic() -> None:
    bad = struct.pack(HEADER_FORMAT, b"XXXX", SOURCE_SAMPLE_RATE, 1, 16)
    s = _FakeSender(header=bad)
    try:
        cap = NetworkAudioCapture("127.0.0.1", s.port)
        with pytest.raises(NetworkCaptureError) as exc:
            cap.start()
        assert "ポート番号" in str(exc.value)
    finally:
        s.close()


def test_start_rejects_stereo() -> None:
    bad = struct.pack(HEADER_FORMAT, MAGIC, SOURCE_SAMPLE_RATE, 2, 16)
    s = _FakeSender(header=bad)
    try:
        cap = NetworkAudioCapture("127.0.0.1", s.port)
        with pytest.raises(NetworkCaptureError):
            cap.start()
    finally:
        s.close()


def test_start_rejects_wrong_sample_rate() -> None:
    """8000 Hz が来ても弾く。無言で通すとリサンプル比が狂って符号長がずれる."""
    bad = struct.pack(HEADER_FORMAT, MAGIC, 8000, 1, 16)
    s = _FakeSender(header=bad)
    try:
        cap = NetworkAudioCapture("127.0.0.1", s.port)
        with pytest.raises(NetworkCaptureError) as exc:
            cap.start()
        assert "8000" in str(exc.value)
    finally:
        s.close()
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -q`
Expected: FAIL — `ImportError: cannot import name 'NetworkAudioCapture'`

- [ ] **Step 3: 最小実装**

`src/infer/net_audio.py` の import 群に追加:

```python
import contextlib
import socket
import threading
import time
```

`parse_endpoint` の後に定数と `NetworkAudioCapture` の骨格を追加:

```python
# 受信スレッドが一度に読むバイト数。16 kHz int16 なら 0.05 秒ぶん。
_READ_CHUNK = 1600

# 切断されたときの再接続間隔 (秒)。
_RECONNECT_INTERVAL_S = 1.0

# 溜め込みの上限 (秒)。消費側が止まっても青天井にしない。
_MAX_BUFFER_S = 30.0


def _close(sock: socket.socket | None) -> None:
    if sock is None:
        return
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        sock.close()


def _recv_exactly(sock: socket.socket, count: int) -> bytes | None:
    """``count`` バイト読み切る。途中で切断されたら None."""
    parts: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


class NetworkAudioCapture:
    """TCP で送られてくる 16 kHz モノラル PCM を 8 kHz にして供給する.

    使い方は :class:`~src.infer.audio.AudioCapture` と同じ::

        cap = NetworkAudioCapture("192.168.1.20", 45678)
        cap.start()
        blocks = cap.drain()

    接続が切れても ``stop()`` を呼ぶまで裏で再接続を試み続ける。送信側
    (受信機を繋いだ PC) を後から起動しても繋がる。
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        target_sample_rate: int = 8000,
        expected_source_rate: int = SOURCE_SAMPLE_RATE,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.target_sample_rate = target_sample_rate
        self.expected_source_rate = expected_source_rate
        self.connect_timeout_s = connect_timeout_s

        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._buffered = 0
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._source_sr: int | None = None
        self._resampler = None
        self._level_rms = 0.0
        self._dropped_blocks = 0
        self._reconnects = 0
        self._last_error: str | None = None

    # ---- AudioCapture と同じインターフェース ----
    @property
    def source_sample_rate(self) -> int | None:
        return self._source_sr

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._sock is not None

    @property
    def reconnects(self) -> int:
        """接続が切れて繋ぎ直した回数."""
        with self._lock:
            return self._reconnects

    @property
    def dropped_blocks(self) -> int:
        with self._lock:
            return self._dropped_blocks

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    # ---- ライフサイクル ----
    def start(self) -> None:
        """転送元へ接続し、受信スレッドを起こす.

        1 回目の接続とヘッダ検証は同期的に行う。繋がっていないのに
        「デコード中」と表示する状態を作らないため。

        Raises:
            NetworkCaptureError: 最初の接続に失敗した場合.
        """
        if self._running:
            return
        self._sock = self._connect()
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="cw-net-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """受信を止める。二重呼び出しは無害."""
        self._running = False
        with self._lock:
            sock, self._sock = self._sock, None
            self._level_rms = 0.0
        _close(sock)

        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        with self._lock:
            self._chunks.clear()
            self._buffered = 0
        self._resampler = None

    # ---- 接続 ----
    def _connect(self) -> socket.socket:
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_s
            )
        except OSError as exc:
            raise NetworkCaptureError(
                f"転送元に接続できません ({self.host}:{self.port}): {exc}\n"
                "受信機を繋いだ PC で scripts/audio_send.py を動かしているか、\n"
                "ファイアウォールがそのポートを通しているかを確認してください。"
            ) from exc

        sock.settimeout(self.connect_timeout_s)
        try:
            header = _recv_exactly(sock, HEADER_SIZE)
        except OSError as exc:
            _close(sock)
            raise NetworkCaptureError(f"転送元からヘッダを受け取れません: {exc}") from exc
        if header is None:
            _close(sock)
            raise NetworkCaptureError("転送元がヘッダを送る前に切断しました")

        magic, sample_rate, channels, bits = struct.unpack(HEADER_FORMAT, header)
        if magic != MAGIC:
            _close(sock)
            raise NetworkCaptureError(
                f"転送元の応答が想定と違います (magic={magic!r})。"
                "ポート番号を間違えていませんか。"
            )
        if channels != 1 or bits != 16:
            _close(sock)
            raise NetworkCaptureError(
                f"未対応の形式です ({channels}ch / {bits}bit)。1ch / 16bit のみ扱えます。"
            )
        if sample_rate != self.expected_source_rate:
            _close(sock)
            raise NetworkCaptureError(
                f"サンプルレートが違います "
                f"(転送元 {sample_rate} Hz / 期待 {self.expected_source_rate} Hz)。"
                "送信側の設定を確認してください。"
            )

        self._source_sr = sample_rate
        self._make_resampler()
        sock.settimeout(1.0)  # stop() に速やかに反応できるよう短くする
        with self._lock:
            self._last_error = None
        return sock

    def _make_resampler(self) -> None:
        """リサンプラを作り直す.

        soxr.ResampleStream はブロック境界の状態を保持するため通常は使い回すが、
        **再接続でストリームが途切れた場合は作り直す**。前の接続の状態を
        引き継ぐと境界が不正になる。
        """
        from src.infer.audio import _SOXR_AVAILABLE, soxr

        if (
            self._source_sr is not None
            and self._source_sr != self.target_sample_rate
            and _SOXR_AVAILABLE
        ):
            self._resampler = soxr.ResampleStream(
                self._source_sr, self.target_sample_rate,
                num_channels=1, dtype="float32",
            )
        else:
            self._resampler = None

    def _reader_loop(self) -> None:
        """Task 3 で実装する."""
        while self._running:
            time.sleep(0.01)
```

`__all__` に `"NetworkAudioCapture"` を追加。

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -q`
Expected: PASS (17 tests)

- [ ] **Step 5: コミット**

```bash
git add src/infer/net_audio.py tests/test_net_audio.py
git commit -m "feat: LAN 音声の接続とヘッダ検証を実装した"
```

---

### Task 3: 受信・リサンプル・レベル・バッファ上限

**Files:**
- Modify: `src/infer/net_audio.py`
- Test: `tests/test_net_audio.py`

**Interfaces:**
- Consumes: Task 2 の `NetworkAudioCapture`
- Produces:
  - `.drain() -> list[np.ndarray]`（8 kHz、無ければ空リスト）
  - `.level_db_rms -> float`（無音時 `-120.0`）
  - `.dropped_blocks -> int`
  - `__init__` に `max_buffer_s: float = _MAX_BUFFER_S` を追加する（Task 2 のシグネチャを拡張）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_net_audio.py` の末尾に追記:

```python
def _drain_until(cap, min_samples: int, timeout_s: float = 5.0) -> np.ndarray:
    """``min_samples`` 以上たまるまで drain し続けて連結する."""
    got: list[np.ndarray] = []
    total = 0
    deadline = time.monotonic() + timeout_s
    while total < min_samples and time.monotonic() < deadline:
        for block in cap.drain():
            got.append(block)
            total += block.size
        time.sleep(0.01)
    return np.concatenate(got) if got else np.zeros(0, dtype=np.float32)


def test_drain_returns_empty_list_when_nothing_arrived(sender) -> None:
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    try:
        assert cap.drain() == []
    finally:
        cap.stop()


def test_drain_downsamples_16k_to_8k(sender) -> None:
    """16 kHz を 1 秒ぶん送ったら 8 kHz で約半分のサンプル数が返る."""
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    try:
        tone = (0.5 * np.sin(2 * np.pi * 500 * np.arange(16000) / 16000)).astype(np.float32)
        sender.send(tone)
        out = _drain_until(cap, 7000)
        # soxr は数サンプルの遅延を持つため厳密一致ではなく範囲で見る
        assert 7000 <= out.size <= 8100
        assert out.dtype == np.float32
        # 500 Hz のトーンは 8 kHz でも保たれる (振幅が消えていない)
        assert float(np.abs(out).max()) > 0.2
    finally:
        cap.stop()


def test_level_db_rms_reflects_received_audio(sender) -> None:
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    try:
        assert cap.level_db_rms == -120.0  # 受信前は無音扱い
        tone = (0.5 * np.sin(2 * np.pi * 500 * np.arange(16000) / 16000)).astype(np.float32)
        sender.send(tone)
        _drain_until(cap, 7000)
        assert -20.0 < cap.level_db_rms < 0.0
    finally:
        cap.stop()


def test_buffer_cap_drops_oldest(sender) -> None:
    """消費側が止まっても青天井に溜めない."""
    cap = NetworkAudioCapture("127.0.0.1", sender.port, max_buffer_s=0.5)
    cap.start()
    try:
        block = np.zeros(16000, dtype=np.float32)  # 1 秒ぶん
        for _ in range(3):
            sender.send(block)
        deadline = time.monotonic() + 5.0
        while cap.dropped_blocks == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cap.dropped_blocks > 0
    finally:
        cap.stop()
```

`NetworkAudioCapture` に `max_buffer_s` 引数が要るので、Step 3 で追加する。

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -q`
Expected: FAIL — `AttributeError: 'NetworkAudioCapture' object has no attribute 'drain'`

- [ ] **Step 3: 実装**

`__init__` のシグネチャに `max_buffer_s: float = _MAX_BUFFER_S` を追加し、`self.max_buffer_s = max_buffer_s` を保存する。

`_reader_loop` のダミー実装を差し替え、以下のメソッドを追加:

```python
    @property
    def level_db_rms(self) -> float:
        """直近ブロックの RMS dBFS。無音時は -120 dB (AudioCapture と同じ規約)."""
        with self._lock:
            rms = self._level_rms
        if rms < 1e-6:
            return -120.0
        return 20.0 * float(np.log10(rms))

    def drain(self) -> list[np.ndarray]:
        """受信済みを 8 kHz に変換してまとめて返す。無ければ空リスト.

        soxr.ResampleStream は内部状態を保持し、ブロック境界での歪みを生じない。
        scipy.signal.resample_poly はステートレスで境界歪みが出て CW の
        dot/dash を破壊するため、フォールバック以外では使わない。
        """
        with self._lock:
            chunks, self._chunks = self._chunks, []
            self._buffered = 0
        if not chunks:
            return []

        native = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if self._source_sr is None or self._source_sr == self.target_sample_rate:
            return [native]
        if self._resampler is None:
            from src.infer.audio import _resample_to_8k
            return [_resample_to_8k(native, self._source_sr)]

        out = self._resampler.resample_chunk(native, last=False)
        if out.size == 0:
            return []
        return [np.asarray(out, dtype=np.float32)]

    def _reader_loop(self) -> None:
        leftover = b""
        while self._running:
            sock = self._sock
            if sock is None:
                sock = self._reconnect()
                if sock is None:
                    continue
                leftover = b""

            try:
                data = sock.recv(_READ_CHUNK)
            except TimeoutError:
                continue
            except OSError as exc:
                self._note_disconnect(str(exc))
                leftover = b""
                continue

            if not data:  # 相手が閉じた
                self._note_disconnect("転送元が切断しました")
                leftover = b""
                continue

            leftover = self._consume(leftover + data)

    def _note_disconnect(self, reason: str) -> None:
        with self._lock:
            sock, self._sock = self._sock, None
            self._last_error = reason
        _close(sock)

    def _consume(self, buffer: bytes) -> bytes:
        """バイト列からサンプルを取り出して溜める。端数は返す."""
        usable = len(buffer) - (len(buffer) % 2)
        if usable <= 0:
            return buffer

        samples = np.frombuffer(buffer[:usable], dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0

        with self._lock:
            self._chunks.append(samples)
            self._buffered += samples.size
            self._level_rms = rms
            # 消費側が止まっているときに溜め続けない。古いほうから捨てる。
            limit = int(self.max_buffer_s * self.expected_source_rate)
            while self._buffered > limit and len(self._chunks) > 1:
                self._buffered -= self._chunks.pop(0).size
                self._dropped_blocks += 1

        return buffer[usable:]
```

`_reconnect` は Task 4 で実装するので、暫定でこれを追加:

```python
    def _reconnect(self) -> socket.socket | None:
        """Task 4 で実装する."""
        time.sleep(_RECONNECT_INTERVAL_S)
        return None
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -q`
Expected: PASS (21 tests)

- [ ] **Step 5: コミット**

```bash
git add src/infer/net_audio.py tests/test_net_audio.py
git commit -m "feat: LAN 音声の受信と 16k→8k リサンプルを実装した"
```

---

### Task 4: 自動再接続とインターフェース適合

**Files:**
- Modify: `src/infer/net_audio.py`
- Test: `tests/test_net_audio.py`

**Interfaces:**
- Consumes: Task 3 の `NetworkAudioCapture`
- Produces: `.reconnects` が切断→再接続で増える挙動

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_net_audio.py` の末尾に追記（先頭 import に `from src.infer.audio import AudioCapture` を足す）:

```python
def test_reconnects_after_disconnect(sender) -> None:
    """送信側を落として上げ直しても、操作なしで受信が再開する."""
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    try:
        tone = (0.5 * np.sin(2 * np.pi * 500 * np.arange(8000) / 16000)).astype(np.float32)
        sender.send(tone)
        _drain_until(cap, 3000)

        sender.drop_client()

        deadline = time.monotonic() + 10.0
        while cap.reconnects == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cap.reconnects >= 1

        sender.send(tone)
        assert _drain_until(cap, 3000).size >= 3000
    finally:
        cap.stop()


def test_last_error_records_disconnect_reason(sender) -> None:
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    try:
        sender.drop_client()
        deadline = time.monotonic() + 5.0
        while cap.last_error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert cap.last_error is not None
    finally:
        cap.stop()


def test_matches_audio_capture_interface() -> None:
    """ワーカーが使う 5 メンバを AudioCapture と同じ形で持つこと.

    片方だけ変更されると差し替えが壊れるので固定する。
    """
    for name in ("start", "stop", "drain"):
        assert callable(getattr(NetworkAudioCapture, name))
        assert callable(getattr(AudioCapture, name))
    for name in ("source_sample_rate", "level_db_rms"):
        assert isinstance(getattr(NetworkAudioCapture, name), property)
        assert isinstance(getattr(AudioCapture, name), property)
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -k reconnect -q`
Expected: FAIL — `assert cap.reconnects >= 1` が 0 のまま

- [ ] **Step 3: 実装**

`_reconnect` の暫定実装を差し替える:

```python
    def _reconnect(self) -> socket.socket | None:
        """繋ぎ直す。失敗したら少し待って None を返す (次のループで再試行)."""
        time.sleep(_RECONNECT_INTERVAL_S)
        if not self._running:
            return None
        try:
            sock = self._connect()
        except NetworkCaptureError as exc:
            with self._lock:
                self._last_error = str(exc).splitlines()[0]
            return None
        with self._lock:
            self._sock = sock
            self._reconnects += 1
        return sock
```

`_connect` の中で `self._make_resampler()` を呼んでいるため、再接続のたびにリサンプラが作り直される（設計書 §5.1 の要求）。

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_net_audio.py -q`
Expected: PASS (24 tests)

- [ ] **Step 5: コミット**

```bash
git add src/infer/net_audio.py tests/test_net_audio.py
git commit -m "feat: LAN 音声の自動再接続を実装した"
```

---

### Task 5: ワーカーへの差し替え口

**Files:**
- Modify: `src/app/workers.py:9`（import）、`:146` 付近（`__init__` の状態）、`:213-216` 付近（`set_net_source` 追加）、`:275-289`（`start`）
- Test: `tests/test_workers_net_source.py`

**Interfaces:**
- Consumes: Task 4 の `NetworkAudioCapture`、`parse_endpoint`
- Produces: `AudioInferenceWorker.set_net_source(endpoint: str | None) -> None`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_workers_net_source.py` を新規作成:

```python
"""ワーカーが LAN 経由キャプチャへ差し替わることのテスト.

Qt のイベントループも実デバイスも使わない。生成されるキャプチャの型だけを見る。
"""
from __future__ import annotations

import pytest

from src.app.workers import AudioInferenceWorker


def _make_worker(engine) -> AudioInferenceWorker:
    return AudioInferenceWorker(engine=engine, sample_rate=8000, mode="european")


def test_set_net_source_stores_endpoint(monkeypatch) -> None:
    engine = object()
    worker = _make_worker(engine)
    worker.set_net_source("192.168.1.20:45000")
    assert worker._net_endpoint == ("192.168.1.20", 45000)


def test_set_net_source_none_clears(monkeypatch) -> None:
    worker = _make_worker(object())
    worker.set_net_source("192.168.1.20")
    worker.set_net_source(None)
    assert worker._net_endpoint is None


def test_set_net_source_rejects_bad_endpoint() -> None:
    worker = _make_worker(object())
    with pytest.raises(ValueError):
        worker.set_net_source("192.168.1.20:abc")


def test_start_uses_network_capture_when_endpoint_set(monkeypatch) -> None:
    """--net-source 指定時は NetworkAudioCapture を作ること."""
    created: dict[str, object] = {}

    class _FakeNetCapture:
        def __init__(self, host, port, **kwargs):
            created["host"] = host
            created["port"] = port
            self.source_sample_rate = 16000

        def start(self):
            created["started"] = True

        def stop(self):
            pass

    monkeypatch.setattr("src.app.workers.NetworkAudioCapture", _FakeNetCapture)
    monkeypatch.setattr("src.app.workers.QTimer", lambda: _NoopTimer())

    worker = _make_worker(object())
    worker.set_net_source("192.168.1.20:45000")
    worker.start()

    assert created["host"] == "192.168.1.20"
    assert created["port"] == 45000
    assert created["started"] is True


class _NoopTimer:
    """QTimer の代わり。start/stop を受けるだけ."""

    def __init__(self) -> None:
        self.timeout = _NoopSignal()

    def start(self, _interval) -> None:
        pass

    def stop(self) -> None:
        pass


class _NoopSignal:
    def connect(self, _slot) -> None:
        pass


def test_start_uses_local_capture_when_no_endpoint(monkeypatch) -> None:
    created: dict[str, object] = {}

    class _FakeLocalCapture:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            self.source_sample_rate = 48000

        def start(self):
            created["started"] = True

        def stop(self):
            pass

    monkeypatch.setattr("src.app.workers.AudioCapture", _FakeLocalCapture)
    monkeypatch.setattr("src.app.workers.QTimer", lambda: _NoopTimer())

    worker = _make_worker(object())
    worker.start()

    assert created["started"] is True
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_workers_net_source.py -q`
Expected: FAIL — `AttributeError: 'AudioInferenceWorker' object has no attribute 'set_net_source'`

- [ ] **Step 3: 実装**

`src/app/workers.py:9` の import を変更:

```python
from src.infer.audio import AudioCapture
from src.infer.net_audio import NetworkAudioCapture, parse_endpoint
```

`__init__` の `self._device_index: int | None = None` の直後に追加:

```python
        # LAN 経由入力 (--net-source)。None ならローカルデバイスを使う。
        self._net_endpoint: tuple[str, int] | None = None
```

`set_device` の直後に追加:

```python
    @Slot(object)
    def set_net_source(self, endpoint: object) -> None:
        """LAN 経由の転送元を設定する。``None`` でローカルデバイスに戻す.

        Raises:
            ValueError: ``"host:port"`` として解釈できない場合.
        """
        if endpoint is None or (isinstance(endpoint, str) and not endpoint.strip()):
            self._net_endpoint = None
            return
        self._net_endpoint = parse_endpoint(str(endpoint))
```

`start()` のキャプチャ生成部分を差し替える:

```python
    @Slot()
    def start(self) -> None:
        try:
            if self._net_endpoint is not None:
                host, port = self._net_endpoint
                self._capture = NetworkAudioCapture(
                    host, port, target_sample_rate=self.sample_rate,
                )
                self._capture.start()
                sr = self._capture.source_sample_rate
                self.status.emit(
                    f"capture started @ {sr} Hz → {self.sample_rate} Hz "
                    f"(LAN {host}:{port})"
                )
            else:
                self._capture = AudioCapture(
                    device_index=self._device_index,
                    target_sample_rate=self.sample_rate,
                    block_duration_s=0.05,
                )
                self._capture.start()
                sr = self._capture.source_sample_rate
                self.status.emit(f"capture started @ {sr} Hz → {self.sample_rate} Hz")
            self._running = True
            self._decoding = False
            self._timer = QTimer()
            self._timer.timeout.connect(self._tick)
            self._timer.start(20)
        except Exception as exc:                       # noqa: BLE001
            self.error.emit(f"start failed: {exc!r}")
```

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_workers_net_source.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: コミット**

```bash
git add src/app/workers.py tests/test_workers_net_source.py
git commit -m "feat: ワーカーが LAN 経由キャプチャへ差し替わるようにした"
```

---

### Task 6: CLI と画面への配線

**Files:**
- Modify: `scripts/run_app.py`
- Modify: `src/app/main_window.py:66-70`（`__init__`）、`:282`（`_on_start`）、`:302`（`set_device` の隣）、`:335` と `:354`（コンボの有効/無効）、`:621-623`（`main`）
- Test: `tests/test_cli_net_source.py`

**Interfaces:**
- Consumes: Task 5 の `AudioInferenceWorker.set_net_source`
- Produces: `run_app.main(argv)` が `--net-source` を受け、`main_window.main(checkpoint_path, net_source)` へ渡す

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_cli_net_source.py` を新規作成:

```python
"""run_app.py の --net-source 引数のテスト (アプリは起動しない)."""
from __future__ import annotations

import scripts.run_app as run_app


def test_net_source_is_passed_through(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(*, checkpoint_path=None, net_source=None):
        captured["checkpoint_path"] = checkpoint_path
        captured["net_source"] = net_source
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    assert run_app.main(["--ckpt", "models/x.pt", "--net-source", "192.168.1.20"]) == 0
    assert captured["checkpoint_path"] == "models/x.pt"
    assert captured["net_source"] == "192.168.1.20"


def test_net_source_defaults_to_none(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(*, checkpoint_path=None, net_source=None):
        captured["net_source"] = net_source
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    assert run_app.main([]) == 0
    assert captured["net_source"] is None
```

- [ ] **Step 2: 失敗を確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_cli_net_source.py -q`
Expected: FAIL — `unrecognized arguments: --net-source`

- [ ] **Step 3: 実装**

`scripts/run_app.py` を差し替え:

```python
"""CW デコーダ アプリ起動 (ライブ連続モード).

使い方::

    python scripts/run_app.py                     # チェックポイント無し (未学習モデル)
    python scripts/run_app.py --ckpt models/best.pt

受信機を繋いだ PC が別なとき (その PC で scripts/audio_send.py を動かしておく)::

    python scripts/run_app.py --ckpt models/full/best_infer.pt --net-source 192.168.1.20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.app.main_window import main as run_app_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CW デコーダ アプリ")
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="チェックポイントパス (省略時は未学習モデルで起動)",
    )
    parser.add_argument(
        "--net-source", type=str, default=None, metavar="HOST[:PORT]",
        help="LAN 経由の転送元 (既定ポート 45678)。指定時は入力デバイスの代わりに使う",
    )
    args = parser.parse_args(argv)
    return run_app_main(
        checkpoint_path=args.ckpt,
        net_source=args.net_source,
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

`src/app/main_window.py` の `CWDecoderWindow.__init__` に引数を追加:

```python
    def __init__(
        self,
        engine: InferenceEngine,
        settings: AppSettings | None = None,
        net_source: str | None = None,
    ) -> None:
```

`self._engine = engine` の直後に追加:

```python
        # LAN 経由入力 (--net-source)。指定時は入力デバイス選択を使わない。
        self._net_source = net_source
```

`_populate_devices()` の呼び出し後（`__init__` 内でウィジェット構築が終わった箇所）に追加:

```python
        if self._net_source:
            self.device_combo.setEnabled(False)
            self.device_combo.setToolTip(
                f"LAN 経由入力を使用中 ({self._net_source})。入力デバイスは無効です。"
            )
```

`_on_start` の `self._worker.set_device(device_index)` の直後に追加:

```python
        self._worker.set_net_source(self._net_source)
```

`:335` の `self.device_combo.setEnabled(False)` はそのまま、`:354` の
`self.device_combo.setEnabled(True)` を次に変える（LAN 使用時は戻さない）:

```python
        self.device_combo.setEnabled(not self._net_source)
```

`main()` を差し替え:

```python
def main(
    checkpoint_path: str | None = None,
    net_source: str | None = None,
) -> int:
    """エントリポイント."""
    from PySide6.QtWidgets import QApplication
    import sys

    settings = load_settings()
    if checkpoint_path:
        settings.checkpoint_path = checkpoint_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if settings.checkpoint_path and Path(settings.checkpoint_path).exists():
        engine = InferenceEngine.from_checkpoint(settings.checkpoint_path, device=device)
    else:
        engine = InferenceEngine.untrained(device=device)

    app = QApplication.instance() or QApplication(sys.argv)
    window = CWDecoderWindow(engine, settings, net_source=net_source)
    window.show()
    return app.exec()
```

`scripts/__init__.py` は**存在しないが作らなくてよい**。既存の `tests/test_keying_corpus.py` が
`from scripts.generate_corpus_scripts import main` で通っており、pytest の rootdir 解決で
`scripts` パッケージが見えている。Step 4 で import に失敗した場合のみ空ファイルを作る。

- [ ] **Step 4: テストが通ることを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_cli_net_source.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: 既存テストの退行がないことを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q > pytest_out.txt 2>&1; grep -cE "^(FAILED|ERROR)" pytest_out.txt`
Expected: `0`（exit 127 は既知のクラッシュなので無視。`FAILED`/`ERROR` がゼロなら合格）

- [ ] **Step 6: コミット**

```bash
git add scripts/run_app.py src/app/main_window.py tests/test_cli_net_source.py
git commit -m "feat: --net-source で LAN 経由入力を選べるようにした"
```

---

### Task 7: 送信側スクリプト

**Files:**
- Create: `scripts/audio_send.py`
- Modify: `docs/INSTALL.md`（末尾に節を追加）

**Interfaces:**
- Consumes: Task 1 の `DEFAULT_PORT` / `SOURCE_SAMPLE_RATE` / `encode_header` / `encode_samples`、`src/infer/audio.py` の `AudioCapture` / `list_input_devices`
- Produces: なし（実行可能スクリプト）

- [ ] **Step 1: スクリプトを書く**

`scripts/audio_send.py` を新規作成:

```python
"""受信機を繋いだ PC で動かし、音声を LAN 越しに非圧縮で送り出す.

GPU を積んだ PC と受信機を繋いだ PC が別なときに使う。GPU 側では
``run_app.py --net-source この PC の IP`` で受け取る。

**リモートデスクトップのマイク転送を使わない理由**: RDP は音声入力を狭帯域
コーデックで圧縮する。同じリポジトリの ``voice_to_string`` が 2026-08-01 に
実測しており、エネルギーの 99% が 1000 Hz 以下になる。このスクリプトは RDP を
迂回して 16 kHz モノラルの PCM をそのまま流す。

使い方 (受信機を繋いだ PC で)::

    python scripts/audio_send.py --list              # 入力デバイス一覧
    python scripts/audio_send.py --device 13         # 送信開始 (Ctrl+C で終了)
    python scripts/audio_send.py --device 13 --port 45678

GPU 側の PC では::

    python scripts/run_app.py --ckpt models/full/best_infer.pt --net-source 192.168.1.20

BPF はここでは掛けない。cw-decorder 側の BPF 設定がそのまま効くようにするため
生の音を流す。**GPU 側で BPF を必ず ON にすること** (BPF 未通過の音は
TER 97% まで崩壊する。docs/phase4_data_collection.md 参照)。

注意: 音声は暗号化せずに流す。家庭内 LAN での利用を前提とし、外部へ公開する
ポートには絶対に割り当てないこと。``--bind`` で待ち受けるアドレスを絞れる。
"""
from __future__ import annotations

import argparse
import contextlib
import select
import socket
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.infer.audio import AudioCapture, list_input_devices  # noqa: E402
from src.infer.net_audio import (  # noqa: E402
    DEFAULT_PORT,
    SOURCE_SAMPLE_RATE,
    encode_header,
    encode_samples,
)

# 送出の周期 (秒)。細かすぎると CPU を食い、粗すぎると遅延が増える。
SEND_INTERVAL_S = 0.05


def client_is_gone(conn: socket.socket) -> bool:
    """受け側が切断していないかを、書き込みを待たずに調べる.

    TCP は相手が消えても ``sendall`` がすぐには失敗しない。放っておくと
    死んだ接続へ送り続け、``accept`` に戻らないので**次の接続が繋がらなく
    なる** (voice_to_string で実際に踏んだ)。受け側はデータを送ってこないので、
    読める状態になっていること自体が切断 (EOF) の合図になる。
    """
    readable, _, errored = select.select([conn], [], [conn], 0)
    if errored:
        return True
    if not readable:
        return False
    try:
        return conn.recv(1) == b""
    except OSError:
        return True


def print_devices() -> int:
    devices = list_input_devices()
    if not devices:
        print("入力デバイスが見つかりませんでした (PortAudio 未導入?)。", file=sys.stderr)
        return 1
    print("入力デバイス:")
    for info in devices:
        print(f"  [{info.index}] {info.name}  ({info.channels}ch, {info.default_samplerate:.0f} Hz)")
    print()
    print("受信機を繋いだ端子のデバイス番号を --device に指定してください。")
    return 0


def local_addresses() -> list[str]:
    """GPU 側の PC から指定してもらう IP の候補を集める."""
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        return []
    return [a for a in addrs if not a.startswith("127.")]


def serve(capture: AudioCapture, server: socket.socket, *, quiet: bool) -> None:
    """クライアント 1 つを受け付けて、切れるまで送り続ける."""
    server.settimeout(1.0)
    while True:
        try:
            conn, peer = server.accept()
        except TimeoutError:
            continue

        print(f"\n接続: {peer[0]}:{peer[1]}  — 送信を開始します")
        conn.settimeout(5.0)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        capture.drain()  # 溜まっていた古い音は捨ててから始める
        sent_samples = 0
        last_report = time.monotonic()
        try:
            conn.sendall(encode_header(SOURCE_SAMPLE_RATE))
            while True:
                time.sleep(SEND_INTERVAL_S)
                if client_is_gone(conn):
                    print("\n受け側が切断しました。次の接続を待ちます。")
                    break

                blocks = capture.drain()
                if blocks:
                    block = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
                    if block.size:
                        conn.sendall(encode_samples(block))
                        sent_samples += block.size

                now = time.monotonic()
                if not quiet and now - last_report >= 1.0:
                    last_report = now
                    print(
                        f"\r  送信 {sent_samples / SOURCE_SAMPLE_RATE:7.1f} 秒"
                        f"   lvl {capture.level_db_rms:6.1f} dB        ",
                        end="",
                        flush=True,
                    )
        except (OSError, TimeoutError) as exc:
            print(f"\n切断されました ({exc})。次の接続を待ちます。")
        finally:
            with contextlib.suppress(OSError):
                conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="受信音を LAN 越しに非圧縮で送り出す")
    parser.add_argument("--list", action="store_true", help="入力デバイス一覧を表示して終了")
    parser.add_argument("--device", type=int, default=None, help="入力デバイス番号 (未指定なら既定デバイス)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"待ち受けポート (既定 {DEFAULT_PORT})")
    parser.add_argument("--bind", default="0.0.0.0", help="待ち受けアドレス (既定 0.0.0.0 = 全インターフェース)")
    parser.add_argument("--quiet", action="store_true", help="レベル表示を出さない")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        return print_devices()

    try:
        capture = AudioCapture(
            device_index=args.device,
            target_sample_rate=SOURCE_SAMPLE_RATE,
            block_duration_s=SEND_INTERVAL_S,
        )
        capture.start()
    except Exception as exc:                           # noqa: BLE001
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((args.bind, args.port))
        server.listen(1)
    except OSError as exc:
        capture.stop()
        print(f"エラー: ポートを開けません ({args.bind}:{args.port}): {exc}", file=sys.stderr)
        return 1

    print(f"入力デバイス : {args.device if args.device is not None else '既定'}")
    print(f"取り込み     : {capture.source_sample_rate} Hz → {SOURCE_SAMPLE_RATE} Hz モノラル")
    print(f"待ち受け     : {args.bind}:{args.port}")
    for addr in local_addresses():
        print(f"  GPU 側の PC では:  python scripts/run_app.py --net-source {addr}:{args.port}")
    print("\n接続を待っています… (Ctrl+C で終了)")

    try:
        serve(capture, server, quiet=args.quiet)
    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        capture.stop()
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 引数解析が動くことを確認**

Run: `PYTHONIOENCODING=utf-8 python scripts/audio_send.py --help`
Expected: usage が表示され、`--list` `--device` `--port` `--bind` `--quiet` が並ぶ

- [ ] **Step 3: デバイス一覧が出ることを確認**

Run: `PYTHONIOENCODING=utf-8 python scripts/audio_send.py --list`
Expected: 入力デバイスが列挙される（この PC に入力が無ければ「見つかりませんでした」で戻り値 1。どちらでも可）

- [ ] **Step 4: `docs/INSTALL.md` に節を追加**

`docs/INSTALL.md` の末尾に追記:

```markdown
## 受信機を繋いだ PC が別のとき（LAN 経由）

GPU を積んだ PC と受信機を繋いだ PC が別な場合、音声を LAN 越しに送れます。

**リモートデスクトップのマイク転送は使わないでください。** 音声が狭帯域
コーデックで圧縮され、エネルギーの 99% が 1000 Hz 以下になります。

### 受信機を繋いだ PC で

```
python scripts/audio_send.py --list       # 入力デバイス番号を調べる
python scripts/audio_send.py --device 13  # 送信開始 (Ctrl+C で終了)
```

起動すると、GPU 側で打つコマンドがそのまま表示されます。

### GPU を積んだ PC で

```
python scripts/run_app.py --ckpt models/full/best_infer.pt --net-source 192.168.1.20
```

`--net-source` を指定すると入力デバイス選択は無効になります。ポートを変えた
場合は `--net-source 192.168.1.20:45000` のように書きます。

送信側を止めて起動し直しても、GPU 側は操作なしで復帰します。

> **BPF を必ず ON にしてください。** BPF 未通過の音をモデルに入れると認識が
> 崩壊します（実測で TER 97%）。送信側は生の音を流し、整形は cw-decorder 側で
> 行う設計です。

> 音声は暗号化せずに流れます。家庭内 LAN での利用を前提としており、外部へ
> 公開するポートには割り当てないでください。
```

- [ ] **Step 5: 全テストの退行がないことを確認**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q > pytest_out.txt 2>&1; grep -cE "^(FAILED|ERROR)" pytest_out.txt`
Expected: `0`

- [ ] **Step 6: コミット**

```bash
git add scripts/audio_send.py docs/INSTALL.md
git commit -m "feat: 受信音を LAN 越しに送る audio_send.py を追加した"
```

---

## 実機での確認（実装完了後、ユーザーと一緒に）

自動テストでは実ネットワークを使わないため、最後に実機で次を確認する。

1. 無線機 PC で `python scripts/audio_send.py --device <番号>` を起動し、レベル表示が動くこと
2. GPU PC で `python scripts/run_app.py --ckpt models/full/best_infer.pt --net-source <IP>` を起動し、ステータスバーに `capture started @ 16000 Hz → 8000 Hz (LAN <IP>:45678)` が出ること
3. **BPF を ON** にして、実際の CW が読めること
4. 送信側を Ctrl+C で止め、再度起動すると GPU 側が操作なしで復帰すること
5. IP を間違えて起動したとき、原因の分かるエラーが出ること

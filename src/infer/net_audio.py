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

import contextlib
import socket
import struct
import threading
import time

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
        max_buffer_s: float = _MAX_BUFFER_S,
    ) -> None:
        self.host = host
        self.port = port
        self.target_sample_rate = target_sample_rate
        self.expected_source_rate = expected_source_rate
        self.connect_timeout_s = connect_timeout_s
        self.max_buffer_s = max_buffer_s

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
        sock = self._connect()
        with self._lock:
            self._sock = sock
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
        try:
            self._make_resampler()
        except Exception:
            _close(sock)
            raise
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

    # ---- 受信データの取り出し ----
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

    # ---- 受信スレッド ----
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

    def _reconnect(self) -> socket.socket | None:
        """繋ぎ直す。失敗したら少し待って None を返す (次のループで再試行)."""
        time.sleep(_RECONNECT_INTERVAL_S)
        if not self._running:
            return None
        try:
            sock = self._connect()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc).splitlines()[0]
            return None
        # 接続待ちの間に stop() が呼ばれていることがある (create_connection の
        # タイムアウトは stop() の join より長い)。掴んだソケットを閉じて捨てる。
        if not self._running:
            _close(sock)
            return None
        with self._lock:
            self._sock = sock
            self._reconnects += 1
        return sock


__all__ = [
    "DEFAULT_PORT",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "MAGIC",
    "SOURCE_SAMPLE_RATE",
    "NetworkAudioCapture",
    "NetworkCaptureError",
    "encode_header",
    "encode_samples",
    "parse_endpoint",
]

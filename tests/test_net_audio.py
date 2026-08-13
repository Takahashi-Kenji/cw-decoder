"""LAN 経由の音声入力のテスト.

実ネットワークには出ない。127.0.0.1 にダミーの送信サーバを立てて検証する。
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from src.infer.audio import AudioCapture
from src.infer.net_audio import (
    DEFAULT_PORT,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    SOURCE_SAMPLE_RATE,
    NetworkAudioCapture,
    NetworkCaptureError,
    encode_header,
    encode_samples,
    parse_endpoint,
)
from src.infer.net_audio import _RECONNECT_INTERVAL_S


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

    def send_bytes_chunked(self, data: bytes, chunk_sizes: list[int]) -> None:
        """接続確立を待ってから、生バイト列を ``chunk_sizes`` の順 (循環) で送る.

        int16 境界 (2 バイト) をまたぐ細かい・奇数バイトのチャンクを意図的に
        作れるよう、``encode_samples`` を経由せず生バイトを直接渡す。
        """
        deadline = time.monotonic() + 5.0
        conn: socket.socket | None = None
        while time.monotonic() < deadline:
            with self._lock:
                conn = self._conn
            if conn is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("ダミー送信側に接続が来ませんでした")

        offset = 0
        i = 0
        while offset < len(data):
            size = max(1, chunk_sizes[i % len(chunk_sizes)])
            i += 1
            piece = data[offset:offset + size]
            with contextlib.suppress(OSError):
                conn.sendall(piece)
            offset += size

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


def test_drain_reassembles_odd_byte_chunks_without_sample_shift(sender) -> None:
    """奇数バイト刻みで届いても標本のずれ・欠落が無いこと (バイト境界の回帰).

    ``_consume`` は int16 (2 バイト単位) の端数を次回に持ち越す。この処理が
    壊れると、全標本がバイトずれして轟音になるという壊れ方をする。既知の
    ランプ波形を、int16 境界をまたぐ細かい (奇数を含む) チャンクで流し込み、
    受信側で完全に復元できることを検証する。

    ``target_sample_rate`` を送信元と同じ 16000 Hz にしてリサンプルを経由させず、
    ``_consume`` のバイト再構成そのものだけを検証する。
    """
    cap = NetworkAudioCapture(
        "127.0.0.1", sender.port, target_sample_rate=SOURCE_SAMPLE_RATE
    )
    cap.start()
    try:
        n = 4000
        ramp = np.arange(-2000, 2000, dtype="<i2")
        assert ramp.size == n
        raw = ramp.tobytes()
        # int16 境界をまたぐ奇数バイトを含む細かいチャンクで送る
        chunk_sizes = [3, 1, 5, 7, 2, 9, 1]
        sender.send_bytes_chunked(raw, chunk_sizes)
        out = _drain_until(cap, n)
        assert out.size >= n
        received = np.round(out[:n] * 32768.0).astype("<i2")
        assert np.array_equal(received, ramp)
    finally:
        cap.stop()


def test_drain_preserves_500hz_tone_across_small_chunks(sender) -> None:
    """500 Hz トーンを実運用同様の小刻み送信 (0.05 秒間隔) で流し込んでも、
    8 kHz 出力の主成分が保たれること.

    このプロジェクトの中核リスクは「ブロック境界のリサンプル歪みが dot/dash を
    壊すこと」。振幅の下限だけでなく、FFT で主成分の周波数を確認する。送信・
    ``drain()`` 呼び出しを本番の ``_tick``/``audio_send.py`` と同様に細かく
    刻んで交互に行うことで、``drain()`` 呼び出しごとにリサンプラを作り直す
    ような実装退行があれば境界で歪みが生じ、主成分が 500 Hz 付近から外れる
    はずである (一括送信・一括 drain だと境界を跨がないため検出できない)。
    """
    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    try:
        duration_s = 2.0
        n = int(SOURCE_SAMPLE_RATE * duration_s)
        tone = (
            0.5 * np.sin(2 * np.pi * 500 * np.arange(n) / SOURCE_SAMPLE_RATE)
        ).astype(np.float32)
        # 実運用 (audio_send.py の SEND_INTERVAL_S) と同じ 0.05 秒刻みで送信し、
        # そのつど drain() する。一括送信後に一括 drain するとブロック境界を
        # またがず、境界歪みの退行を検出できない。
        block = int(SOURCE_SAMPLE_RATE * 0.05)
        collected: list[np.ndarray] = []
        for start in range(0, n, block):
            sender.send(tone[start:start + block])
            time.sleep(0.02)
            collected.extend(cap.drain())

        # 送信済み分の取りこぼしを回収
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            got = cap.drain()
            if got:
                collected.extend(got)
                continue
            if sum(b.size for b in collected) >= int(n * 0.5 * 0.8):
                break
            time.sleep(0.02)

        out = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
        assert out.size > 1000
        spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
        freqs = np.fft.rfftfreq(out.size, d=1.0 / 8000)
        peak_freq = float(freqs[int(np.argmax(spectrum))])
        assert 450.0 <= peak_freq <= 550.0
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


def test_stop_during_reconnect_header_wait_does_not_leak_socket(sender) -> None:
    """再接続の接続確立待ち中に stop() が呼ばれても、後から確立した接続を
    握ったままにしない (レビュー指摘1 の回帰テスト).

    ``stop()`` は受信スレッドを ``thread.join(timeout=2.0)`` で待つが、
    ``_connect()`` は転送元からのヘッダ待ちの ``recv()`` でそれより長く
    ブロックしうる (既定の ``connect_timeout_s`` は 5 秒)。ヘッダ送信を
    わざと遅らせて join がタイムアウトで先に返る状況を作り、その後に
    接続が確立しても ``is_connected`` が False のままであることを確認する。
    """
    header_delay_s = 2.5  # stop() の join(timeout=2.0) より長く取る
    slow_conns: list[socket.socket] = []  # ヘッダ送信後すぐに閉じると EOF で

    slow_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    slow_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    slow_server.bind(("127.0.0.1", 0))
    slow_server.listen(1)
    slow_server.settimeout(5.0)
    slow_port = slow_server.getsockname()[1]

    def _accept_and_delay() -> None:
        try:
            conn, _ = slow_server.accept()
        except OSError:
            return
        time.sleep(header_delay_s)
        with contextlib.suppress(OSError):
            conn.sendall(encode_header())
        # すぐに閉じると相手側が EOF で正規の切断経路 (_note_disconnect) を
        # 踏んでしまい、検証したいバグ (stop() 後に is_connected が True の
        # ままになる) を見えなくしてしまう。接続は開けたまま、後で片付ける。
        slow_conns.append(conn)

    server_thread = threading.Thread(target=_accept_and_delay, daemon=True)

    cap = NetworkAudioCapture("127.0.0.1", sender.port)
    cap.start()
    try:
        assert cap.is_connected

        sender.drop_client()
        deadline = time.monotonic() + 5.0
        while cap.is_connected and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not cap.is_connected

        # 遅いヘッダを返すサーバへ向け先を変える。_reconnect() は
        # self.host / self.port をそのつど読むので、次の試行から効く。
        # 切断検知の直後、_reconnect() 内の _RECONNECT_INTERVAL_S 待ちが
        # 明けるより十分前に書き換える。
        server_thread.start()
        cap.host, cap.port = "127.0.0.1", slow_port

        # _reconnect() は _RECONNECT_INTERVAL_S (1.0s) 待ってから接続を試み、
        # ヘッダ待ちのブロッキングに入る。そのタイミングで stop() を呼ぶ。
        time.sleep(_RECONNECT_INTERVAL_S + 0.3)
        cap.stop()
    finally:
        with contextlib.suppress(OSError):
            slow_server.close()
        server_thread.join(timeout=2.0)

    # ヘッダがようやく届いて接続が確立しても、stop() 後は握ってはいけない。
    time.sleep(header_delay_s)
    try:
        assert not cap.is_connected
    finally:
        for conn in slow_conns:
            with contextlib.suppress(OSError):
                conn.close()

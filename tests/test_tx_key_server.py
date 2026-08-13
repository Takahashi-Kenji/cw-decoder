"""打鍵サーバのテスト.

**実機は使わない。** 電波が出るため。シリアルは ``--dry-run`` 相当の偽物で代える。
"""
from __future__ import annotations

import contextlib
import math
import socket
import threading
import time

import pytest

from src.tx import key_server, protocol
from src.tx.key_server import (
    KeySession,
    RequestRejected,
    ServerConfig,
    prepare,
    visible_length,
)
from src.tx.keyer import RecordingKeySink
from src.tx.protocol import LineReader, decode_message, encode_message
from src.tx.serial_key import SerialKeyConfig, SerialKeyError


def test_マーカーを1文字と数える() -> None:
    assert visible_length("A{HORE}B") == 3


def test_マーカーの無い文はそのまま数える() -> None:
    assert visible_length("CQ DE JA1ABC") == 12


def test_送信専用マーカーも1文字と数える() -> None:
    # {KAKKO}/{TOJI} は SPECIAL_INPUT_MARKERS ではなく TX_ONLY_MARKERS 由来。
    # ここを数え損ねると "{KAKKO}アイ{TOJI}" が 4 文字ではなく 15 文字になる。
    assert visible_length("{KAKKO}アイ{TOJI}") == 4


def test_送れる文が要素列になる() -> None:
    sequence = prepare("CQ DE JA1ABC K", 20.0, max_duration_s=120.0)
    assert sequence.durations.size > 0
    assert sequence.total_seconds > 0.0


def test_送信専用の記号を含む文が要素列になる() -> None:
    """送信専用の記号 (設計書 2026-08-13-tx-only-chars-design.md) は key_server 経由でも拒否されない."""
    sequence = prepare('CQ (TEST) "73" K', 20.0, max_duration_s=120.0)
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


@pytest.mark.parametrize("wpm", [math.nan, math.inf, -math.inf])
def test_有限でないwpmを撥ねる(wpm: object) -> None:
    with pytest.raises(RequestRejected) as caught:
        prepare("CQ K", wpm, max_duration_s=120.0)
    assert caught.value.code == protocol.CODE_BAD_REQUEST


@pytest.mark.parametrize(
    "wpm",
    [
        int("1" + "0" * 330),
        -int("1" + "0" * 330),
    ],
)
def test_巨大な整数のwpmを撥ねる(wpm: object) -> None:
    """float() でオーバーフローする巨大な整数も CODE_BAD_REQUEST で撥ねる。"""
    with pytest.raises(RequestRejected) as caught:
        prepare("CQ K", wpm, max_duration_s=120.0)
    assert caught.value.code == protocol.CODE_BAD_REQUEST


@pytest.mark.parametrize("wpm", [20, 20.0, 20.5])
def test_通常のwpmは通す(wpm: object) -> None:
    """正常な有限 WPM 値は prepare() を通す。"""
    sequence = prepare("CQ K", wpm, max_duration_s=120.0)
    assert sequence.durations.size > 0
    assert sequence.total_seconds > 0.0


def test_dry_runの設定は結線を晒さない() -> None:
    config = ServerConfig()
    assert config.dry_run is True
    assert config.wiring() == {"port": "", "key": "", "ptt": ""}


def test_実結線の設定は結線を返す() -> None:
    config = ServerConfig(serial=SerialKeyConfig(port="COM3"))
    assert config.dry_run is False
    assert config.wiring() == {"port": "COM3", "key": "DTR", "ptt": "RTS"}


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
        peer.sock.sendall("{壊れている\n".encode())
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


def test_打鍵中のsendをbusyで撥ねるがdoneが先に来る() -> None:
    """**``send`` 1 回につき、先に返るのは ``done`` ただ 1 つ。**

    以前は打鍵中の ``send`` にその場で ``error(busy)`` を返しており、1 回の
    ``send`` に対して ``error`` → ``done`` の 2 通が返っていた。アプリ
    (``NetKeyClient.send``) は 1 通目の ``error`` を ``NetKeyRejected``
    (「**打鍵されていない**」) として投げて待つのをやめるため、後から届く
    ``done`` が次の要求の応答として読まれ、以後のやり取りが 1 つずれる。

    busy 自体は返す (2 本目の要求への応答として正しい) が、**順序は要求の順**
    である — 先に出した ``send`` の ``done`` が先、次に 2 本目への ``busy``。
    """
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("CQ CQ CQ DE JA1ABC JA1ABC K", 12.0))   # 長め
        peer.send(protocol.send("E", 40.0))
        first = peer.recv(timeout=60.0)
        assert first["type"] == "done"                                  # 1 本目は完走する
        second = peer.recv(timeout=10.0)
        assert second["type"] == "error"
        assert second["code"] == protocol.CODE_BUSY                     # 2 本目は撥ねる
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_打鍵中のcheckもbusyで撥ねるがdoneが先に来る() -> None:
    """符号化が打鍵の時間刻みを邪魔しないようにするため (順序は上と同じ理由)."""
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("CQ CQ CQ DE JA1ABC JA1ABC K", 12.0))
        peer.send(protocol.check("E", 40.0))
        first = peer.recv(timeout=60.0)
        assert first["type"] == "done"
        second = peer.recv(timeout=10.0)
        assert second["code"] == protocol.CODE_BUSY
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_打鍵中の壊れた行への応答もdoneの後に回る() -> None:
    """壊れた行だけ別扱いにしない (``_next_message`` の経路も同じ不変条件)."""
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("CQ CQ CQ DE JA1ABC JA1ABC K", 12.0))
        peer.sock.sendall("{壊れている\n".encode())
        first = peer.recv(timeout=60.0)
        assert first["type"] == "done"
        second = peer.recv(timeout=10.0)
        assert second["code"] == protocol.CODE_BAD_REQUEST
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


def test_打鍵中に切断してもスレッドが残らない() -> None:
    """**線が上げっぱなしにならないこと。** ``done`` を送れなくても、である。

    Task 6 で足す「切断で中止し ``aborted`` が真になる」テストとは観点が違う。
    ここで見るのは「アプリが黙って消えても、セッションのスレッドが残らず、
    電鍵と PTT の線が落ちるか」。
    """
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
        peer.send(protocol.send("CQ CQ CQ DE JA1ABC JA1ABC K", 12.0))   # 長め
        time.sleep(0.3)                        # 打鍵が始まってから切る
        peer.close()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert sink.ended_safely()
    finally:
        with contextlib.suppress(OSError):
            peer.close()
        thread.join(timeout=5.0)


class StuckSink:
    """``key(True)`` で固まる偽の窓口 (USB 切断でシリアルが固まった状況)."""

    def __init__(self, release: threading.Event) -> None:
        self._release = release

    def key(self, on: bool) -> None:
        if on:
            self._release.wait(timeout=30.0)

    def ptt(self, on: bool) -> None:
        pass


def test_打鍵スレッドが止まらなければ応答を必ず返して以後を撥ねる(monkeypatch) -> None:
    """**返事を 1 通も返さないのが最悪。** アプリは ``send`` の中で永久に待つ.

    ``join`` が空振りしたとき、以前は ``_result`` が ``None`` のままで応答が
    1 通も出ず、しかも ``_keying`` だけ下りるので**次の ``send`` が 2 本目の
    打鍵スレッドを起こせた**。
    """
    monkeypatch.setattr(key_server, "JOIN_TIMEOUT_S", 0.2)
    release = threading.Event()
    sink = StuckSink(release)

    app_sock, server_sock = socket.socketpair()
    session = KeySession(
        ServerConfig(lifeline_timeout_s=0.3),
        server_sock,
        sink_opener=lambda config: contextlib.nullcontext(sink),
    )
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    try:
        peer.recv()
        peer.send(protocol.send("E E", 20.0))
        # 拍を打たないので心綱が切れ、中止しようとするが打鍵スレッドは死なない
        answer = peer.recv(timeout=15.0)
        assert answer["type"] == "error"
        assert answer["code"] == protocol.CODE_INTERNAL

        # **以後の要求はすべて撥ねる。** 同じポートを二度と触らせない
        peer.send(protocol.check("E", 20.0))
        assert peer.recv(timeout=5.0)["code"] == protocol.CODE_INTERNAL
        peer.send(protocol.send("E", 20.0))
        assert peer.recv(timeout=5.0)["code"] == protocol.CODE_INTERNAL

        # **打鍵中扱いのまま。** serve() が次の接続に busy を返し続けるため
        assert session.is_keying is True
    finally:
        release.set()
        peer.close()
        thread.join(timeout=5.0)


def test_想定外の例外はserialと混ぜずトレースを残す(capsys) -> None:
    """バグを「COM の問題」に化けさせない (運用者が結線を疑い続けることになる)."""

    class ExplodingSink:
        def key(self, on: bool) -> None:
            raise ZeroDivisionError("打鍵側のバグ")

        def ptt(self, on: bool) -> None:
            pass

    app_sock, server_sock = socket.socketpair()
    session = KeySession(
        ServerConfig(),
        server_sock,
        sink_opener=lambda config: contextlib.nullcontext(ExplodingSink()),
    )
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    peer = Peer(app_sock)
    try:
        peer.recv()
        peer.send(protocol.send("E", 20.0))
        answer = peer.recv(timeout=10.0)
        assert answer["type"] == "error"
        assert answer["code"] == protocol.CODE_INTERNAL
        assert answer["code"] != protocol.CODE_SERIAL
        assert "ZeroDivisionError" in answer["message"]
    finally:
        peer.close()
        thread.join(timeout=5.0)
    assert "ZeroDivisionError" in capsys.readouterr().err       # トレースが残る


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
            except TimeoutError:
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


# ---- レビュー対応: 「send 1 回につき done は 1 つだけ」の不変条件 ----
#
# 当初、心綱切れの理由を独立した error メッセージとして done の前 (brief 原案)
# または後 (最初の実装) に追送していたが、いずれも「send 1 回につき応答は
# done 1 通だけ」という前提でループするクライアントと噛み合わず、以後の
# やり取りが恒久的に 1 つずつずれることが分かった (コーディネーターのレビュー
# で実測)。理由は done 自身の reason 欄に載せることで解決した。
# 以下はその不変条件を固定するテスト。


def test_心綱で中止した後も次の要求に正しく応答する() -> None:
    peer, thread = run_session(ServerConfig(lifeline_timeout_s=0.3))
    try:
        peer.recv()
        peer.send(protocol.send(_長い文(), 12.0))
        # 拍を送らない。切断もしない (ケーブルが抜けた状態の模擬)
        done = peer.recv(timeout=10.0)
        assert done["type"] == "done"
        assert done["aborted"] is True
        peer.send(protocol.check("CQ K", 20.0))
        checked = peer.recv(timeout=10.0)
        assert checked["type"] == "checked"          # error ではない
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_stopで中止した後も次の要求に正しく応答する() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send(_長い文(), 12.0))
        time.sleep(0.3)
        peer.send(protocol.stop())
        done = peer.recv(timeout=10.0)
        assert done["type"] == "done"
        assert done["aborted"] is True
        peer.send(protocol.check("CQ K", 20.0))
        checked = peer.recv(timeout=10.0)
        assert checked["type"] == "checked"
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_正常完了の後も次の要求に正しく応答する() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("E E", 40.0))
        done = peer.recv(timeout=10.0)
        assert done["type"] == "done"
        assert done["aborted"] is False
        peer.send(protocol.check("CQ K", 20.0))
        checked = peer.recv(timeout=10.0)
        assert checked["type"] == "checked"
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_心綱の中止は理由が分かる() -> None:
    peer, thread = run_session(ServerConfig(lifeline_timeout_s=0.3))
    try:
        peer.recv()
        peer.send(protocol.send(_長い文(), 12.0))
        done = peer.recv(timeout=10.0)
        assert done["aborted"] is True
        assert done["reason"] == "lifeline"
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_stopの中止は理由が分かる() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send(_長い文(), 12.0))
        time.sleep(0.3)
        peer.send(protocol.stop())
        done = peer.recv(timeout=10.0)
        assert done["aborted"] is True
        assert done["reason"] == "stop"
    finally:
        peer.close()
        thread.join(timeout=5.0)


def test_正常完了には理由が付かない() -> None:
    peer, thread = run_session(ServerConfig())
    try:
        peer.recv()
        peer.send(protocol.send("E E", 40.0))
        done = peer.recv(timeout=10.0)
        assert done["aborted"] is False
        assert done["reason"] is None
    finally:
        peer.close()
        thread.join(timeout=5.0)

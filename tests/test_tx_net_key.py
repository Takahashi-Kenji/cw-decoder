"""アプリ側クライアントのテスト.

**本物の打鍵サーバを ``--dry-run`` 相当で起こして繋ぐ。** 偽物にすると
拍や切断の扱いを取り逃がす。

``reason`` (``done`` に載る中止理由) は Task 6 でサーバ側に追加された。
「送信を待つあいだに拍が途絶えたら lifeline、stop で止めたら stop、正常完了
なら None」を確かめる。**不変条件: ``send`` 1 回につきサーバが返すのは
``done`` ただ 1 つ**であり、クライアント側も応答を取りこぼさない・
食い違わせない作りであることを、想定外の応答を寄越す生ソケット相手の
テストで確かめる。
"""
from __future__ import annotations

import contextlib
import socket
import threading

import pytest

from src.tx import protocol
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


def test_connectがbusyのerrorを受け取るとNetKeyRejectedになる() -> None:
    """欠陥1 の再現・回帰防止.

    ``check``/``send`` は応答が ``error`` のとき ``_raise_if_error`` で
    ``code``/``message`` を ``NetKeyRejected`` として拾うが、以前の
    ``connect()`` にはこの経路が無く、``busy`` で撥ねられても
    「名乗りが来ませんでした: 'error'」という汎用文になっていた。
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def responder() -> None:
        conn, _ = server.accept()
        conn.sendall(
            protocol.encode_message(
                protocol.error(protocol.CODE_BUSY, "別のアプリが繋がっています")
            )
        )
        conn.close()

    threading.Thread(target=responder, daemon=True).start()
    client = NetKeyClient(host, port)
    try:
        with pytest.raises(NetKeyRejected) as caught:
            client.connect()
        assert caught.value.code == "busy"
        assert "別のアプリ" in caught.value.args[0]
        # 撥ねられたときもソケットを閉じ、is_connected が偽になること
        assert client.is_connected is False
    finally:
        client.close()
        server.close()


def test_connectで想定外のerror以外のtypeはNetKeyErrorのまま() -> None:
    """``error`` でもない想定外の ``type`` は、これまでどおり汎用の NetKeyError."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def responder() -> None:
        conn, _ = server.accept()
        conn.sendall(b'{"type":"surprise"}\n')
        conn.close()

    threading.Thread(target=responder, daemon=True).start()
    client = NetKeyClient(host, port)
    try:
        with pytest.raises(NetKeyError) as caught:
            client.connect()
        assert not isinstance(caught.value, NetKeyRejected)
        assert client.is_connected is False
    finally:
        client.close()
        server.close()


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
        # 正常完了なら reason は必ず None (Task 6 で追加された done.reason)
        assert result.reason is None
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
        assert result.reason is None
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
        assert result.reason == "stop"
    finally:
        client.close()
        server.close()


def test_心綱で中止すると理由がlifelineになる() -> None:
    """**``ping_interval_s`` を打鍵側の期限より長くしても壊れないか。**

    クライアントの拍が打鍵側の心綱の期限 (0.2 秒) に間に合わないほど疎に
    設定し、打鍵側が自分の判断で中止することを確かめる。``done`` はデータが
    届き次第すぐ読めるので、``ping_interval_s`` を長くしても待ちすぎには
    ならない (壊れない) ことも合わせて確認する。
    """
    host, port, server = start_server(ServerConfig(lifeline_timeout_s=0.2))
    client = NetKeyClient(host, port, ping_interval_s=2.0)
    try:
        client.connect()
        result = client.send("CQ CQ CQ DE JA1ABC JA1ABC PSE K", 5.0)
        assert result.aborted is True
        assert result.reason == "lifeline"
    finally:
        client.close()
        server.close()


def test_送信中に別スレッドからcloseしてもNetKeyErrorになる() -> None:
    """**``close()`` を別スレッドから呼んだとき、受信中のスレッドで何が起きるか。**

    生の ``OSError`` が漏れて呼び出し側の ``except NetKeyError`` を素通り
    しないことを確かめる (ソケットが閉じた直後の ``settimeout``/``recv`` は
    ``OSError`` を投げうる)。
    """
    host, port, server = start_server()
    client = NetKeyClient(host, port)
    try:
        client.connect()
        closer = threading.Timer(0.3, client.close)
        closer.start()
        try:
            with pytest.raises(NetKeyError):
                client.send("CQ CQ CQ DE JA1ABC JA1ABC PSE K", 5.0)
        finally:
            closer.cancel()
    finally:
        with contextlib.suppress(Exception):
            client.close()
        server.close()


def test_checkで想定外のtypeを受け取ると接続を閉じる() -> None:
    """**サーバが期待と違う ``type`` を返したとき、読み取り位置がずれないか。**

    本物の打鍵サーバは決してこうしないが、壊れた・悪意ある相手を想定して
    生ソケットで模す。想定外の応答は例外にするだけでなく、以後このずれた
    接続を使い回さないよう閉じることを確かめる。
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def responder() -> None:
        conn, _ = server.accept()
        conn.sendall(protocol.encode_message(protocol.hello("dummy", {}, True)))
        conn.recv(4096)  # check 要求を読み捨てる
        conn.sendall(b'{"type":"surprise"}\n')
        conn.close()

    threading.Thread(target=responder, daemon=True).start()
    client = NetKeyClient(host, port)
    try:
        client.connect()
        with pytest.raises(NetKeyError):
            client.check("CQ K", 20.0)
        assert client.is_connected is False
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

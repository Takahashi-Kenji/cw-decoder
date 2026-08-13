"""端から端まで — 本物の CLI を別プロセスで起こし、本物の TCP で叩く.

**COM ポートは要らない。** ``--dry-run`` があるため、この PC で全経路を通せる。
実機でしか確かめられないのは DTR/RTS の実際の挙動・極性・PTT・実測のずれだけ。

不変条件の検証
--------------
``send`` 1 回につき打鍵側が返すのは ``done`` ただ 1 つ、という不変条件がある
(``src/tx/key_server.py`` の ``KeySession._handle_send`` docstring 参照)。
以前、心綱で中止したとき ``done`` の後に ``error`` が追送され、**次の要求の
応答がずれる**という欠陥が実在した。単体テストでは見つからず、2 回目の要求で
初めて現れる種類の欠陥だったため、ここでは**中止させたあと、同じ接続でもう
一度 ``check`` を送り、``checked`` が正しく返ること**を、心綱・``stop``・
正常完了の 3 経路すべてで確かめる。
"""
from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import threading
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
    """CLI を別プロセスで起こす.

    **待ち受けの確認そのものが 1 本の TCP 接続を作ってしまう。** 「繋がるまで
    試す」ために ``socket.create_connection`` で繋いでは閉じる。以前は
    打鍵側の「同時に扱うのは 1 つ」判定 (``active.is_alive()`` だけ) が
    このプローブ接続の後始末が終わるまでの一瞬 ``busy`` を誤って返す欠陥が
    あり (レビューで実測: 15 回中 4 回再現)、ここに決め打ちの待ち
    (``_ACCEPT_SETTLE_S``) を入れて回避していた。**欠陥は
    ``scripts/cw_key_server.py`` の ``serve()``/``_peer_already_disconnected``
    側で直したので、この待ちはもう要らない** (外して安定することをテストで
    確認済み)。
    """
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
        # 正常完了なら reason は必ず None (aborted のときだけ意味を持つ)
        assert result.reason is None

        # **不変条件の検証 (正常完了経路)**: done を受け取ったあと、同じ接続で
        # もう一度 check を送り、checked が正しく返ること。
        checked_again = client.check("E", 40.0)
        assert checked_again.seconds > 0.0
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
        assert result.reason == "lifeline"

        # **不変条件の検証 (心綱経路)**: 中止させたあと、同じ接続でもう一度
        # check を送り、checked が正しく返ること。以前は done の後に error が
        # 追送され、次の要求の応答が恒久的に 1 つずつずれる欠陥があった。
        checked = client.check("E", 40.0)
        assert checked.seconds > 0.0
    finally:
        client.close()


def test_stopで止めると打鍵側が中止する(打鍵側) -> None:
    """**運用者が [中止] を押した状況の模擬。** 別スレッドから ``stop()`` を呼ぶ."""
    host, port = 打鍵側
    client = NetKeyClient(host, port)
    results: list[object] = []
    try:
        client.connect()

        def 送る() -> None:
            results.append(client.send("CQ CQ CQ DE JA1ABC PSE K", 12.0))

        sender = threading.Thread(target=送る, daemon=True)
        sender.start()
        time.sleep(0.3)                       # 少し打たせてから止める
        client.stop()
        sender.join(timeout=10.0)

        assert len(results) == 1
        result = results[0]
        assert result.aborted is True
        assert result.reason == "stop"

        # **不変条件の検証 (stop 経路)**: 中止させたあと、同じ接続でもう一度
        # check を送り、checked が正しく返ること。
        checked = client.check("E", 40.0)
        assert checked.seconds > 0.0
    finally:
        client.close()


def test_2つ目の接続はbusyで断られる(打鍵側) -> None:
    """**同時に扱うのは 1 つ。** 二重打鍵しないため (設計書 §7.5).

    以前は ``connect()`` が ``hello`` 以外の ``type`` を一律
    ``名乗りが来ませんでした: 'error'`` という汎用の ``NetKeyError`` にして
    しまい、``busy`` の理由が握りつぶされていた (欠陥 1)。``check``/``send``
    と同じく ``_raise_if_error`` を通すようにしたので、2 本目の接続は
    ``NetKeyRejected`` で ``code == "busy"`` になることを確かめる。
    """
    host, port = 打鍵側
    first = NetKeyClient(host, port)
    first.connect()
    try:
        second = NetKeyClient(host, port)
        with pytest.raises(NetKeyRejected) as caught:
            second.connect()
        assert caught.value.code == "busy"
        assert second.is_connected is False
        second.close()
        assert first.check("E", 40.0).seconds > 0.0     # 先の接続は無事
    finally:
        first.close()


def test_送信中に切ると打鍵側が次の接続を受けられる(打鍵側) -> None:
    """切断で中止し、**サーバは生き残る**."""
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

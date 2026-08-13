"""打鍵 CLI のテスト.

**実機は使わない。** ``--dry-run`` と偽の窓口で代える。
"""
from __future__ import annotations

import contextlib
import importlib.util
import socket
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from src.tx import protocol as cli_protocol
from src.tx.key_server import ServerConfig
from src.tx.keyer import RecordingKeySink
from src.tx.net_key import NetKeyClient, NetKeyRejected
from src.tx.serial_key import SerialKeyConfig

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


# ---- ここから先は brief に無い追加テスト ----
#
# 「自分で考えてほしいこと」で指摘された、argparse の type=float が
# "nan" / "inf" を素通ししてしまう問題への対応を確かめる。
# --max-duration-s に nan が通ると `total_seconds > max_duration_s` が常に
# 偽になり長さの上限が、--key-watchdog-s に nan が通ると `duration >
# key_watchdog_s` が常に偽になり番犬が、--lifeline-timeout に nan が通ると
# 心綱の期限判定が、それぞれ無効化される。運用者の打ち間違いで安全装置が
# 黙って効かなくなるのは避けたい。


@pytest.mark.parametrize(
    "flag",
    [
        "--max-duration-s",
        "--lifeline-timeout",
        "--ptt-lead-ms",
        "--ptt-tail-ms",
        "--key-watchdog-s",
        "--wpm",
    ],
)
@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf"])
def test_安全装置に関わる引数はnanとinfを受け付けない(flag: str, bad_value: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--dry-run", flag, bad_value])


@pytest.mark.parametrize(
    "flag",
    [
        "--max-duration-s",
        "--lifeline-timeout",
        "--key-watchdog-s",
        "--wpm",
    ],
)
def test_安全装置に関わる引数は0以下を受け付けない(flag: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--dry-run", flag, "0"])


def test_ptt先行と後追いは0を受け付ける() -> None:
    """PTT 先行・後追いは「無し」を表す 0 が正当な値である."""
    args = cli.build_parser().parse_args(["--dry-run", "--ptt-lead-ms", "0", "--ptt-tail-ms", "0"])
    config = cli.config_from_args(args)
    assert config.ptt_lead_s == 0.0
    assert config.ptt_tail_s == 0.0


def test_ptt先行と後追いは負を受け付けない() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--dry-run", "--ptt-lead-ms", "-1"])


def test_check_linesとdry_runを同時に渡すと止める(capsys) -> None:
    """``--check-lines`` は結線確認モードであり、実際の結線が要る。

    ``--dry-run`` はシリアルに一切触らないので、両方を渡されたら
    実行せずに止める (黙って dry-run 扱いで何かを表示したりしない)。
    """
    assert cli.main(["--dry-run", "--check-lines"]) == 1
    assert "check-lines" in capsys.readouterr().err


def test_busy応答の送信に失敗しても接続を必ず閉じる(monkeypatch) -> None:
    """serve() の busy 応答経路. sendall が失敗しても close() は必ず呼ぶ.

    (元の brief 実装は ``with contextlib.suppress(OSError): sendall(); close()``
    という書き方で、``sendall`` が例外を出すと同じ ``with`` ブロック内の
    ``close()`` まで実行が届かない隙間があった。busy で撥ねた接続を
    閉じ損ねると、その分の fd が接続のたびに漏れ続ける。)
    """

    events: list[str] = []

    class _FakeConn:
        def sendall(self, data: bytes) -> None:
            events.append("send")
            raise OSError("送信失敗")

        def close(self) -> None:
            events.append("close")

        def setsockopt(self, *args: object) -> None:
            pass  # 通常経路 (1 回目) で呼ばれる。busy 経路では呼ばれない

    class _AlwaysAliveThread:
        """常に is_alive() が真を返す偽スレッド. 2 回目の接続を busy 経路へ導く."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def start(self) -> None:
            pass

    class _FakeServer:
        def __init__(self) -> None:
            self._calls = 0

        def settimeout(self, seconds: float) -> None:
            pass

        def accept(self):
            self._calls += 1
            if self._calls <= 2:
                return _FakeConn(), ("127.0.0.1", self._calls)
            raise KeyboardInterrupt   # ループを止めるための番人

    monkeypatch.setattr(cli.threading, "Thread", _AlwaysAliveThread)

    with pytest.raises(KeyboardInterrupt):
        cli.serve(ServerConfig(), _FakeServer())

    # 1 回目は「生きているセッション無し」なので通常経路 (busy ではない)。
    # 2 回目が busy 経路を通り、sendall が失敗しても close() が呼ばれる。
    assert events == ["send", "close"]


def test_打鍵中なら相手が切れていても次の接続をbusyで撥ねる() -> None:
    """**二重打鍵しないのを OS ではなくコードで担保する** (設計 §7.5).

    アプリが打鍵中に強制終了して運用者がすぐ繋ぎ直すと、古いセッションがまだ
    ``Keyer.send`` の中にいるのに新しい接続が受理され、2 本の打鍵スレッドが
    同じ ``KeySink`` を叩いていた (レビューで 3/3 再現)。実機では COM の
    二重オープンを OS が弾くが、それは**コードが担保していない**ということ。

    ここでは ``key(True)`` で固まる窓口を差し、「確実に打鍵中」の状態を作って
    から 1 本目を強制切断し、直後の接続が ``busy`` になることを見る。
    """
    gate = threading.Event()

    class HeldSink:
        """最初の ON で止まる窓口 (打鍵中の状態を確実に作るため)."""

        def key(self, on: bool) -> None:
            if on:
                gate.wait(timeout=30.0)

        def ptt(self, on: bool) -> None:
            pass

    sessions: list[object] = []
    real_session = cli.KeySession

    def factory(config, conn, **kwargs):
        # serve() が渡す sink_opener は、ここで差す固まる窓口で上書きする
        # (打鍵中の状態を確実に作るのがこのテストの目的)。
        kwargs["sink_opener"] = lambda _config: contextlib.nullcontext(HeldSink())
        session = real_session(config, conn, **kwargs)
        sessions.append(session)
        return session

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()
    stop_event = threading.Event()
    serve_thread = threading.Thread(
        target=cli.serve, args=(ServerConfig(), server), kwargs={"stop_event": stop_event}, daemon=True
    )
    try:
        cli.KeySession = factory                      # type: ignore[assignment]
        serve_thread.start()

        first = NetKeyClient(host, port)
        first.connect()
        first._send(cli_protocol.send("E E", 20.0))   # 打鍵させる (応答は待たない)
        deadline = time.monotonic() + 5.0
        while not (sessions and sessions[0].is_keying):   # type: ignore[attr-defined]
            assert time.monotonic() < deadline, "打鍵が始まらない"
            time.sleep(0.01)

        first.close()                                 # アプリが落ちたのと同じ

        second = NetKeyClient(host, port)
        try:
            with pytest.raises(NetKeyRejected) as caught:
                second.connect()
            assert caught.value.code == "busy"
        finally:
            second.close()
    finally:
        cli.KeySession = real_session                 # type: ignore[assignment]
        gate.set()
        stop_event.set()
        serve_thread.join(timeout=10.0)
        with contextlib.suppress(OSError):
            server.close()


def test_繋いですぐ閉じても直後の接続がbusyにならない() -> None:
    """欠陥2 の再現・回帰防止 (busy 判定のレース).

    ``serve()`` は 2 つ目の接続を断るかどうかを ``active.is_alive()`` だけで
    判定していた。相手が切断してもセッションのスレッドが実際に終わる
    (相手の切断を検知しきる) までのわずかな間、``busy`` を返し続けてしまう。

    ここでは「繋いですぐ (``hello`` を読まずに) 閉じる」→「直後にもう一度
    繋ぐ」を繰り返し、後者が毎回 ``busy`` にならず ``hello`` を受け取れる
    ことを確かめる。``hello`` を読まずに閉じると、送信側 (打鍵側) の受信
    バッファに未読データが残った状態での ``close()`` になり TCP が RST を
    送るため、レビューで実測した状況 (15 回中 4 回誤 busy) をそのまま再現
    できる。最低 20 回繰り返して安定することを確認する (要求どおり)。
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    # 強制的にソケットを閉じて accept() に例外を出させる止め方だと、例外の
    # 種類がプラットフォーム依存になり (このリポジトリの開発機では Windows の
    # WinError 10038) スレッドの外に漏れて警告になる。stop_event で次の
    # accept() のタイムアウト (最大 1 秒) にきれいに抜けてもらう。
    stop_event = threading.Event()
    serve_thread = threading.Thread(
        target=cli.serve, args=(ServerConfig(), server), kwargs={"stop_event": stop_event}, daemon=True
    )
    serve_thread.start()
    try:
        for i in range(20):
            # 繋いですぐ、hello を読まずに閉じる (RST を誘発する)
            with socket.create_connection((host, port), timeout=1.0):
                pass

            client = NetKeyClient(host, port)
            try:
                hello = client.connect()          # busy になってはいけない
                assert hello.dry_run is True, f"{i} 回目で失敗"
            finally:
                client.close()
    finally:
        stop_event.set()
        serve_thread.join(timeout=5.0)
        with contextlib.suppress(OSError):
            server.close()


class TestSharedSink:
    """ポートを開きっぱなしにする.

    **送信のたびに開き直してはいけない。** 変換器によっては開くたびに再認識が
    走り、実測で約 7 秒 線が命令に反応しない。運用者の環境では 6.3 秒の電文の
    うち最後の 1.3 秒しか電波にならなかった (2026-08-11)。
    """

    def _fake_serial(self, monkeypatch):
        """開かれた回数と書かれた線を記録する偽の serial モジュール."""
        記録 = {"opens": 0, "closes": 0, "lines": []}

        class 偽Serial:
            def __init__(self, *a, **kw):
                記録["opens"] += 1

            def __setattr__(self, name, value):
                if name in ("dtr", "rts"):
                    記録["lines"].append((name, value))
                object.__setattr__(self, name, value)

            def close(self):
                記録["closes"] += 1

        mod = types.ModuleType("serial")
        mod.Serial = 偽Serial
        monkeypatch.setitem(sys.modules, "serial", mod)
        return 記録

    def test_送信を繰り返してもポートは一度しか開かない(self, monkeypatch) -> None:
        記録 = self._fake_serial(monkeypatch)
        config = ServerConfig(serial=SerialKeyConfig(port="COM_TEST"))
        with cli.shared_sink(config, settle_s=0.0) as opener:
            for _ in range(3):
                with opener(config) as sink:
                    sink.key(True)
                    sink.key(False)
        assert 記録["opens"] == 1

    def test_送信ごとに閉じない(self, monkeypatch) -> None:
        記録 = self._fake_serial(monkeypatch)
        config = ServerConfig(serial=SerialKeyConfig(port="COM_TEST"))
        with cli.shared_sink(config, settle_s=0.0) as opener:
            with opener(config) as sink:
                sink.key(True)
            assert 記録["closes"] == 0, "送信の終わりに閉じてはいけない"
        assert 記録["closes"] == 1, "セッションの終わりには閉じること"

    def test_送信の終わりに両線が落ちる(self, monkeypatch) -> None:
        """閉じなくても、上げっぱなしにしない約束は守る."""
        記録 = self._fake_serial(monkeypatch)
        config = ServerConfig(serial=SerialKeyConfig(port="COM_TEST"))
        with cli.shared_sink(config, settle_s=0.0) as opener:
            with opener(config) as sink:
                sink.key(True)
                sink.ptt(True)
            末尾 = 記録["lines"][-2:]
            assert ("dtr", False) in 末尾 and ("rts", False) in 末尾

    def test_dry_runはシリアルに触らない(self, monkeypatch) -> None:
        記録 = self._fake_serial(monkeypatch)
        with cli.shared_sink(ServerConfig(), settle_s=0.0) as opener, opener(ServerConfig()) as sink:
            sink.key(True)
            sink.key(False)
        assert 記録["opens"] == 0

    def test_待ち時間は設定できる(self) -> None:
        args = cli.build_parser().parse_args(["--dry-run", "--open-settle-s", "0"])
        assert args.open_settle_s == 0.0
        assert cli.build_parser().parse_args(["--dry-run"]).open_settle_s == cli.DEFAULT_OPEN_SETTLE_S

"""アプリ側 (GPU PC) から打鍵側 (無線機 PC) を叩くクライアント.

**この PC には COM ポートが無い。** 打鍵は無線機を繋いだ PC で行う。
ここは確定したテキストを渡し、拍を打ち、``stop`` を送るだけ。

打鍵側は :file:`scripts/cw_key_server.py`。

繋ぎ直しの方針
--------------
**待機中は呼び出し側が繋ぎ直してよい。** 打鍵側を後から起こしても繋がる。

**送信中は繋ぎ直さない。** 切れたら中止として扱う。繋ぎ直して続きから打つような
ことはしない — 何がどこまで出たのか分からない状態で電波を出さないため。

応答を取りこぼさない・食い違わせない
------------------------------------
打鍵側が ``done`` の前後に別の ``error`` を送る欠陥が 2 度見つかっている
(中止の理由の追送と、打鍵中に届いた要求への即時 ``busy``)。どちらも以後の
要求への応答が恒久的に 1 つずつずれる。打鍵側はこれを直し、**``send`` 1 回に
つき返すのは ``done`` ただ 1 つ**という不変条件を持つ (打鍵中に届いた要求への
応答は ``done`` の後に回る。``key_server.py`` の ``_reply``)。

クライアント側もこれを鵜呑みにしない。``check``/``send`` の応答として
想定外の ``type`` が来たら、以後この接続をずれたまま使い回さないよう
即座に閉じる (:meth:`NetKeyClient.close`)。呼び出し側は
:class:`NetKeyError` を受けたら繋ぎ直す (待機中のみ許される、上記参照)。
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
    """打鍵の実測.

    Attributes:
        reason: 中止した理由 (``"lifeline"`` / ``"stop"``)。**``aborted`` が
            真のときだけ意味を持つ。** 正常完了なら常に ``None``。
            打鍵側の ``done`` メッセージの ``reason`` 欄をそのまま写す。
    """

    elements_sent: int
    aborted: bool
    watchdog_tripped: bool
    seconds: float
    max_error_ms: float
    mean_error_ms: float
    reason: str | None = None


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
            NetKeyRejected: 打鍵側が撥ねた (``busy`` 等。``code``/``message`` が
                届く。``check``/``send`` と同じ ``_raise_if_error`` を通す)。
            NetKeyError: 繋がらない、``error`` でもない想定外の応答だった.
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
            # **後始末を先に。** _raise_if_error / 次の NetKeyError のどちらで
            # 抜けても、この接続はもう使い回さないので必ず閉じておく。
            self.close()
            self._raise_if_error(message)   # busy 等なら NetKeyRejected で抜ける
            raise NetKeyError(f"名乗りが来ませんでした: {message.get('type')!r}")
        return Hello(
            protocol_version=int(message.get("protocol", 0)),
            tokens_fingerprint=str(message.get("tokens_fingerprint", "")),
            wiring=dict(message.get("wiring") or {}),
            dry_run=bool(message.get("dry_run")),
        )

    def close(self) -> None:
        """閉じる. 二重呼び出しは無害. **別スレッドから呼んでよい。**

        受信中のスレッドがちょうど ``recv``/``settimeout`` の最中でも、
        ``_recv`` 側がすべての段階を ``OSError`` として捕まえるので、
        生の ``OSError`` が呼び出し側に漏れることはない (:meth:`_recv` 参照)。
        """
        with self._write_lock:
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
            NetKeyError: 切れた、想定外の応答だった (このとき接続は閉じる).
        """
        self._send(protocol.check(text, wpm))
        message = self._recv(timeout=self.connect_timeout_s)
        self._raise_if_error(message)
        if message.get("type") != "checked":
            self.close()
            raise NetKeyError(f"想定外の応答です: {message.get('type')!r}")
        return CheckResult(
            chars=int(message["chars"]),
            elements=int(message["elements"]),
            seconds=float(message["seconds"]),
        )

    def send(self, text: str, wpm: float) -> SendResult:
        """打鍵させ、終わるまで待つ. **待つあいだ拍を打ち続ける。**

        Raises:
            NetKeyRejected: 打鍵側が撥ねた (打鍵されていない。まだ打鍵に
                入っていない要求の即時撥ね — 不正な文字・長さ超過等).
            NetKeyError: 途中で切れた、想定外の応答だった (このとき接続は閉じる).

        **不変条件: ``send`` 1 回につき打鍵側が返すのは ``done`` ただ 1 つ**
        (即時撥ねの場合を除く)。中止した理由 (拍の途絶・``stop``) は
        ``done.reason`` に載る。``error`` を単独で待ち受ける経路は無い。
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
                    reason=message.get("reason"),
                )
            self.close()
            raise NetKeyError(f"想定外の応答です: {message.get('type')!r}")

    def stop(self) -> None:
        """中止させる. **別のスレッドから呼んでよい** (画面の [中止] ボタン).

        ``send`` の受信ループとソケット書き込みが衝突しないよう、
        書き込みは :attr:`_write_lock` で直列化している (:meth:`_send` 参照)。
        応答は待たない — 中止の結果は進行中の ``send`` が返す ``done`` の
        ``reason`` に ``"stop"`` として載る。
        """
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

        **別スレッドから ``close()`` されたときの安全策。** ループの毎回
        ``self._sock`` を読み直し (ループ突入前に一度だけ束縛した古い
        ``sock`` を使い続けない)、``settimeout`` も ``recv`` と同じ
        ``try`` に入れている。閉じた直後のソケットに対する ``settimeout``
        自体が ``OSError`` (Bad file descriptor 等) を投げうるため、
        ``try`` の外に置くと素の ``OSError`` が呼び出し側に漏れてしまう。
        """
        while not self._pending:
            sock = self._sock
            if sock is None:
                raise NetKeyError("繋がっていません")
            try:
                sock.settimeout(timeout)
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

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

import contextlib
import math
import re
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from src.tokens.morse_tokens import SPECIAL_INPUT_MARKERS
from src.tx import protocol
from src.tx.encoder import ElementSequence, build_sequence, find_unsendable
from src.tx.fingerprint import tokens_fingerprint
from src.tx.keyer import Keyer, KeyingReport, KeySink, RecordingKeySink
from src.tx.serial_key import SerialKeyConfig, SerialKeyError, SerialKeySink

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
    if isinstance(wpm, bool) or not isinstance(wpm, (int, float)):
        raise RequestRejected(protocol.CODE_BAD_REQUEST, f"wpm が不正です: {wpm!r}")

    # float 変換を試す。巨大な整数でオーバーフロー例外が出ることがある
    try:
        wpm_value = float(wpm)
    except (OverflowError, ValueError):
        raise RequestRejected(protocol.CODE_BAD_REQUEST, f"wpm が不正です: {wpm!r}") from None

    if not math.isfinite(wpm_value) or wpm_value <= 0:
        raise RequestRejected(protocol.CODE_BAD_REQUEST, f"wpm が不正です: {wpm!r}")

    bad = find_unsendable(text)
    if bad:
        raise RequestRejected(
            protocol.CODE_UNSENDABLE,
            "送信できない文字が含まれています: " + "".join(b.char for b in bad),
            unsendable=[{"index": b.index, "char": b.char} for b in bad],
        )

    sequence = build_sequence(text, wpm_value)
    if sequence.total_seconds > max_duration_s:
        raise RequestRejected(
            protocol.CODE_TOO_LONG,
            f"長すぎます ({sequence.total_seconds:.1f} 秒 > {max_duration_s:.1f} 秒)",
        )
    return sequence


# 読み待ちの刻み。心綱の期限を見張るため、待ちっぱなしにしない。
POLL_INTERVAL_S = 0.1

# 待機中を表す番人。切断 (None) と区別する。
_IDLE = object()

# 打鍵中に持ち越せる応答の数の上限 (``_reply``)。
_MAX_DEFERRED = 32

# 打鍵スレッドの停止を待つ上限 (秒)。これを超えたらセッションを毒とみなす
# (:class:`KeySession` の docstring)。**テストから短くできるよう定数にしてある。**
JOIN_TIMEOUT_S = 5.0

SinkOpener = Callable[["ServerConfig"], AbstractContextManager[KeySink]]


def open_sink(config: ServerConfig) -> AbstractContextManager[KeySink]:
    """送信 1 回分の窓口を開く.

    **dry-run ではシリアルに一切触らない。** COM ポートの無い PC でも
    LAN の経路を端から端まで試験できるようにするため。

    **実結線でこれを毎回呼んではいけない。** 変換器によっては開くたびに
    再認識が走り、実測で約 7 秒 線が命令に反応しなくなる (2026-08-11、
    運用者の環境で確認)。その間の打鍵は電波にならず、6.3 秒の電文が
    まるごと消えた。実結線では :func:`persistent_opener` を使い、
    ポートを 1 回だけ開いて保つこと。
    """
    if config.serial is None:
        return contextlib.nullcontext(RecordingKeySink())
    return SerialKeySink(config.serial)


class _PersistentSink:
    """開きっぱなしの窓口を、送信 1 回分の顔で配るための包み.

    ``__exit__`` で**閉じない**。閉じて開き直すと変換器の再認識が走り、
    次の送信の頭が消えるため (:func:`open_sink` の docstring 参照)。

    **閉じないが、必ず両線は落とす。** 上げっぱなしにしないという約束は
    ポートを保つかどうかとは無関係に守る。
    """

    def __init__(self, sink: KeySink) -> None:
        self._sink = sink

    def __enter__(self) -> KeySink:
        return self._sink

    def __exit__(self, *exc: object) -> None:
        # 落とす操作そのものが失敗しても (USB を抜かれた等)、後片付けを
        # 止めない。``serial_key.py`` の ``_all_lines_off`` と同じ約束。
        with contextlib.suppress(Exception):
            self._sink.key(False)
        with contextlib.suppress(Exception):
            self._sink.ptt(False)


def persistent_opener(sink: KeySink) -> SinkOpener:
    """既に開いている窓口を配る opener を作る.

    ポートを開くのは呼び出し側の責任。ここは配るだけで、開きも閉じもしない。
    """
    holder = _PersistentSink(sink)
    return lambda config: holder


class KeySession:
    """1 接続分の処理.

    **同時に扱う接続は 1 つ。** 二重打鍵しないため。

    打鍵スレッドが死ななかったとき
    ------------------------------
    打鍵スレッドが ``thread.join`` のタイムアウトを超えて生き残ることがある
    (例: USB 切断で ``SerialKeySink`` が固まる。``serial_key.py`` 参照)。
    このときセッションを**毒**とみなす (:attr:`is_keying` が真のまま):

    * ``send`` には ``error(internal)`` を 1 通返す。**返事を 1 通も返さないと、
      アプリは ``send`` の中で拍を打ち続けたまま永久に止められない**
    * 以後の ``check`` / ``send`` はすべて撥ねる。**同じ物理ポートを二度と
      触らせない**
    * :attr:`is_keying` が真のままなので、:file:`scripts/cw_key_server.py` の
      ``serve()`` は次の接続に ``busy`` を返し続ける (別の ``KeySession`` が
      同じポートを開きに行くのを防ぐ)

    **回復にはこの CLI の再起動が要る。** 物理ポートを掴んだままのスレッドが
    残っている以上、それが正しい。
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
        self._keying = False
        # 打鍵スレッドが join のタイムアウトを超えて生き残った (毒). クラスの
        # docstring 参照。**一度立ったら下ろさない。**
        self._stuck = False
        # 打鍵中に届いた要求への応答。**``done`` の後まで持ち越す** (``_reply``)
        self._deferred: list[dict[str, Any]] = []
        self._abort = threading.Event()
        self._result: dict[str, Any] | None = None
        # 中止の理由 (``"stop"`` / ``"lifeline"`` / ``None``)。``_dispatch`` の
        # stop 処理と ``_pump_while_keying`` の心綱処理が書き込み、``worker``
        # (打鍵スレッド) が ``done`` を組み立てるときに読む。**``done`` に載せる
        # だけで、これ単体を送信することは絶対にしない** (別送すると `send` 1 回
        # につき応答が 2 通になり、以後のやり取りが恒久的に 1 つずつずれる。
        # レビューで実測して判明した)
        self._abort_reason: str | None = None

    # ---- 状態 ----
    @property
    def is_keying(self) -> bool:
        """打鍵中 (または打鍵スレッドが残ったまま) か.

        **別スレッドから読んでよい** (``serve()`` が busy の判定に使う)。
        真偽値の読み書きは GIL の下で不可分であり、ここで見たいのは
        「打鍵しているかもしれないか」という安全側の粗い判定である。

        毒 (:attr:`_stuck`) を含めるのは、**打鍵スレッドが物理ポートを掴んだまま
        残っている間は、次の接続にポートを触らせてはいけない**ため。
        """
        return self._keying or self._stuck

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
                self._reply(protocol.error(protocol.CODE_BAD_REQUEST, str(exc)))
                return _IDLE
        line = self._pending.pop(0)
        try:
            return protocol.decode_message(line)
        except protocol.ProtocolError as exc:
            self._reply(protocol.error(protocol.CODE_BAD_REQUEST, str(exc)))
            return _IDLE

    def _send(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):
            self.conn.sendall(protocol.encode_message(message))

    def _reply(self, message: dict[str, Any]) -> None:
        """要求への応答を送る. **打鍵中は ``done`` の後まで持ち越す。**

        **``send`` 1 回につきサーバが返すのは ``done`` ただ 1 つ**という不変条件を
        守るため。打鍵中に届いた ``check``/``send``/壊れた行へその場で ``error``
        を返すと、``done`` を待っているアプリがその ``error`` を ``send`` の応答
        として読んでしまい (``NetKeyClient.send`` は ``NetKeyRejected`` =
        「打鍵されていない」として投げる)、後から届く ``done`` がバッファに
        残って以後の応答が恒久的に 1 つずれる (レビューで実測して判明した)。

        持ち越しても**要求と応答の順序は保たれる** — ``done`` (先の ``send`` の
        応答) の後に、打鍵中に届いた要求への応答が届いた順に並ぶ。
        """
        if not self._keying:
            self._send(message)
            return
        if len(self._deferred) >= _MAX_DEFERRED:
            # 打鍵中に要求を溜め続ける相手 (壊れた/悪意ある実装) に、
            # 無限に記憶を使わせない。まともなアプリはここに届かない
            return
        self._deferred.append(message)

    def _flush_deferred(self) -> None:
        """持ち越した応答を届いた順に送る. **``done`` を送った後で呼ぶ。**"""
        deferred, self._deferred = self._deferred, []
        for message in deferred:
            self._send(message)

    # ---- 割り振り ----
    def _dispatch(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        if kind == "ping":
            return                            # 拍。返事は要らない
        if kind == "stop":
            self._abort_reason = "stop"       # 打鍵していなければ意味を持たない
            self._abort.set()                 # 打鍵していなければ何も起きない
            return
        if self._stuck:
            # **毒。** 打鍵スレッドが物理ポートを掴んだまま残っている
            self._reply(
                protocol.error(
                    protocol.CODE_INTERNAL,
                    "前の打鍵が終わっていません。この接続はもう使えません。"
                    "打鍵側 (cw_key_server.py) を再起動してください",
                )
            )
            return
        if self._keying:
            # **打鍵中は check も撥ねる。** 符号化が時間の刻みを邪魔しないため。
            # 応答は ``done`` の後まで持ち越す (``_reply``)
            self._reply(protocol.error(protocol.CODE_BUSY, "打鍵中です"))
            return
        if kind == "check":
            self._handle_check(message)
            return
        if kind == "send":
            self._handle_send(message)
            return
        self._reply(
            protocol.error(protocol.CODE_BAD_REQUEST, f"知らない type です: {kind!r}")
        )

    def _handle_check(self, message: dict[str, Any]) -> None:
        """**打鍵しない検査。** シリアルには触らない."""
        try:
            sequence = prepare(
                message.get("text"), message.get("wpm"), self.config.max_duration_s
            )
        except RequestRejected as exc:
            self._reply(exc.to_message())
            return
        self._reply(
            protocol.checked(
                chars=visible_length(message["text"]),
                elements=int(sequence.durations.size),
                seconds=sequence.total_seconds,
            )
        )

    # ---- 打鍵 ----
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
            self._reply(exc.to_message())
            return

        self._abort.clear()
        self._result = None
        self._abort_reason = None
        self._keying = True

        def worker() -> None:
            try:
                report = self._key(sequence)
            except SerialKeyError as exc:
                self._result = protocol.error(protocol.CODE_SERIAL, str(exc))
            except Exception as exc:
                # **想定外の例外でも `_result` を必ず埋める。** 前回の結果が
                # 残ったまま `_keying` だけ下りると、古い ``done`` を誤って
                # 送り返すことになる。
                #
                # **`CODE_SERIAL` にしない。** 何でもシリアルの失敗に見せると、
                # ただのバグが「COM の問題」に化けて運用者が結線を疑い続ける。
                # **トレースは必ず残す** — これはバグであり、握り潰すと直せない。
                traceback.print_exc(file=sys.stderr)
                self._result = protocol.error(
                    protocol.CODE_INTERNAL,
                    f"打鍵側で想定外の失敗が起きました: {type(exc).__name__}: {exc}",
                )
            else:
                self._result = protocol.done(
                    elements_sent=report.elements_sent,
                    aborted=report.aborted,
                    watchdog_tripped=report.watchdog_tripped,
                    seconds=sequence.total_seconds,
                    max_error_ms=report.max_error_ms,
                    mean_error_ms=report.mean_error_ms,
                    # **`aborted` が真のときだけ理由を載せる。** 拍の期限が
                    # 切れた瞬間に打鍵が自然完了する稀な競合でも、正常完了の
                    # ``done`` に理由が紛れ込まないようにするため
                    reason=self._abort_reason if report.aborted else None,
                )

        thread = threading.Thread(target=worker, name="cw-keying", daemon=True)
        thread.start()
        self._pump_while_keying(thread)
        self._keying = False
        if self._stuck and self._result is None:
            # **打鍵スレッドが死ななかった。** ここで黙ると、アプリは `send` の
            # 中で拍を打ち続けたまま永久に止められない (応答が 1 通も来ない)。
            # クラスの docstring 参照
            self._result = protocol.error(
                protocol.CODE_INTERNAL,
                "打鍵スレッドが止まりません (シリアルが固まっている可能性があります)。"
                "無線機側で送信を止め、打鍵側 (cw_key_server.py) を再起動してください",
            )
        # **`send` 1 回につき送るメッセージは、ここで送る `done` ただ 1 つ。**
        # 中止の理由は `done` の `reason` 欄に載っているので、別送しない
        # (別送すると、アプリが `done` を読んだ時点で応答を待つのをやめるため、
        # 取り残されたメッセージが次の要求の応答として誤って読まれ、以後の
        # やり取りが恒久的に 1 つずつずれる)
        if self._result is not None:
            self._send(self._result)
        # 打鍵中に届いた要求への応答は、この `done` の後に回してある
        self._flush_deferred()

    def _key(self, sequence: ElementSequence) -> KeyingReport:
        """要素列を叩く. **どの経路で抜けても両線を落とす** (``Keyer`` が担保)."""
        keyer = Keyer(key_watchdog_s=self.config.key_watchdog_s)
        # ``Keyer.send`` (変更禁止) は ``Sequence[float]`` / ``Sequence[bool]`` を
        # 要求するが、``ElementSequence`` は numpy 配列で持つ (合成器側が
        # ベクトル化しているため)。``Keyer.send`` は要素を 1 個ずつ舐めるだけ
        # (ここは numpy ベクトル化の対象ではない) なので、ここで list に変換
        # しても速度上の問題は無い。``tolist()`` は値そのままの Python
        # 組み込み型 (float / bool) に変換するだけで、実行時の打鍵内容は変わらない。
        with self._sink_opener(self.config) as sink:
            return keyer.send(
                sequence.durations.tolist(),
                sequence.is_on.tolist(),
                sink,
                abort=self._abort,
                ptt_lead_s=self.config.ptt_lead_s,
                ptt_tail_s=self.config.ptt_tail_s,
                use_ptt=self.config.serial is None or self.config.serial.uses_ptt,
            )

    def _pump_while_keying(self, thread: threading.Thread) -> None:
        """打鍵が終わるまで、届くメッセージを捌きつつ**心綱を見張る**.

        **ここでソケットを触るのは主スレッドだけ。** 打鍵スレッド (``worker``)
        は ``sink`` (シリアル/記録) にしか触らず、ソケットには触れない。

        中止の経路は 3 つあり、**すべて ``self._abort`` に合流する**:

        1. ``stop`` — 運用者が止めた (``_dispatch`` 側で ``self._abort.set()``)
        2. 切断 — アプリが落ちた
        3. **拍の途絶** — LAN が止まった、アプリが固まった

        3 が要るのは、**ケーブルが抜けても TCP は黙ったままだから**である。
        ソケットが閉じたことはアプリが落ちれば分かるが、線が抜けたときには
        分からない。「止まったら止める」を字義どおり満たすには拍が要る。

        待機中には期限を掛けない (掛けると接続が常に落ちる)。ここは打鍵中だけ。

        期限の更新は ``ping`` だけでなく**届いたメッセージすべて**で行う。
        アプリが生きて話しかけてきている証拠であることに変わりはなく、
        ``check``/``send`` (busy で撥ねられる) や壊れた行であっても同じ。
        拍だけを特別扱いする理由が無い。

        **ここでは ``self._send`` を一切呼ばない。** 中止の理由は
        ``self._abort_reason`` に記録するだけにとどめ、``done`` に載せて
        ``_handle_send`` が 1 通で送る。ここで別のメッセージを送ってしまうと、
        ``send`` 1 回につきサーバが返すメッセージが 2 通になり、クライアント
        (``send`` の応答を 1 通で待ち切る設計) が 1 通目を読んだ時点で待つのを
        やめるため、取り残された 2 通目が次の要求への応答として誤って読まれ、
        以後のやり取りが恒久的に 1 つずつずれる (レビューで実測して判明した)。
        打鍵中に届いた要求への応答も同じ理由で ``done`` の後まで持ち越す
        (``_reply``)。
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
                self._abort_reason = "lifeline"
                self._abort.set()
                break
        # **心綱で抜けたときも、切断で抜けたときも、打鍵スレッドが実際に
        # 止まるまで待つ。** ``Keyer`` は要素の境目でしか ``abort`` を見ない
        # ため、``break`` した時点ではまだ線が上がっている可能性がある。
        # ここで待たずに戻ると、``_result`` が未確定のまま古い値を送ったり、
        # 次の要求を受け付けて二重打鍵したりする恐れがある。
        thread.join(timeout=JOIN_TIMEOUT_S)
        if thread.is_alive():
            # **待っても死ななかった。** ここから先、このセッションは毒である
            # (クラスの docstring 参照)。**黙って `_keying` を下ろしてはいけない** —
            # 下ろすと次の `send` が 2 本目の打鍵スレッドを起こす
            self._stuck = True
            print(
                "打鍵スレッドが止まりません。この接続はもう使いません "
                "(cw_key_server.py の再起動が要ります)。",
                file=sys.stderr,
            )


__all__ = [
    "JOIN_TIMEOUT_S",
    "POLL_INTERVAL_S",
    "KeySession",
    "RequestRejected",
    "ServerConfig",
    "SinkOpener",
    "open_sink",
    "persistent_opener",
    "prepare",
    "visible_length",
]

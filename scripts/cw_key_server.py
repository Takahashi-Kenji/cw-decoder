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
import math
import select
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tx.key_server import (  # noqa: E402
    KeySession,
    RequestRejected,
    ServerConfig,
    SinkOpener,
    open_sink,
    persistent_opener,
    prepare,
)
from src.tx.keyer import Keyer  # noqa: E402
from src.tx.protocol import (  # noqa: E402
    CODE_BUSY,
    DEFAULT_KEY_PORT,
    encode_message,
    error,
)
from src.tx.serial_key import SerialKeyConfig, SerialKeyError, SerialKeySink  # noqa: E402

# 結線確認モードで線を上げている時間 (秒)。テスターの針が振れる程度に長く取る。
CHECK_LINE_HOLD_S = 1.0

# ポートを開いてから線が命令に反応し始めるまでの待ち (秒)。
# **変換器によっては開くたびに再認識が走る。** 運用者の環境で実測 約 7 秒。
# その間の打鍵は電波にならず、6.3 秒の電文がまるごと消えた (2026-08-11)。
# ポートは起動時に 1 回だけ開いて保つので、この待ちは 1 セッションに 1 回。
DEFAULT_OPEN_SETTLE_S = 8.0


def _finite_float(min_value: float, *, inclusive: bool) -> Callable[[str], float]:
    """安全装置に関わる数値引数の ``type=``.

    **``argparse`` の既定の ``type=float`` は ``"nan"`` と ``"inf"`` をそのまま
    通してしまう。** 例えば ``--max-duration-s nan`` を渡すと
    ``sequence.total_seconds > max_duration_s`` が常に偽になり、1 回の送信の
    長さの上限が黙って効かなくなる。``--key-watchdog-s`` も同様に
    ``duration > key_watchdog_s`` が常に偽になり、キー ON が続いてよい上限
    (番犬) が効かなくなる。``--lifeline-timeout`` も心綱の期限判定が効かなく
    なる。運用者の打ち間違いで安全装置が黙って無効になるのは避けたいので、
    ここで有限性と範囲を確かめる。

    Args:
        min_value: 下限。
        inclusive: ``True`` なら ``min_value`` 自身を許す (``>=``)。
            ``False`` なら厳密に超える値だけを許す (``>``)。
    """

    def parser(text: str) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"数値ではありません: {text!r}") from exc
        if not math.isfinite(value):
            raise argparse.ArgumentTypeError(
                f"有限の数値である必要があります (nan/inf は使えません): {text!r}"
            )
        if inclusive:
            if value < min_value:
                raise argparse.ArgumentTypeError(f"{min_value} 以上である必要があります: {text!r}")
        elif value <= min_value:
            raise argparse.ArgumentTypeError(f"{min_value} より大きい必要があります: {text!r}")
        return value

    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LAN 越しに受け取ったテキストを打鍵する (無線機を繋いだ PC で動かす)"
    )
    parser.add_argument("--port", default=None, help="COM ポート名 (--dry-run 時は不要)")
    parser.add_argument("--key-line", default="DTR", choices=["DTR", "RTS"], help="電鍵の線")
    parser.add_argument("--ptt-line", default="RTS", choices=["DTR", "RTS", "NONE"], help="PTT の線")
    parser.add_argument("--key-invert", action="store_true", help="電鍵の極性を反転する")
    parser.add_argument("--ptt-invert", action="store_true", help="PTT の極性を反転する")
    # PTT 先行・後追いは「無し」を表す 0 が正当な値なので下限は inclusive (>= 0)。
    parser.add_argument(
        "--ptt-lead-ms", type=_finite_float(0.0, inclusive=True), default=50.0, help="PTT 先行 (ミリ秒)"
    )
    parser.add_argument(
        "--ptt-tail-ms", type=_finite_float(0.0, inclusive=True), default=100.0, help="PTT 後追い (ミリ秒)"
    )
    # 以下の 4 つは 0 やそれ以下だと安全装置として意味を成さない (0 秒の上限は
    # 「常に超過」、0 秒の番犬は「ON にした瞬間に発動」になる) ので、
    # 厳密に正の値だけを許す (exclusive, > 0)。
    parser.add_argument(
        "--key-watchdog-s",
        type=_finite_float(0.0, inclusive=False),
        default=5.0,
        help="キー ON が続いてよい上限 (秒)",
    )
    parser.add_argument(
        "--max-duration-s",
        type=_finite_float(0.0, inclusive=False),
        default=120.0,
        help="1 回の送信の長さの上限 (秒)",
    )
    parser.add_argument(
        "--lifeline-timeout",
        type=_finite_float(0.0, inclusive=False),
        default=1.0,
        help="拍が途絶えたとみなす時間 (秒)",
    )
    parser.add_argument("--listen", type=int, default=DEFAULT_KEY_PORT, help=f"待ち受けポート (既定 {DEFAULT_KEY_PORT})")
    parser.add_argument("--bind", default="0.0.0.0", help="待ち受けアドレス (既定 0.0.0.0)")
    parser.add_argument("--dry-run", action="store_true", help="シリアルに一切触らない")
    parser.add_argument("--check-lines", action="store_true", help="結線確認モード (無線機の電源を切って使う)")
    parser.add_argument("--text", default=None, help="単発送信モード。LAN を待たない")
    parser.add_argument(
        "--wpm", type=_finite_float(0.0, inclusive=False), default=20.0, help="--text 使用時の速度"
    )
    parser.add_argument(
        "--open-settle-s",
        type=_finite_float(0.0, inclusive=True),
        default=DEFAULT_OPEN_SETTLE_S,
        help=f"ポートを開いてから叩き始めるまでの待ち (秒、既定 {DEFAULT_OPEN_SETTLE_S:.0f})",
    )
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


def send_once(
    config: ServerConfig, text: str, wpm: float, opener: SinkOpener | None = None
) -> int:
    """単発送信モード. LAN を待たず 1 回打って終わる.

    **カナ変換は無い。** カタカナと欧文で書くこと (``{HORE}コンニチハ{RATA}``)。
    中止は Ctrl+C。``Keyer`` の ``finally`` が両線を落とす。

    Args:
        opener: 窓口を配るもの。``None`` なら :func:`open_sink` (送信ごとに開く)。
            **実結線では :func:`shared_sink` が作る opener を渡すこと。**
    """
    # 既定は呼び出し時に引く (モジュール変数の差し替えをテストで効かせるため)
    opener = opener or open_sink
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
        with opener(config) as sink:
            # ``Keyer.send`` (変更禁止) は ``Sequence[float]`` / ``Sequence[bool]``
            # を要求するが、``ElementSequence`` は numpy 配列で持つ (合成器側が
            # ベクトル化しているため)。``tolist()`` は値そのままの Python 組み
            # 込み型に変換するだけで、打鍵内容は変わらない
            # (``src/tx/key_server.py`` の ``KeySession._key`` と同じ理由)。
            report = keyer.send(
                sequence.durations.tolist(),
                sequence.is_on.tolist(),
                sink,
                ptt_lead_s=config.ptt_lead_s,
                ptt_tail_s=config.ptt_tail_s,
                use_ptt=config.serial is None or config.serial.uses_ptt,
            )
    except SerialKeyError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # ``Keyer.send`` は ``finally`` で必ず両線を落としてから例外を
        # 送出する (``with open_sink(...)`` の ``__exit__`` も同様)。
        # ここではその後始末が済んだあとの後片付け (終了コードとメッセージ)
        # だけを行う。
        print("\n中止しました。")
        return 1
    print(f"完了 : {report.elements_sent} 要素   ずれ 最大 {report.max_error_ms:.1f} ms / 平均 {report.mean_error_ms:.1f} ms")
    return 0


def check_lines(
    config: ServerConfig, cycles: int | None = None, opener: SinkOpener | None = None
) -> int:
    """結線確認モード. 電鍵線と PTT 線をゆっくり交互に上下させる.

    **必ず無線機の電源を切るかダミーロードで使うこと。**
    テスターか LED を当てて、線の割り当てと極性を確かめる。

    Args:
        cycles: 繰り返す回数。``None`` なら Ctrl+C まで。
        opener: 窓口を配るもの。``None`` なら :func:`open_sink`。
    """
    opener = opener or open_sink
    print("結線確認モード — **無線機の電源が切れていることを確かめてください。**")
    print("電鍵線と PTT 線を 1 秒ずつ交互に上げます。Ctrl+C で終了。\n")
    count = 0
    try:
        with opener(config) as sink:
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


def _peer_already_disconnected(conn: socket.socket) -> bool:
    """相手が既に切断しているかを、詰まらずに (ノンブロッキングで) 調べる.

    **なぜ ``active.is_alive()`` だけでは足りないか。** 前の接続のセッションは
    別スレッドで動いており、相手の切断 (TCP の EOF/RST) を検知して
    ``run()`` から抜けるまでは ``is_alive()`` が真のままである。相手が実際には
    もう切断していても、そのスレッドがまだ ``recv`` を呼ぶ順番のスケジュール
    が回ってきていないだけの一瞬に次の接続が来ると、誤って ``busy`` を返して
    しまう (レビューで実測: 15 回中 4 回)。

    **``select`` はスレッドのスケジュールを介さず、カーネルのソケット状態を
    直接見る。** 相手の切断 (EOF/RST) はカーネルが非同期に受け取るので、
    ``select`` で読み取り可能と分かった時点で、まだ前のセッションのスレッドが
    それを ``recv`` していなくても「もう繋がっていない」と判定できる。
    決め打ちの ``sleep`` で間を置く方法とは違い、レースそのものを取り除く。

    読めるデータが実際にあるだけ (相手が生きていて何か送ってきた) の場合は
    ``False`` を返す — ``MSG_PEEK`` はキューからデータを取り除かないので、
    前のセッションのスレッドが後で普通に ``recv`` すればそのまま読める。

    何らかの理由で判定できない場合 (テスト用の偽ソケット等、``fileno``/
    ``select`` に対応していない) は、二重打鍵を防ぐという本来の目的を
    壊さないよう、**安全側 (まだ繋がっている扱い) に倒す。**
    """
    try:
        if conn.fileno() == -1:
            return True                       # 既に閉じられている
        readable, _, _ = select.select([conn], [], [], 0)
        if not readable:
            return False                      # 何も来ていない = 開いたまま
        return conn.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True                           # 壊れている (RST 等) = 切断済み
    except Exception:
        return False                          # 判定不能 = 安全側 (まだ繋がっている扱い)


@contextlib.contextmanager
def shared_sink(config: ServerConfig, settle_s: float):
    """ポートを 1 回だけ開き、セッション中ずっと保つ.

    **送信のたびに開き直してはいけない。** 変換器によっては開くたびに再認識が
    走り、実測で約 7 秒 線が命令に反応しない。その間の打鍵は電波にならず、
    運用者の環境では 6.3 秒の電文のうち最後の 1.3 秒しか出なかった。

    dry-run では従来どおり毎回 ``RecordingKeySink`` を配る (シリアルに触らない)。

    Yields:
        送信 1 回分の窓口を配る opener。
    """
    if config.serial is None:
        yield open_sink
        return

    sink = SerialKeySink(config.serial)
    sink.open()                      # 開いた直後に両線を落とす (SerialKeySink の約束)
    try:
        if settle_s > 0:
            print(f"変換器の準備を待っています… {settle_s:.0f} 秒")
            time.sleep(settle_s)
        yield persistent_opener(sink)
    finally:
        sink.close()                 # 両線を落としてから閉じる


def serve(
    config: ServerConfig,
    server: socket.socket,
    *,
    opener: SinkOpener | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """接続を受け付ける.

    **同時に扱うのは 1 つ。** 二重打鍵しないため、先に繋いだ側を優先し、
    2 つ目には ``busy`` を返して閉じる。**待たせない** — アプリ側は繋がったか
    どうかをすぐ知る必要がある (画面のボタンの有効・無効がそれで決まる)。

    busy の判定は 2 段で行う。

    1. **打鍵中なら、相手が切れていようと無条件に ``busy``。**
       ``KeySession.is_keying`` を見る。アプリが打鍵中に強制終了して運用者が
       すぐ繋ぎ直すと、古いセッションがまだ ``Keyer.send`` の中にいるのに
       新しい接続を受けてしまい、2 本の打鍵スレッドが同じ ``KeySink`` を叩く
       (レビューで 3/3 再現)。実機では COM の二重オープンを OS が弾くが、
       **「二重打鍵しない」を担保するのが OS であってコードでないのは設計 §7.5
       に反する。**
    2. 打鍵していなければ ``active.is_alive()`` と
       :func:`_peer_already_disconnected` を見る。前者だけだと、相手が切断しても
       セッションのスレッドが切断検知を終えるまでの一瞬 ``busy`` を誤って返す
       レースがある (詳細は同関数の docstring 参照)。

    Args:
        opener: 窓口を配るもの。``None`` なら :func:`open_sink` (送信ごとに開く)。
            **実結線では :func:`shared_sink` が作る opener を渡すこと。**
        stop_event: 立っていれば、次の ``accept`` のタイムアウト (最大 1 秒) で
            抜ける。**テストの後片付け用。** ``None`` (既定) なら通常運用どおり
            ``Ctrl+C`` (``KeyboardInterrupt``) でしか止まらない。ソケットを
            外から強制的に閉じて ``accept`` に例外を出させる後片付け方法だと、
            プラットフォームによって例外の種類が変わるうえ (このリポジトリの
            開発機では Windows の ``WinError 10038``)、その例外がスレッドの
            外に漏れて警告になる。``stop_event`` はそれを避けるための、待ち
            受けループを固まらせない・決め打ちの ``sleep`` も要らない止め方。
    """
    opener = opener or open_sink
    server.settimeout(1.0)
    active: threading.Thread | None = None
    active_conn: socket.socket | None = None
    active_session: KeySession | None = None
    while stop_event is None or not stop_event.is_set():
        try:
            conn, peer = server.accept()
        except TimeoutError:
            continue

        still_keying = active_session is not None and active_session.is_keying
        still_connected = (
            active is not None
            and active.is_alive()
            and active_conn is not None
            and not _peer_already_disconnected(active_conn)
        )
        if still_keying or still_connected:
            reason = "まだ打鍵中です" if still_keying else "既に別のアプリが繋がっています"
            print(f"拒否: {peer[0]}:{peer[1]} ({reason})")
            # **sendall が失敗しても close は必ず呼ぶ。** 同じ ``with`` の中で
            # 両方を ``suppress`` すると、``sendall`` の例外が ``close`` を
            # 読み飛ばしてしまい、拒否した接続の fd が閉じずに残る隙間ができる。
            try:
                with contextlib.suppress(OSError):
                    conn.sendall(
                        encode_message(error(CODE_BUSY, "別のアプリが繋がっています"))
                    )
            finally:
                with contextlib.suppress(OSError):
                    conn.close()
            continue

        print(f"\n接続: {peer[0]}:{peer[1]}")
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # **セッションはスレッドの外で作って持っておく。** 打鍵中かどうかを
        # 次の接続の判定 (上記) で見るため。スレッドの中で作ると、参照を持てない
        session = KeySession(config, conn, sink_opener=opener)

        def run_session(session: KeySession = session, conn: socket.socket = conn) -> None:
            try:
                session.run()
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                print(f"セッションが落ちました: {exc}", file=sys.stderr)
            finally:
                with contextlib.suppress(OSError):
                    conn.close()
                print("切断されました。次の接続を待ちます。")

        active = threading.Thread(target=run_session, name="cw-session", daemon=True)
        active_conn = conn
        active_session = session
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

    if args.dry_run and args.check_lines:
        # --check-lines は実際の結線を上下させて人が目で確かめるモードなので、
        # シリアルに一切触らない --dry-run と組み合わせても意味を成さない。
        # 黙って dry-run 扱いで「確認した気になる」出力を出すのは危険なので止める。
        print(
            "エラー: --check-lines は --dry-run と併用できません (結線確認には実際の"
            "ポートが要ります)。",
            file=sys.stderr,
        )
        return 1

    try:
        config = config_from_args(args)
    except SerialKeyError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    # **ポートはここで 1 回だけ開き、終了まで保つ。** 送信のたびに開き直すと
    # 変換器の再認識が走り、その間 (実測 約 7 秒) の打鍵が電波にならない。
    try:
        with shared_sink(config, args.open_settle_s) as opener:
            return _run(args, config, opener)
    except SerialKeyError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, config: ServerConfig, opener: SinkOpener) -> int:
    """ポートが開いた状態で、選ばれたモードを実行する."""
    if args.check_lines:
        # ここに来る時点で --dry-run との併用は上の分岐で撥ねているので、
        # config.dry_run は常に偽 (= config.serial は常に設定済み) である。
        return check_lines(config, opener=opener)

    if args.text is not None:
        return send_once(config, args.text, args.wpm, opener=opener)

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
        serve(config, server, opener=opener)
    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

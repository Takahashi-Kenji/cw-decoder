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

from src.infer.audio import (  # noqa: E402
    _SOXR_AVAILABLE,
    AudioCapture,
    list_input_devices,
)
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

    if not _SOXR_AVAILABLE:
        # なぜ必須か: _resample_to_8k() は出力先を 8000 Hz にハードコードしている。
        # soxr が無いと AudioCapture.drain() のフォールバックがそれを呼び、
        # target_sample_rate=16000 を無視して黙って 8 kHz を返す。このスクリプトは
        # それを SOURCE_SAMPLE_RATE (16000 Hz) だと申告して送ってしまうため、
        # 受信側はヘッダを信じて全 dot/dash を 2 倍長として扱い、デコードが
        # 無言で全崩壊する。実害が音として気付きにくいため、起動時に必ず止める。
        print(
            "エラー: soxr が見つかりません。\n"
            "このスクリプトはデバイスのネイティブサンプルレートを"
            f" {SOURCE_SAMPLE_RATE} Hz へ正しく変換するために soxr のステートフル"
            "リサンプラを必須にしています。無いまま起動すると、サンプルレートの"
            "申告と実際の音声が食い違い、受信側のデコードが無言で全崩壊します。\n"
            "  pip install soxr\n"
            "を実行してから再度起動してください。",
            file=sys.stderr,
        )
        return 1

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

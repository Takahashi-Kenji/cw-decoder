"""実受信機チェーンからラベル付き録音を集める CLI (L4).

無線機 A (ダミーロード) で学習用の CW を送信し、無線機 B (アンテナ) で受けた音を
録る。**送ったテキストがそのままラベル**になるので書き起こしが要らない。

::

    無線機 A ──電波──> 無線機 B
      ↑ 電鍵                │ 音声
    cw_key_server.py     audio_send.py
      ↑ LAN 45679           │ LAN 45678
      └──── GPU PC (このスクリプト) ────┘

準備 (無線機 PC でこの順に。**飛ばさない**)::

    python scripts/cw_key_server.py --port COM3 --check-lines   # 無線機の電源を切って結線確認
    python scripts/cw_key_server.py --port COM3 --dry-run       # 打鍵せず経路確認
    python scripts/cw_key_server.py --port COM3                 # 本番
    python scripts/audio_send.py --device N                     # 無線機 B の音声を送る

使い方 (GPU PC)::

    # 電波を出さずに計画だけ見る
    python scripts/collect_l4.py --host 192.168.0.10 --count 20 --plan-only

    # 1 件だけ試して録れ高を確かめる (**一晩回す前に必ず**)
    python scripts/collect_l4.py --host 192.168.0.10 --count 1 --session-id test

    # 一晩 (20 秒 + 間 10 秒 で 8 時間 ≒ 900 件)
    python scripts/collect_l4.py --host 192.168.0.10 --count 900 \\
        --session-id 20260825_agcfast_500 --note agc=fast --note filter_hz=500

**安全**: 送信するのは無線機 A (ダミーロード) 側だけ。出力は最小に絞り、
セッションの先頭に自局コールを打つ (``--id-call``)。連続送信のデューティは
``--gap-s`` で調整する (終段の熱)。

**取れないもの**: 打鍵はオートキーヤーなので人間の手打ちの癖は入らない。
これは受信機チェーンの癖 (AGC・フィルタ・実ノイズ床) を集めるための経路である。

**``scripts/audit_labels.py`` をこのデータに使わないこと。** あれは包絡線から
符号を組み立て直してラベルと突き合わせる道具で、綺麗な自己打鍵録音を前提にしている。
バンドノイズに埋めた信号では雑音の尖りが ON として拾われ (実測: ON 230 個のうち
135 個が 20 ms 未満)、ラベル 19 に対し「音 50」のような無意味な数字が出る。
L4 のラベルは**送ったテキストそのもの**なので、確かめるべきは
「その音が読めるか」であり、``scripts/eval_model.py --keyed-set`` で TER を見る。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.collect import (                      # noqa: E402
    CollectConfig,
    build_session_plan,
    collect_item,
    collect_with_retry,
    dry_run_error,
    fingerprint_error,
    level_verdict,
    load_plan_file,
    measure_recording,
    write_sample,
)
from src.finetune.preprocess import (                   # noqa: E402
    DEFAULT_BPF_BANDWIDTH_HZ,
    DEFAULT_BPF_CENTER_HZ,
    DEFAULT_PAD_SEC,
)
from src.infer.net_audio import (                       # noqa: E402
    NetworkAudioCapture,
    NetworkCaptureError,
)
from src.tx import protocol                             # noqa: E402
from src.tx.net_key import NetKeyClient, NetKeyError, NetKeyRejected  # noqa: E402


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="実受信機チェーンから学習データを集める")
    p.add_argument("--host", required=True, help="無線機 PC のアドレス")
    p.add_argument("--key-port", type=int, default=protocol.DEFAULT_KEY_PORT)
    p.add_argument("--audio-port", type=int, default=45678)
    p.add_argument("--out-dir", type=Path, default=Path("data/l4"))
    p.add_argument("--session-id", default="l4", help="セッション名 (出力の接頭辞)")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--modes", nargs="+", default=["european", "japanese"],
                   choices=["european", "japanese"])
    p.add_argument("--wpm-list", type=float, nargs="+", default=[15.0, 18.0, 22.0, 26.0, 30.0],
                   help="この順に巡らせる (乱数で引かない)")
    p.add_argument("--seed", type=int, default=42, help="本文の乱数シード (同じなら同じ計画)")
    p.add_argument("--text-file", type=Path, default=None,
                   help="打鍵内容を書いたファイル (1 行 1 件。"
                        "「本文」/「モード<TAB>本文」/「モード<TAB>速度<TAB>本文」)。"
                        "指定すると --count と --seed は使わない")
    p.add_argument("--lead-in-s", type=float, default=1.5,
                   help="**打鍵の前にノイズ床を録る長さ。** 0 にすると音が"
                        "「無音 → いきなり符号」で始まり、実運用に無い形になる "
                        "(実測: 93%% の録音で 1 文字目が脱落した)")
    p.add_argument("--tail-s", type=float, default=1.0, help="打ち終わってからも録る長さ")
    p.add_argument("--gap-s", type=float, default=10.0, help="次の項目までの間 (終段の熱)")
    p.add_argument("--min-level-db", type=float, default=-60.0,
                   help="この dBFS 未満は「無線機 B が聞けていない」として捨てる")
    p.add_argument("--id-call", default=None,
                   help="セッション先頭に打つ自局コール (識別)")
    p.add_argument("--note", action="append", default=[], metavar="KEY=VALUE",
                   help="TXT ヘッダに残すセッション条件 (agc=fast、filter_hz=500 等)")
    p.add_argument("--bpf-center", type=float, default=DEFAULT_BPF_CENTER_HZ)
    p.add_argument("--bpf-bandwidth", type=float, default=DEFAULT_BPF_BANDWIDTH_HZ)
    p.add_argument("--no-bpf", action="store_true",
                   help="BPF をかけない (受信側で既に通している場合のみ)")
    p.add_argument("--pad-sec", type=float, default=DEFAULT_PAD_SEC)
    p.add_argument("--plan-only", action="store_true", help="電波を出さず計画だけ表示")
    p.add_argument("--preflight", action="store_true",
                   help="**電波を出さずに**接続を確認する (名乗り・符号表の指紋・受信レベル)")
    p.add_argument("--allow-fingerprint-mismatch", action="store_true",
                   help="両 PC の符号表が食い違っていても進める (承知の上で)")
    p.add_argument("--buffer-s", type=float, default=120.0,
                   help="受信バッファの長さ (秒)。1 件の長さより十分に大きく取る")
    return p


def parse_notes(notes: list[str]) -> dict[str, str]:
    """``KEY=VALUE`` の並びを dict にする."""
    out: dict[str, str] = {}
    for spec in notes:
        key, sep, value = spec.partition("=")
        if not sep or not key:
            raise SystemExit(f"--note は KEY=VALUE の形で渡してください: {spec!r}")
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_args().parse_args(argv)
    meta = parse_notes(args.note) | {"session": args.session_id}

    if args.text_file is not None:
        plan = load_plan_file(
            args.text_file, session_id=args.session_id,
            modes=tuple(args.modes), wpm_list=tuple(args.wpm_list),
        )
    else:
        plan = build_session_plan(
            count=args.count, modes=tuple(args.modes), wpm_list=tuple(args.wpm_list),
            seed=args.seed, session_id=args.session_id,
        )

    if args.plan_only:
        for item in plan:
            print(f"{item.name}  {item.wpm:5.1f}wpm  {item.text}")
        print(f"\n[plan] {len(plan)} 件 (電波は出していません)")
        return 0

    config = CollectConfig(
        lead_in_s=args.lead_in_s, tail_s=args.tail_s, gap_s=args.gap_s,
        min_level_db=args.min_level_db,
    )
    bpf = None if args.no_bpf else (args.bpf_center, args.bpf_bandwidth)

    capture = NetworkAudioCapture(args.host, args.audio_port, max_buffer_s=args.buffer_s)
    keyer = NetKeyClient(args.host, args.key_port)

    try:
        capture.start()
    except NetworkCaptureError as exc:
        print(f"[err] 音声側に繋がりません: {exc}", flush=True)
        print("      無線機 B を繋いだ PC で scripts/audio_send.py を動かしてください",
              flush=True)
        return 2

    try:
        hello = keyer.connect()
        print(f"[key] {args.host}:{args.key_port} {hello.describe_wiring()}", flush=True)
        error = fingerprint_error(hello.fingerprint_matches, args.allow_fingerprint_mismatch)
        if error is not None:
            keyer.close()
            capture.stop()
            print(f"[err] {error}", flush=True)
            return 2
        if not hello.fingerprint_matches:
            print("[warn] 符号表が食い違ったまま進みます", flush=True)
        # **dry-run のまま集めない。** 電鍵が動かないので、録れるのは
        # ノイズ床だけなのにラベルは全文が付く。確認 (--preflight) は通す。
        dry_error = dry_run_error(hello.dry_run)
        if dry_error is not None and not args.preflight:
            keyer.close()
            capture.stop()
            print(f"[err] {dry_error}", flush=True)
            return 2
        if dry_error is not None:
            print(f"[warn] {dry_error}", flush=True)
    except NetKeyError as exc:
        capture.stop()
        print(f"[err] 打鍵側に繋がりません: {exc}", flush=True)
        print("      無線機 A を繋いだ PC で scripts/cw_key_server.py を動かしてください",
              flush=True)
        return 2

    # 音声が届き始めるのを待つ (無線機 B の音声送信が上がっていないと全件無駄になる)
    deadline = time.time() + 10.0
    while not capture.is_connected and time.time() < deadline:
        time.sleep(0.2)
    if not capture.is_connected:
        keyer.close()
        capture.stop()
        print(f"[err] 音声側に繋がりません: {args.host}:{args.audio_port} "
              f"({capture.last_error})", flush=True)
        return 2
    print(f"[audio] {args.host}:{args.audio_port} 接続 "
          f"(元 {capture.source_sample_rate} Hz)", flush=True)

    if args.preflight:
        # **電波を出さない。** check は打鍵側の検算だけで、電鍵は動かない。
        item = plan[0]
        try:
            expected = keyer.check(item.text, item.wpm)
            print(f"[check] {item.mode} {item.wpm:.0f}wpm  "
                  f"{expected.chars} 文字 / {expected.elements} 要素 / "
                  f"{expected.seconds:.1f} 秒 — 打鍵側は受け付けます", flush=True)
        except NetKeyRejected as exc:
            print(f"[err] 打鍵側が撥ねました: {exc}", flush=True)
        for _ in range(6):
            time.sleep(0.5)
            level = capture.level_db_rms
            print(f"[level] {level:7.1f} dB  {level_verdict(level)}", flush=True)
        capture.drain()
        print(f"[drop] 取りこぼし {capture.dropped_blocks} ブロック / "
              f"再接続 {capture.reconnects} 回", flush=True)
        keyer.close()
        capture.stop()
        print("\n[preflight] 電波は出していません。"
              "上が全部正常なら --count 1 で 1 件録ってみてください", flush=True)
        return 0

    saved = 0
    rejected: dict[str, int] = {}
    try:
        if args.id_call:
            print(f"[id] {args.id_call}", flush=True)
            keyer.send(args.id_call, args.wpm_list[0])
            time.sleep(args.gap_s)

        for i, item in enumerate(plan, start=1):
            out_path = args.out_dir / args.session_id / f"{item.name}.wav"
            if out_path.exists():
                continue                       # 途中から再開できる
            def reconnect(keyer: NetKeyClient = keyer) -> None:
                # **待機中の繋ぎ直しは許されている** (送信中は許されない)。
                keyer.close()
                keyer.connect()
                print("[retry] 打鍵側へ繋ぎ直しました", flush=True)

            try:
                result = collect_with_retry(
                    item, keyer, capture, config, sleep=time.sleep,
                    reconnect=reconnect, error_types=(NetKeyError,),
                    no_retry_types=(NetKeyRejected,),
                )
            except NetKeyRejected as exc:
                print(f"[skip] {item.name}: 打鍵側が撥ねた ({exc})", flush=True)
                rejected["rejected_by_keyer"] = rejected.get("rejected_by_keyer", 0) + 1
                continue

            if result.rejected is not None:
                key = result.rejected.name
                rejected[key] = rejected.get(key, 0) + 1
                print(f"[skip] {item.name}: {result.rejected.value}", flush=True)
            else:
                write_sample(
                    args.out_dir / args.session_id, item, result.wave,
                    config.sample_rate, meta, bpf=bpf, pad_sec=args.pad_sec,
                )
                saved += 1
                send = result.send
                m = measure_recording(result.wave, config.sample_rate)
                print(f"[save] {item.name}  {item.wpm:.0f}wpm  "
                      f"{send.seconds:5.1f}s  トーン {m['tone_hz']}Hz  "
                      f"レベル {m['level_db']}dB  深さ {m['contrast_db']}dB  "
                      f"({i}/{len(plan)})", flush=True)
            if args.gap_s > 0 and i < len(plan):
                time.sleep(args.gap_s)
    except KeyboardInterrupt:
        print("\n[stop] 中断しました", flush=True)
    except NetKeyError as exc:
        print(f"[err] 打鍵が切れました: {exc}", flush=True)
    finally:
        keyer.close()
        capture.stop()

    print(f"\n[done] 保存 {saved} 件 → {args.out_dir / args.session_id}", flush=True)
    for key, n in sorted(rejected.items()):
        print(f"  捨てた: {key} {n} 件", flush=True)
    if saved:
        print("\n**取り込んだら scripts/audit_labels.py を通すこと** "
              "(ラベルと音が合っているかは毎回確かめる)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CW デコーダ アプリ起動 (ライブ連続モード).

使い方::

    python scripts/run_app.py                     # チェックポイント無し (未学習モデル)
    python scripts/run_app.py --ckpt models/full/best_infer.pt

受信機を繋いだ PC が別なとき (その PC で scripts/audio_send.py を動かしておく)::

    python scripts/run_app.py --ckpt models/full/best_infer.pt --net-source 192.168.1.20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.app.main_window import main as run_app_main  # noqa: E402
from src.infer.net_audio import parse_endpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CW デコーダ アプリ")
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="チェックポイントパス (省略時は未学習モデルで起動)",
    )
    parser.add_argument(
        "--net-source", type=str, default=None, metavar="HOST[:PORT]",
        help="LAN 経由の転送元 (既定ポート 45678)。指定時は入力デバイスの代わりに使う",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda", "auto"],
        help="デコードを走らせるデバイス (既定 cpu。GPU はローカル LLM に空ける)",
    )
    args = parser.parse_args(argv)

    if args.net_source is not None:
        try:
            parse_endpoint(args.net_source)
        except ValueError as exc:
            print(f"エラー: --net-source の指定が不正です: {exc}", file=sys.stderr)
            return 1

    return run_app_main(
        checkpoint_path=args.ckpt,
        net_source=args.net_source,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())

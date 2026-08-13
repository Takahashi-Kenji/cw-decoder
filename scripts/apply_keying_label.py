"""録音 TXT に打鍵原稿のラベルを転記する CLI.

アプリの録音ボタンで保存した ``data/real/<時刻>_<モード>.txt`` の本文
(モデルの誤デコード) を、実際に打鍵した原稿テキストで置き換える。

使い方::

    python scripts/apply_keying_label.py data/real/20260712_190000_japanese.txt \\
        --script data/keying_scripts/script_ja_01.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.keying_scripts import apply_script_label, read_script_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="録音 TXT へ原稿ラベルを転記")
    p.add_argument("txt", type=Path, help="録音 TXT のパス")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--script", type=Path, help="原稿ファイル (script_ja_01.txt 等)")
    group.add_argument("--text", type=str, help="ラベルテキストを直接指定")
    args = p.parse_args(argv)

    text = args.text if args.text is not None else read_script_text(args.script)
    apply_script_label(args.txt, text)
    print(f"[done] {args.txt} の本文を更新: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

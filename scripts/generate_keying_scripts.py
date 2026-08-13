"""打鍵原稿の生成 CLI (自己打鍵データ収集用).

使い方::

    # 和文 20 件 + 欧文 10 件を data/keying_scripts/ に生成
    python scripts/generate_keying_scripts.py

    # 和文だけ 30 件
    python scripts/generate_keying_scripts.py --japanese 30 --european 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.keying_scripts import (  # noqa: E402
    estimate_duration_sec,
    read_script_text,
    write_script_files,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="打鍵原稿の生成")
    p.add_argument("--japanese", type=int, default=20, help="和文原稿の件数")
    p.add_argument("--european", type=int, default=10, help="欧文原稿の件数")
    p.add_argument("--out-dir", type=Path, default=Path("data/keying_scripts"))
    p.add_argument("--wpm", type=float, default=24.0, help="時間見積もり用の WPM")
    p.add_argument("--seed", type=int, default=20260712)
    args = p.parse_args(argv)

    total = 0
    for mode, count in (("japanese", args.japanese), ("european", args.european)):
        if count <= 0:
            continue
        paths = write_script_files(
            args.out_dir, mode=mode, count=count, seed=args.seed, wpm=args.wpm  # type: ignore[arg-type]
        )
        for path in paths:
            text = read_script_text(path)
            dur = estimate_duration_sec(text, mode, args.wpm)  # type: ignore[arg-type]
            print(f"{path.name}  ({dur:4.1f}s)  {text}")
        total += len(paths)
    print(f"[done] {total} 件を {args.out_dir} に生成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

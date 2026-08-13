"""手書きコーパス (``src/finetune/keying_corpus.py``) から打鍵原稿を書き出す CLI.

乱数生成の ``generate_keying_scripts.py`` と違い、本文は固定である。
弱点トークン (6要素符号) の出現数を設計値どおりに保つのが目的。

使い方::

    # 欧文30件を data/keying_scripts/ に script_eu_t01..t30.txt として生成
    python scripts/generate_corpus_scripts.py

    # 中身の確認だけ (書き出さない)
    python scripts/generate_corpus_scripts.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.keying_corpus import (  # noqa: E402
    ALL_SCRIPTS,
    EUROPEAN_SCRIPTS,
    JAPANESE_SCRIPTS,
    KeyingScript,
    code_histogram,
    resolve_code,
)
from src.finetune.keying_scripts import estimate_duration_sec  # noqa: E402
from src.tokens.morse_tokens import text_to_codes  # noqa: E402


def _render(script: KeyingScript) -> str:
    """原稿ファイルの中身を組み立てる (既存の打鍵原稿と同じ ``ヘッダ --- 本文`` 形式)."""
    dur = estimate_duration_sec(script.text, script.mode, script.wpm)
    return (
        f"mode: {script.mode}\n"
        f"estimated_duration_s: {dur:.1f}\n"
        f"wpm: {script.wpm:.0f}\n"
        "---\n"
        f"{script.text}\n"
    )


def _report(label: str, scripts: tuple[KeyingScript, ...], keys: list[str]) -> None:
    """モード別に注目トークンの出現数を表示する."""
    if not scripts:
        return
    counter: Counter[str] = code_histogram(scripts)
    mode = scripts[0].mode
    shown = "  ".join(f"{k}={counter[resolve_code(k, mode)]}" for k in keys)
    print(f"{label}: {shown}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="手書きコーパスから打鍵原稿を生成")
    p.add_argument("--out-dir", type=Path, default=Path("data/keying_scripts"))
    p.add_argument(
        "--dry-run", action="store_true", help="ファイルを書かず内容だけ表示する"
    )
    args = p.parse_args(argv)

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    total_sec = 0.0
    for script in ALL_SCRIPTS:
        # 生成時に必ずトークン化できることを確認する (FT のラベル検証で落とさない)
        text_to_codes(script.text, script.mode)
        dur = estimate_duration_sec(script.text, script.mode, script.wpm)
        total_sec += dur
        path = args.out_dir / f"script_{script.name}.txt"
        if not args.dry_run:
            path.write_text(_render(script), encoding="utf-8")
        print(f"{path.name}  {script.wpm:2.0f}wpm  {dur:5.1f}s  {script.text}")

    print()
    _report("欧文 弱点", EUROPEAN_SCRIPTS, ["?", ".", ",", "-", "@", "/", "=", "+"])
    _report("欧文 数字", EUROPEAN_SCRIPTS, list("0123456789"))
    _report("和文 弱点", JAPANESE_SCRIPTS, ["、", "。", "?", "-", "@", "ー", "゛", "゜"])
    _report("和文 数字", JAPANESE_SCRIPTS, list("0123456789"))
    action = "確認" if args.dry_run else f"{args.out_dir} に生成"
    print(f"[done] {len(ALL_SCRIPTS)} 件を{action} (合計 {total_sec / 60:.1f} 分)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

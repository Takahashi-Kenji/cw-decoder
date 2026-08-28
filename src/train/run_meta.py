"""学習 1 回分のメタ情報 (実験台帳に転記するためのもの).

なぜ要るか
----------
**現行モデルの学習コマンドが正確に残っていない。** 設計書には「102,000 step
(フル + 補充)」としかなく、対照実験に設定ドリフトの疑いが残っている
(docs/model_baseline_and_improvement_plan.md §2.1)。

夜間に学習を回し放題にするなら、**どの run が何だったかを機械可読で残す**しかない。
``save_checkpoint`` の ``extra`` と ``<ckpt-dir>/meta.json`` の両方に同じ dict を入れ、
ckpt を開かなくても台帳へ転記できるようにする。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

META_FILENAME = "meta.json"


def git_revision(cwd: Path | None = None) -> dict[str, Any]:
    """HEAD の SHA と作業ツリーの汚れ具合を返す.

    git が無い / リポジトリでない場合も**落とさない** (学習を止める理由にならない)。
    """
    def run(*cmd: str) -> str | None:
        try:
            out = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    rev = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "git_rev": rev,
        # None (git が使えない) と "" (綺麗) を区別する
        "git_dirty": None if status is None else bool(status),
    }


def build_run_meta(
    argv: list[str] | None = None,
    *,
    cwd: Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """実験台帳に載せる 1 行分の情報を作る.

    ``fields`` には seed・データ件数など run ごとの値を渡す。
    **時刻は入れない** (同じコマンドの再実行で差分が出ると比較しにくいため。
    時刻はファイルの mtime と台帳側で持つ)。
    """
    meta: dict[str, Any] = {"argv": list(argv if argv is not None else sys.argv)}
    meta.update(git_revision(cwd))
    meta.update(fields)
    return meta


def write_run_meta(ckpt_dir: Path, meta: dict[str, Any]) -> Path:
    """``<ckpt-dir>/meta.json`` に書き出してパスを返す."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / META_FILENAME
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["META_FILENAME", "build_run_meta", "git_revision", "write_run_meta"]

"""学習 run のメタ情報のテスト.

**「当時の正確なコマンドが記録されていない」を二度と起こさない**ための仕組みなので、
argv がそのまま残ることを固定する。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.train.run_meta import META_FILENAME, build_run_meta, write_run_meta


class TestBuildRunMeta:
    def test_argv_をそのまま残す(self) -> None:
        argv = ["scripts/train.py", "--steps", "102000", "--seed", "42"]
        assert build_run_meta(argv)["argv"] == argv

    def test_追加のフィールドを載せる(self) -> None:
        meta = build_run_meta([], seed=42, n_real=90, n_synth=None)
        assert meta["seed"] == 42
        assert meta["n_real"] == 90
        assert meta["n_synth"] is None

    def test_git_情報のキーが必ずある(self) -> None:
        """git が無い環境でも落とさない (学習を止める理由にならない)."""
        meta = build_run_meta([])
        assert "git_rev" in meta and "git_dirty" in meta

    def test_時刻は入れない(self) -> None:
        """同じコマンドの再実行で差分が出ると比較しにくい."""
        assert build_run_meta([], seed=1) == build_run_meta([], seed=1)


class TestWriteRunMeta:
    def test_ckpt_dir_に書く(self, tmp_path: Path) -> None:
        path = write_run_meta(tmp_path, {"argv": ["a"], "seed": 42})
        assert path == tmp_path / META_FILENAME
        assert json.loads(path.read_text(encoding="utf-8")) == {"argv": ["a"], "seed": 42}

    def test_ディレクトリが無ければ作る(self, tmp_path: Path) -> None:
        path = write_run_meta(tmp_path / "new" / "dir", {"seed": 1})
        assert path.exists()

    def test_日本語が読める形で入る(self, tmp_path: Path) -> None:
        path = write_run_meta(tmp_path, {"note": "対照群"})
        assert "対照群" in path.read_text(encoding="utf-8")

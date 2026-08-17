"""配布物の中でモデルを見つけられることを固定する.

**配布版は利用者が ``--ckpt`` を打たない。** アイコンを叩くだけで動く必要が
あるので、同梱したモデルを自力で見つけられなければならない。ここが壊れると
「未学習モデルで起動」して、それらしいゴミを延々と表示する
(**エラーにならないぶん質が悪い**)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.app import resources


class TestBundleDetection:
    def test_source_tree_is_not_frozen(self) -> None:
        assert not resources.is_frozen()

    def test_bundle_dir_is_project_root_when_running_from_source(self) -> None:
        assert (resources.bundle_dir() / "src" / "app").is_dir()

    def test_frozen_uses_meipass(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """PyInstaller の展開先を見ること."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert resources.is_frozen()
        assert resources.bundle_dir() == tmp_path


class TestDefaultModel:
    def test_prefers_bundled_onnx(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """同梱の ONNX があればそれを使う."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        onnx = tmp_path / resources.BUNDLED_MODEL_REL
        onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.write_bytes(b"dummy")

        assert resources.default_model_path() == onnx

    def test_returns_none_when_nothing_is_there(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert resources.default_model_path() is None

    def test_source_tree_finds_something_or_nothing_but_never_raises(self) -> None:
        """開発環境では ONNX かチェックポイントのどちらかが見つかりうる.

        どちらも無い環境 (新規クローン) でも**例外にしないこと**。
        起動できないより、未学習モデルで起動して画面を出す方がよい。
        """
        got = resources.default_model_path()
        assert got is None or got.exists()


class TestResolveModel:
    def test_saved_path_wins_when_it_exists(self, tmp_path: Path) -> None:
        saved = tmp_path / "my.onnx"
        saved.write_bytes(b"x")
        assert resources.resolve_model_path(str(saved)) == saved

    def test_falls_back_when_saved_path_is_gone(self, tmp_path: Path) -> None:
        """**設定に残った古いパスで起動を諦めないこと。**

        インストーラで入れ直した・モデルを移動した、で簡単に起きる。
        """
        got = resources.resolve_model_path(str(tmp_path / "消えたモデル.onnx"))
        assert got != tmp_path / "消えたモデル.onnx"

    def test_empty_setting_falls_back_to_default(self) -> None:
        assert resources.resolve_model_path(None) == resources.default_model_path()
        assert resources.resolve_model_path("") == resources.default_model_path()

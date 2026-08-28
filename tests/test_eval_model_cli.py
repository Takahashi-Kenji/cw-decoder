"""`scripts/eval_model.py` の CLI 補助ロジックのテスト.

held-out を**名前付きで複数**渡せることを保証する。
旧 21 件は凍結して連続性を保ち、新しく録った分を別セットとして足すため。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.eval_model import LEGACY_SET_NAME, parse_keyed_sets


class TestParseKeyedSets:
    def test_旧形式は既定の名前になる(self) -> None:
        got = parse_keyed_sets(Path("data/keying_scripts"), None)
        assert got == {LEGACY_SET_NAME: Path("data/keying_scripts")}

    def test_名前付きを複数(self) -> None:
        got = parse_keyed_sets(None, ["v1=data/keying_scripts", "v2=data/heldout/v2"])
        assert got == {
            "v1": Path("data/keying_scripts"),
            "v2": Path("data/heldout/v2"),
        }

    def test_旧形式と名前付きの併用(self) -> None:
        got = parse_keyed_sets(Path("old"), ["v2=new"])
        assert got == {LEGACY_SET_NAME: Path("old"), "v2": Path("new")}

    def test_パスに等号が含まれてもよい(self) -> None:
        """区切りは最初の = だけ (Windows のパスに = が入りうる)."""
        got = parse_keyed_sets(None, ["v1=C:/a=b/c"])
        assert got == {"v1": Path("C:/a=b/c")}

    def test_名前が無ければエラー(self) -> None:
        with pytest.raises(SystemExit):
            parse_keyed_sets(None, ["data/keying_scripts"])

    def test_名前が空ならエラー(self) -> None:
        with pytest.raises(SystemExit):
            parse_keyed_sets(None, ["=data/keying_scripts"])

    def test_名前の重複はエラー(self) -> None:
        """**黙って上書きしない。** 片方が測られないまま採否を決めてしまう."""
        with pytest.raises(SystemExit):
            parse_keyed_sets(None, ["v1=a", "v1=b"])

    def test_旧形式の名前と衝突したらエラー(self) -> None:
        with pytest.raises(SystemExit):
            parse_keyed_sets(Path("old"), [f"{LEGACY_SET_NAME}=new"])

    def test_何も無ければ空(self) -> None:
        assert parse_keyed_sets(None, None) == {}

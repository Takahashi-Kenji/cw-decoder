"""配布物の作りを固定するテスト.

**実際のビルドはしない** (数分かかる)。設定ファイル同士の食い違いだけを見る。

守るのは 3 つ。

* **README が展開フォルダの直下に置かれること** — spec の ``datas`` に書くと
  onedir では ``_internal`` の下に入ってしまう (2026-08-18 に実際に踏んだ)
* **取説の同梱先と、スタートメニューの指し先が一致すること** — ずれても
  インストーラは ``Check: FileExists`` で**黙ってショートカットだけ作らない**。
  配って初めて「取説が開けない」と分かる
* **README が実在するファイル名を案内していること** — 保存先の名前を書き換えた
  ときに README だけ古くなるのを防ぐ
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_ISS = _ROOT / "packaging" / "cw-decoder.iss"
_SPEC = _ROOT / "packaging" / "cw-decoder.spec"
_README = _ROOT / "packaging" / "dist_files" / "README.txt"

# PyInstaller の onedir は datas を必ずこの下に置く。
_INTERNAL = "_internal"


class TestManualIsReachable:
    def test_the_spec_bundles_the_manual(self) -> None:
        spec = _SPEC.read_text(encoding="utf-8")
        assert '"manual"' in spec, "spec が取説を同梱していない"
        assert 'ROOT / "docs" / "manual"' in spec

    def test_the_manual_actually_exists(self) -> None:
        """**同梱元が無ければビルドは止まる**が、気づくのは早いほうがよい."""
        assert (_ROOT / "docs" / "manual" / "index.html").is_file()

    def test_the_shortcut_points_at_the_bundled_manual(self) -> None:
        """スタートメニューの指し先が ``_internal\\manual\\index.html`` であること."""
        icons = [
            line for line in _ISS.read_text(encoding="utf-8").splitlines()
            if line.startswith("Name:") and "取扱説明書" in line
        ]
        assert len(icons) == 1, "取説のショートカットが 1 つでない"
        match = re.search(r'Filename:\s*"([^"]+)"', icons[0])
        assert match is not None
        assert match.group(1) == rf"{{app}}\{_INTERNAL}\manual\index.html"


class TestDistributedReadme:
    def test_it_exists_and_is_utf8(self) -> None:
        assert _README.is_file()
        assert _README.read_text(encoding="utf-8").startswith("cw-decoder")

    def test_the_spec_does_not_bundle_it(self) -> None:
        """**spec に書かないこと.** onedir では ``_internal`` の下に入ってしまう."""
        assert "dist_files" not in _SPEC.read_text(encoding="utf-8")

    def test_the_build_script_copies_it_to_the_root(self, tmp_path, monkeypatch) -> None:
        from scripts import build_installer

        monkeypatch.setattr(build_installer, "BUNDLE", tmp_path)
        placed = build_installer.place_readme()
        assert placed == tmp_path / "README.txt"
        assert placed.is_file()

    @pytest.mark.parametrize(
        "expected",
        [
            rf"{_INTERNAL}\manual\index.html",   # 取説の場所
            ".cw-decorder",                      # 利用者データの場所
            "cw-decoder.exe",                    # 起動の仕方
        ],
    )
    def test_it_tells_where_things_are(self, expected: str) -> None:
        assert expected in _README.read_text(encoding="utf-8")

    def test_the_files_it_names_are_the_real_ones(self) -> None:
        """**保存先の名前を書き換えたら README も直すこと.**"""
        from src.infer.settings import DEFAULT_CONFIG_PATH
        from src.infer.word_correct import DEFAULT_JA_LEXICON_PATH
        from src.tx.profile import DEFAULT_PROFILE_PATH
        from src.tx.templates import DEFAULT_TEMPLATES_PATH

        text = _README.read_text(encoding="utf-8")
        for path in (
            DEFAULT_CONFIG_PATH, DEFAULT_JA_LEXICON_PATH,
            DEFAULT_PROFILE_PATH, DEFAULT_TEMPLATES_PATH,
        ):
            assert path.name in text, f"{path.name} が README に無い"

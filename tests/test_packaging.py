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
        from dataclasses import dataclass

        from scripts import build_installer

        # 配布物が 3 つになったので (受信アプリ / 打鍵サーバ / 音声送出)、
        # 置き場所は Target が持つ。README を置くのは**本体だけ**。
        @dataclass(frozen=True)
        class _FakeTarget:
            bundle = tmp_path
            title = "テスト用"

        placed = build_installer.place_readme(_FakeTarget())
        assert placed == tmp_path / "README.txt"
        assert placed.is_file()

    def test_only_the_app_gets_the_readme(self) -> None:
        """打鍵サーバ・音声送出には置かない (窓が自分で名乗るため)."""
        from scripts.build_installer import TARGETS

        assert [t.key for t in TARGETS if t.readme] == ["app"]

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


def test_spec_compiles() -> None:
    """spec は Python として構文が通ること.

    2026-08-28 に f 文字列の途中に実改行が入った spec を配布ブランチに push し、
    ビルドで初めて気づいた。テストで止める。
    """
    from pathlib import Path
    spec = Path(__file__).resolve().parent.parent / "packaging" / "cw-decoder.spec"
    # BOM 付き UTF-8 (PyInstaller は utf-8-sig で読む)
    compile(spec.read_text(encoding="utf-8-sig"), str(spec), "exec")


class TestPykakasiDataIsBundled:
    """**pykakasi はデータファイルを持つ。** Python モジュールだけでは動かない.

    2026-08-31 に運用者から「CW サーバに繋がったのに [確認] が押せない」と
    報告された。原因は ``pykakasi`` の辞書 (``kanwadict4.db`` ほか 9 ファイル、
    約 9.5 MB) が配布物に入っていなかったこと。

    症状の出かたが分かりにくい:

    1. 日本語を書くと ``textChanged`` → ``refresh_kana`` → ``to_sendable_kana``
    2. その中の ``pykakasi.kakasi()`` が ``FileNotFoundError`` を投げる
    3. カナ欄が空のままになる
    4. ``[確認]`` は ``bool(wire_text(panel))`` で有効になるので**押せない**

    窓を出さない配布版では例外がどこにも表示されないため、利用者からは
    「ボタンがアクティブにならない」としか見えない。

    **PyInstaller はデータファイルを自動では拾わない** (import を辿るだけ)。
    """

    def test_the_library_really_needs_data_files(self) -> None:
        """前提の確認: データが無いと例外になること (仕様が変わったら気づく)."""
        pytest.importorskip("pykakasi")
        import pykakasi.properties as properties

        original = properties.Configurations.data_path
        try:
            properties.Configurations.data_path = Path(__file__).parent / "存在しない"
            import pykakasi

            with pytest.raises(FileNotFoundError):
                pykakasi.kakasi()
        finally:
            properties.Configurations.data_path = original

    def test_the_spec_collects_them(self) -> None:
        assert "pykakasi" in _SPEC.read_text(encoding="utf-8"), (
            "spec が pykakasi のデータを集めていない。"
            "送信の日本語→カナ変換が配布版で動かなくなる"
        )

    def test_collection_finds_the_dictionaries(self) -> None:
        """集める仕組みが**いま実際に効いていること** (名前だけでは足りない)."""
        pytest.importorskip("PyInstaller")
        from PyInstaller.utils.hooks import collect_data_files

        names = {Path(src).name for src, _dest in collect_data_files("pykakasi")}
        assert "kanwadict4.db" in names, f"辞書が集まっていない: {sorted(names)}"
        assert len([n for n in names if n.endswith(".db")]) >= 9

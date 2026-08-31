"""配布物 (PyInstaller) の除外設定で画面が壊れないことを固定する.

**これは「押しても何も起きない」を防ぐ歯止めである。**

spec の ``excludes`` は効き目が大きいぶん危ない。パッケージ名で丸ごと外すと、
その中の**軽い 1 モジュールだけを使っている画面**が道連れになる。しかも
配布版は窓を出さない (``console=False``) ので、``ModuleNotFoundError`` は
どこにも表示されない。利用者から見ると**ボタンを押しても何も起きない**。

実際に踏んだ (2026-08-31): ``src.synth`` を丸ごと除外していたため、
``src.tx.encoder`` → ``src.synth.keying`` が解決できず、[交信…] が無反応だった。
開発環境では ``src.synth`` があるので**まったく再現しない**。

判定は**別プロセス**で行い、spec の除外を ``sys.meta_path`` で再現する。
除外リストは spec から読む。**書き写さない** — 写せば必ず食い違う
(``tests/data/eGov_morse_reference.py`` の件と同じ)。
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = _ROOT / "packaging" / "cw-decoder.spec"

# 打鍵サーバ・音声送出も同じ落とし穴を持つ (別々の配布物なので除外も別々)。
# **入口のモジュールが読めなければ、起動した瞬間に落ちる。**
_OTHER_SPECS = {
    "cw-key-server": "scripts.cw_key_server",
    "cw-audio-send": "scripts.audio_send",
}

# 配布版で**画面から辿れる**モジュール。ここが import できなければ、
# 対応するボタンが無反応になる。
GUI_MODULES = [
    "src.app.main_window",
    "src.app.tx_dialog",
    "src.app.settings_dialog",
    "src.app.profile_dialog",
    "src.app.template_dialog",
]


def _spec_excludes(spec: Path | None = None) -> list[str]:
    """spec の ``EXCLUDES`` を読む (spec は PyInstaller 用なので import できない)."""
    spec = spec or _SPEC
    # spec は BOM 付き (PyInstaller のひな形がそう作る) なので utf-8-sig で読む
    tree = ast.parse(spec.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EXCLUDES" for t in node.targets
        ):
            return [str(v) for v in ast.literal_eval(node.value)]
    raise AssertionError(f"{spec} に EXCLUDES が見つかりません")


def _import_without_excluded(module: str, excludes: list[str]) -> tuple[bool, str]:
    """``excludes`` を無いことにして ``module`` を import する (別プロセス)."""
    script = (
        "import sys, importlib.abc\n"
        f"sys.path.insert(0, {str(_ROOT)!r})\n"
        f"BLOCKED = {excludes!r}\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        for b in BLOCKED:\n"
        "            if fullname == b or fullname.startswith(b + '.'):\n"
        "                raise ModuleNotFoundError(\n"
        "                    f'No module named {fullname!r} (配布物では除外されている)')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"import {module}\n"
        "print('IMPORT_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=_ROOT, timeout=300,
    )
    return "IMPORT_OK" in proc.stdout, proc.stderr


class TestGuiSurvivesDistributionExcludes:
    @pytest.mark.parametrize("module", GUI_MODULES)
    def test_importable_without_excluded_modules(self, module: str) -> None:
        ok, err = _import_without_excluded(module, _spec_excludes())
        assert ok, (
            f"{module} は配布版で import できない。"
            "spec の excludes が画面の依存を巻き込んでいる:\n"
            f"{err}"
        )


class TestSpecExcludesAreReadable:
    def test_excludes_found(self) -> None:
        """spec の書式が変わって除外を読めなくなったら気づけるように."""
        excludes = _spec_excludes()
        assert "torch" in excludes, "EXCLUDES を読めていない (torch が無い)"


class TestHelperToolsSurviveTheirExcludes:
    """打鍵サーバ・音声送出も、それぞれの除外設定で起動できること.

    **入口のモジュールが読めなければ、窓が開いた瞬間に落ちる。** 受信アプリと
    違ってボタンは無いので「押しても無反応」ではなく即死になるが、原因が
    見えない点は同じである。
    """

    @pytest.mark.parametrize(("spec_name", "module"), sorted(_OTHER_SPECS.items()))
    def test_entry_point_imports(self, spec_name: str, module: str) -> None:
        spec = _ROOT / "packaging" / f"{spec_name}.spec"
        ok, err = _import_without_excluded(module, _spec_excludes(spec))
        assert ok, (
            f"{spec_name} の excludes が入口の依存を巻き込んでいる:\n{err}"
        )

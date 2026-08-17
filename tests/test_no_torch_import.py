"""GUI を起動しても PyTorch が読み込まれないことを固定する.

**これが配布物の大きさを守る歯止めである。**

PyTorch は 2.8 GB あり、venv 全体 3.9 GB の 7 割を占める。PyInstaller は
「import されているもの」を配布物に入れるので、``import torch`` が 1 行でも
起動経路に残っていると**インストーラが 100 MB から 1 GB 超へ膨らむ**。

しかも厄介なのは、膨らんでも動いてしまうことである。テストは通り、アプリも
動き、ただ配布物だけが 10 倍になる。人間のレビューでは気づきにくいので、
機械で押さえる。

判定は**別プロセス**で行う。pytest 本体は他のテストのために torch を読み込んで
いるので、同じプロセスで ``sys.modules`` を見ても意味がない。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

# 起動経路で読み込まれてはいけないもの。
# torchaudio / torchvision も torch を引き込むので一緒に見る。
FORBIDDEN = ("torch", "torchaudio", "torchvision")


def _import_and_report(statement: str) -> tuple[set[str], str]:
    """別プロセスで ``statement`` を実行し、禁止モジュールの読み込みを調べる."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(_ROOT)!r})\n"
        f"{statement}\n"
        "loaded = sorted(m for m in sys.modules "
        f"if m == {FORBIDDEN[0]!r} or m.split('.')[0] in {FORBIDDEN!r})\n"
        "print('LOADED:' + ','.join(loaded))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=_ROOT, timeout=300,
    )
    if proc.returncode != 0:
        return set(), proc.stderr
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("LOADED:")), "LOADED:"
    )
    names = {n for n in line[len("LOADED:"):].split(",") if n}
    return {n.split(".")[0] for n in names}, proc.stderr


class TestDecodePathIsTorchFree:
    @pytest.mark.parametrize("module", [
        "src.infer.ctc",
        "src.infer.mel_params",
        "src.infer.backend",
        "src.infer.onnx_engine",
        "src.infer.sliding_window",
        "src.tokens.converter",
        "src.infer.word_correct",
    ])
    def test_module_does_not_pull_torch(self, module: str) -> None:
        loaded, err = _import_and_report(f"import {module}")
        assert not err.strip(), f"{module} の import で失敗:\n{err}"
        assert loaded == set(), (
            f"{module} を import したら {sorted(loaded)} が読み込まれました。\n"
            "配布物に PyTorch (2.8 GB) が入ってしまいます。"
            "torch の import は関数の中に閉じ込めてください "
            "(src/infer/backend.py の書き方に倣うこと)。"
        )


class TestGuiEntryIsTorchFree:
    def test_main_window_import_does_not_pull_torch(self) -> None:
        """**主画面のモジュールを読むだけで torch が来ないこと。**"""
        loaded, err = _import_and_report("import src.app.main_window")
        assert not err.strip(), f"main_window の import で失敗:\n{err}"
        assert loaded == set(), (
            f"src.app.main_window を import したら {sorted(loaded)} が"
            "読み込まれました。起動経路から torch を外してください。"
        )

    def test_redecode_worker_import_does_not_pull_torch(self) -> None:
        loaded, err = _import_and_report("import src.app.redecode_worker")
        assert not err.strip(), f"redecode_worker の import で失敗:\n{err}"
        assert loaded == set(), f"redecode_worker が {sorted(loaded)} を読み込みました"


class TestTorchPathStillWorks:
    """**torch を外したのではなく、遅延させただけ**であることを確かめる.

    学習・評価は今までどおり PyTorch で動く必要がある。
    """

    def test_engine_module_still_provides_torch_path(self) -> None:
        loaded, err = _import_and_report("import src.infer.engine")
        assert not err.strip(), err
        assert "torch" in loaded, (
            "src.infer.engine は PyTorch 経路そのものなので、"
            "torch を読み込むのが正しい"
        )

    def test_factory_can_still_build_torch_engine(self) -> None:
        """拡張子が ``.pt`` なら PyTorch 経路が選ばれること."""
        from src.infer.backend import is_onnx_model

        assert not is_onnx_model("models/full/best_infer.pt")

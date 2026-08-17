# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の設定 (Windows 配布用).

    python -m PyInstaller packaging/cw-decoder.spec --noconfirm

**PyTorch を入れないことが最重要である。** PyTorch は 2.8 GB あり、入ると
配布物が 1 GB を超える。``src/infer/backend.py`` が torch の import を関数の
中に閉じ込めてあるので、起動経路からは辿れない。それでも念のため
``excludes`` にも書いておく (依存解析が想定外の経路で拾うことがある)。

歯止め: ``tests/test_no_torch_import.py``。ビルド後の実測は
``scripts/build_installer.py`` が報告する。
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).parent          # noqa: F821  (SPECPATH は PyInstaller が入れる)

# --- 同梱するファイル -------------------------------------------------------
# 配布版は ONNX だけを持つ。**置き場所は src/app/resources.py の
# BUNDLED_MODEL_REL と揃えること** (ずれると「モデルが見つかりません」になる)。
datas = [
    (str(ROOT / "web" / "public" / "model" / "cw.onnx"), "model"),
]

# 取扱説明書 (HTML) と、展開したフォルダの直下に置く README。
#
# **どちらも欠けたまま配ってはいけない。** インストーラは取説へのショートカットを
# `Check: FileExists` 付きで作るので、無ければ**黙ってショートカットだけ消える**。
# 気づけるように、ここで止める。
_manual = ROOT / "docs" / "manual"
if not _manual.is_dir():
    raise SystemExit(f"取扱説明書が見つかりません: {_manual}")
datas.append((str(_manual), "manual"))

# **README はここに書かない。** onedir の datas は必ず `_internal` の下に入るが、
# README は展開したフォルダの**直下**に要る (ZIP にはスタートメニューが無く、
# 取説は _internal の中にあるため、入口を 1 枚置いておきたい)。
# ビルド後に scripts/build_installer.py が置く。

# sounddevice (PortAudio) と soundfile (libsndfile) は DLL を抱えている。
# 自動収集しないと、音が一切入らない実行ファイルができる。
binaries = collect_dynamic_libs("sounddevice") + collect_dynamic_libs("soundfile")

# --- 配布物から外すもの -----------------------------------------------------
EXCLUDES = [
    # 学習・評価まわり。**ここが本体の削減 (2.8 GB)**
    "torch", "torchaudio", "torchvision", "onnx", "sympy", "networkx",
    "src.train", "src.synth", "src.finetune", "src.eval",
    # 開発用
    "pytest", "_pytest", "ruff", "mypy", "PyInstaller",
    "IPython", "jupyter", "notebook",
    # 使っていない GUI / 描画系
    "tkinter", "matplotlib", "PIL",
    # PySide6 のうち使っていないもの (Qt は部品ごとに重い)
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtMultimedia",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtBluetooth",
    "PySide6.QtPdf", "PySide6.QtDesigner", "PySide6.QtTest",
]
# **``shiboken6`` を除外してはいけない。** PySide6 の中核 (Python と Qt を繋ぐ
# バインディング) で、外すと `No module named 'shiboken6.Shiboken'` で
# 起動できない。2026-08-16 に実際にこれで起動しなくなった。
#
# 除外リストは効き目が大きいぶん危ない。**足したら必ず起動を確かめること。**
# 名前を見て「使っていなさそう」で判断すると、今回のように中核を落とす。

a = Analysis(                                        # noqa: F821
    [str(ROOT / "scripts" / "run_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["onnxruntime"],
    hookspath=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)                                    # noqa: F821

exe = EXE(                                           # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cw-decoder",
    debug=False,
    strip=False,
    upx=False,          # UPX は onnxruntime の DLL を壊すことがあるので使わない
    # GUI アプリなので通常はコンソール窓を出さない。
    # **ただし窓が無いと起動時の例外が読めない。** 環境変数で切り替えられる
    # ようにしておく (`$env:CW_BUILD_CONSOLE=1` を付けてビルドすると付く)。
    # 配布用は必ず既定 (False) で作ること。
    console=os.environ.get("CW_BUILD_CONSOLE") == "1",
    icon=None,
)

coll = COLLECT(                                      # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="cw-decoder",
)

# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の設定 (音声送出・Windows 配布用).

    python -m PyInstaller packaging/cw-audio-send.spec --noconfirm

受信機の音を取る PC に入れる。**受信アプリとは別の配布物**である
(運用者の指示、2026-08-31: 「異なった PC で使うケースがある」)。

**コンソールアプリとして作る** (``console=True``)。起動すると窓が開き、
入力デバイスを番号で選ばせ、そのままレベルと送信量を流し続ける。

``soxr`` は**必須**である。無いとサンプルレートの申告と実際の音が食い違い、
受信側のデコードが無言で全崩壊する (``scripts/audio_send.py`` の冒頭を参照)。
そのため ``excludes`` に入れないことはもちろん、DLL ごと確実に収める。
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).parent          # noqa: F821

# sounddevice (PortAudio) は DLL を抱えている。自動収集しないと音が一切入らない。
binaries = collect_dynamic_libs("sounddevice") + collect_dynamic_libs("soxr")

# **パッケージ名で丸ごと外さない** (受信アプリで 3 画面を壊した轍。2026-08-31)。
EXCLUDES = [
    "torch", "torchaudio", "torchvision", "onnx", "onnxruntime", "sympy", "networkx",
    "src.train", "src.finetune", "src.eval", "src.synth", "src.tx", "src.llm", "src.app",
    # 画面まわり (このプログラムは窓を持たない)
    "PySide6", "shiboken6", "tkinter", "matplotlib", "PIL", "pyqtgraph",
    "serial",
    "pytest", "_pytest", "ruff", "mypy", "PyInstaller",
]

a = Analysis(                                        # noqa: F821
    [str(ROOT / "scripts" / "audio_send.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=[],
    hiddenimports=["soxr"],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)                                    # noqa: F821

exe = EXE(                                           # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cw-audio-send",
    debug=False,
    strip=False,
    upx=False,          # UPX は音まわりの DLL を壊すことがあるので使わない
    console=True,       # **窓を出す。** 番号を選ばせ、レベルを流し続ける
    icon=None,
)

coll = COLLECT(                                      # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="cw-audio-send",
)

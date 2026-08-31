# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller の設定 (打鍵サーバ・Windows 配布用).

    python -m PyInstaller packaging/cw-key-server.spec --noconfirm

無線機に電鍵線を繋いだ PC に入れる。**受信アプリとは別の配布物**である
(運用者の指示、2026-08-31: 「異なった PC で使うケースがある」)。

**コンソールアプリとして作る** (``console=True``)。起動すると窓が開き、
COM ポートを番号で選ばせ、そのまま打鍵の様子を流し続ける。窓を閉じるか
Ctrl+C で終わる。設定用の窓を別に作らないのは、動作ログを流す窓がどのみち
要るからである。

PySide6・onnxruntime・モデルは**入らない**。依存は pyserial と numpy だけ
(numpy は ``src.synth.keying`` が打鍵列を組むのに使う)。
"""
from pathlib import Path

ROOT = Path(SPECPATH).parent          # noqa: F821  (SPECPATH は PyInstaller が入れる)

# 受信アプリと同じ轍を踏まないこと。**パッケージ名で丸ごと外さない。**
# `src.synth` を丸ごと除外して 3 画面を壊した (2026-08-31)。
# 歯止めは `tests/test_frozen_imports.py`。
EXCLUDES = [
    # 学習・評価まわり (打鍵には要らない)
    "torch", "torchaudio", "torchvision", "onnx", "onnxruntime", "sympy", "networkx",
    "src.train", "src.finetune", "src.eval", "src.infer", "src.llm", "src.app",
    "src.synth.dataset", "src.synth.synthesizer",
    "src.synth.noise", "src.synth.text_generator",
    # 画面まわり (このプログラムは窓を持たない)
    "PySide6", "shiboken6", "tkinter", "matplotlib", "PIL", "pyqtgraph",
    # 音まわり (打鍵に音は要らない)
    "sounddevice", "soundfile", "soxr", "scipy",
    # 開発用
    "pytest", "_pytest", "ruff", "mypy", "PyInstaller",
]

a = Analysis(                                        # noqa: F821
    [str(ROOT / "scripts" / "cw_key_server.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=["serial.tools.list_ports"],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)                                    # noqa: F821

exe = EXE(                                           # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cw-key-server",
    debug=False,
    strip=False,
    upx=False,
    console=True,        # **窓を出す。** 番号を選ばせ、打鍵のログを流し続ける
    icon=None,
)

coll = COLLECT(                                      # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="cw-key-server",
)

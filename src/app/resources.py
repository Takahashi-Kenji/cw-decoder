"""同梱ファイルの探し方 (開発環境と配布物の両方で動く).

配布版 (PyInstaller) では、利用者は ``--ckpt`` を打たない。アイコンを叩くだけで
動く必要があるので、**同梱したモデルを自力で見つける**。

PyInstaller は実行時に一時領域へ展開し、その場所を ``sys._MEIPASS`` に入れる。
開発環境ではリポジトリのルートを使う。呼ぶ側はどちらか意識しなくてよい。

**このモジュールは torch を import してはいけない。**
"""
from __future__ import annotations

import sys
from pathlib import Path

# 配布物の中でのモデルの位置 (spec の datas と揃えること)
BUNDLED_MODEL_REL = Path("model") / "cw.onnx"

# 開発環境で探す場所。**ONNX を先に見る** (配布物と同じ経路で動作確認したいため)。
SOURCE_MODEL_CANDIDATES: tuple[Path, ...] = (
    Path("web") / "public" / "model" / "cw.onnx",
    Path("models") / "full" / "best_infer.pt",
)


def is_frozen() -> bool:
    """PyInstaller で固めた実行ファイルとして動いているか."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """同梱ファイルの置き場所.

    配布物では PyInstaller の展開先、開発環境ではリポジトリのルート。
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent.parent


def default_model_path() -> Path | None:
    """同梱モデルの場所。無ければ ``None``.

    **見つからなくても例外にしない。** 起動できないより、未学習モデルで画面を
    出して「モデルが見つかりません」と伝える方がよい。
    """
    root = bundle_dir()
    bundled = root / BUNDLED_MODEL_REL
    if bundled.exists():
        return bundled
    for rel in SOURCE_MODEL_CANDIDATES:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def resolve_model_path(saved: str | Path | None) -> Path | None:
    """設定に保存されたパスを、実際に使えるパスへ解決する.

    保存されたパスが**もう無い**ことは簡単に起きる (インストーラで入れ直した、
    モデルを移動した)。そのときは黙って同梱モデルへ戻す。設定に残った古い
    パスのせいで起動できないのは、利用者から見て理不尽である。
    """
    if saved:
        path = Path(saved)
        if path.exists():
            return path
    return default_model_path()


__all__ = [
    "BUNDLED_MODEL_REL",
    "bundle_dir",
    "default_model_path",
    "is_frozen",
    "resolve_model_path",
]

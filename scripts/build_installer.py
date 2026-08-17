"""Windows 配布物 (ZIP + setup.exe) を作る.

    python scripts/build_installer.py              # 全部
    python scripts/build_installer.py --skip-exe   # 既存の dist から ZIP/インストーラだけ

手順:

1. PyInstaller で ``dist/cw-decoder/`` を作る
2. README を展開フォルダの直下に置く
3. **配布物に PyTorch が混じっていないことを確かめる** (混じると 10 倍になる)
4. 持ち運び ZIP を作る
5. Inno Setup があれば ``setup.exe`` も作る (無ければ飛ばして案内を出す)

**大きさは毎回報告する。** 静かに膨らむのがいちばん怖い失敗で、
動いてしまうぶん気づきにくい。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SPEC = _PROJECT_ROOT / "packaging" / "cw-decoder.spec"
ISS = _PROJECT_ROOT / "packaging" / "cw-decoder.iss"
README_SRC = _PROJECT_ROOT / "packaging" / "dist_files" / "README.txt"
DIST = _PROJECT_ROOT / "dist"
BUNDLE = DIST / "cw-decoder"

# 混じってはいけないもの。名前がこれで始まるファイルが 1 つでもあれば失敗。
FORBIDDEN_PREFIXES = ("torch", "libtorch", "torchaudio")

# Inno Setup のコンパイラを探す場所。
#
# **ユーザ領域を忘れないこと。** `winget install JRSoftware.InnoSetup` は
# 管理者でなければ `%LOCALAPPDATA%\Programs\Inno Setup 6` に入れる。
# Program Files しか見ていなかったため、導入済みなのに「見つかりません」と
# 言って setup.exe を作らなかった (2026-08-17 に実際に踏んだ)。
# 環境変数 ISCC で明示指定もできる。
ISCC_CANDIDATES = (
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Default\AppData\Local"))
    / "Programs" / "Inno Setup 6" / "ISCC.exe",
)


def find_iscc() -> Path | None:
    """Inno Setup のコンパイラの場所を返す. 無ければ ``None``.

    順に、環境変数 ``ISCC`` → PATH → 既知の導入先を見る。
    """
    override = os.environ.get("ISCC")
    if override:
        path = Path(override)
        return path if path.exists() else None
    found = shutil.which("ISCC")
    if found:
        return Path(found)
    return next((p for p in ISCC_CANDIDATES if p.exists()), None)


def _size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 / 1024


def build_exe() -> None:
    print("[1/5] PyInstaller でビルドします…")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
         "--distpath", str(DIST), "--workpath", str(_PROJECT_ROOT / "build")],
        cwd=_PROJECT_ROOT, check=True,
    )


def place_readme() -> Path:
    """README を**展開したフォルダの直下**に置く.

    spec の ``datas`` に書くと onedir では ``_internal`` の下に入ってしまう。
    ZIP (持ち運び版) にはスタートメニューが無く、取説も ``_internal`` の中に
    あるので、入口になる 1 枚は直下に要る。
    """
    print("[2/5] README を置きます…")
    if not README_SRC.is_file():
        raise SystemExit(f"配布用 README がありません: {README_SRC}")
    dest = BUNDLE / README_SRC.name
    shutil.copy2(README_SRC, dest)
    try:
        shown = dest.relative_to(_PROJECT_ROOT)
    except ValueError:          # 出力先がプロジェクトの外にある場合
        shown = dest
    print(f"    {shown}")
    return dest


def check_no_torch() -> None:
    """**配布物に PyTorch が混じっていないこと。**

    ここで止めるのは、混じっても動いてしまうからである。テストは通り、
    アプリも起動し、配布物だけが 10 倍になる。人間は気づけない。
    """
    print("[3/5] PyTorch の混入を確かめます…")
    found = [
        f for f in BUNDLE.rglob("*")
        if f.is_file() and f.name.lower().startswith(FORBIDDEN_PREFIXES)
    ]
    if found:
        sample = "\n".join(f"    {f.relative_to(BUNDLE)}" for f in found[:10])
        raise SystemExit(
            f"配布物に PyTorch が混じっています ({len(found)} 件):\n{sample}\n"
            "起動経路のどこかで torch が import されています。"
            "`pytest tests/test_no_torch_import.py` で場所を特定してください。"
        )
    print("    PyTorch 由来のファイル: 0 件")


def make_zip(version: str) -> Path:
    print("[4/5] 持ち運び ZIP を作ります…")
    out = DIST / f"cw-decoder-{version}-portable.zip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(BUNDLE.rglob("*")):
            if f.is_file():
                z.write(f, Path("cw-decoder") / f.relative_to(BUNDLE))
    return out


def make_installer() -> Path | None:
    print("[5/5] インストーラを作ります…")
    iscc = find_iscc()
    if iscc is None:
        print("    Inno Setup が見つかりません。setup.exe は作りませんでした。")
        print("    導入: winget install JRSoftware.InnoSetup")
        print("    別の場所に入れてある場合は 環境変数 ISCC で指定してください。")
        return None
    print(f"    Inno Setup: {iscc}")
    subprocess.run([str(iscc), str(ISS)], cwd=_PROJECT_ROOT, check=True)
    made = sorted(DIST.glob("cw-decoder-*-setup.exe"))
    return made[-1] if made else None


def _version_from_iss() -> str:
    for line in ISS.read_text(encoding="utf-8").splitlines():
        if line.startswith("#define AppVersion"):
            return line.split('"')[1]
    return "0.0.0"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-exe", action="store_true", help="PyInstaller を飛ばす")
    args = p.parse_args(argv)

    if not args.skip_exe:
        build_exe()
    if not BUNDLE.is_dir():
        raise SystemExit(f"{BUNDLE} がありません。--skip-exe を外して実行してください。")

    place_readme()
    check_no_torch()
    version = _version_from_iss()
    zip_path = make_zip(version)
    setup_path = make_installer()

    print("\n" + "=" * 56)
    print(f"  展開後      : {_size_mb(BUNDLE):7.1f} MB   {BUNDLE}")
    print(f"  持ち運び ZIP : {_size_mb(zip_path):7.1f} MB   {zip_path.name}")
    if setup_path:
        print(f"  インストーラ  : {_size_mb(setup_path):7.1f} MB   {setup_path.name}")
    print("=" * 56)
    if shutil.which("signtool") is None:
        print("\n注意: 署名していないため、初回起動時に SmartScreen の警告が出ます。")
        print("      「詳細情報」→「実行」で進めます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

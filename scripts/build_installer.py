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
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

README_SRC = _PROJECT_ROOT / "packaging" / "dist_files" / "README.txt"
DIST = _PROJECT_ROOT / "dist"


@dataclass(frozen=True)
class Target:
    """配布物 1 つ分.

    **3 つある** (運用者の指示、2026-08-31:「異なった PC で使うケースがある」)。
    受信アプリ / 打鍵サーバ / 音声送出 をそれぞれ別のインストーラで配る。
    """

    key: str                 # --target に書く名前
    name: str                # dist の下のフォルダ名 = exe の名前
    title: str               # 画面に出す名前
    readme: bool = False     # 展開フォルダ直下に README を置くか (本体だけ)

    @property
    def spec(self) -> Path:
        return _PROJECT_ROOT / "packaging" / f"{self.name}.spec"

    @property
    def iss(self) -> Path:
        return _PROJECT_ROOT / "packaging" / f"{self.name}.iss"

    @property
    def bundle(self) -> Path:
        return DIST / self.name


TARGETS: tuple[Target, ...] = (
    Target("app", "cw-decoder", "受信アプリ", readme=True),
    Target("key-server", "cw-key-server", "打鍵サーバ"),
    Target("audio-send", "cw-audio-send", "音声送出"),
)

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


def build_exe(target: Target) -> None:
    print(f"[1/5] PyInstaller でビルドします ({target.title})…")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(target.spec), "--noconfirm",
         "--distpath", str(DIST), "--workpath", str(_PROJECT_ROOT / "build")],
        cwd=_PROJECT_ROOT, check=True,
    )


def place_readme(target: Target) -> Path:
    """README を**展開したフォルダの直下**に置く.

    spec の ``datas`` に書くと onedir では ``_internal`` の下に入ってしまう。
    ZIP (持ち運び版) にはスタートメニューが無く、取説も ``_internal`` の中に
    あるので、入口になる 1 枚は直下に要る。
    """
    print("[2/5] README を置きます…")
    if not README_SRC.is_file():
        raise SystemExit(f"配布用 README がありません: {README_SRC}")
    dest = target.bundle / README_SRC.name
    shutil.copy2(README_SRC, dest)
    try:
        shown = dest.relative_to(_PROJECT_ROOT)
    except ValueError:          # 出力先がプロジェクトの外にある場合
        shown = dest
    print(f"    {shown}")
    return dest


def check_no_torch(target: Target) -> None:
    """**配布物に PyTorch が混じっていないこと。**

    ここで止めるのは、混じっても動いてしまうからである。テストは通り、
    アプリも起動し、配布物だけが 10 倍になる。人間は気づけない。
    """
    print("[3/5] PyTorch の混入を確かめます…")
    found = [
        f for f in target.bundle.rglob("*")
        if f.is_file() and f.name.lower().startswith(FORBIDDEN_PREFIXES)
    ]
    if found:
        sample = "\n".join(f"    {f.relative_to(target.bundle)}" for f in found[:10])
        raise SystemExit(
            f"配布物に PyTorch が混じっています ({len(found)} 件):\n{sample}\n"
            "起動経路のどこかで torch が import されています。"
            "`pytest tests/test_no_torch_import.py` で場所を特定してください。"
        )
    print("    PyTorch 由来のファイル: 0 件")


def make_zip(target: Target, version: str) -> Path:
    print("[4/5] 持ち運び ZIP を作ります…")
    out = DIST / f"{target.name}-{version}-portable.zip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(target.bundle.rglob("*")):
            if f.is_file():
                z.write(f, Path(target.name) / f.relative_to(target.bundle))
    return out


def make_installer(target: Target) -> Path | None:
    print("[5/5] インストーラを作ります…")
    iscc = find_iscc()
    if iscc is None:
        print("    Inno Setup が見つかりません。setup.exe は作りませんでした。")
        print("    導入: winget install JRSoftware.InnoSetup")
        print("    別の場所に入れてある場合は 環境変数 ISCC で指定してください。")
        return None
    print(f"    Inno Setup: {iscc}")
    subprocess.run([str(iscc), str(target.iss)], cwd=_PROJECT_ROOT, check=True)
    made = sorted(DIST.glob(f"{target.name}-*-setup.exe"))
    return made[-1] if made else None


def _version_from_iss(target: Target) -> str:
    for line in target.iss.read_text(encoding="utf-8").splitlines():
        if line.startswith("#define AppVersion"):
            return line.split('"')[1]
    return "0.0.0"


def build_target(target: Target, skip_exe: bool) -> tuple[Path, Path | None]:
    """1 つ分を作る. 戻り値は (ZIP, setup.exe か None)."""
    print("")
    print(f"### {target.title} ({target.name})")
    if not skip_exe:
        build_exe(target)
    if not target.bundle.is_dir():
        raise SystemExit(f"{target.bundle} がありません。--skip-exe を外して実行してください。")

    if target.readme:
        place_readme(target)
    else:
        print("[2/5] README は本体だけなので飛ばします")
    check_no_torch(target)
    version = _version_from_iss(target)
    return make_zip(target, version), make_installer(target)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-exe", action="store_true", help="PyInstaller を飛ばす")
    p.add_argument(
        "--target", default="all",
        choices=["all", *(t.key for t in TARGETS)],
        help="作るもの (既定 all = 3 つとも)",
    )
    args = p.parse_args(argv)

    chosen = TARGETS if args.target == "all" else tuple(
        t for t in TARGETS if t.key == args.target
    )
    made: list[tuple[Target, Path, Path | None]] = []
    for target in chosen:
        zip_path, setup_path = build_target(target, args.skip_exe)
        made.append((target, zip_path, setup_path))

    print("")
    print("=" * 56)
    for target, zip_path, setup_path in made:
        print(f"  {target.title}")
        print(f"    展開後      : {_size_mb(target.bundle):7.1f} MB   {target.bundle.name}")
        print(f"    持ち運び ZIP : {_size_mb(zip_path):7.1f} MB   {zip_path.name}")
        if setup_path:
            print(f"    インストーラ  : {_size_mb(setup_path):7.1f} MB   {setup_path.name}")
    print("=" * 56)
    if shutil.which("signtool") is None:
        print("")
        print("注意: 署名していないため、初回起動時に SmartScreen の警告が出ます。")
        print("      「詳細情報」→「実行」で進めます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

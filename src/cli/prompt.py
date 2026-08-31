"""起動時の対話 (配布版の打鍵サーバ / 音声送出で使う).

**exe をダブルクリックした人が、コマンドを覚えずに始められること**が目的
(運用者の指示、2026-08-31)。COM ポート番号も音声デバイス番号もその PC でしか
分からないので、一覧を出して番号で選ばせ、選んだ値を覚えて次回は Enter だけで
済むようにする。

**ハードウェアには触らない。** 候補・前回値・入出力の関数を受け取り、選択を
返すだけにしてある。実機が無くてもテストできる (``tests/test_cli_prompt.py``)。

**引数が与えられているときは呼ばないこと。** ``--port`` や ``--device`` を
書いた従来の使い方、自動化、テストから対話に落ちてはいけない。判断は呼ぶ側
(``scripts/``) が行う。
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 覚えた値の置き場所。設定・語彙と同じ場所に揃える (アンインストールで消えない)。
REMEMBERED_DIR = Path.home() / ".cw-decorder"

ReadFn = Callable[[str], str]
WriteFn = Callable[[str], Any]

# 「はい」とみなす答え。**空 (Enter) も「はい」**。
_YES = {"", "y", "yes", "は", "はい"}


@dataclass(frozen=True)
class Choice:
    """選択肢 1 つ.

    ``value`` が実際に使う値、``label`` が画面に出す 1 行、``short`` は
    「前回: …」に出す短い名前 (省略時は ``value``)。

    **``value`` を一覧に出さないこと。** 値がただの番号である音声デバイスでは、
    行番号と並んで数字が 2 つ出てしまい、どちらを打つのか読み取れない
    (2026-08-31 に運用者が実際に取り違えた)。値を見せたい COM ポートのような
    ものは、``label`` の先頭に自分で入れる。
    """

    value: str
    label: str = ""
    short: str = ""

    @property
    def name(self) -> str:
        """「前回: …」に出す短い名前."""
        return self.short or self.value

    def line(self, number: int) -> str:
        return f"    {number}) {self.label or self.value}".rstrip()


def ask_choice(
    title: str,
    choices: Sequence[Choice],
    remembered: str | None,
    read: ReadFn,
    write: WriteFn,
) -> Choice:
    """番号で 1 つ選ばせる. Enter だけなら既定 (前回値、無ければ先頭).

    ``remembered`` が候補に無いとき (前回の機器が繋がっていない) は、それを
    既定にしない。**無い機器の名前を既定として見せる方が紛らわしい。**

    Raises:
        LookupError: 候補が 1 つも無いとき。
    """
    if not choices:
        raise LookupError(f"{title}が 1 つも見つかりません")

    default = next((c for c in choices if c.value == remembered), choices[0])
    write(f"\n  使える{title}:")
    for i, choice in enumerate(choices, start=1):
        write(choice.line(i))

    kind = "前回" if default.value == remembered else "既定"
    hint = f"{kind}: {default.name}"
    while True:
        answer = read(f"  番号を選んでください [{hint}] > ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        write(f"  1〜{len(choices)} の番号を入れてください。もう一度どうぞ。")


def ask_yes(question: str, read: ReadFn, write: WriteFn) -> bool:
    """はい/いいえ。**Enter は「はい」**.

    電波が出る前の確認に使う。止めたい人は ``n`` と打つ。
    """
    answer = read(f"  {question} [Enter で続行 / n でやめる] > ").strip().lower()
    if answer in _YES:
        return True
    write("  やめました。")
    return False


def load_remembered(path: Path) -> dict[str, Any]:
    """前回の選択を読む. 無い・壊れていれば空を返す (起動は止めない)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_remembered(path: Path, data: dict[str, Any]) -> None:
    """今回の選択を覚える. 書けなくても起動を止めない."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


__all__ = [
    "REMEMBERED_DIR",
    "Choice",
    "ask_choice",
    "ask_yes",
    "load_remembered",
    "save_remembered",
]

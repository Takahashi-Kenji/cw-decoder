"""運用者の経歴と読み辞書 (送信用).

なぜ設定ファイルと分けるか
--------------------------
``settings.json`` は「アプリの挙動の設定」であるのに対し、こちらは**内容**である。
コールサインや自己紹介文は、閾値や BPF の帯域とは性質が違う。混ぜると、
設定をリセットしたいだけなのに自分の情報まで消えることになる。

なぜ欧文用と和文用を分けるか
----------------------------
**運用者が必要としているのは「同じものの 2 通りの書き方」ではなく、
独立した 2 つの値である** (2026-08-12 の聞き取り)。

===========  ==============  ==================
欄           欧文用          和文用
===========  ==============  ==================
名前         ``TARO``       ``タロウ``
QTH          ``YOKOHAMA``     ``ヨコハマシ``
リグ         ``FT991``       ``「FT991」``
===========  ==============  ==================

``「FT991」`` は ``FT991`` の「読み」ではない。別の値である。

**この構造にしたことで、引き継いだ欠陥のうち経歴に由来する分が消えた。**
以前は ``field_readings()`` が「表示形 → 読み」の辞書を作り、``reading.py`` が
それを本文全体に当てていた。そのため欧文の本文の ``TARO`` が ``タロウ`` に
置き換わり、カタカナが混ざってモードが和文に倒れ、**文中の欧文がまるごと
送れなくなっていた** (2026-08-11 に実測)。差し込みのときにモードで側を選べば
済むので、変換辞書に注ぐ理由がそもそも無い。

**``reading_dictionary`` の分は残る。** そちらは運用者が意図して入れたもので、
編集画面が警告する (設計書 §4.3)。

コールサインを分けない理由
--------------------------
**和文の交信でもコールサインは欧文で送る。** 分ける意味が無いので 1 つだけ持つ。

古いファイル
------------
**移行は書かない** (設計書 §2.4)。旧い形 (``display`` / ``reading``) のファイルは
**空の欄として読まれる**。運用者の環境にはまだこのファイル自体が無く、
書き直せば済む。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from src.tokens.morse_tokens import DisplayMode

# 経歴の保存先。**テストは必ず一時パスを渡すこと** (利用者の実ファイルを壊さない)。
DEFAULT_PROFILE_PATH = Path.home() / ".cw-decorder" / "operator.json"

# 値が無いときに打つ文字。**欄による例外を作らない** (設計書 §2.2)。
# ``?`` は和文表にも欧文表にもあるのでどちらのモードでも送れ、CW では
# 「分からない」を意味するので交信としても通る。
UNKNOWN = "?"


@dataclass
class BilingualField:
    """欧文用と和文用の、独立した 2 つの値.

    **もう一方で代用しない。** 和文側が空でも欧文側の値は使わない
    (書き忘れと区別がつかなくなる)。空なら :data:`UNKNOWN` を打つ。
    """

    european: str = ""
    japanese: str = ""

    def for_mode(self, mode: DisplayMode | str) -> str:
        """モードに合う側を返す. 空なら ``""`` (``?`` に倒すのは呼び出し側).

        ``japanese`` のときだけ和文側。``european`` と ``any`` は欧文側である
        (``any`` の型は欧文の略語を主とするため。設計書 §3)。
        """
        return self.japanese if mode == "japanese" else self.european


@dataclass
class OperatorProfile:
    """自局の情報."""

    # **コールサインだけは 1 つ。** 和文交信でも欧文で送るため
    callsign: str = ""
    name: BilingualField = field(default_factory=BilingualField)
    qth: BilingualField = field(default_factory=BilingualField)
    rig: BilingualField = field(default_factory=BilingualField)
    antenna: BilingualField = field(default_factory=BilingualField)
    power: BilingualField = field(default_factory=BilingualField)
    # 自由記述 (趣味・経歴・よく話す話題)。LLM 返信案に文脈として渡す
    notes: str = ""
    # 運用者が育てる読み辞書。語 → 読み (カタカナ)。
    # **これは本文全体に当たる。** 欧文の語を入れると欧文の交信が壊れるので、
    # 編集画面が警告する (設計書 §4.3)。経歴の欄とは別物である。
    reading_dictionary: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorProfile:
        """辞書から作る. **読めない欄は空として扱う** (壊れても落ちない).

        旧い形 (``{"display": ..., "reading": ...}``) は
        ``european``/``japanese`` を持たないので空の欄になる。移行はしない。
        """
        valid = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in valid:
                continue
            if key in _BILINGUAL_FIELDS:
                if isinstance(value, dict):
                    kwargs[key] = BilingualField(
                        european=str(value.get("european", "")),
                        japanese=str(value.get("japanese", "")),
                    )
                continue
            if key == "reading_dictionary":
                if isinstance(value, dict):
                    kwargs[key] = {str(k): str(v) for k, v in value.items()}
                continue
            # callsign / notes は文字列。旧い形では dict が入っているので捨てる
            if isinstance(value, str):
                kwargs[key] = value
        return cls(**kwargs)


# 2 値を持つ欄の名前。**ここが唯一の一覧** (画面・差し込み・読み込みが参照する)。
_BILINGUAL_FIELDS: tuple[str, ...] = ("name", "qth", "rig", "antenna", "power")

BILINGUAL_FIELDS = _BILINGUAL_FIELDS


def load_profile(path: Path | str = DEFAULT_PROFILE_PATH) -> OperatorProfile:
    """経歴を読み込む. 無い/壊れていれば空の経歴を返す.

    **壊れていても上書きしない** — 呼び出し側が保存するまでファイルは残る
    (手で直せる余地を残す。設計書 §6)。
    """
    path = Path(path)
    if not path.exists():
        return OperatorProfile()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return OperatorProfile()
    if not isinstance(data, dict):
        return OperatorProfile()
    return OperatorProfile.from_dict(data)


def save_profile(
    profile: OperatorProfile, path: Path | str = DEFAULT_PROFILE_PATH
) -> None:
    """経歴を保存する.

    **途中で失敗しても既存のファイルを壊さない** (一時ファイルに書いて置き換える。
    型の保存 ``src/tx/templates.py`` と同じ流儀)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
    tmp.replace(path)


__all__ = [
    "BILINGUAL_FIELDS",
    "DEFAULT_PROFILE_PATH",
    "UNKNOWN",
    "BilingualField",
    "OperatorProfile",
    "load_profile",
    "save_profile",
]

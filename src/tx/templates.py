"""返信の型 — 前もって書いた文に欄を差し込む.

なぜ型が要るか
--------------
**こちらが主導する場面には受信内容が無い** (CQ、こちらから呼ぶ、締め)。
LLM に作らせる材料が無いので、前もって書いた文を呼ぶのが確実である。
相手に応答する場面でも、良い案が出たら型として取っておけば次から即座に使える。

欄の差し込みは符号化より前
--------------------------
``{相手コール}`` のような欄は**日本語ボックスへ入れる前に**差し込む。
逆にすると欄がカナ変換器 (pykakasi) を通って壊れる。

知らない ``{…}`` は触らない
---------------------------
**これが効くのは ``{HORE}`` と ``{RATA}`` である。** 型の中に書いたマーカーが
差し込みで壊れてはいけない。``PLACEHOLDERS`` に載っている名前だけを置き換える。

正規表現は入れ子の ``{`` を許さない
------------------------------------
``[^}]*`` だと、閉じていない ``{`` が後方の正しい欄まで飲み込んでしまう
(例: ``{相手コール DE {自局コール}`` で ``{自局コール}`` ごと 1 つの
「不明な欄」として食われ、置換もされず ``?`` にも
現れなくなる)。これは「漏れに気づけなくなる」の最悪形 (漏れの検出自体が
漏れる) なので、``[^{}]*`` にして欄名の中に ``{`` を許さない。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tokens.morse_tokens import DisplayMode
from src.tx.encoder import find_unsendable
from src.tx.profile import UNKNOWN, OperatorProfile
from src.tx.reading import to_sendable_kana

# 差し込める欄。**増やすほど型が書きにくくなるので絞る** (設計書 §4.1)。
PLACEHOLDERS: frozenset[str] = frozenset(
    {
        # 経歴から
        "自局コール", "名前", "QTH", "リグ", "アンテナ", "出力",
        # 拾った値、または運用者が打った値
        "相手コール", "相手名前",
        # **その交信のもの。経歴には置かない** (運用のたびに経歴を開くことになる)。
        # 気温は数字で書く (`20`)。数字は両方の表にあるので和文でも通る
        "天気", "気温",
        # **こちらが相手に与える RST。** 相手からもらった値ではない
        "RST",
    }
)

# 経歴の欄名 → 差し込む欄名
# **``callsign`` はここに入れない。** 1 つしかなく `for_mode` を持たない
_PROFILE_FIELDS: dict[str, str] = {
    "name": "名前",
    "qth": "QTH",
    "rig": "リグ",
    "antenna": "アンテナ",
    "power": "出力",
}

# **欄名に ``{`` を含めない** (中括弧の入れ子/未閉じで後方の正しい欄を
# 飲み込まないため。上のモジュール docstring 参照)。
_FIELD_RE = re.compile(r"\{([^{}]*)\}")

# 型の保存先。**テストは必ず一時パスを渡すこと** (利用者の実ファイルを壊さない)。
# 設定 (settings.json) ではなく内容なので、経歴と同じく別ファイルにする。
DEFAULT_TEMPLATES_PATH = Path.home() / ".cw-decorder" / "templates.json"

# 型を検証するときに欄へ入れる仮の値。
#
# **数字にするのには理由がある。** 数字は欧文表にも和文表にもあるので、
# どちらのモードの型に差し込んでもモードを倒さない。カタカナを入れると
# 欧文の型が和文と判定され、文中の欧文がまるごと「送れない」になる。
#
# 欄が空のせいで「送れない」と言わないようにするため、必ず何かを入れる
# (空の値は fill が `?` に倒す)。
_PROBE_VALUES: dict[str, str] = dict.fromkeys(PLACEHOLDERS, "0")


@dataclass(frozen=True)
class ReplyTemplate:
    """返信の型.

    Args:
        name: 画面の一覧に出す名前。
        mode: ``"european"`` / ``"japanese"`` / ``"any"``。
            **デコーダのモードに合う型だけを一覧に出す**ために使う。
        text: 本文。漢字かな交じりで書いてよい (差し込みの後にカナ変換を通る)。
    """

    name: str
    mode: str = "any"
    text: str = ""


def fill(text: str, values: dict[str, str]) -> str:
    """欄を差し込む. **知らない ``{…}`` には触らない。空の値は ``?`` になる。**

    ``{HORE}`` のような符号のマーカーは ``PLACEHOLDERS`` に無いのでそのまま残る。

    **空でも埋める。** 以前は空の値を差し込まず ``missing_placeholders`` が
    名指ししていたが、その仕組みは廃止した (設計書 §2.2)。運用者は送信文に
    出た ``?`` を見て、必要なら直してから確認・送信する。
    ``?`` は両方の符号表にあるので、どちらのモードでも送れる。

    1 回の呼び出しは ``re.sub`` の 1 パスで完結し、差し込んだ値の中身を
    再スキャンしない。ただし呼び出し側が ``fill`` の結果を別の ``fill`` に
    もう一度通すと、値の中にたまたま ``{欄名}`` の形が含まれていれば
    そこで展開される。差し込みは 1 回で完結させること。
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in PLACEHOLDERS:
            return match.group(0)          # {HORE} などはそのまま
        return values.get(name, "") or UNKNOWN

    return _FIELD_RE.sub(replace, text)


def profile_values(profile: OperatorProfile, mode: str) -> dict[str, str]:
    """経歴を差し込む値にする. **モードで側を選ぶ.**

    ============  ========
    型の ``mode``  使う側
    ============  ========
    ``japanese``  和文用
    ``european``  欧文用
    ``any``       欧文用
    ============  ========

    ``any`` が欧文側なのは、``any`` の型が欧文の略語 (``PSE AGN AGN K``) を
    主とするため (設計書 §3)。

    **空の欄も入れる。値は ``?`` になる** (:data:`~src.tx.profile.UNKNOWN`)。
    欄による例外を作らない。**もう一方の側で代用しない** — 和文側が空でも
    欧文側の値は使わない (書き忘れと区別がつかなくなる)。

    以前は「空の欄は入れない」で、埋まらなかった ``{…}`` を
    ``missing_placeholders`` が名指ししていた。**その仕組みは廃止した** —
    運用者は送信文に出た ``?`` を見て、必要なら直してから確認・送信する
    (2026-08-12 の運用者の判断)。``[確認]`` の関門は変わらない。

    ``callsign`` は 1 つしかないので、どちらのモードでも同じものを使う
    (和文の交信でもコールサインは欧文で送る)。
    """
    values: dict[str, str] = {"自局コール": profile.callsign or UNKNOWN}
    for attr, placeholder in _PROFILE_FIELDS.items():
        values[placeholder] = getattr(profile, attr).for_mode(mode) or UNKNOWN
    return values


def _str_field(item: dict[str, Any], key: str, default: str) -> str:
    """辞書から文字列の欄を取り出す. 型が違えば (無ければ) 既定値にする.

    **``item.get(key, default)`` では足りない。** キーが存在して値が
    ``None`` の場合 (``{"name": null}`` のような壊れ方)、``dict.get`` は
    デフォルトを使わず ``None`` を返すので、素直に ``str()`` を通すと
    ``"None"`` という文字列を書き込んでしまう。キーが無い場合と型が
    違う場合をまとめて既定値に倒す。
    """
    value = item.get(key)
    return value if isinstance(value, str) else default


def load_templates(path: Path | str = DEFAULT_TEMPLATES_PATH) -> list[ReplyTemplate]:
    """型を読み込む. 無い/壊れていれば空を返す.

    **壊れていても上書きしない。** 手で直せる余地を残す。
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("templates")
    if not isinstance(raw, list):
        return []
    out: list[ReplyTemplate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            ReplyTemplate(
                name=_str_field(item, "name", ""),
                mode=_str_field(item, "mode", "any"),
                text=_str_field(item, "text", ""),
            )
        )
    return out


def save_templates(
    templates: list[ReplyTemplate], path: Path | str = DEFAULT_TEMPLATES_PATH
) -> None:
    """型を保存する. **日本語はそのまま書く** (人が開いて読めるように).

    **一時ファイルに書いてから置き換える。** 書き込みの途中で失敗しても
    (ディスク満杯・強制終了等)、既存の正しいファイルを壊してはいけない。
    ``Path.replace`` (``os.replace``) は同一ドライブ上ならアトミックなので、
    置き換えの瞬間に壊れた状態を経由しない。失敗したときは一時ファイルを
    片付けてから例外を再送出する (ゴミを残さない)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "templates": [
            {"name": t.name, "mode": t.mode, "text": t.text} for t in templates
        ]
    }
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(path)


def unsendable_in_template(
    template: ReplyTemplate, profile: OperatorProfile | None = None
) -> str:
    """型に送れない文字があれば、その文字を並べて返す。無ければ空文字.

    **欄はすべて仮の値で埋めてから調べる。** 埋まっていないことは
    ``fill`` が ``?`` に倒すので、ここの関心ではない。

    一番よく起きるのは**ホレの中に欧文を書いてしまう**ことである
    (``{HORE}コンニチハ RST 599{RATA}``)。RST は欧文で送るものなので
    和文モードの中では送れない。交信中に気づくのでは遅いので保存時に見せる。

    Args:
        profile: 経歴。**渡さないと、経歴に読み (カタカナ) を入れている
            運用者にだけ起きる失敗を原理的に検出できない。** 仮の値 (数字)
            はどちらの表にもあるのでモードを倒さず、``{リグ}`` をホレの外に
            置いた型が「送れる」と判定されてしまう。実際にはそこへ
            ``エフティー キュウキュウイチ`` が入って欧文が全滅する
            (2026-08-11 のレビューで例文「設備の紹介」が該当した)。
            経歴を渡すと**実際に差し込まれる値**で調べ、さらに
            ``to_sendable_kana`` にも渡すので読み辞書の効き方まで同じになる。
    """
    values = dict(_PROBE_VALUES)
    if profile is not None:
        # 経歴から埋まる欄は実際の値で。残り (相手コール等) は仮の値のまま
        values.update(profile_values(profile, template.mode))
    filled = fill(template.text, values)
    # **実際に通る道と同じ道を通す。** 小書きカナ (ャュョッ) は
    # ``reading.SMALL_KANA_MAP`` で大書きに倒されてから符号化される。
    # 生の型だけを見ると ``キョウテン`` を「送れない」と誤判定する。
    #
    # ``to_sendable_kana`` は空文字を早期リターンするので、本文が空の
    # 型でも pykakasi を呼ばずに済む (import が重いので余計な起動を避ける)。
    converted = to_sendable_kana(filled, profile).text
    return "".join(bad.char for bad in find_unsendable(converted))


def templates_for_mode(
    templates: list[ReplyTemplate], mode: DisplayMode
) -> list[ReplyTemplate]:
    """そのモードで使える型だけを、元の順のまま返す.

    **``auto`` ではすべての型を出す。** ``auto`` は「相手に合わせて欧文と
    和文を切り替える」**主力の運用モード**であり、どちらの型も使う。ここを
    未知のモード扱いにしていたため、``auto`` の運用者からは ``any`` 以外の型が
    すべて消えていた (例文 10 個中 9 個。しかも理由が出なかった)。

    知らないモード文字列を渡しても (あるいは型の ``mode`` が手編集で
    壊れていても) 落ちない。一致する型が無ければ ``any`` の型だけが残る
    (fail-closed: **本当に未知のもの**は出さない側に倒す)。``auto`` は
    ``DisplayMode`` の正規の値なので、この fail-closed には当たらない。
    """
    if mode == "auto":
        return list(templates)
    return [t for t in templates if t.mode == mode or t.mode == "any"]


__all__ = [
    "DEFAULT_TEMPLATES_PATH",
    "PLACEHOLDERS",
    "ReplyTemplate",
    "fill",
    "load_templates",
    "profile_values",
    "save_templates",
    "templates_for_mode",
    "unsendable_in_template",
]

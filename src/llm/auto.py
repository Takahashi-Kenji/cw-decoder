"""清書をいつ・どこまで送るかを決める (純粋関数, Qt 非依存).

2 つのモードがある。どちらも**増分**で、一度清書した分は送り直さない。

**まとめて清書** (手動ボタン)
    未清書分を 1 回で全部送る。腰を据えて読むとき用。

**自動清書**
    **区切りが来た時点で、そこまでを送る。** 区切りは改行と、和文の区切点 ``、``
    段落 ``。``。文として完結したところで区切れるので、途中で切って清書するより
    自然な日本語になる。直前の分は「参考」として添え、話のつながりを保つ。

    当初は改行だけで発火させたが、**改行は無音 3 秒以上でしか入らず滅多に起きない**
    (運用者の指摘、2026-08-08)。句読点なら送信の途中でも区切れる。

    区切りが来ない長い送信のために、``interval_s`` 経過でも送る保険を持つ。
    **区切りによる発火は間隔で待たせない。**

なぜ増分か
----------
従来は確定テキスト全体を毎回送り直していた。同じ内容を何度も清書するので

* 結果が毎回揺れて、読んでいる途中で書き換わる
* ローカル LLM (Ollama) では 1 回に数秒かかるのに、その大半が再清書に消える
"""
from __future__ import annotations

from dataclasses import dataclass

# 清書を区切る文字。
#
# ``、`` は和文の区切点、``。`` は段落 (文の終わり)。どちらも符号表にある
# 実在の文字で、文の切れ目を表す。改行 (無音 3 秒) は滅多に起きず、それだけでは
# 自動清書がほとんど発火しなかった (運用者の指摘、2026-08-08)。
#
# **``。`` は 2026-08-12 まで ``」`` と書かれていた。** ``・-・-・・`` を ``」``
# と誤記していたためで、**同じ符号を指している**。デコード結果の表記が変わった
# のでここも直す。直さないと自動清書が文末で発火しなくなる。
#
# **``?`` を境界にしてはいけない。** 符号表には ``?`` もあるが、``?`` は
# 「読めなかった」印としても出力される。境界にすると確信度の低い箇所ごとに
# 発火してしまう。
BOUNDARY_CHARS = "\n、。"

# 参考として LLM に添える直前テキストの長さ (文字)。
# 長すぎると小さいモデルが参考部分まで出力してしまい、短すぎると話が繋がらない。
DEFAULT_LEAD_CHARS = 120


@dataclass
class AutoRefineState:
    """増分清書の進捗.

    ``refined_len`` は確定テキストのうち既に LLM へ送った文字数。
    確定テキストが作り直された (クリア・モード変更) 場合は 0 に戻す。
    """

    refined_len: int = 0
    last_time: float | None = None


@dataclass(frozen=True)
class RefineRequest:
    """次に送る清書要求."""

    text: str               # 清書させる本文
    lead: str               # 文脈として添える直前部分 (出力させない)
    refined_len: int        # 送信後に ``AutoRefineState.refined_len`` へ入れる値
    # "line_break" | "punctuation" | "interval" | "manual" (ログ・テスト用)
    reason: str


def pending_text(committed_text: str, state: AutoRefineState) -> str:
    """まだ清書していない部分を返す.

    確定テキストが縮んだ・別物に置き換わった場合は全体を未清書として扱う.
    """
    if state.refined_len > len(committed_text):
        return committed_text
    return committed_text[state.refined_len:]


def lead_text(
    committed_text: str, state: AutoRefineState, max_chars: int = DEFAULT_LEAD_CHARS
) -> str:
    """未清書分の直前にある確定テキスト (文脈用) を返す."""
    end = min(state.refined_len, len(committed_text))
    return committed_text[max(0, end - max_chars):end]


def should_refine(
    *, has_pending: bool, now: float, interval_s: float, state: AutoRefineState
) -> bool:
    """間隔での発火条件 (区切りが来ないときの保険).

    未清書分があり、かつ前回送信から ``interval_s`` 秒以上経過していれば送る。
    一度も送っていない場合は初回として間隔を免除する。
    """
    if not has_pending:
        return False
    if state.last_time is None:
        return True
    return now - state.last_time >= interval_s


def plan_refine_all(
    committed_text: str,
    state: AutoRefineState,
    *,
    lead_chars: int = DEFAULT_LEAD_CHARS,
) -> RefineRequest | None:
    """**まとめて清書**: 未清書分を 1 回で全部送る.

    送るものが無ければ ``None``。
    """
    pending = pending_text(committed_text, state)
    if not pending.strip():
        return None
    return RefineRequest(
        text=pending.strip(),
        lead=lead_text(committed_text, state, lead_chars),
        refined_len=len(committed_text),
        reason="manual",
    )


def plan_auto_refine(
    committed_text: str,
    state: AutoRefineState,
    *,
    now: float,
    interval_s: float,
    lead_chars: int = DEFAULT_LEAD_CHARS,
) -> RefineRequest | None:
    """**自動清書**: 区切りごとに送る. 送るものが無ければ ``None``.

    優先順:

    1. 未清書分に区切り (``BOUNDARY_CHARS``) があれば、**最後の区切りまでを送る**。
       途中は残す (文の途中で切って清書させない)。間隔では待たせない
    2. 区切りが無く ``interval_s`` 経過していれば、未清書分を全部送る (保険)
    """
    pending = pending_text(committed_text, state)
    if not pending.strip():
        return None

    lead = lead_text(committed_text, state, lead_chars)
    index = max(pending.rfind(ch) for ch in BOUNDARY_CHARS)
    if index >= 0:
        # 改行は行の区切りなので送らない。句読点は文の一部なので送る
        is_newline = pending[index] == "\n"
        chunk = pending[:index] if is_newline else pending[: index + 1]
        # 中身が区切り記号と空白だけなら送らない。``、`` だけを清書させても
        # 意味がなく、LLM の往復を無駄にする
        if chunk.strip(BOUNDARY_CHARS + " 　"):
            return RefineRequest(
                text=chunk.strip(),
                lead=lead,
                # 区切りの次の文字から未清書にする
                refined_len=state.refined_len + index + 1,
                reason="line_break" if is_newline else "punctuation",
            )
        # 区切りしかない (空行・先頭の読点) — 何も送らずに次を待つ
        return None

    if should_refine(
        has_pending=True, now=now, interval_s=interval_s, state=state
    ):
        return RefineRequest(
            text=pending.strip(),
            lead=lead,
            refined_len=len(committed_text),
            reason="interval",
        )
    return None


__all__ = [
    "BOUNDARY_CHARS",
    "DEFAULT_LEAD_CHARS",
    "AutoRefineState",
    "RefineRequest",
    "lead_text",
    "pending_text",
    "plan_auto_refine",
    "plan_refine_all",
    "should_refine",
]

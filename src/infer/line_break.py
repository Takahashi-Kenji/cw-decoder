"""確定テキストの改行 (送信のターンの切れ目と、文の終わりで行を分ける).

改行を入れる理由は 2 つある。

1. **トークン間隔** — 3 秒以上の無音は送信のターンの切れ目とみなす
2. **段落 ``。``** — 文の終わり (運用者の要望、2026-08-12)

1 だけでは、1 回の送信が長いと 1 行に延々と続く。

判定は**トークンの時刻差だけ**で行い、音声は見ない。確定テキストは毎回トークン列
全体から作り直されるので、判定がトークン列だけから決まることが重要である
(何度作り直しても同じ結果になり、「確定した文字は書き換わらない」という保証を壊さない)。

トーンを使わないのはこのため。`SlidingWindowDecoder` のリングバッファは
``window_s`` (既定 30 秒) しか保持しないので、古いトークンのトーンは作り直し時には
もう取れず、確定時に測って覚えておく状態管理が必要になる。

ブラウザ版の ``web/src/worker/render-mode.ts`` と同じロジック。
**ただし段落による改行 (2) はまだデスクトップ版だけである。** ブラウザ版は
装飾の位置 (``committedFallbacks``) を最終テキストの文字列インデックスで持って
おり、改行を挿入すると**それ以降の位置を全部ずらす必要がある**。入れるときは
位置の付け替えまで込みで行うこと。

    **判定はトークン間の「ピーク間距離」であって純粋な無音長ではない。**
    CTC の貪欲デコードが返す frame_start/frame_end はピーク位置でほぼ同一 (幅 0〜20 ms)
    であり、符号の実際の長さを持たない。したがってここで測る間隔には文字自身の長さが
    含まれる。文字は長くても 0.5 秒程度なので、閾値 3 秒なら実際の無音は 2.5 秒以上あり、
    ターンの切れ目の判定としては実用上問題ない。**ただし語間 (7 dot ≒ 0.4 秒) の判定には
    この値は使えない** (文字の長さに埋もれる。実測で全文字間にスペースが入った)。
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from src.infer.sliding_window import CommittedToken
from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import Mode

# 改行を入れる無音の長さ (秒) の既定値。
#
# 根拠: held-out 実録音 21 件で「1 回の送信の中に現れる無音」を測ると
# 中央値 650 ms・99%tile 2.0 秒・**最大 4.4 秒**だった。3.0 秒だと送信の途中でも
# まれに改行が入るが、**余分な改行が入っても文字は失われない**ので許容している。
DEFAULT_LINE_BREAK_GAP_S = 3.0


# 段落 (文の終わり)。**ここでも行を分ける** (運用者の要望、2026-08-12)。
#
# 無音による区切りは 3 秒以上の間が空いたときにしか入らないので、1 回の送信が
# 長いと 1 行に延々と続く。段落は文の終わりを表す実在の符号なので、これで
# 分ければ送信の途中でも読みやすい区切りが入る。
#
# **``、`` (区切点) では分けない。** 文の途中に何度も現れるので、分けると
# 1 文が細切れになる。
#
# 判定は変換後のテキストだけを見るので、無音による区切りと同じく
# **作り直しても結果が動かない** (確定した文字は書き換わらない)。
_DANRAKU = "。"
_AFTER_DANRAKU_RE = re.compile(f"(?<={_DANRAKU})")


def _split_after_danraku(text: str) -> list[str]:
    """段落の直後で行を分ける. 段落が無ければそのまま 1 行として返す.

    空行を作らないこと。段落が末尾に来たとき、連続したとき、直後に語間の
    空白が続くときのいずれでも余分な行を生まない。
    """
    if _DANRAKU not in text:
        # **段落が無いときは触らない。** 空文字列や空白だけの segment を
        # 落としてしまわないようにするため (行数が変わると改行の意味が変わる)。
        return [text]
    return [piece for piece in (p.strip() for p in _AFTER_DANRAKU_RE.split(text)) if piece]


def split_at_gaps(
    tokens: Sequence[CommittedToken], gap_samples: int
) -> list[list[CommittedToken]]:
    """確定トークン列を、``gap_samples`` 以上の無音で区切る.

    Args:
        tokens: 確定トークン列 (絶対サンプル位置つき).
        gap_samples: この長さ以上の無音で区切る. 0 以下なら区切らない.

    Returns:
        区間ごとのトークン列. 入力が空なら空リスト.
    """
    if not tokens:
        return []
    if gap_samples <= 0:
        return [list(tokens)]
    segments: list[list[CommittedToken]] = [[tokens[0]]]
    for prev, cur in zip(tokens, tokens[1:]):
        gap = cur.absolute_sample_start - prev.absolute_sample_end
        if gap >= gap_samples:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return segments


def render_committed(
    tokens: Sequence[CommittedToken],
    converter: TokenConverter,
    gap_samples: int,
    initial_mode: Mode = "european",
) -> tuple[str, Mode]:
    """確定トークン列を、無音で区切って改行付きのテキストに変換する.

    Args:
        tokens: 確定トークン列.
        converter: 変換器.
        gap_samples: 改行を入れる無音の長さ (サンプル). 0 以下なら改行しない.
        initial_mode: 走査開始時のサブモード.

    Returns:
        (改行を含むテキスト, 末尾のサブモード).

    Note:
        **区間をまたいでモードを引き継ぐ。** 自動モードでは和文/欧文の状態が
        区間をまたいで続くため、引き継がないと区切りのたびに欧文へ戻ってしまう。
        (ブラウザ版は Mode が 2 値で自動モードが無いためこの処理が無い)
    """
    lines: list[str] = []
    mode: Mode = initial_mode
    for segment in split_at_gaps(tokens, gap_samples):
        res = converter.convert(
            [t.token_id for t in segment],
            [t.confidence for t in segment],
            initial_mode=mode,
        )
        lines.extend(_split_after_danraku(res.text))
        mode = res.final_mode
    return "\n".join(lines), mode


__all__ = ["DEFAULT_LINE_BREAK_GAP_S", "split_at_gaps", "render_committed"]

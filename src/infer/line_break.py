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
from src.tokens.morse_tokens import ID_TO_TOKEN, JAPANESE_TABLE, TOKEN_TO_ID, Mode

# 「。」を「デ」に直すための符号 ID (_danraku_to_de を参照)。
_DANRAKU_ID = TOKEN_TO_ID["・-・-・・"]
_TE_ID = TOKEN_TO_ID["・-・--"]
_DAKUTEN_ID = TOKEN_TO_ID["・・"]
# 単独の短点 (和文表では「ヘ」)。_merge_stray_dot を参照。
_DOT_ID = TOKEN_TO_ID["・"]


def _merge_stray_dot(tokens: list[CommittedToken]) -> list[CommittedToken]:
    """単独の短点 (「ヘ」) を次の文字と合成して正規の符号に戻す.

    **癖のある打鍵向けの任意機能** (運用者の提案、2026-08-29):「癖のある打点は、
    短点 (ヘ) とそれにつづく文字の組み合わせで 1 つの文字になる。補正でヘが
    出てきたら、次の文字と合成してモールスコードを正規に補正したらどうか」。

    実録音 20260829_153839 の測定:

    * 「ヘ」の後の間隔は中央 **4.8 短点**、**その他の文字間は中央 8.2 短点**。
      この局は文字間を広く取る癖があり、その中で「ヘ」の後だけ半分 =
      文字間ではなく要素間である
    * 「ヘ」+次の符号を繋ぐと **19/19 が正規の和文符号**になった
    * 合成すると ``ヘソヘニタクモノハコヘムラヘニドリ`` が
      ``センタクモノハコイランドリ`` (洗濯物はコインランドリ) になる

    **「ヘヘ」は合成しない。** 「ヌヘヘ」は笑い (欧文の HI HI) で、繋ぐと
    濁点になって壊れる (運用者:「ぬへへは、そのまま、ぬへへでいいです」)。

    繋いだ符号が和文表に無いときも触らない (根拠が無いため)。
    """
    out: list[CommittedToken] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        if token.token_id == _DOT_ID and nxt is not None and nxt.token_id != _DOT_ID:
            joined = "・" + ID_TO_TOKEN[nxt.token_id].code
            merged_id = TOKEN_TO_ID.get(joined)
            if merged_id is not None and joined in JAPANESE_TABLE:
                out.append(CommittedToken(
                    token_id=merged_id,
                    confidence=min(token.confidence, nxt.confidence),
                    absolute_sample_start=token.absolute_sample_start,
                    absolute_sample_end=nxt.absolute_sample_end,
                ))
                index += 2
                continue
        out.append(token)
        index += 1
    return out


def _danraku_to_de(
    segment: Sequence[CommittedToken],
) -> tuple[list[int], list[float]]:
    """区間の**末尾以外**の「。」を「デ」(テ + 濁点) に直す.

    デ = テ(・-・--) + 濁点(・・) = ・-・--・・ (7 要素) と
    。 =                        ・-・-・・   (6 要素) は 1 要素しか違わず、
    モデルが確信度 0.99〜1.00 で読み違える (音の実測では 7 要素が正しく
    打たれていた。閾値では救えない)。

    運用者の指示 (2026-08-29):「「。」の後、改行する時間を経過すれば「。」に
    して、そうでなければ「デ」にしたほうがいい」「改行する時間を 1 秒に設定
    すればいいだけだから、その改行時間で判定するのでいい」。

    **新しい設定は増やさない。** ``split_at_gaps`` が既に改行時間で区切って
    いるので、「区間の末尾の『。』だけが段落」と見れば判定は済む。
    段落として残ったものは改行の印としても働き続ける。

    **辞書には頼らない** (辞書補正は ``ヨウカイデス`` を ``マイク デス`` に
    する。運用者:「補正が強すぎて使えない」)。ここは符号 1 要素違いという
    事実と、間隔という測れる量だけで決めている。
    """
    ids: list[int] = []
    confidences: list[float] = []
    last = len(segment) - 1
    for index, token in enumerate(segment):
        if token.token_id == _DANRAKU_ID and index != last:
            ids.extend((_TE_ID, _DAKUTEN_ID))
            confidences.extend((token.confidence, token.confidence))
        else:
            ids.append(token.token_id)
            confidences.append(token.confidence)
    return ids, confidences

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
    merge_stray_dot: bool = False,
) -> tuple[str, Mode]:
    """確定トークン列を、無音で区切って改行付きのテキストに変換する.

    Args:
        tokens: 確定トークン列.
        converter: 変換器.
        gap_samples: 改行を入れる無音の長さ (サンプル). 0 以下なら改行しない.
        initial_mode: 走査開始時のサブモード.
        merge_stray_dot: 単独の短点 (「ヘ」) を次の文字と合成するか
            (``_merge_stray_dot``)。**癖のある打鍵向けの任意機能**で既定は
            ``False``。和文のときだけ効く。

    Returns:
        (改行を含むテキスト, 末尾のサブモード).

    Note:
        **区間をまたいでモードを引き継ぐ。** 自動モードでは和文/欧文の状態が
        区間をまたいで続くため、引き継がないと区切りのたびに欧文へ戻ってしまう。
        (ブラウザ版は Mode が 2 値で自動モードが無いためこの処理が無い)
    """
    lines: list[str] = []
    mode: Mode = initial_mode
    # 和文のときだけ「。」→「デ」を当てる。**欧文表には「。」も「テ」も無い**
    # ので、欧文で当てると読めない文字が増えるだけになる。改行を切っている
    # (gap<=0) ときは判定の物差しが無いので触らない。
    fix_danraku = converter.mode == "japanese" and gap_samples > 0
    merge_dot = merge_stray_dot and converter.mode == "japanese"
    for segment in split_at_gaps(tokens, gap_samples):
        if merge_dot:
            segment = _merge_stray_dot(list(segment))
        if fix_danraku:
            ids, confidences = _danraku_to_de(segment)
        else:
            ids = [t.token_id for t in segment]
            confidences = [t.confidence for t in segment]
        res = converter.convert(ids, confidences, initial_mode=mode)
        lines.extend(_split_after_danraku(res.text))
        mode = res.final_mode
    return "\n".join(lines), mode


__all__ = ["DEFAULT_LINE_BREAK_GAP_S", "split_at_gaps", "render_committed"]

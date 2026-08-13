"""確定テキストを CW 定型語彙で直す (LLM を使わない即時補正).

なぜ必要か
----------
LLM 清書は文脈を読めるが、ローカル (Ollama) では 1 回に数秒かかる。受信しながら
「早く言葉として成立させる」用途には遅い。一方、CW の誤りの多くは**定型語彙の
狭い世界の中**で起きるので、辞書と動的計画法だけで直せる。こちらはマイクロ秒で
終わるため hop (0.5 秒) ごとに走らせられる。

2026-08-07 の実測 (held-out 実録音の欧文 10 件、``models/full/best_infer.pt``)::

    補正なし (現行)          CER 19.25%
    このモジュール               17.15%  (-2.09pt)   ← 出荷値

戦略ごとの寄与 (歯止めを入れる前の掃引。相対的な効き方の目安)::

    切り直しのみ                 17.15%  (-2.09pt)
    寄せのみ                     17.99%  (-1.26pt)
    切り直し + 寄せ              16.74%  (-2.51pt)

**測定の限界を承知しておくこと。** 欧文の held-out は 10 件 57 語しかなく、
しきい値も同じデータで選んでいるので数字は楽観側に出る。ただし
``max_distance`` を 1.2〜2.0、``margin`` を 0.2〜0.5 と振っても結果が動かない
ので、たまたま合った値ではない。**和文は語彙が別なので未対応** (CER は ±0.00pt、
つまり悪くもならない)。

2 つの誤りを別々に直す
----------------------
**1. 語の切れ目 (多数派)**  ``CQDE`` → ``CQ DE``
    WORD_BREAK トークンの再現率は 57〜69% しかなく、語が繋がって出る。
    語彙語の並びに**厳密一致で**分割できるときだけ切る。曖昧な分割まで許すと
    総当たりで何にでも切れてしまうため。

**2. 文字の誤り**  ``CQRT`` → ``QRT``、``NAM`` → ``NAME``
    寄せ先は**符号 (・-) の距離**で選ぶ。素の編集距離ではない。CW の誤りは
    D(-・・) ↔ B(-・・・) のように**点 1 個の差**で起きるので、符号空間で測ると
    正解が最近傍に来る。文字空間で測ると D と B は「ただの別の文字」になってしまう。

壊さないための歯止め
--------------------
``?`` を消す方向の変更なので、確信度閾値を下げたときと同じ罠がある
(``.claude/CLAUDE.md``: 閾値 0.5 → 0.0 で CER が 3.9pt 悪化した)。
「?」の裏に正解があるとは限らず、たいてい間違った文字が出てくるだけだった。
そこで次の 3 つで守る。

* **数字を含む語は触らない。** コールサイン (JH0ILL) と RST (599) がこれ。
  存在しないコールサインを自信ありげに作るのは ``?`` より悪い。
* **プロサイン ``[SK]`` 等は触らない。** 語ではなく運用記号である。
* **2 位との差 (margin) を要求する。** 語彙は 2〜3 文字の略語が密集していて
  最近傍がすぐ入れ替わる。差が無いときは「分からない」として元のまま残す。
* **末尾の ``?`` が本物なら守る** (``is_real_question`` 参照)。

実測では、正しかった語を壊した数は 0 だった。

``?`` の二重の意味
------------------
``?`` は「読めなかった」印であると同時に、**符号表にある実在の文字** (・・--・・)
でもある。``QSL?`` の ``?`` は本物の疑問符で、消すと意味が変わる。テキストだけを
見て両者は区別できない。

そこで「**末尾の ``?`` を外すと語彙語になるなら、その ``?`` は本物**」と判定する。
語の残りが既に正しいなら、その ``?`` はその語のデコード失敗ではないからである。
この歯止めが無いと ``QSL?`` → ``QSL`` と削ってしまい、実測で 0.42pt 悪化した
(-2.09pt が -1.67pt に落ちた)。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.tokens.morse_tokens import EUROPEAN_TABLE

# 欧文 CW の定型語彙。**ここが唯一の真正ソース**で、LLM プロンプトの参考語彙
# (src/llm/prompt.py) もこれを参照する。二重定義すると必ず食い違う。
#
# コールサイン・RST・数字は**入れない**。それらは辞書で直してはいけないもので、
# 語彙に入れると寄せ先の候補になってしまう。
EUROPEAN_LEXICON: tuple[str, ...] = tuple("""
CQ DE K KN AR SK BK RST RPT OM YL XYL TNX TKS FB HW GM GA GE GN ES
UR DR PSE WX RIG ANT PWR NAME QTH GL CUL AGN NR QRZ QSL QSO QRM QRN QSB
QRP QSY QTC QRT R RR TU CFM SRI HR NW BTU VY GUD GB DX WKD WID ABT MNI HPE
CUAGN SIG SIGS RCVR TX RX JST OP
""".split())

_LEXSET = frozenset(EUROPEAN_LEXICON)

# 文字 → 符号 (EUROPEAN_TABLE の逆引き)。符号定義は morse_tokens.py が唯一の
# 真正ソースなので、ここでは持たずに毎回逆引きする (アーキテクチャ原則 2)。
_CHAR_TO_CODE: dict[str, str] = {
    char: code for code, char in EUROPEAN_TABLE.items() if len(char) == 1
}

# 既定のしきい値。2026-08-07 の掃引で決めた。
# max_distance は 1.2〜2.0、margin は 0.2〜0.5 のどこでも CER が同じ
# (16.74%) だったので、knife-edge に合わせた値ではない。安全側の 1.2 / 0.2 を採る。
DEFAULT_MAX_DISTANCE = 1.2
DEFAULT_MARGIN = 0.2

# 分割してよい最小の部分長。1 を許すと K や R (1 文字の語彙語) がどこにでも
# 現れて、意味のない分割を量産する。
_MIN_SEGMENT_LEN = 2


@dataclass(frozen=True)
class CorrectedSpan:
    """補正した範囲 (補正後テキストの文字位置) と、補正前の姿."""

    start: int
    end: int
    original: str


@dataclass(frozen=True)
class CorrectionResult:
    """補正後テキストと、どこを触ったか."""

    text: str
    spans: tuple[CorrectedSpan, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.spans)


def _levenshtein(a: str, b: str) -> int:
    """素の編集距離 (符号文字列どうしの比較に使う)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=4096)
def substitution_cost(a: str, b: str) -> float:
    """文字 ``a`` を ``b`` に置換する費用を**符号の近さ**で返す.

    同一なら 0。``?`` は「読めなかった」印なのでどの文字にも安く化ける (0.3)。
    それ以外は符号の編集距離を長さで正規化し、0.2〜1.0 に写す
    (0.2 の下駄は「別の文字である」こと自体の費用)。
    """
    if a == b:
        return 0.0
    if a == "?":
        return 0.3
    code_a, code_b = _CHAR_TO_CODE.get(a), _CHAR_TO_CODE.get(b)
    if code_a is None or code_b is None:
        return 1.0
    return 0.2 + 0.8 * _levenshtein(code_a, code_b) / max(len(code_a), len(code_b))


@lru_cache(maxsize=8192)
def word_distance(word: str, candidate: str) -> float:
    """符号の近さで重み付けした語間距離."""
    prev = [float(j) for j in range(len(candidate) + 1)]
    for i, ch in enumerate(word, 1):
        cur = [float(i)]
        for j, cc in enumerate(candidate, 1):
            cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0,
                           prev[j - 1] + substitution_cost(ch, cc)))
        prev = cur
    return prev[-1]


def is_protected(word: str) -> bool:
    """辞書で触ってはいけない語か.

    数字を含む語 (コールサイン・RST・時刻) と、プロサイン表記 ``[SK]``。
    """
    return any(ch.isdigit() for ch in word) or ("[" in word) or ("]" in word)


def nearest_word(
    word: str,
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    margin: float = DEFAULT_MARGIN,
) -> str | None:
    """語彙中の最近傍を返す. 十分近く、かつ 2 位と差があるときだけ.

    既に語彙にある語はそのまま返す (``None`` ではない) ので、呼び出し側は
    「変わったか」を文字列比較で判定できる。
    """
    if word in _LEXSET:
        return word
    if not word:
        return None
    scored = sorted((word_distance(word, cand), cand) for cand in EUROPEAN_LEXICON)
    best_distance, best_word = scored[0]
    second = scored[1][0] if len(scored) > 1 else float("inf")
    if best_distance <= max_distance and second - best_distance >= margin:
        return best_word
    return None


def segment_word(word: str, *, max_parts: int = 4) -> list[str]:
    """繋がった語を語彙語の並びに切り直す (厳密一致のみ).

    ``CQDE`` → ``["CQ", "DE"]``。切れなければ ``[word]`` を返す。

    厳密一致に限るのは、曖昧な分割を許すと総当たりで何にでも切れてしまうため。
    部分数が最小になる分割を選ぶ (``CQCQDE`` を 3 語に切り、6 語には切らない)。
    """
    n = len(word)
    if n < _MIN_SEGMENT_LEN * 2 or word in _LEXSET:
        return [word]
    # best[i] = word[:i] を切り終えた (部分数, 並び). None は到達不能.
    best: list[tuple[int, list[str]] | None] = [None] * (n + 1)
    best[0] = (0, [])
    for i in range(_MIN_SEGMENT_LEN, n + 1):
        for j in range(max(0, i - 6), i - _MIN_SEGMENT_LEN + 1):
            prefix = best[j]
            if prefix is None or word[j:i] not in _LEXSET:
                continue
            candidate = (prefix[0] + 1, [*prefix[1], word[j:i]])
            if best[i] is None or candidate[0] < best[i][0]:  # type: ignore[index]
                best[i] = candidate
    tail = best[n]
    if tail is not None and 2 <= tail[0] <= max_parts:
        return tail[1]
    return [word]


def correct_text(
    text: str,
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    margin: float = DEFAULT_MARGIN,
) -> CorrectionResult:
    """確定テキストを語ごとに切り直し・寄せする.

    改行と語間スペースは保つ (改行は送信のターンの切れ目を表すため潰せない)。

    Returns:
        補正後テキストと、触った範囲。範囲は**補正後**テキストの文字位置で、
        UI が色を変えるのに使う。
    """
    if not text:
        return CorrectionResult(text)

    out: list[str] = []
    spans: list[CorrectedSpan] = []
    position = 0
    # 改行と空白の並びを保つため、区切り文字ごと分解して走査する。
    for line_index, line in enumerate(text.split("\n")):
        if line_index:
            out.append("\n")
            position += 1
        # 先頭・末尾の空白も保つので split ではなく手で刻む
        for chunk in _split_keeping_spaces(line):
            if not chunk.strip():
                out.append(chunk)
                position += len(chunk)
                continue
            fixed = _correct_word(chunk, max_distance=max_distance, margin=margin)
            if fixed != chunk:
                spans.append(CorrectedSpan(position, position + len(fixed), chunk))
            out.append(fixed)
            position += len(fixed)
    return CorrectionResult("".join(out), tuple(spans))


def _split_keeping_spaces(line: str) -> list[str]:
    """行を「語」と「空白の並び」に交互に刻む (復元可能な分解)."""
    parts: list[str] = []
    current = ""
    current_is_space: bool | None = None
    for ch in line:
        is_space = ch == " "
        if current_is_space is None or is_space == current_is_space:
            current += ch
            current_is_space = is_space
        else:
            parts.append(current)
            current, current_is_space = ch, is_space
    if current:
        parts.append(current)
    return parts


def is_real_question(word: str) -> bool:
    """末尾の ``?`` が「読めなかった印」ではなく本物の疑問符か.

    ``?`` を外すと語彙語になるなら本物とみなす (``QSL?``、``QRZ?``)。
    語の残りが既に正しいのだから、その ``?`` はその語のデコード失敗ではない。
    """
    return len(word) > 1 and word.endswith("?") and word[:-1] in _LEXSET


def _correct_word(word: str, *, max_distance: float, margin: float) -> str:
    """1 語を切り直し → 寄せ の順に直す."""
    if is_protected(word) or is_real_question(word):
        return word
    pieces = segment_word(word)
    fixed = [
        nearest_word(piece, max_distance=max_distance, margin=margin) or piece
        for piece in pieces
    ]
    return " ".join(fixed)


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_MAX_DISTANCE",
    "EUROPEAN_LEXICON",
    "CorrectedSpan",
    "CorrectionResult",
    "correct_text",
    "is_protected",
    "is_real_question",
    "nearest_word",
    "segment_word",
    "substitution_cost",
    "word_distance",
]

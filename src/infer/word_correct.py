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

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.tokens.converter import FALLBACK_CHAR as UNREADABLE
from src.tokens.morse_tokens import EUROPEAN_TABLE, JAPANESE_CHAR_TO_CODES

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

# 和文 CW の定型語彙。**ここが唯一の真正ソース**で、LLM プロンプトの参考語彙
# (src/llm/prompt.py) もこれを参照する。
#
# **カテゴリで分けるのには理由がある。** カテゴリごとに「寄せ先にしてよいか」が
# 違うためで、単なる整理ではない (``_JA_NO_TARGET_CATEGORIES``)。
# 語を足す運用者は「どのカテゴリか」を決めるだけでよい。
#
# 数字を含む語は入れない (欧文と同じ。数字を含む語は触らないと決めているので、
# 語彙に入れると寄せ先の候補になってしまう)。
JAPANESE_PARTICLES: tuple[str, ...] = ("ハ", "ノ", "ガ", "ヲ", "ニ", "デ", "ト", "モ", "カ")
"""助詞 (1 文字)。**分割にだけ使い、寄せ先にはしない。**

1 文字の曖昧一致を許すとどのカナも別のカナに化ける。完全一致で切り出すだけ。
``ヘ`` は符号が ``・`` の 1 要素しかなく誤一致の温床なので**入れない**。
"""

JAPANESE_LEXICON: dict[str, tuple[str, ...]] = {
    # **小文字 (ッ ャ ュ ョ) は和文モールスに存在しない。** 符号表は大文字のみなので
    # 「いってきます」は ``イツテキマス`` と書く。運用者が挙げた語を入れる際に
    # 実際にここで直した (2026-08-14)。``TestLexiconIsEncodable`` が歯止め。
    "挨拶・応答": tuple("""
    コンニチハ コンバンハ オハヨウ ゴザイマス アリガトウ ゴザイマシタ ドウモ
    ドウゾ ヨロシク オネガイ シマス イタシマス サヨウナラ オヤスミナサイ
    ハジメマシテ リヨウカイ ワカリマセン シツレイ オゲンキ マタ
    ドウイタシマシテ スミマセン ゴメンナサイ ハイ イイエ ワカリマシタ
    オツカレサマ ゴクロウサマ イツテキマス イツテラツシヤイ タダイマ
    オカエリナサイ マタアシタ マタネ ヒサシブリ オメデトウ キヲツケテ
    ゴキゲンヨウ オセワニナリマス カシコマリマシタ
    """.split()),
    "交信": tuple("""
    コチラ ソチラ ホウ コチラハ シンゴウ レポート ナマエ オナマエ バシヨ ムセン コウシン
    トレマス ハイツテ イマス ヨク ツヨイ ヨワイ ノイズ オオイ コンデイシヨン
    モウ イチド ハヤイ ユツクリ スコシ オワリ マス デス デシタ マシタ ネ
    シテ シテマス シテオリマス オリマス テマス ナリマス アリマス
    ヨイ アイマシヨウ アイマス シカタ ナイ ザンネン
    キコエマス キコエマスカ キコエマセン ミエマス ミエマスカ
    モウイチド オネガイシマス マツテクダサイ イツテクダサイ
    ダイジヨウブ モンダイアリマセン ソウデス チガイマス タダシイ
    カクニン カクニンシマス ヘンシン オウトウ
    ドナタ ドコ イツ ナニ ナゼ ドウシテ ドノヨウニ
    コレ ソレ アレ ココ ソコ アソコ
    ミギ ヒダリ マエ ウシロ ウエ シタ チカイ トオイ スグ
    ハジメマス ツヅケマス チユウシシマス オワリマシタ マタアイマシヨウ
    """.split()),
    # CW の笑いは欧文の ``HI HI``。和文表で読むと ``ヌヘヘ`` や ``ホヘヘ`` になる
    # (運用者、2026-08-14)。**符号としては同じ音**なので、これを語彙に入れておかないと
    # 辞書が別の語に引き寄せてしまう。表示を「(笑)」にするのは LLM 側の仕事。
    "慣用・笑い": tuple("""
    ヌヘヘ ホヘヘ ヌヘ アハハ エヘヘ ハイハイ ウエーン ニコニコ シクシク
    """.split()),
    "設備": tuple("""
    アンテナ ダイポール ロングワイヤ ジーピー バーチカル ビーム
    リグ デンリヨク シユツリヨク ワツト タカサ メートル シユウハスウ
    トランシーバー ムセンキ ジユシンキ ソウシンキ アンプ リニアアンプ
    チユーナー アンテナチユーナー マイク スピーカー ヘツドホン イヤホン
    デンゲン バツテリー ケーブル ドウジクケーブル コネクター セツゾク
    アース セツチ マスト タワー ローテーター エレメント ラジアル バラン
    フイルター メーター エスダブリユーアール ソクテイキ ダミーロード
    キヨクメン モード チヤンネル メモリー ゲイン インピーダンス
    ボルト アンペア モービル ハンデイ コテイキ ヨビデンゲン
    """.split()),
    "天気・気候": tuple("""
    テンキ ハレ クモリ アメ ユキ カゼ キオン サムイ アツイ ヒンヤリ ムシアツイ
    スズシイ アタタカイ オオアメ タイヘン
    カイセイ ウスグモリ ニワカアメ コサメ ゴウウ ライウ ユウダチ
    ミゾレ アラレ ヒヨウ フブキ カミナリ イナズマ キリ ツユ シモ コオリ
    ツヨイカゼ キヨウフウ ボウフウ タイフウ タツマキ オオユキ セキセツ
    コウスイ コウスイリヨウ シツド キアツ コウキアツ テイキアツ
    ゼンセン キダン テンキヨホウ ヨホウ ケイホウ チユウイホウ
    サイコウキオン サイテイキオン カンソウ ジメジメ ポカポカ
    ダンボウ レイボウ キセツ ハル ナツ アキ フユ ツユイリ ツユアケ
    ヒデリ コウズイ ナダレ
    """.split()),
    "時・暮らし": tuple("""
    キヨウ キノウ サクヤ アシタ マイニチ サイキン ホンジツ シゴト ヤスミ
    タノシイ レンシユウ ワブン オウブン
    イマ アサ ヒル ユウガタ ヨル シンヤ ケサ コンヤ マイアサ マイバン
    マイシユウ センシユウ コンシユウ ライシユウ センゲツ コンゲツ ライゲツ
    キヨネン コトシ ライネン サキホド ノチホド アトデ シバラク
    ジカン ジコク ナンジ ヘイジツ シユウマツ ニチヨウビ キユウジツ シユクジツ
    ハヤオキ ヨフカシ イソガシイ ヒマ ゲンキ ツカレマシタ
    シユツキン タイキン ザイタク シユツチヨウ ツウキン
    カゾク シユミ サンポ カイモノ リヨコウ シヨクジ スイミン
    ウンヨウ ジユシン ソウシン コンテスト ロールコール キヨクスウ
    ワブンツウシン オウブンツウシン モールス チヨウシユ レンラク マタコンヤ
    """.split()),
    # 地理・呼称の**一般語**と主要地名。閉じた集合なので寄せ先にしてよい。
    # 個別のローカル地名・実在の人名は下の「地名・人名」に置くこと (方針が違う)。
    # ``ク`` (区) のような 1 文字語は入れない。助詞と同じで、どこでも切れてしまう。
    "地理・呼称": tuple("""
    トドウフケン シチヨウソン トシ マチ ムラ チイキ チホウ ケンナイ ケンガイ
    シナイ グン ジユウシヨ キヨクチ イドウチ ジモト シユツシン
    ホツカイドウ トウホク カントウ ホクリク コウシンエツ トウカイ キンキ
    カンサイ チユウゴク シコク キユウシユウ オキナワ
    トウキヨウ ヨコハマ ナゴヤ キヨウト オオサカ コウベ フクオカ サツポロ センダイ
    セイ ミヨウジ ナマエノホウ フルネーム ニツクネーム ハンドルネーム ハンドル
    オペレーター オペレーターネーム ホンニン トモダチ ナカマ センパイ コウハイ
    シシヨウ カイイン ブチヨウ カイチヨウ ダンセイ ジヨセイ
    カタ サン サマ クン チヤン
    """.split()),
    "助詞": JAPANESE_PARTICLES,
    # 運用者が実際の交信で使う固有名詞を足す欄。
    # **寄せ先にはしない** (開いた集合なので、正しい固有名詞を語彙語に
    # 引き寄せて壊す方が害が大きい)。分割の手掛かりとしてだけ使う。
    "地名・人名": (),
}

# 寄せ先 (fuzzy match の行き先) にしないカテゴリ。
_JA_NO_TARGET_CATEGORIES = frozenset({"助詞", "地名・人名"})

_JA_MAX_WORD_LEN = 8

# 語を 1 つ増やすごとの費用 (符号要素)。**細かく刻む分割を嫌うため**。
#
# これが無いと、長い塊が凡庸な曖昧一致の連鎖で「説明できてしまう」。
# 2026-08-14 の実測では ``サクヤハソチラノホウハンオアメデ`` が
# ``サクヤ ハレ ドウゾ ハ オオアメ デ`` に砕かれた。費用そのものは小さくても、
# **語数が増えること自体を費用に数える**と、まとまりのある解釈が勝つ。
_JA_PIECE_PENALTY = 2.0

# 助詞を 1 つ使うごとの選好上のペナルティ (符号要素)。**予算には数えない。**
#
# 助詞は接着剤であって、それ自体が分割を正当化すべきではない。これが無いと
# ``モロムマス`` が ``モ`` (助詞) + ``テマス`` と費用 0 で説明できてしまい、
# 正しい ``シテマス`` (費用 1) に勝つ (2026-08-14 実測)。
_JA_PARTICLE_PENALTY = 1.5

# 塊全体で許す編集量の割合 (符号要素)。個々の語が閾値を通っても、
# **連鎖した合計**がこれを超える分割は採らない。1 語ごとの閾値だけでは
# 長い塊を守れない。
#
# **辞書は安全な直しだけを行い、迷うものは候補つきで LLM に回す**というのが
# 全体の方針なので、ここは思い切って厳しくてよい。0.12 では
# ``サクヤハソチラノホウハンオアメデ`` (余分な ``ン`` が 1 つ混ざっている) が
# ``サクヤ ハレ ドウゾ ハ …`` に砕かれた。締めると「説明できない」と判断して
# そのまま残し、候補を添えて次の層へ渡す。
_JA_TOTAL_RATIO = 0.05

# 曖昧一致を許す最小の文字数。**2 文字は危険**。
#
# 和文は 2 文字の並びが別の 1〜2 文字と符号列まで一致することがある::
#
#     ヨ(--) + イ(・-) = --・- = ネ
#
# 実測 (2026-08-14, 正解ラベル 70 件) で ``ヨイ テンキ`` が ``ネ テンキ`` に
# 化けた。語彙に無い 2 文字語は、語彙にある 2 文字語と見分けがつかない。
# 完全一致はどの長さでも許す (それは「元から語彙にある」ということなので安全)。
_JA_MIN_FUZZY_LEN = 3

# 分割を採用するために必要な「芯となる語」の最小文字数。
#
# 短い語 (イマ シタ コレ ソレ サン …) が語彙に増えると、**完全一致だけで**
# 正しい語が砕けるようになる。2026-08-14 に語彙を 4 倍に広げた直後、
# ``カイマシタ`` (買いました) が ``カ イマ シタ`` に割れた。
# **語彙が増えるほどこの条件は効いてくる。** 3 文字以上の語を芯に持たない分割は
# 採らない (``コレデ`` を ``コレ デ`` に切るような、意味を変えない切り直しも
# 諦めることになるが、``カイマシタ`` を壊さない方が大事)。
_JA_MIN_CONTENT_LEN = 3

# 和文で分割・寄せを試みる最大の塊長。長すぎる塊は DP が重くなるうえ、
# そこまで繋がっているものは分割の信頼度も低い。
_JA_MAX_CHUNK_LEN = 24

# 和文の句読点。塊の切れ目として扱う (「コチラノ、ムンキニ」が 1 語にならないように)。
_JA_PUNCT = "、。"

_KATAKANA_RE = re.compile(r"[゠-ヿ]")

# **無条件に置き換えるカナ** (運用者の指示、2026-08-14)。
#
# ``ヱ`` は日常の和文にまず出てこない。そして符号を見ると読み違えの姿である::
#
#     ヱ   = ・--・・
#     イマ = ・- + -・・- = ・--・・-    ← 長音 1 本の差
#
# ``ヱ`` が出たら ``イマ`` の末尾の長音が落ちたものと決め打ってよい、という
# 運用者の判断。曖昧一致に任せず必ず置き換える (``ヱ`` は 1 文字なので
# 曖昧一致の対象にならず、辞書では直せない)。
JAPANESE_FORCED_SUBSTITUTIONS: dict[str, str] = {
    "ヱ": "イマ",
}

# 利用者が足した語彙の既定の保存先。返信の型 (src/tx/templates.py) と同じ流儀。
# **テストは必ず一時パスを渡すこと** (利用者の実ファイルに依存しない)。
DEFAULT_JA_LEXICON_PATH = Path.home() / ".cw-decorder" / "lexicon_ja.json"

# 文字 → 符号 (EUROPEAN_TABLE の逆引き)。符号定義は morse_tokens.py が唯一の
# 真正ソースなので、ここでは持たずに毎回逆引きする (アーキテクチャ原則 2)。
_CHAR_TO_CODE: dict[str, str] = {
    char: code for code, char in EUROPEAN_TABLE.items() if len(char) == 1
}

# 和文の 文字 → 符号。**合成カナ (ガ・パ) は「基本カナの符号 + 濁点の符号」**を
# 連結したものになる (和文表に単独の符号を持たないため)。
#
# これは欠点ではなく利点である。``ガ`` と ``カ`` の距離が「濁点 1 個ぶん」として
# 自然に出るので、**濁点の付き外れという和文で頻出する誤りが、そのまま符号距離に
# 乗る**。欧文の「点 1 個の差」と同じ構造になる。
_JA_CHAR_TO_CODE: dict[str, str] = {
    char: "".join(codes) for char, codes in JAPANESE_CHAR_TO_CODES.items()
}

_SCRIPT_TABLES: dict[str, dict[str, str]] = {
    "european": _CHAR_TO_CODE,
    "japanese": _JA_CHAR_TO_CODE,
}

# 既定のしきい値。2026-08-07 の掃引で決めた。
# max_distance は 1.2〜2.0、margin は 0.2〜0.5 のどこでも CER が同じ
# (16.74%) だったので、knife-edge に合わせた値ではない。安全側の 1.2 / 0.2 を採る。
DEFAULT_MAX_DISTANCE = 1.2
DEFAULT_MARGIN = 0.2

# 和文の閾値は**単位が違う** (符号要素数)。符号列の長さに対する割合で見る。
# 0.20 は「3 カナの語 (約 15 要素) で 3 要素まで許す」あたり。
DEFAULT_JA_DISTANCE_RATIO = 0.20
# 2 位との差。1 要素の差がつかない候補は「分からない」として寄せない。
DEFAULT_JA_MARGIN = 1.0

# ``?`` を符号列に展開したときの目印。任意個の要素を吸収するワイルドカード。
_WILDCARD = "\x00"

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
    """補正後テキストと、どこを触ったか、直せなかった語."""

    text: str
    spans: tuple[CorrectedSpan, ...] = ()
    # 辞書では決めきれなかった語と、符号距離が近い候補。**LLM へ渡す材料**。
    unresolved: tuple[UnresolvedWord, ...] = ()

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


@lru_cache(maxsize=8192)
def substitution_cost(a: str, b: str, script: str = "european") -> float:
    """文字 ``a`` を ``b`` に置換する費用を**符号の近さ**で返す.

    同一なら 0。``?`` は「読めなかった」印なのでどの文字にも安く化ける (0.3)。
    それ以外は符号の編集距離を長さで正規化し、0.2〜1.0 に写す
    (0.2 の下駄は「別の文字である」こと自体の費用)。

    Args:
        script: ``"european"`` / ``"japanese"``。**既定は欧文** なので、
            引数を渡さない既存の呼び出しは挙動が変わらない。
    """
    if a == b:
        return 0.0
    if a == UNREADABLE:
        return 0.3
    table = _SCRIPT_TABLES.get(script, _CHAR_TO_CODE)
    code_a, code_b = table.get(a), table.get(b)
    if code_a is None or code_b is None:
        return 1.0
    return 0.2 + 0.8 * _levenshtein(code_a, code_b) / max(len(code_a), len(code_b))


@lru_cache(maxsize=16384)
def japanese_code_of(word: str) -> str | None:
    """和文の語を**符号列**に展開する (文字の境界を落とす).

    **これが和文の距離の土台である。** 和文で頻出する誤りは文字の置き換わりでは
    なく、**1 文字が要素の途中で切られて 2 文字になる**ことだからである。

    実例 (2026-08-14 の実受信、運用者が発見)::

        ロ(・-・-) + ム(-) = ・-・-- = テ

    ``ムンキ`` は ``テンキ`` の ``テ`` が途中で切れた姿であって、``テ`` が ``ム`` に
    化けたのではない。文字単位で測ると「1 文字消して 1 文字置換」で高くつくが、
    符号列に展開して並べれば**距離ゼロ**になる。要素間が詰まった打鍵
    (この録音は 0.66 dot) では、この切れ方が誤りの主役になる。

    ``?`` は「何かあったが読めなかった」印なので**空に展開する**。欠けた要素は
    編集距離の挿入として埋まるので、短い取りこぼしほど安くなる。

    Returns:
        符号列。表に無い文字が混ざっていれば ``None``。
    """
    parts: list[str] = []
    for ch in word:
        if ch == UNREADABLE:
            parts.append(_WILDCARD)
            continue
        code = _JA_CHAR_TO_CODE.get(ch)
        if code is None:
            return None
        parts.append(code)
    return "".join(parts)


def _levenshtein_wild(a: str, b: str) -> int:
    """符号列の編集距離. ``a`` 中の ``_WILDCARD`` は任意個の要素を費用 1 で吸収する.

    ``?`` は「何かあったが読めなかった」印なので、そこに何要素あったかは
    分からない。1 文字ぶん (4〜5 要素) を挿入として数えると高くつきすぎて、
    ``?`` を含む語がどこにも寄らなくなる。**「不明な区間が 1 つある」ことだけを
    費用 1 として数える**のが実態に合う。

    実例: ``?ロムマス`` と ``シテマス`` は距離 1 になる
    (``?`` が ``シ`` を吸収し、``ロ+ム`` が ``テ`` に一致する)。
    """
    if _WILDCARD not in a:
        return _levenshtein(a, b)
    n = len(b)
    prev = list(range(n + 1))
    for ca in a:
        if ca == _WILDCARD:
            # 任意個を吸収: これまでの最小 + 1 が、以降どの位置でも取れる
            running = prev[0] + 1
            cur = [running]
            for j in range(1, n + 1):
                running = min(running, prev[j] + 1)
                cur.append(running)
        else:
            cur = [prev[0] + 1]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=16384)
def word_distance(word: str, candidate: str, script: str = "european") -> float:
    """語間距離.

    欧文は**文字単位**で、各置換の費用を符号の近さで重み付けする。
    和文は**符号列そのもの**の編集距離 (単位は符号要素) を返す。
    和文だけ測り方が違うのは ``japanese_code_of`` の説明のとおりで、
    文字の境界がそもそも当てにならないためである。

    **単位が違うので閾値も別** (``DEFAULT_JA_*`` を使うこと)。
    """
    if script == "japanese":
        code_a, code_b = japanese_code_of(word), japanese_code_of(candidate)
        if code_a is None or code_b is None:
            return float("inf")
        return float(_levenshtein_wild(code_a, code_b))
    prev = [float(j) for j in range(len(candidate) + 1)]
    for i, ch in enumerate(word, 1):
        cur = [float(i)]
        for j, cc in enumerate(candidate, 1):
            cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0,
                           prev[j - 1] + substitution_cost(ch, cc, script)))
        prev = cur
    return prev[-1]


def japanese_max_distance(candidate: str, ratio: float = DEFAULT_JA_DISTANCE_RATIO) -> float:
    """和文で許す最大距離 (符号要素数). 語の長さに比例させる.

    絶対値で切ると、短い語 (マス) では緩すぎ、長い語 (ゴザイマシタ) では
    厳しすぎる。符号列の長さに対する割合で見るのが素直である。
    """
    code = japanese_code_of(candidate)
    if not code:
        return 0.0
    return max(1.0, ratio * len(code))


def is_protected(word: str) -> bool:
    """辞書で触ってはいけない語か.

    * 数字を含む語 (コールサイン・RST・時刻)
    * プロサイン表記 ``[SK]``
    * **``?`` を含む語** — ``?`` は符号表にある実在の文字 (``・・--・・``) であり、
      語彙語はどれも ``?`` を持たない。寄せると必ず ``?`` が消えて意味が変わる
      (``QSL?`` → ``QSL``、``ドウゾ?`` → ``ドウゾ``)。

    読めなかった箇所は ``_`` で表すので (``FALLBACK_CHAR``)、ここで ``?`` を
    守っても「読めなかった語を直す」邪魔にはならない。**記号を分ける前は両者が
    同じ ``?`` だったため、「末尾の ``?`` を外して語彙語になれば本物」という
    推測に頼っていた** (``is_real_question``)。記号を分けたので推測は要らない。
    """
    return (
        any(ch.isdigit() for ch in word)
        or ("[" in word)
        or ("]" in word)
        or ("?" in word)
    )


def japanese_target_words(extra: frozenset[str] = frozenset()) -> frozenset[str]:
    """**寄せ先にしてよい**和文語の集合.

    助詞 (1 文字なので何にでも化ける) と地名・人名 (開いた集合なので、正しい
    固有名詞を語彙語に引き寄せて壊す) は除く。
    """
    return _japanese_target_words(extra)


@lru_cache(maxsize=32)
def _japanese_target_words(
    extra: frozenset[str], min_len: int = _JA_MIN_FUZZY_LEN
) -> frozenset[str]:
    """**短い語は寄せ先にしない。**

    2 文字の語 (サン ハレ イマ …) は符号列も短く、少し違うだけの 3〜4 文字語を
    片端から引き寄せる。2026-08-14 に語彙を広げた直後、``ムンキニ`` が
    ``サン ニ`` に化けた (``サン`` = 敬称)。**寄せ先は 3 文字以上に限る。**
    完全一致で切り出す分には短い語も使ってよい (``_japanese_all_words``)。

    ``min_len`` を下げてよいのは**語の途中に句読点がある**ときだけ
    (``_segment_japanese`` 参照)。句読点そのものが読み違えの強い証拠なので、
    短い語への寄せを許してもよい。
    """
    words: set[str] = set()
    for category, entries in JAPANESE_LEXICON.items():
        if category in _JA_NO_TARGET_CATEGORIES:
            continue
        words.update(entries)
    words.update(extra)
    return frozenset(w for w in words if len(w) >= min_len)


@lru_cache(maxsize=16)
def _japanese_targets_by_code_length(
    extra: frozenset[str],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """寄せ先を**符号長ごとに**まとめる (探索を絞るため).

    編集距離は長さの差以上になる (``distance >= |len_a - len_b|``) ので、
    許容距離を超える長さの候補は計算するまでもなく落とせる。
    語彙が 400 語に育つと、全候補との距離計算が現実的な時間に収まらない
    (2026-08-14 に GUI が凍結した)。
    """
    buckets: dict[int, list[str]] = {}
    for word in _japanese_target_words(extra):
        code = japanese_code_of(word)
        if code:
            buckets.setdefault(len(code), []).append(word)
    return tuple(
        (length, tuple(sorted(words))) for length, words in sorted(buckets.items())
    )


def _japanese_candidates_near_length(
    code_length: int, ratio: float, extra: frozenset[str]
) -> tuple[str, ...]:
    """符号長が近い寄せ先だけを返す.

    ``|len_a - len_b| <= ratio * len_b`` を満たしうる ``len_b`` の範囲は
    ``[len_a / (1 + ratio), len_a / (1 - ratio)]``。両端に 1 の余裕を持たせる
    (``japanese_max_distance`` の下限 1.0 のぶん)。
    """
    low = code_length / (1.0 + ratio) - 1.0
    high = code_length / max(1.0 - ratio, 0.01) + 1.0
    result: list[str] = []
    for length, words in _japanese_targets_by_code_length(extra):
        if low <= length <= high:
            result.extend(words)
    return tuple(result)


@lru_cache(maxsize=16)
def _japanese_all_words(extra: frozenset[str]) -> frozenset[str]:
    """完全一致で切り出してよい語 (助詞・固有名詞を含む)."""
    words: set[str] = set()
    for entries in JAPANESE_LEXICON.values():
        words.update(entries)
    # 句読点も「そのまま切り出してよい要素」として扱う。**寄せ先にはしない。**
    # これが無いと、文末の ``。`` を含む塊が「説明できない」と判定されてしまう。
    words.update(_JA_PUNCT)
    return frozenset(words | set(extra))


def load_user_lexicon(path: Path | str = DEFAULT_JA_LEXICON_PATH) -> dict[str, tuple[str, ...]]:
    """利用者が足した和文語彙を読む. 無ければ空を返す.

    形式は組み込み語彙と同じ ``{カテゴリ: [語, ...]}``。壊れていても落とさない
    (受信中に例外を上げるより、補正が効かない方がまし)。
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for category, words in data.items():
        if isinstance(words, list):
            result[str(category)] = tuple(str(w) for w in words if str(w).strip())
    return result


def user_target_words(user_lexicon: Mapping[str, Sequence[str]]) -> frozenset[str]:
    """利用者語彙のうち**寄せ先にしてよい**もの (地名・人名などは除く)."""
    words: set[str] = set()
    for category, entries in user_lexicon.items():
        if category in _JA_NO_TARGET_CATEGORIES:
            continue
        words.update(entries)
    return frozenset(words)


@lru_cache(maxsize=65536)
def nearest_word(
    word: str,
    *,
    script: str = "european",
    max_distance: float = DEFAULT_MAX_DISTANCE,
    margin: float = DEFAULT_MARGIN,
    ja_distance_ratio: float = DEFAULT_JA_DISTANCE_RATIO,
    ja_margin: float = DEFAULT_JA_MARGIN,
    ja_min_target_len: int = _JA_MIN_FUZZY_LEN,
    extra: frozenset[str] = frozenset(),
) -> str | None:
    """語彙中の最近傍を返す. 十分近く、かつ 2 位と差があるときだけ.

    既に語彙にある語はそのまま返す (``None`` ではない) ので、呼び出し側は
    「変わったか」を文字列比較で判定できる。

    **分割の DP から 1 塊あたり百回以上呼ばれるので、キャッシュは必須。**
    """
    if not word:
        return None
    if script == "japanese":
        if word in _japanese_all_words(extra):
            return word
        code = japanese_code_of(word)
        if code is None:
            return None
        # 符号長でふるいにかける。``?`` を含むと長さの下限が崩れるので絞らない。
        if _WILDCARD in code or ja_min_target_len != _JA_MIN_FUZZY_LEN:
            vocabulary: tuple[str, ...] = tuple(
                sorted(_japanese_target_words(extra, ja_min_target_len))
            )
        else:
            vocabulary = _japanese_candidates_near_length(
                len(code), ja_distance_ratio, extra
            )
        scored = sorted((word_distance(word, cand, script), cand) for cand in vocabulary)
        if not scored:
            return None
        best_distance, best_word = scored[0]
        second = scored[1][0] if len(scored) > 1 else float("inf")
        # 和文の閾値は語の長さに比例させる (単位は符号要素)。
        if (
            best_distance <= japanese_max_distance(best_word, ja_distance_ratio)
            and second - best_distance >= ja_margin
        ):
            return best_word
        return None
    if word in _LEXSET:
        return word
    scored = sorted((word_distance(word, cand), cand) for cand in EUROPEAN_LEXICON)
    best_distance, best_word = scored[0]
    second = scored[1][0] if len(scored) > 1 else float("inf")
    if best_distance <= max_distance and second - best_distance >= margin:
        return best_word
    return None


@dataclass(frozen=True)
class Candidate:
    """寄せ先の候補と、その符号距離."""

    word: str
    distance: float


@dataclass(frozen=True)
class UnresolvedWord:
    """直せなかった語と、符号距離が近い候補.

    **``candidates`` は既定で空。** 候補の計算は 1 語あたり 60 ms かかり、
    必要になるのは LLM を呼ぶときだけなので、hop ごとに走る ``correct_text``
    では計算しない。埋めるには ``with_candidates(...)`` を通すこと。

    **LLM に渡すための材料** (docs の B 案)。語彙を丸ごとプロンプトに入れる代わりに、
    詰まった語ぶんだけ候補を添える。プロンプトが小さいまま済み、LLM を
    「候補から選ぶ」役に限定できるので捏造が減る。

    距離は閾値で切らずに付ける。**選ぶのは文脈を読める側の仕事**だからである。
    """

    word: str
    candidates: tuple[Candidate, ...]


def with_candidates(
    unresolved: tuple[UnresolvedWord, ...],
    *,
    extra: frozenset[str] = frozenset(),
) -> tuple[UnresolvedWord, ...]:
    """直せなかった語に候補を埋めて返す (**LLM を呼ぶときだけ使う**).

    候補の計算は 1 語あたり 60 ms ほどかかる。``correct_text`` は hop
    (0.5 秒) ごとに GUI スレッドで走るので、そこでは計算しない。
    """
    return tuple(
        UnresolvedWord(
            entry.word,
            candidates_for(
                entry.word, script="japanese", extra=extra,
                # 長い塊ほど中に語がたくさん入っているので候補も増やす
                limit=min(6, max(3, len(entry.word) // 3)),
            ),
        )
        for entry in unresolved
    )


def _substring_distance(needle_code: str, hay_code: str) -> int:
    """``needle_code`` が ``hay_code`` の**どこかに**現れるかの編集距離.

    先頭・末尾を自由にした編集距離 (approximate substring matching)。
    先頭行を 0 で埋めると「どこから始めてもよい」、最終行の最小を取ると
    「どこで終わってもよい」になる。
    """
    if not needle_code:
        return 0
    prev = [0] * (len(hay_code) + 1)
    for ca in needle_code:
        cur = [prev[0] + 1]
        for j, cb in enumerate(hay_code, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(prev)


# 候補として出す上限 (語の符号長に対する割合)。これを超える語は「その塊に
# 現れているとは言えない」として出さない。**無関係な候補は無いよりも悪い**
# (LLM を誤導する) ので、出さない方に倒す。
_JA_CANDIDATE_RATIO = 0.30


def candidates_for(
    word: str,
    *,
    script: str = "japanese",
    limit: int = 3,
    extra: frozenset[str] = frozenset(),
) -> tuple[Candidate, ...]:
    """その塊の**どこかに現れていそうな**語を、近い順に返す.

    **語全体との距離ではない。** 直せなかった塊は文節がまるごと繋がっていること
    が多く (``サクヤハソチラノホウハンオアメデ``)、1 語と丸ごと比べても
    ``エスダブリユーアール`` のような無意味な候補しか出ない。2026-08-14 に
    実際にそうなった。**無関係な候補は無いよりも悪い** ので、部分一致で探し、
    見合うものが無ければ何も返さない。
    """
    if not word:
        return ()
    if script != "japanese":
        scored = sorted((word_distance(word, cand, script), cand)
                        for cand in EUROPEAN_LEXICON)
        return tuple(Candidate(word=w, distance=d) for d, w in scored[:limit])
    hay = japanese_code_of(word)
    if hay is None:
        return ()
    hay = hay.replace(_WILDCARD, "")
    # 順位は **符号長で正規化した距離**、同点なら**長い語を優先**する。
    # 生の距離で並べると短い語 (エヘヘ) が偶然の一致で上位に来る。長い語が
    # 一致するのは偶然では起きにくいので、そちらの方が手掛かりとして強い。
    results: list[tuple[float, int, str, int]] = []
    for candidate in _japanese_target_words(extra):
        needle = japanese_code_of(candidate)
        if not needle:
            continue
        distance = _substring_distance(needle, hay)
        if distance <= _JA_CANDIDATE_RATIO * len(needle):
            results.append((distance / len(needle), -len(needle), candidate, distance))
    results.sort()
    return tuple(
        Candidate(word=word, distance=float(distance))
        for _, _, word, distance in results[:limit]
    )


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


def script_of(word: str) -> str | None:
    """語をどちらの符号表で扱うか. ``None`` なら触らない.

    **モードではなく文字種で決める。** こうすると自動モードでも、和文の中の
    欧文区間 (``「…」``) でも、配線を足さずに正しい辞書が選ばれる。
    """
    if _KATAKANA_RE.search(word):
        return "japanese"
    if any(ch.isascii() and ch.isalpha() for ch in word):
        return "european"
    return None


def _ja_min_fuzzy_len(segment: str) -> int:
    """この断片に曖昧一致を許す最小長.

    通常は 3 文字 (``_JA_MIN_FUZZY_LEN``)。ただし**短くて句読点を含む断片は
    2 文字**まで許す。``デ`` と ``。`` は符号が 1 要素しか違わず、語の途中に
    句読点が現れること自体が読み違えの強い証拠だからである。

    例外を**短い断片に限る**のは実測による。長さを問わず許したところ
    ``、タカサ`` が ``イナズマ`` に化けた (2026-08-14)。句読点 1 文字ぶんの
    読み違えを直すのが目的なので、断片が長ければその根拠は弱い。

    読めなかった印 (``_``) を含む断片は例外にしない。``_`` は任意個の要素を
    吸収するワイルドカードなので、短い語への寄せまで許すと何にでも化ける。
    """
    if UNREADABLE in segment:
        return _JA_MIN_FUZZY_LEN
    if len(segment) <= 3 and any(ch in _JA_PUNCT for ch in segment):
        return 2
    return _JA_MIN_FUZZY_LEN


def _segment_japanese(
    chunk: str,
    *,
    ja_distance_ratio: float,
    ja_margin: float,
    extra: frozenset[str],
) -> list[str] | None:
    """和文の塊を語彙語の並びに切り直す (曖昧一致つき DP).

    欧文は厳密一致だけで足りるが、和文は**語の切れ目が当てにならない**ので
    切り直しと寄せを同時に解く必要がある (``テンキハ`` と ``ムンキニ`` は
    同じ問題の別の顔である)。

    採用する条件は 3 つ。どれも「正しい語を壊さない」ための歯止めである。

    1. **塊全体を語彙で説明できること。** 未知の断片が残る分割は採らない。
       これが無いと ``イチノセキ`` が ``イチ ノ セキ`` に割れる
    2. **2 文字以上の語を 1 つ以上含むこと。** 助詞だけの並びはどうにでも
       切れるので採らない (``ハノガ``)
    3. **1 文字の曖昧一致を許さない。** 1 文字カナはどれも符号が近く、
       許すとどのカナも別のカナに化ける。助詞は完全一致のみ

    Returns:
        採用できる分割、または ``None`` (そのままにする)。
    """
    n = len(chunk)
    if n < 2 or n > _JA_MAX_CHUNK_LEN:
        return None
    chunk_code = japanese_code_of(chunk)
    if not chunk_code:
        return None
    all_words = _japanese_all_words(extra)
    # best[i] = (選好値, 語数, 語の並び, 芯となる語を含むか, 素の編集量)
    #
    # **選好値と予算は別物**。選好値には助詞のペナルティを含めるが、予算
    # (``_JA_TOTAL_RATIO``) は素の編集量だけで見る。こうしないと正しい
    # ``テンキ`` + ``ハ`` が予算超過で落ちる。
    #
    # **比べるのは (編集量, 語数) の順**。語数そのものも費用に足していたところ、
    # 費用 0 の正しい分割 (``ジーピー`` + ``、`` + ``タカサ``、3 語) より
    # 曖昧一致を含む 2 語 (``ジーピー`` + ``、タカサ``→``イナズマ``) が
    # 勝ってしまった (2026-08-14 実測)。**語数は同点のときだけ見る。**
    best: list[tuple[float, int, list[str], bool, float] | None] = [None] * (n + 1)
    best[0] = (0.0, 0, [], False, 0.0)
    for i in range(1, n + 1):
        for j in range(max(0, i - _JA_MAX_WORD_LEN), i):
            prefix = best[j]
            if prefix is None:
                continue
            segment = chunk[j:i]
            # **助詞は塊の先頭には来ない。**
            #
            # 助詞は語の後ろに付くものなので、塊が助詞で始まる分割は不自然である。
            # これが無いと ``モロムマス`` が ``モ`` (助詞) + ``テマス`` と
            # 費用 0 で説明できてしまい、正しい ``シテマス`` (費用 1) に勝つ
            # (2026-08-14 実測)。塊まるごとが助詞 1 文字の場合は別扱い
            # (そもそも n < 2 で分割しない)。
            if j == 0 and i < n and segment in JAPANESE_PARTICLES:
                continue
            # **助詞を使う分割には選好上のペナルティを与える。**
            #
            # 助詞は接着剤であって、それ自体が分割を正当化すべきではない。
            # これが無いと ``モロムマス`` が ``モ`` + ``テマス`` (費用 0) と
            # 説明できてしまい、正しい ``シテマス`` (費用 1) に勝つ。
            # **予算には数えない** ので、``テンキ`` + ``ハ`` のような正しい
            # 分割は今までどおり通る (競合が無ければ選ばれる)。
            penalty = _JA_PARTICLE_PENALTY if segment in JAPANESE_PARTICLES else 0.0
            if segment in all_words:
                cost = 0.0
                fixed = segment
                has_content = prefix[3] or len(segment) >= _JA_MIN_CONTENT_LEN
            elif len(segment) >= _ja_min_fuzzy_len(segment):
                # **語の途中の句読点は読み違えの強い証拠**なので、そこだけ
                # 短い語への寄せも許す (``。ス`` → ``デス``、符号距離 1)。
                # デ = ・-・--・・ と 。 = ・-・-・・ は長音 1 本しか違わない。
                match = nearest_word(
                    segment, script="japanese",
                    ja_distance_ratio=ja_distance_ratio, ja_margin=ja_margin,
                    ja_min_target_len=_ja_min_fuzzy_len(segment),
                    extra=extra,
                )
                if match is None:
                    continue
                cost = word_distance(segment, match, "japanese")
                fixed = match
                has_content = True
            else:
                continue
            candidate = (
                prefix[0] + cost + penalty,
                prefix[1] + 1,
                [*prefix[2], fixed],
                has_content,
                prefix[4] + cost,
            )
            current = best[i]
            if current is None or candidate[:2] < current[:2]:
                best[i] = candidate
    tail = best[n]
    if tail is None or not tail[3]:
        return None
    # 連鎖した合計が塊全体に対して大きすぎる分割は採らない (**素の編集量**で見る)。
    if tail[4] > _JA_TOTAL_RATIO * len(chunk_code):
        return None
    return tail[2]


def correct_text(
    text: str,
    *,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    margin: float = DEFAULT_MARGIN,
    ja_distance_ratio: float = DEFAULT_JA_DISTANCE_RATIO,
    ja_margin: float = DEFAULT_JA_MARGIN,
    japanese_enabled: bool = True,
    japanese_extra: frozenset[str] = frozenset(),
) -> CorrectionResult:
    """確定テキストを語ごとに切り直し・寄せする.

    改行と語間スペースは保つ (改行は送信のターンの切れ目を表すため潰せない)。
    辞書は**モードではなく語の文字種**で選ぶ (``script_of``)。

    Args:
        japanese_enabled: 和文の補正を行うか。**欧文とは別に切れる**
            (運用者の要望、2026-08-14)。

    Returns:
        補正後テキストと、触った範囲、直せなかった語 (候補つき)。範囲は**補正後**
        テキストの文字位置で、UI が色を変えるのに使う。
    """
    if not text:
        return CorrectionResult(text)

    out: list[str] = []
    spans: list[CorrectedSpan] = []
    unresolved: list[UnresolvedWord] = []
    position = 0
    # 改行と空白の並びを保つため、区切り文字ごと分解して走査する。
    for line_index, line in enumerate(text.split("\n")):
        if line_index:
            out.append("\n")
            position += 1
        # 先頭・末尾の空白も保つので split ではなく手で刻む
        parts = _split_keeping_spaces(line)
        index = 0
        while index < len(parts):
            chunk = parts[index]
            if not chunk.strip():
                out.append(chunk)
                position += len(chunk)
                index += 1
                continue
            consumed, fixed, pending = _correct_from(
                parts, index,
                max_distance=max_distance, margin=margin,
                ja_distance_ratio=ja_distance_ratio, ja_margin=ja_margin,
                japanese_enabled=japanese_enabled, japanese_extra=japanese_extra,
            )
            original = "".join(parts[index:index + consumed])
            unresolved.extend(pending)
            if fixed != original:
                spans.append(CorrectedSpan(position, position + len(fixed), original))
            out.append(fixed)
            position += len(fixed)
            index += consumed
    return CorrectionResult("".join(out), tuple(spans), tuple(unresolved))


# 一度に繋いで試す塊の数。**繋ぐほど「説明できてしまう」危険も増える**ので、
# 3 で打ち切る。実際、余計なスペースは 1 語を 2〜3 個に割る程度である。
_JA_MAX_MERGE = 3


def _correct_from(
    parts: list[str],
    index: int,
    *,
    max_distance: float,
    margin: float,
    ja_distance_ratio: float,
    ja_margin: float,
    japanese_enabled: bool,
    japanese_extra: frozenset[str],
) -> tuple[int, str, tuple[UnresolvedWord, ...]]:
    """``parts[index]`` から始まる 1 語ぶん (または繋いだ数語ぶん) を直す.

    **送られてくる語間スペースは当てにならない。** 過去の分析で誤りの約 30% が
    語間スペースの過剰挿入だった。空白で区切った塊ごとに独立して処理すると、
    語の途中に余計なスペースが入った語は**原理的に一致しない** (``テン キハ``)。
    そこで隣り合う塊を繋いだものも試す。

    **繋ぐのは「繋ぐと説明できて、個別では説明できなかった」ときだけ。**
    無条件に繋ぐと、正しく分かれている語まで作り直してしまう。

    Returns:
        (消費した ``parts`` の数, 直した文字列, 直せなかった語)
    """
    single = _correct_chunk(
        parts[index],
        max_distance=max_distance, margin=margin,
        ja_distance_ratio=ja_distance_ratio, ja_margin=ja_margin,
        japanese_enabled=japanese_enabled, japanese_extra=japanese_extra,
    )
    if not japanese_enabled:
        return 1, single[0], single[1]
    # 繋ぐのは「個別では説明できなかった」ときだけ。ただし **1〜2 文字の塊は
    # そもそも単独では説明しようがない** (芯となる 3 文字語を含めない) ので、
    # 直せなかった語として報告もされない。``シ テマス`` の ``シ`` がこれで、
    # 隣と繋がないと永久に直らない。
    # **それ自体が語彙にある語なら繋がない。** ``モウ`` は正しい語なので、
    # ``モウ イチド`` を ``モウイチド`` に繋ぎ直す理由は無い (意味は同じでも
    # 元のテキストを変えることになる)。繋ぐのは ``シ`` や ``クモ`` のような、
    # 単独では語になっていない断片だけ。
    chunk = parts[index]
    too_short = (
        script_of(chunk) == "japanese"
        and len(chunk) < _JA_MIN_CONTENT_LEN
        and chunk not in _japanese_all_words(japanese_extra)
    )
    if not single[1] and not too_short:
        return 1, single[0], single[1]

    # ``_split_keeping_spaces`` は語と空白を交互に並べるので、
    # k 語を繋ぐには parts[index : index + 2k - 1] を消費する。
    for count in range(_JA_MAX_MERGE, 1, -1):
        last = index + 2 * (count - 1)
        if last >= len(parts):
            continue
        group = parts[index:last + 1]
        words = group[::2]
        spacers = group[1::2]
        if any(part.strip() for part in spacers):
            continue
        if any(script_of(word) != "japanese" or is_protected(word) for word in words):
            continue
        merged = "".join(words)
        fixed, pending = _correct_chunk(
            merged,
            max_distance=max_distance, margin=margin,
            ja_distance_ratio=ja_distance_ratio, ja_margin=ja_margin,
            japanese_enabled=True, japanese_extra=japanese_extra,
        )
        # **繋いだ結果は文字を変えてはいけない。切り直すだけ。**
        #
        # 繋いだ文字列は長いぶん編集量の予算 (``_JA_TOTAL_RATIO`` × 符号長) も
        # 増えるので、個別なら通らない曖昧一致が通ってしまう。実測 (2026-08-14)
        # で ``イカガ デス`` が ``イチド デス``、``ミナサン オゲンキ`` が
        # ``ヘンシン オゲンキ`` に化けた。**繋ぐ目的はスペースの直しであって、
        # 文字を直すことではない。** 文字の直しは塊ごとの処理に任せる。
        if pending or fixed.replace(" ", "") != merged:
            continue
        # **繋いだ結果が元の境界を 1 つも跨がないなら、何も繋いでいない。**
        #
        # その場合の「説明できた」は、繋いだことで芯となる 3 文字語が
        # 隣から供給されただけで、実際には元の塊をそのまま切り直している。
        # 実測 (2026-08-14) で ``コレデ オワリ`` が ``コレ デ オワリ`` になり、
        # ``カイマシタ ソチラ`` なら ``カ イマ シタ ソチラ`` になりうる。
        # **繋ぐ目的は余計なスペースを跨ぐことなので、跨がないなら採らない。**
        if _merge_spans_boundary(words, fixed):
            return len(group), fixed, ()
    return 1, single[0], single[1]


def _merge_spans_boundary(words: list[str], fixed: str) -> bool:
    """切り直しの結果が、元のスペース位置を 1 つ以上跨いでいるか."""
    original: set[int] = set()
    offset = 0
    for word in words[:-1]:
        offset += len(word)
        original.add(offset)
    produced: set[int] = set()
    offset = 0
    pieces = fixed.split(" ")
    for piece in pieces[:-1]:
        offset += len(piece)
        produced.add(offset)
    return bool(original - produced)


def _correct_chunk(
    chunk: str,
    *,
    max_distance: float,
    margin: float,
    ja_distance_ratio: float,
    ja_margin: float,
    japanese_enabled: bool,
    japanese_extra: frozenset[str],
) -> tuple[str, tuple[UnresolvedWord, ...]]:
    """空白で区切られた 1 塊を、**文字種に応じた辞書で**直す."""
    script = script_of(chunk)
    if script == "european":
        return _correct_word(chunk, max_distance=max_distance, margin=margin), ()
    if script != "japanese" or not japanese_enabled:
        return chunk, ()
    return _correct_japanese_chunk(
        chunk,
        ja_distance_ratio=ja_distance_ratio,
        ja_margin=ja_margin,
        extra=japanese_extra,
    )


@lru_cache(maxsize=8192)
def _correct_japanese_chunk(
    chunk: str,
    *,
    ja_distance_ratio: float,
    ja_margin: float,
    extra: frozenset[str],
) -> tuple[str, tuple[UnresolvedWord, ...]]:
    """和文の塊を直す. 句読点は切れ目として保つ.

    **キャッシュが要る。** ``correct_text`` は確定テキスト**全体**に対して
    hop (0.5 秒) ごとに呼ばれる。確定テキストは末尾に伸びていくだけなので、
    塊ごとに覚えておけば 2 回目以降は末尾の 1 塊しか計算しない。
    これが無いと交信が進むほど遅くなり、GUI スレッドが詰まる
    (2026-08-14 に実際に凍結した。83 文字で 1.8 秒、1,328 文字で 29 秒)。
    """
    out: list[str] = []
    unresolved: list[UnresolvedWord] = []
    # **句読点で切らない。** 以前は無条件に区切り記号として保護していたが、
    # ``デ`` (・-・--・・) と ``。`` (・-・-・・) は符号が 1 要素しか違わず、
    # **語の途中に現れた句読点は読み違えであることが多い** (運用者、2026-08-14)。
    # 保護してしまうと直せない。正しい句読点は語彙側で「そのまま切り出せる要素」
    # として扱うので消えない。
    for part in [_apply_forced_substitutions(chunk)]:
        if not part:
            continue
        if is_protected(part):
            out.append(part)
            continue
        pieces = _segment_japanese(
            part, ja_distance_ratio=ja_distance_ratio,
            ja_margin=ja_margin, extra=extra,
        )
        if pieces is None:
            out.append(part)
            # 直せなかった語は候補を添えて LLM へ回す (B 案)。
            # **既に語彙にある語は「直せなかった」ではない** ので出さない
            # (分割が不要なだけ)。出すと LLM に無用な候補を渡すことになる。
            if len(part) >= 2 and part not in _japanese_all_words(extra):
                # **候補はここでは計算しない。** 必要になるのは LLM を呼ぶときだけ
                # なのに、``correct_text`` は hop (0.5 秒) ごとに走る。
                # 1 塊あたり 60 ms かかり、GUI スレッドには重すぎた。
                unresolved.append(UnresolvedWord(part, ()))
            continue
        out.append(_join_pieces(pieces))
    return "".join(out), tuple(unresolved)


def _apply_forced_substitutions(text: str) -> str:
    """``JAPANESE_FORCED_SUBSTITUTIONS`` を無条件に適用する."""
    for source, target in JAPANESE_FORCED_SUBSTITUTIONS.items():
        text = text.replace(source, target)
    return text


def _join_pieces(pieces: list[str]) -> str:
    """語を空白で繋ぐ. **句読点の前後には空白を入れない.**

    ``シテマス 。`` や ``デ、 サヨウナラ`` にしないため。塊は空白で切ってから
    渡されるので、塊の中の句読点は必ず前後の文字と地続きだったものである。
    """
    text = ""
    previous = ""
    for piece in pieces:
        if text and piece not in _JA_PUNCT and previous not in _JA_PUNCT:
            text += " "
        text += piece
        previous = piece
    return text


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
    """末尾の ``?`` が本物の疑問符か.

    **記号を分けたので、いまは常に本物である** (読めなかった箇所は ``_``)。
    ``is_protected`` が ``?`` を含む語をまとめて守るため、この関数は
    補正の判断には使われていない。後方互換のために残してある。
    """
    return len(word) > 1 and word.endswith("?") and word[:-1] in _LEXSET


def _correct_word(word: str, *, max_distance: float, margin: float) -> str:
    """1 語を切り直し → 寄せ の順に直す."""
    if is_protected(word):
        return word
    pieces = segment_word(word)
    fixed = [
        nearest_word(piece, max_distance=max_distance, margin=margin) or piece
        for piece in pieces
    ]
    return " ".join(fixed)


__all__ = [
    "DEFAULT_JA_DISTANCE_RATIO",
    "DEFAULT_JA_LEXICON_PATH",
    "DEFAULT_JA_MARGIN",
    "DEFAULT_MARGIN",
    "DEFAULT_MAX_DISTANCE",
    "EUROPEAN_LEXICON",
    "JAPANESE_LEXICON",
    "JAPANESE_PARTICLES",
    "Candidate",
    "CorrectedSpan",
    "CorrectionResult",
    "UnresolvedWord",
    "candidates_for",
    "correct_text",
    "japanese_code_of",
    "japanese_max_distance",
    "japanese_target_words",
    "load_user_lexicon",
    "script_of",
    "user_target_words",
    "with_candidates",
    "is_protected",
    "is_real_question",
    "nearest_word",
    "segment_word",
    "substitution_cost",
    "word_distance",
]

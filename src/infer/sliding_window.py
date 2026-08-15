"""スライディングウィンドウ再デコード + prefix commit (Phase 6).

ライブ音声をリングバッファに保持し、HOP_S ごとに窓全体を ``decode_chunk`` で
再デコードする. ``[max(head_guard, last_commit), now - commit_lag)`` の区間に
入るトークンのみを **不変 (immutable)** に確定する. 確定済みは後から変化しない
ため UI 表示がちらつかない (design.md §5.4/5.5 の文脈損失問題の構造的解消).

**唯一の例外が 2 段階確定** (:meth:`SlidingWindowDecoder.refine_closed_turns`)。
ターンが終わった瞬間に、そのターンだけを全文脈でデコードし直して置き換える。
書き換えは「ターン終了時に 1 回だけ」に限る (2026-08-12、運用者の承認済み)。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.infer.engine import FrameToken, InferenceEngine
from src.tokens.morse_tokens import (
    DAKUTEN_CHAR,
    HANDAKUTEN_CHAR,
    JAPANESE_TABLE,
    TOKEN_TO_ID,
)

# 濁点・半濁点のトークン ID。
#
# **符号定義は morse_tokens.py が唯一の真正ソース**なので、ここでは持たずに
# 毎回引き当てる (アーキテクチャ原則 2)。欧文モードでも同じトークン ID が
# 使われる (``・・`` は欧文の ``I``) が、待たされるのは 1 hop なので実害は無い。
VOICING_MARK_TOKEN_IDS: frozenset[int] = frozenset(
    TOKEN_TO_ID[code]
    for code, char in JAPANESE_TABLE.items()
    if char in (DAKUTEN_CHAR, HANDAKUTEN_CHAR) and code in TOKEN_TO_ID
)

# デコードの前に足す助走の無音 (秒)。
#
# **モデルは音の立ち上がりで幻覚を出す。** held-out 実録音 21 件の実測で、
# 先頭に余計なトークンが出たのは 18/21 件。別実装のデコーダは同じ音で 0/10 件だった
# 。学習データの録音が常に符号で始まるため、
# 「静かなところから符号が始まる」を知らないものと思われる。
#
# **0.3 秒の無音を足すだけで TER 24.21% → 22.32% (-1.9pt)**、先頭の誤りは
# 18/21 → 13/21 に減る。0.6 秒・1.0 秒はかえって悪い (22.74% / 23.37%)。
# 無音の中に出たトークンを捨てても数字は変わらない = モデルは無音では
# 何も出さず、立ち上がりでだけ出す。
DEFAULT_LEAD_IN_S = 0.3

# 確信度が閾値未満のトークンに与える「余分な猶予」(秒)。
#
# 閾値未満のトークンは画面に読めなかった印 (``_``) として出る。そのまま確定して
# しまうと、右文脈が増えたあとの読みを永久に取り逃がす。**確定を少し待たせれば、
# より多くの右文脈で読み直した結果を確定できる** (運用者の要望、2026-08-14)。
#
# **無限には待たない。** 猶予を過ぎたら確信度が低いままでも確定する。待ち続けると
# 画面が進まなくなり、「読めなかった」ことすら分からなくなる。
#
# held-out 21 件の掃引 (2026-08-14、2 段階確定なし)::
#
#     猶予 0.0 秒 (従来)  TER 25.73%   欧文 18.41% / 和文 34.13%
#     猶予 0.5 秒         TER 25.50%   (-0.22pt)
#     猶予 1.0 秒         TER 24.38%   (-1.34pt)
#     猶予 1.5 秒         TER 23.71%   (-2.01pt)  ← 採用
#     猶予 2.0 秒         TER 23.49%   (-2.24pt)
#
# 単調に改善し、**和文で -3.84pt** (34.13% → 30.29%) と効きが大きい。2.0 秒の方が
# わずかに良いが、読めなかった文字の確定が最大 2 秒遅れる代償があるため、改善幅の
# 大半を取れる 1.5 秒を採る (運用者の判断)。
#
# **2 段階確定が届く場面では効果が消える** (3 回目がターン全体を読み直して上書き
# するため、猶予 0.0〜2.0 のどれでも TER 21.25% で同じ)。効くのは 3 回目が
# 諦める場面 — 長い送信や、音がリングから落ちたターン — である。
DEFAULT_LOW_CONFIDENCE_EXTRA_LAG_S = 1.5


@dataclass(frozen=True)
class CommittedToken:
    """確定 (immutable) トークン. 絶対サンプル位置付き."""

    token_id: int
    confidence: float
    absolute_sample_start: int
    absolute_sample_end: int


@dataclass(frozen=True)
class DecodeView:
    """1 回の再デコード結果のスナップショット."""

    committed: list[CommittedToken] = field(default_factory=list)       # 確定済み全体
    newly_committed: list[CommittedToken] = field(default_factory=list)  # 今回新規確定
    provisional: list[CommittedToken] = field(default_factory=list)      # 暫定 (グレー表示)


class SlidingWindowDecoder:
    def __init__(
        self,
        engine: InferenceEngine,
        window_s: float = 30.0,
        # AppSettings / ブラウザ版と揃えること。効くのは単独値ではなく
        # `commit_lag_s + hop_s / 2` (= 実効右文脈) で、目標は 2.25 秒
        # (docs/commit_lag_sweep_result.md §8)。実運用の呼び出し側は明示的に
        # 渡すが、既定が取り残されると直接生成したときだけ挙動が変わる
        hop_s: float = 0.5,
        commit_lag_s: float = 2.0,
        head_guard_s: float = 1.0,
        decode_left_context_s: float = 5.0,
        commit_jitter_margin_s: float = 0.02,
        sample_rate: int = 8000,
        lead_in_s: float = DEFAULT_LEAD_IN_S,
        low_confidence_threshold: float = 0.5,
        low_confidence_extra_lag_s: float = DEFAULT_LOW_CONFIDENCE_EXTRA_LAG_S,
    ) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.hop_samples_audio = int(hop_s * sample_rate)
        self.commit_lag_samples = int(commit_lag_s * sample_rate)
        self.head_guard_samples = int(head_guard_s * sample_rate)
        self.left_context_samples = int(decode_left_context_s * sample_rate)
        self.jitter_margin_samples = int(commit_jitter_margin_s * sample_rate)
        # 音の立ち上がりの幻覚よけ (:data:`DEFAULT_LEAD_IN_S`)
        self.lead_in_samples = int(lead_in_s * sample_rate)
        # 読めなかった印になるトークンに与える余分な猶予
        self.low_confidence_threshold = low_confidence_threshold
        self.low_confidence_extra_lag = int(low_confidence_extra_lag_s * sample_rate)
        self._ring = np.zeros(0, dtype=np.float32)
        self._total_consumed = 0                 # 累積投入サンプル数 (= 現在時刻)
        self._committed: list[CommittedToken] = []
        # 確定済み末尾の絶対サンプル位置 (中点ウォーターマーク基準). 無ければ None.
        self._last_commit_end: int | None = None
        # 2 段階確定で直し終えたターン (先頭トークンの絶対開始位置で覚える)。
        # 書き換えは「ターン終了時に 1 回だけ」の約束なので、二度直さない
        self._refined_turns: set[int] = set()

    def reset(self) -> None:
        self._ring = np.zeros(0, dtype=np.float32)
        self._total_consumed = 0
        self._committed = []
        self._last_commit_end = None
        self._refined_turns = set()

    def push(self, audio: np.ndarray) -> None:
        """音声を追加 (デコードはしない). 窓長を超えた古い分は捨てる."""
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        audio = audio.astype(np.float32, copy=False)
        self._total_consumed += audio.size
        self._ring = np.concatenate([self._ring, audio])
        if self._ring.size > self.window_samples:
            self._ring = self._ring[-self.window_samples:]

    def refine_closed_turns(
        self,
        gap_samples: int,
        *,
        all_turns: bool = False,
        guard_s: float = 0.25,
        lead_in_s: float = DEFAULT_LEAD_IN_S,
    ) -> bool:
        """終わったターンを全文脈でデコードし直して置き換える (2 段階確定).

        **「確定済みは後から変化しない」の唯一の例外である** (2026-08-12、
        運用者の承認済み)。ライブ確定は実効 2.25 秒の右文脈しか使えず、
        held-out 実録音 21 件の実測でオフライン全文脈より TER が 3.2pt 悪い
        (27.4% → 24.2%)。ターン (``gap_samples`` 以上の無音で区切られた一続き)
        が終わった瞬間に、そのターンの音をリングバッファから取り出して
        ``decode_chunk`` に丸ごと通し、そのターンのトークンを置き換える。

        書き換えは**ターン終了時に 1 回だけ** (``_refined_turns`` で覚える)。
        表示のちらつきを「ターン中は不変・終了時に 1 回」に限るための約束である。

        ターンが「終わった」とみなす条件:

        * 次のターンが始まっている (後ろに ``gap_samples`` 以上の無音を挟んで
          別のトークンがある)、または
        * 末尾から ``gap_samples + commit_lag_samples`` 以上の無音が続いた
          (これ以降に確定するトークンは必ず新しいターンになる)、または
        * ``all_turns=True`` (停止時。もう音は増えない)

        守りの規則:

        * **音がリングから落ちたターンは諦める** (第 1 段の結果を残す。
          黙って消すより、精度がライブ確定のままの方がよい)
        * **置き換えが空なら元を残す** (見えていた文字を黙って消さない)
        * 結果が元と同じなら ``False`` を返す (無駄な再描画を起こさない)

        デコードに渡す区間は**隣のターンの手前までの無音を含む** (固定長の
        余白ではない)。単独のターンなら区間はリング全体になり、オフラインの
        一括デコードと**同じ入力**になる。固定 0.5 秒の余白では held-out で
        オフラインに 1.3pt 届かなかった (無音の量が違うと特徴量の正規化が
        変わり、同じ音でも結果がずれる)。

        Args:
            gap_samples: ターンの切れ目とみなす無音の長さ (サンプル)。
                ``line_break`` の改行と同じ値を渡すこと (ターンの定義を揃える)。
            all_turns: 停止時に ``True``。最後のターンも待たずに直す。
            guard_s: 隣のターンのトークンからこれだけ離す (秒)。隣の符号の
                尻尾を拾って**別のターンの文字を重複させない**ため。
            lead_in_s: デコードの前に足す無音の長さ (秒)。
                :data:`DEFAULT_LEAD_IN_S` の注記を参照。

        Returns:
            トークンが実際に変わったら ``True`` (呼び出し側は表示を作り直す)。
        """
        if gap_samples <= 0 or not self._committed:
            return False
        hop = self.engine.frame_hop_samples
        ring_start_abs = self._total_consumed - self._ring.size
        guard = int(guard_s * self.sample_rate)

        # ターンに刻む (line_break.split_at_gaps と同じ規則)
        turns: list[list[CommittedToken]] = [[self._committed[0]]]
        for prev, cur in zip(self._committed, self._committed[1:], strict=False):
            if cur.absolute_sample_start - prev.absolute_sample_end >= gap_samples:
                turns.append([cur])
            else:
                turns[-1].append(cur)

        changed = False
        rebuilt: list[CommittedToken] = []
        for index, turn in enumerate(turns):
            start_abs = turn[0].absolute_sample_start
            end_abs = turn[-1].absolute_sample_end
            closed = (
                index < len(turns) - 1
                or all_turns
                or self._total_consumed - end_abs
                >= gap_samples + self.commit_lag_samples
            )
            if (
                not closed
                or start_abs in self._refined_turns
                # **ターン自体の音が落ちたときだけ諦める。** 余白ぶんまで
                # 要求すると、録音の先頭近くのターン (余白の取りようがない) を
                # 全部諦めてしまう — 実際にそうなっていた。held-out 21 件の
                # 余白掃引で「余白を増やすほど 1 段階の値に戻る」という
                # 逆向きの結果が出て発覚した (余白 2.5 秒で全件スキップ)
                or start_abs < ring_start_abs
            ):
                rebuilt.extend(turn)
                continue

            prev_end = turns[index - 1][-1].absolute_sample_end if index > 0 else None
            next_start = (
                turns[index + 1][0].absolute_sample_start
                if index < len(turns) - 1 else None
            )
            span_start = (
                ring_start_abs if prev_end is None
                else max(ring_start_abs, prev_end + guard)
            )
            span_end = (
                self._total_consumed if next_start is None
                else next_start - guard
            )
            wave = self._ring[span_start - ring_start_abs:span_end - ring_start_abs]
            # **助走の無音を足す** (:data:`DEFAULT_LEAD_IN_S`)。位置は足した
            # ぶんだけ戻す。
            lead_in = int(lead_in_s * self.sample_rate)
            if lead_in > 0:
                wave = np.concatenate([np.zeros(lead_in, dtype=np.float32), wave])
            origin = span_start - lead_in
            refined = [
                CommittedToken(
                    token_id=tok.token_id,
                    confidence=tok.confidence,
                    absolute_sample_start=origin + tok.frame_start * hop,
                    absolute_sample_end=origin + tok.frame_end * hop,
                )
                for tok in self.engine.decode_chunk(wave)
                # **無音の中に出たトークンは捨てる。** そこに音が無いのだから
                # 符号ではありえない (実測では出ないが、出たら幻覚である)
                if origin + tok.frame_start * hop >= span_start
            ]
            self._refined_turns.add(start_abs)
            if not refined:
                rebuilt.extend(turn)
                continue
            # **置き換え後の開始位置も覚える。** 置き換えでターンの先頭が
            # 動くと、次の呼び出しで別のターンに見えて二度直してしまう
            # (実際にテストで踏んだ)
            self._refined_turns.add(refined[0].absolute_sample_start)
            if [(t.token_id, t.absolute_sample_start, t.absolute_sample_end) for t in refined] != [
                (t.token_id, t.absolute_sample_start, t.absolute_sample_end) for t in turn
            ]:
                changed = True
            rebuilt.extend(refined)
        self._committed = rebuilt
        return changed

    def recent_audio(self, seconds: float) -> np.ndarray:
        """直近 ``seconds`` 秒の音声を返す (窓に残っている範囲で).

        受信信号の速度 (WPM) を測るために使う (``src/infer/wpm.py``)。
        **溜め直さずここから取る。** 同じ音を二重に持つ意味が無いうえ、
        片方だけスキッシュを通すといった食い違いが起きる。

        窓長 (``window_s``) より長くは遡れない。返すのはビューではなくコピーで、
        呼び出し側が書き換えてもリングバッファは壊れない。
        """
        n = int(seconds * self.sample_rate)
        if n <= 0 or self._ring.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self._ring[-n:].copy()

    def redecode(self) -> DecodeView:
        """確定済み末尾 - left_context 以降を再デコードし確定/暫定を更新.

        確定済み領域は不変なので再計算しない (CPU 負荷削減). 末尾 commit_lag
        圏内のトークンは右文脈不足のため暫定とする.
        """
        return self._decode(commit_limit_abs=self._total_consumed - self.commit_lag_samples)

    def finalize(self) -> DecodeView:
        """送信終了時の最終確定. commit_lag を無視し残り全部を確定する.

        ストリーム終端では右文脈がもう増えないため、暫定として保留していた
        末尾トークンをここで確定させる (ワーカーの stop() から呼ぶ).
        """
        return self._decode(commit_limit_abs=self._total_consumed + 1)

    def _decode(self, commit_limit_abs: int) -> DecodeView:
        if self._ring.size == 0:
            return DecodeView(committed=list(self._committed))

        hop = self.engine.frame_hop_samples
        ring_start_abs = self._total_consumed - self._ring.size

        # --- デコード区間の動的短縮 (RTF 対策) ---
        last_end = self._last_commit_end
        anchor = last_end if last_end is not None else ring_start_abs
        decode_start_abs = max(ring_start_abs, anchor - self.left_context_samples)
        sub = self._ring[decode_start_abs - ring_start_abs:]

        # **区間の頭に音がいきなり来るときだけ助走の無音を足す**
        # (:data:`DEFAULT_LEAD_IN_S`)。それ以外は既に左文脈があるので要らない。
        #
        # 該当するのは「まだ何も確定しておらず、リングの先頭から復号する」
        # 場面 = **交信の冒頭**である。ここでモデルが幻覚を出すと、その 1 文字が
        # 確定してしまい後から消せない (第 2 段の書き直しまで残る)。
        lead_in = self.lead_in_samples if decode_start_abs == ring_start_abs else 0
        if lead_in > 0:
            sub = np.concatenate([np.zeros(lead_in, dtype=np.float32), sub])
        frame_tokens = self.engine.decode_chunk(sub)
        # 位置は助走ぶん戻す。**無音の中に出たものは捨てる** (そこに音は無い)
        origin_abs = decode_start_abs - lead_in

        # head guard: デコード区間先頭の不採用区間. ただし区間がストリーム先頭
        # (decode_start_abs == 0) のときは先頭信号を捨てない (design.md §4.4).
        head_cut_abs = (
            decode_start_abs + self.head_guard_samples if decode_start_abs > 0 else 0
        )

        newly: list[CommittedToken] = []
        provisional: list[CommittedToken] = []
        for index, tok in enumerate(frame_tokens):
            abs_start = origin_abs + tok.frame_start * hop
            abs_end = origin_abs + tok.frame_end * hop
            if abs_start < decode_start_abs:          # 助走の無音の中 = 幻覚
                continue
            ct = CommittedToken(
                token_id=tok.token_id,
                confidence=tok.confidence,
                absolute_sample_start=abs_start,
                absolute_sample_end=abs_end,
            )
            # 右文脈不足 (commit 境界を終了がまたぐ) → 暫定
            if abs_end >= commit_limit_abs:
                provisional.append(ct)
                continue
            # **読めなかった印になる文字は、もう少し待って読み直す。**
            #
            # 確信度が閾値未満のトークンは画面に ``_`` として出る。そのまま
            # 確定すると、右文脈が増えたあとの読みを永久に取り逃がす。
            # 猶予を過ぎたら確信度が低いままでも確定する (待ち続けると画面が
            # 進まない)。``finalize()`` は commit_limit を全部の先に置くので
            # ここも通り抜ける。
            if (
                self.low_confidence_extra_lag > 0
                and tok.confidence < self.low_confidence_threshold
                and abs_end >= commit_limit_abs - self.low_confidence_extra_lag
            ):
                provisional.append(ct)
                continue
            # 左文脈なし → 不採用
            if abs_start < head_cut_abs:
                continue
            # 中点ウォーターマーク: 確定済み末尾を中点が超えるもののみ新規確定.
            # 既確定トークンの再出現 (midpoint < last_end) は確実にスキップされ、
            # 文字間ギャップが小さくても新規トークンは脱落しない.
            midpoint = (abs_start + abs_end) // 2
            if last_end is not None and midpoint <= last_end + self.jitter_margin_samples:
                continue
            # **濁点が付くカナは、濁点が確定できるまで確定させない。**
            #
            # 濁点・半濁点は基本カナとは別のトークンである。基本カナだけが先に
            # 確定境界を越えると、画面には清音が出る (``デ`` が ``テ`` に見える)。
            # 濁点が確定するのは 1〜3 hop 後で、そこで濁音に直る。運用者からは
            # 「正しかった文字が誤りに変わり、しばらくして直る」ように見える
            # (2026-08-14 の実受信 105 秒で 9 回)。
            #
            # **遅らせるだけで、確定済みは書き換えない** (原則は守られる)。
            # 待つのは濁点が暫定圏にいる間だけなので、通常 1 hop で済む。
            if _next_is_pending_voicing_mark(
                frame_tokens, index, origin_abs, hop, commit_limit_abs
            ):
                provisional.append(ct)
                continue
            self._committed.append(ct)
            newly.append(ct)
            last_end = abs_end
            self._last_commit_end = abs_end

        return DecodeView(
            committed=list(self._committed),
            newly_committed=newly,
            provisional=provisional,
        )


def _next_is_pending_voicing_mark(
    frame_tokens: Sequence[FrameToken],
    index: int,
    origin_abs: int,
    hop: int,
    commit_limit_abs: int,
) -> bool:
    """次のトークンが**まだ確定できない**濁点・半濁点か.

    濁点が確定圏に入っていれば ``False`` (カナと濁点をまとめて確定してよい)。
    終端の ``finalize()`` は ``commit_limit_abs`` を全部の先に置くので、
    ここは常に ``False`` になり待たされない。
    """
    if index + 1 >= len(frame_tokens):
        return False
    nxt = frame_tokens[index + 1]
    if nxt.token_id not in VOICING_MARK_TOKEN_IDS:
        return False
    return origin_abs + nxt.frame_end * hop >= commit_limit_abs


__all__ = [
    "VOICING_MARK_TOKEN_IDS",
    "CommittedToken",
    "DecodeView",
    "SlidingWindowDecoder",
]

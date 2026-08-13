"""符号トークン列 → 表示テキスト変換器.

NN が出力する符号トークン ID 列を、モード別変換表 (欧文 / 和文) で
表示テキストに変換する. 確信度閾値未満および表ルックアップ失敗時は
``?`` で置換し、種別 (``LOW_CONFIDENCE`` / ``TABLE_MISS``) をログに残す.

和文モードでは濁点 (``゛``) ・半濁点 (``゜``) を直前カナと合成する.

``convert_timed`` メソッドは ``FrameToken`` 列を受け取り、トークン間の
フレームギャップから語間スペース (7 dot 相当の無音) を自動検出して
スペース文字を挿入する.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from src.tokens.morse_tokens import (
    BLANK_TOKEN_ID,
    DAKUTEN_CHAR,
    DAKUTEN_COMPOSE,
    DisplayMode,
    EUROPEAN_TABLE,
    HANDAKUTEN_CHAR,
    HANDAKUTEN_COMPOSE,
    HORE_CODE,
    ID_TO_TOKEN,
    JAPANESE_TABLE,
    Mode,
    RATA_CODE,
    WORD_BREAK_TOKEN_ID,
)

FALLBACK_CHAR: Literal["?"] = "?"
FallbackKind = Literal["TABLE_MISS", "LOW_CONFIDENCE"]


class TimedToken(Protocol):
    """``FrameToken`` 互換の時刻情報付きトークン."""

    token_id: int
    confidence: float
    frame_start: int
    frame_end: int


@dataclass(frozen=True)
class FallbackEvent:
    """``?`` 出力イベント (デバッグ用ログ)."""

    # ``?`` を出した時点の出力スロット番号 (``out_chars`` の添字) であって、
    # 最終テキストの文字列インデックスではない。``[SK]`` ``[ホレ]`` のように
    # 表示値が複数文字のトークンがあるため両者は一致しない
    # (``"[SK]?"`` の ``?`` はスロット 1 だが文字列位置は 4)。
    # Python 側はこの値をログ出力にしか使っていないので実害はないが、
    # 文字列添字として使ってはいけない (TypeScript 版の FallbackEvent.position は
    # UI が直接添字参照するため、あちらは文字列インデックスになっている)。
    position: int
    input_index: int                    # 入力 token_ids のインデックス
    token_id: int
    code: str                           # ``"<UNKNOWN>"`` または符号
    kind: FallbackKind
    confidence: float | None = None


@dataclass
class ConvertResult:
    text: str
    fallback_log: list[FallbackEvent] = field(default_factory=list)
    final_mode: Mode = "european"


class TokenConverter:
    """符号トークン列 → 文字列変換器.

    Args:
        mode: ``"european"``、``"japanese"``、または ``"auto"`` (自動切替).
        confidence_threshold: この値**未満**の確信度のトークンは ``?`` に置換.
            ``[0.0, 1.0]`` の範囲. デフォルト 0.5.
        prosign_threshold: auto モードでモード切替に使うプロサイン
            (ホレ / 和文中のラタ) にだけ適用する閾値. ``None`` なら
            ``confidence_threshold`` と同じ (従来の挙動).

            **なぜ分けるか**: ホレ/ラタは 1 トークンの誤りが後続の全文字の
            解釈を変えるため、他のトークンと重みが違う。2026-08-04 の実測では
            実録音でホレの確信度が 0.42〜0.46 に集まり、既定の 0.50 で棄却されて
            自動モード切替が働かなかった (合成音では 1.00 なのでモデルの
            能力の問題ではない)。固定モード (european / japanese) では
            モード切替が無いため、この値は無視される.
        switch_on_japanese_only: auto モードの欧文中に**欧文表に無く和文表にある**
            符号が来たら和文へ切り替えるか. 既定 ``True``.

            自動切替は実質的に和文のためにあるが、ホレ 1 個の検出に賭けるのは脆い。
            和文専用符号は 23 種あり高頻度カナ (コ ソ ロ ノ ス ア シ ヒ 等) を含むため
            和文の送信なら数文字で必ず当たる。しかもこれらは欧文モードでは
            TABLE_MISS で ``?`` になるため、切り替えたほうが確実に良くなる。
            和文から欧文へはラタ (``・・・-・``) で戻る (従来どおり).
    """

    def __init__(
        self,
        mode: DisplayMode,
        confidence_threshold: float = 0.5,
        prosign_threshold: float | None = None,
        switch_on_japanese_only: bool = True,
    ) -> None:
        if mode not in ("european", "japanese", "auto"):
            raise ValueError(f"Unknown mode: {mode!r}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        if prosign_threshold is not None and not 0.0 <= prosign_threshold <= 1.0:
            raise ValueError(
                f"prosign_threshold must be in [0, 1], got {prosign_threshold}"
            )
        self.mode: DisplayMode = mode
        self._auto: bool = mode == "auto"
        self.confidence_threshold: float = confidence_threshold
        self.prosign_threshold: float | None = prosign_threshold
        self.switch_on_japanese_only: bool = switch_on_japanese_only
        # auto では走査中に表を切替えるため、固定表は持たない (european を既定保持).
        self._table: dict[str, str] = (
            JAPANESE_TABLE if mode == "japanese" else EUROPEAN_TABLE
        )

    @staticmethod
    def _table_for(active: Mode) -> dict[str, str]:
        return JAPANESE_TABLE if active == "japanese" else EUROPEAN_TABLE

    def _threshold_for(self, code: str, active: Mode) -> float:
        """このトークンに適用する確信度閾値を返す.

        auto モードで**実際にモード切替を起こす**トークンだけ
        ``prosign_threshold`` を使う。欧文中のラタ (= プロサイン SN) は
        切替を起こさない通常のトークンなので対象外.
        """
        if self.prosign_threshold is None or not self._auto:
            return self.confidence_threshold
        switches_mode = code == HORE_CODE or (
            code == RATA_CODE and active == "japanese"
        )
        return self.prosign_threshold if switches_mode else self.confidence_threshold

    def convert(
        self,
        token_ids: list[int],
        confidences: list[float] | None = None,
        initial_mode: Mode | None = None,
        keep_leading_space: bool = False,
    ) -> ConvertResult:
        """トークン ID 列を表示テキストに変換.

        ``confidences`` を渡すと閾値判定を有効化. ``None`` ならすべて確信度 1.0 扱い.
        ``initial_mode`` は走査開始時のサブモード. 自動モードで暫定列を確定列の末尾
        モードから継続させる用途. ``None`` のとき auto は ``"european"`` から、
        固定モードは ``self.mode`` から開始する.
        固定モード (european/japanese) では ``initial_mode`` は無視される.

        ``keep_leading_space`` は先頭の WORD_BREAK をスペースとして残すかどうか.
        既定 ``False`` では行頭の余計なスペースを避けるため捨てる. ライブ経路は
        確定列と暫定列を別々に変換して連結するため、境界が語間に落ちると
        スペースが消える (``"GL 73 CQ"`` → ``"GL 73CQ"``). 暫定列の変換で
        ``True`` を渡すとこれを防げる. **確定列が既にスペースで終わっている場合は
        ``False`` を渡すこと** (二重スペースになる). 判断は呼び出し側が行う.
        """
        if confidences is not None and len(confidences) != len(token_ids):
            raise ValueError(
                f"confidences length {len(confidences)} != token_ids length "
                f"{len(token_ids)}"
            )

        if self._auto:
            active: Mode = initial_mode if initial_mode is not None else "european"
        else:
            # 固定モードでは initial_mode を無視 (常に自モードの表を使う).
            active = self.mode  # type: ignore[assignment]  # _auto が False なので "auto" ではない
        table = self._table_for(active)

        out_chars: list[str] = []
        log: list[FallbackEvent] = []
        # 和文合成用: 直前の出力が「濁点/半濁点と合成可能なカナ」だった位置
        composable_at: int | None = None

        for i, tid in enumerate(token_ids):
            if tid == BLANK_TOKEN_ID:
                continue

            # WORD_BREAK は語間スペースとして出力 (連続するスペースは1つに集約)
            if tid == WORD_BREAK_TOKEN_ID:
                if out_chars:
                    if out_chars[-1] != " ":
                        out_chars.append(" ")
                elif keep_leading_space:
                    # 確定列との連結でスペースが消えるのを防ぐ (docstring 参照)
                    out_chars.append(" ")
                composable_at = None
                continue

            conf = confidences[i] if confidences is not None else 1.0
            token = ID_TO_TOKEN.get(tid)

            if token is None:
                self._emit_fallback(
                    out_chars, log, i, tid, "<UNKNOWN>", "TABLE_MISS", conf
                )
                composable_at = None
                continue

            if conf < self._threshold_for(token.code, active):
                self._emit_fallback(
                    out_chars, log, i, tid, token.code, "LOW_CONFIDENCE", conf
                )
                composable_at = None
                continue

            # --- 自動モード: プロサインによる切替 (確信度を満たしたトークンのみ) ---
            if self._auto:
                if token.code == HORE_CODE:
                    out_chars.append("[ホレ]")
                    active = "japanese"
                    table = self._table_for(active)
                    composable_at = None
                    continue
                if token.code == RATA_CODE and active == "japanese":
                    out_chars.append("[ラタ]")
                    active = "european"
                    table = self._table_for(active)
                    composable_at = None
                    continue
                # それ以外 (欧文中の RATA=SN を含む) は通常変換へ落ちる.

                # 欧文表に無く和文表にある符号が来たら和文へ切り替える.
                # 自動切替は実質的に和文のためにあり、ホレ 1 個の検出に賭けるのは
                # 脆い (実運用でホレが拾えず切り替わらない事例が発生)。
                # 和文専用符号は 23 種あり高頻度カナ (コ ソ ロ ノ ス ア シ ヒ 等) を
                # 含むため数文字で必ず当たる。しかもこれらは欧文モードでは
                # TABLE_MISS で「?」になるので、切り替えたほうが確実に良くなる.
                if (
                    self.switch_on_japanese_only
                    and active == "european"
                    and token.code not in EUROPEAN_TABLE
                    and token.code in JAPANESE_TABLE
                ):
                    active = "japanese"
                    table = self._table_for(active)
                    composable_at = None

            display = table.get(token.code)
            if display is None:
                self._emit_fallback(
                    out_chars, log, i, tid, token.code, "TABLE_MISS", conf
                )
                composable_at = None
                continue

            if active == "japanese" and display in (DAKUTEN_CHAR, HANDAKUTEN_CHAR):
                compose_map = (
                    DAKUTEN_COMPOSE if display == DAKUTEN_CHAR else HANDAKUTEN_COMPOSE
                )
                if composable_at is not None:
                    composed = compose_map.get(out_chars[composable_at])
                    if composed is not None:
                        out_chars[composable_at] = composed
                        composable_at = None
                        continue
                # 直前カナが無い or 合成対象外 → 単独の濁点/半濁点は意味を成さない
                self._emit_fallback(
                    out_chars, log, i, tid, token.code, "TABLE_MISS", conf
                )
                composable_at = None
                continue

            out_chars.append(display)
            composable_at = (
                len(out_chars) - 1
                if active == "japanese"
                and (display in DAKUTEN_COMPOSE or display in HANDAKUTEN_COMPOSE)
                else None
            )

        return ConvertResult(
            text="".join(out_chars), fallback_log=log, final_mode=active
        )

    def convert_timed(
        self,
        frame_tokens: Sequence[TimedToken],
        gap_threshold_frames: int | None = None,
        word_break_flags: Sequence[bool] | None = None,
    ) -> ConvertResult:
        """``FrameToken`` 列を変換し、語間スペースを挿入.

        スペース挿入の判定は次の優先順:

        1. ``word_break_flags`` が指定されていれば、それを真値として使用
           (推奨: ``detect_word_breaks_from_audio`` などで事前計算).
        2. それ以外は ``gap_threshold_frames`` を用いた token 間フレームギャップ判定.
           ただし CTC モデルの出力タイミングは音声の実際の無音時間と一致しない
           ことが多く、不確かな結果になり得る.

        Args:
            frame_tokens: ``FrameToken`` 互換のリスト.
            gap_threshold_frames: フレームギャップ閾値.
                ``None`` の場合は分布から自動推定.
            word_break_flags: ``frame_tokens`` と同じ長さの bool 列.
                ``True`` の位置の token の **前** にスペースを入れる.

        Returns:
            ``ConvertResult`` — ``text`` にスペースが挿入される.
        """
        ids = [t.token_id for t in frame_tokens]
        confs = [t.confidence for t in frame_tokens]
        base = self.convert(ids, confs)
        if len(frame_tokens) < 2 or not base.text:
            return base

        # 各 token 間のフレームギャップを計算
        gaps: list[int] = []
        for prev, cur in zip(frame_tokens[:-1], frame_tokens[1:], strict=True):
            gap = max(0, cur.frame_start - prev.frame_end)
            gaps.append(gap)

        # word_break_flags 優先 (信頼性が高い)
        if word_break_flags is not None:
            if len(word_break_flags) != len(frame_tokens):
                raise ValueError(
                    f"word_break_flags length {len(word_break_flags)} != "
                    f"frame_tokens length {len(frame_tokens)}"
                )
            # break_after_i: token i と i+1 の間にスペースを入れるか
            break_after = [bool(word_break_flags[i + 1]) for i in range(len(gaps))]
        else:
            if gap_threshold_frames is None:
                gap_threshold_frames = self._estimate_word_gap_threshold(gaps)
            break_after = [g > gap_threshold_frames for g in gaps]

        return self._reinsert_spaces(base, frame_tokens, break_after)

    @staticmethod
    def _estimate_word_gap_threshold(gaps: Sequence[int]) -> int:
        """ギャップ分布から語間閾値を推定.

        多くのギャップは文字間 (3 dot)、少数が語間 (7 dot).
        中央値を文字間相当とみなし、その 1.8 倍を語間閾値とする (安全マージン).
        """
        if not gaps:
            return 30
        sorted_gaps = sorted(gaps)
        median = sorted_gaps[len(sorted_gaps) // 2]
        # 文字間に対し語間は 7/3 ≒ 2.33 倍。1.8 倍を境界とする (3.3 dot 相当).
        # ただし median が 0 や極小の場合は固定値で補正.
        return max(15, int(median * 1.8))

    def _reinsert_spaces(
        self,
        base: ConvertResult,
        frame_tokens: Sequence[TimedToken],
        break_after: Sequence[bool],
    ) -> ConvertResult:
        """``base.text`` の対応する位置にスペースを挿入した新しい ``ConvertResult`` を返す.

        和文の濁点・半濁点による合成は 2 token が 1 文字に縮約されるため、
        合成された区間 (≒ ゼロ長文字遷移) ではスペース挿入をスキップする.
        """
        ids = [t.token_id for t in frame_tokens]
        confs = [t.confidence for t in frame_tokens]
        # 各 token を変換した後の出力文字数を再現.
        idx_to_pos: list[int] = []
        for n in range(1, len(ids) + 1):
            partial = self.convert(ids[:n], confs[:n])
            idx_to_pos.append(len(partial.text))

        out: list[str] = list(base.text)
        # 逆順に挿入 (後ろから入れれば前の位置がずれない)
        for i in range(len(break_after) - 1, -1, -1):
            if not break_after[i]:
                continue
            # 合成濁点で出力が増えない遷移はスキップ
            if i + 1 < len(idx_to_pos) and idx_to_pos[i] == idx_to_pos[i + 1]:
                continue
            insert_at = idx_to_pos[i]
            # 既にスペースなら重複しない
            if 0 < insert_at <= len(out) and (
                insert_at == len(out) or out[insert_at] != " "
            ):
                if out and out[insert_at - 1] != " ":
                    out.insert(insert_at, " ")

        return ConvertResult(text="".join(out), fallback_log=base.fallback_log, final_mode=base.final_mode)

    @staticmethod
    def _emit_fallback(
        out_chars: list[str],
        log: list[FallbackEvent],
        input_index: int,
        token_id: int,
        code: str,
        kind: FallbackKind,
        confidence: float,
    ) -> None:
        log.append(
            FallbackEvent(
                position=len(out_chars),
                input_index=input_index,
                token_id=token_id,
                code=code,
                kind=kind,
                confidence=confidence,
            )
        )
        out_chars.append(FALLBACK_CHAR)


__all__ = [
    "FALLBACK_CHAR",
    "ConvertResult",
    "FallbackEvent",
    "FallbackKind",
    "TimedToken",
    "TokenConverter",
]

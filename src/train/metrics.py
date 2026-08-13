"""評価指標 (Token Error Rate / Character Error Rate) と集計.

要件 §3.3.3:

- 評価指標: TER (トークン誤り率) と CER (文字誤り率)
- SNR 別・WPM 別に集計

Phase A (評価基盤) の追加分:

- ``align_sequences``: 編集操作 (置換/挿入/削除) の復元
- ``TokenErrorAnalysis``: token 別エラー集計 + confusion matrix
- ``DetailedEvalReport``: サンプル別詳細を含む JSON 化可能なレポート

用語は **リファレンス基準** で統一する:

- substitution: ref の token が別 token として出力された
- deletion: ref の token が出力されなかった (脱落)
- insertion: ref に無い token が余分に出力された
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from src.tokens.morse_tokens import EUROPEAN_TABLE, ID_TO_TOKEN, JAPANESE_TABLE


def levenshtein_distance(seq_a: Sequence, seq_b: Sequence) -> int:
    """編集距離 (置換・挿入・削除 各コスト 1)."""
    if seq_a is seq_b:
        return 0
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # 削除
                curr[j - 1] + 1,   # 挿入
                prev[j - 1] + cost,  # 置換
            )
        prev, curr = curr, prev
    return prev[m]


def error_rate(pred: Sequence, ref: Sequence) -> float:
    """編集距離 / リファレンス長. リファレンス空時は 0.0 を返す."""
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein_distance(pred, ref) / len(ref)


@dataclass
class EvalRecord:
    """1 サンプル分の評価記録."""

    ref_tokens: list[int]
    pred_tokens: list[int]
    ref_text: str
    pred_text: str
    snr_db: float | None = None
    wpm: float | None = None
    eff_snr_db: float | None = None  # 受信機 BPF 後の実効 SNR (dB)

    @property
    def token_distance(self) -> int:
        return levenshtein_distance(self.pred_tokens, self.ref_tokens)

    @property
    def char_distance(self) -> int:
        return levenshtein_distance(self.pred_text, self.ref_text)


@dataclass
class AggregateMetrics:
    n_samples: int = 0
    total_ref_tokens: int = 0
    total_token_errors: int = 0
    total_ref_chars: int = 0
    total_char_errors: int = 0

    @property
    def ter(self) -> float:
        if self.total_ref_tokens == 0:
            return 0.0
        return self.total_token_errors / self.total_ref_tokens

    @property
    def cer(self) -> float:
        if self.total_ref_chars == 0:
            return 0.0
        return self.total_char_errors / self.total_ref_chars

    def add(self, record: EvalRecord) -> None:
        self.n_samples += 1
        self.total_ref_tokens += len(record.ref_tokens)
        self.total_token_errors += record.token_distance
        self.total_ref_chars += len(record.ref_text)
        self.total_char_errors += record.char_distance


@dataclass
class EvalReport:
    overall: AggregateMetrics = field(default_factory=AggregateMetrics)
    by_snr: dict[float, AggregateMetrics] = field(default_factory=dict)
    by_wpm: dict[float, AggregateMetrics] = field(default_factory=dict)
    by_eff_snr: dict[float, AggregateMetrics] = field(default_factory=dict)

    def add(
        self,
        record: EvalRecord,
        snr_bin: float | None = None,
        wpm_bin: float | None = None,
        eff_snr_bin: float | None = None,
    ) -> None:
        self.overall.add(record)
        if snr_bin is not None:
            self.by_snr.setdefault(snr_bin, AggregateMetrics()).add(record)
        if wpm_bin is not None:
            self.by_wpm.setdefault(wpm_bin, AggregateMetrics()).add(record)
        if eff_snr_bin is not None:
            self.by_eff_snr.setdefault(eff_snr_bin, AggregateMetrics()).add(record)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Overall  n={self.overall.n_samples:5d}  "
            f"TER={self.overall.ter * 100:6.2f}%  CER={self.overall.cer * 100:6.2f}%",
        ]
        if self.by_snr:
            lines.append("By SNR:")
            for snr in sorted(self.by_snr):
                m = self.by_snr[snr]
                lines.append(
                    f"  SNR={snr:+5.1f}dB  n={m.n_samples:5d}  "
                    f"TER={m.ter * 100:6.2f}%  CER={m.cer * 100:6.2f}%"
                )
        if self.by_wpm:
            lines.append("By WPM:")
            for wpm in sorted(self.by_wpm):
                m = self.by_wpm[wpm]
                lines.append(
                    f"  WPM={wpm:5.1f}    n={m.n_samples:5d}  "
                    f"TER={m.ter * 100:6.2f}%  CER={m.cer * 100:6.2f}%"
                )
        if self.by_eff_snr:
            lines.append("By EffSNR:")
            for eff in sorted(self.by_eff_snr):
                m = self.by_eff_snr[eff]
                lines.append(
                    f"  EffSNR={eff:+5.1f}dB n={m.n_samples:5d}  "
                    f"TER={m.ter * 100:6.2f}%  CER={m.cer * 100:6.2f}%"
                )
        return lines


# ============================================================
# 編集操作の復元 (Phase A)
# ============================================================
EditOpKind = Literal["equal", "substitution", "insertion", "deletion"]

# confusion matrix / JSON でのセンチネル表記
DEL_KEY: str = "<DEL>"   # ref にあるが pred に無い (脱落)
INS_KEY: str = "<INS>"   # pred にあるが ref に無い (挿入)


@dataclass(frozen=True)
class EditOp:
    """1 個の編集操作.

    ``ref_index`` / ``pred_index`` は該当列のインデックス. その操作が
    片方の列を消費しない場合 (insertion なら ref 側、deletion なら pred 側) は
    ``None``.
    """

    kind: EditOpKind
    ref_index: int | None = None
    pred_index: int | None = None
    ref_token: Any = None
    pred_token: Any = None


def align_sequences(pred: Sequence, ref: Sequence) -> list[EditOp]:
    """``pred`` と ``ref`` を編集距離最小のアライメントで対応付ける.

    ``levenshtein_distance(pred, ref)`` と整合する (非 equal な op の数が
    編集距離に一致する). 同コストの経路が複数ある場合は
    **置換/一致 → 削除 → 挿入** の優先順で選ぶ (決定的).

    Returns:
        先頭から末尾へ並んだ ``EditOp`` のリスト.
    """
    n, m = len(pred), len(ref)
    # dp[i][j] = pred[:i] と ref[:j] の編集距離
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i          # pred の余り = 挿入
    for j in range(1, m + 1):
        dp[0][j] = j          # ref の余り = 脱落
    for i in range(1, n + 1):
        pi = pred[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if pi == ref[j - 1] else 1
            row[j] = min(
                prev_row[j - 1] + cost,   # 一致 / 置換
                row[j - 1] + 1,           # 削除 (ref を消費)
                prev_row[j] + 1,          # 挿入 (pred を消費)
            )

    ops: list[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if pred[i - 1] == ref[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                ops.append(EditOp(
                    kind="equal" if cost == 0 else "substitution",
                    ref_index=j - 1, pred_index=i - 1,
                    ref_token=ref[j - 1], pred_token=pred[i - 1],
                ))
                i -= 1
                j -= 1
                continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(EditOp(
                kind="deletion", ref_index=j - 1, pred_index=None,
                ref_token=ref[j - 1], pred_token=None,
            ))
            j -= 1
            continue
        ops.append(EditOp(
            kind="insertion", ref_index=None, pred_index=i - 1,
            ref_token=None, pred_token=pred[i - 1],
        ))
        i -= 1
    ops.reverse()
    return ops


# ============================================================
# token 表示ラベル
# ============================================================
def describe_token(token_id: int) -> dict[str, Any]:
    """token ID から符号文字列と欧文/和文の表示文字を引く.

    未知 ID でも例外を投げず ``code="<UNKNOWN>"`` を返す (評価が落ちないため).
    """
    token = ID_TO_TOKEN.get(token_id)
    if token is None:
        return {
            "token_id": token_id, "code": "<UNKNOWN>",
            "european": None, "japanese": None,
        }
    return {
        "token_id": token_id,
        "code": token.code,
        "european": EUROPEAN_TABLE.get(token.code),
        "japanese": JAPANESE_TABLE.get(token.code),
    }


def token_label(token_id: int) -> str:
    """``符号(欧文/和文)`` 形式の人間可読ラベル. 例: ``・-(A/イ)``."""
    info = describe_token(token_id)
    displays = [d for d in (info["european"], info["japanese"]) if d]
    uniq = list(dict.fromkeys(displays))
    code = str(info["code"])
    return f"{code}({'/'.join(uniq)})" if uniq else code


def _confusion_key(token_id: int | None, sentinel: str) -> str:
    return sentinel if token_id is None else str(token_id)


def _confusion_code(token_id: int | None, sentinel: str) -> str:
    return sentinel if token_id is None else str(describe_token(token_id)["code"])


# ============================================================
# token 別エラー集計 + confusion matrix
# ============================================================
@dataclass
class TokenErrorCounts:
    """1 token 分のエラー内訳.

    ``correct`` / ``substituted`` / ``deleted`` は **ref にこの token が居た** 回数の内訳.
    ``inserted`` / ``mistaken`` は **pred にこの token を誤って出した** 回数.
    """

    correct: int = 0
    substituted: int = 0    # ref がこの token → 別 token として出力された
    deleted: int = 0        # ref がこの token → 出力されなかった
    inserted: int = 0       # ref に無いのに余分に出力された
    mistaken: int = 0       # 他 token の代わりにこの token を出力した

    @property
    def ref_count(self) -> int:
        """ref にこの token が出現した回数."""
        return self.correct + self.substituted + self.deleted

    @property
    def pred_count(self) -> int:
        """pred にこの token が出現した回数."""
        return self.correct + self.mistaken + self.inserted

    @property
    def wrongly_output(self) -> int:
        """誤って出力した回数 (挿入 + 他 token の身代わり)."""
        return self.inserted + self.mistaken

    @property
    def error_count(self) -> int:
        return self.substituted + self.deleted + self.inserted

    @property
    def recall(self) -> float:
        return self.correct / self.ref_count if self.ref_count else 0.0

    @property
    def precision(self) -> float:
        return self.correct / self.pred_count if self.pred_count else 0.0

    def to_dict(self, token_id: int) -> dict[str, Any]:
        info = describe_token(token_id)
        return {
            "token_id": token_id,
            "code": info["code"],
            "european": info["european"],
            "japanese": info["japanese"],
            "ref_count": self.ref_count,
            "pred_count": self.pred_count,
            "correct": self.correct,
            "substituted": self.substituted,
            "deleted": self.deleted,
            "inserted": self.inserted,
            "mistaken": self.mistaken,
            "error_count": self.error_count,
            "recall": self.recall,
            "precision": self.precision,
        }


@dataclass
class TokenErrorAnalysis:
    """token 別 substitution/insertion/deletion と confusion matrix の集計器."""

    counts: dict[int, TokenErrorCounts] = field(
        default_factory=lambda: defaultdict(TokenErrorCounts)
    )
    # (ref_token_id, pred_token_id) → 回数. 挿入は ref=None、脱落は pred=None.
    confusion: dict[tuple[int | None, int | None], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    n_samples: int = 0

    def add_ops(self, ops: Sequence[EditOp]) -> None:
        """復元済みの編集操作列を集計に加える."""
        for op in ops:
            match op.kind:
                case "equal":
                    self.counts[op.ref_token].correct += 1
                    self.confusion[(op.ref_token, op.pred_token)] += 1
                case "substitution":
                    self.counts[op.ref_token].substituted += 1
                    self.counts[op.pred_token].mistaken += 1
                    self.confusion[(op.ref_token, op.pred_token)] += 1
                case "deletion":
                    self.counts[op.ref_token].deleted += 1
                    self.confusion[(op.ref_token, None)] += 1
                case "insertion":
                    self.counts[op.pred_token].inserted += 1
                    self.confusion[(None, op.pred_token)] += 1

    def add_record(self, record: EvalRecord) -> None:
        """1 サンプルの評価記録を集計に加える."""
        self.n_samples += 1
        self.add_ops(align_sequences(record.pred_tokens, record.ref_tokens))

    @property
    def totals(self) -> dict[str, Any]:
        correct = sum(c.correct for c in self.counts.values())
        substitutions = sum(c.substituted for c in self.counts.values())
        deletions = sum(c.deleted for c in self.counts.values())
        insertions = sum(c.inserted for c in self.counts.values())
        ref_tokens = correct + substitutions + deletions
        token_errors = substitutions + deletions + insertions
        return {
            "n_samples": self.n_samples,
            "correct": correct,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "ref_tokens": ref_tokens,
            "pred_tokens": correct + substitutions + insertions,
            "token_errors": token_errors,
            "ter": token_errors / ref_tokens if ref_tokens else 0.0,
        }

    def token_errors_to_list(self) -> list[dict[str, Any]]:
        """token ID 昇順の内訳リスト (JSON 化用)."""
        return [self.counts[tid].to_dict(tid) for tid in sorted(self.counts)]

    def top_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """誤り数の多い token から順に返す."""
        ranked = sorted(
            self.counts,
            key=lambda tid: (-self.counts[tid].error_count, tid),
        )
        return [
            self.counts[tid].to_dict(tid)
            for tid in ranked[:limit]
            if self.counts[tid].error_count > 0
        ]

    def confusion_to_dict(self) -> dict[str, Any]:
        """confusion matrix を JSON 化可能な形へ (回数の多い順)."""
        entries: list[dict[str, Any]] = []
        for (ref_id, pred_id), count in self.confusion.items():
            if ref_id is None:
                kind: EditOpKind = "insertion"
            elif pred_id is None:
                kind = "deletion"
            elif ref_id == pred_id:
                kind = "equal"
            else:
                kind = "substitution"
            entries.append({
                "ref": _confusion_key(ref_id, INS_KEY),
                "pred": _confusion_key(pred_id, DEL_KEY),
                "ref_code": _confusion_code(ref_id, INS_KEY),
                "pred_code": _confusion_code(pred_id, DEL_KEY),
                "ref_label": INS_KEY if ref_id is None else token_label(ref_id),
                "pred_label": DEL_KEY if pred_id is None else token_label(pred_id),
                "kind": kind,
                "count": count,
            })
        entries.sort(key=lambda e: (-e["count"], e["ref"], e["pred"]))
        return {"n_samples": self.n_samples, "entries": entries}


# ============================================================
# 詳細評価レポート
# ============================================================
@dataclass
class SampleEval:
    """1 サンプルの詳細評価 (アライメント済み)."""

    name: str
    mode: str | None
    record: EvalRecord
    ops: list[EditOp]

    @property
    def ter(self) -> float:
        return error_rate(self.record.pred_tokens, self.record.ref_tokens)

    @property
    def cer(self) -> float:
        return error_rate(self.record.pred_text, self.record.ref_text)

    def _count(self, kind: EditOpKind) -> int:
        return sum(1 for op in self.ops if op.kind == kind)

    @property
    def substitutions(self) -> int:
        return self._count("substitution")

    @property
    def insertions(self) -> int:
        return self._count("insertion")

    @property
    def deletions(self) -> int:
        return self._count("deletion")

    def to_dict(self) -> dict[str, Any]:
        rec = self.record
        return {
            "name": self.name,
            "mode": self.mode,
            "ter": self.ter,
            "cer": self.cer,
            "n_ref_tokens": len(rec.ref_tokens),
            "n_pred_tokens": len(rec.pred_tokens),
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "ref_tokens": list(rec.ref_tokens),
            "pred_tokens": list(rec.pred_tokens),
            "ref_codes": [describe_token(t)["code"] for t in rec.ref_tokens],
            "pred_codes": [describe_token(t)["code"] for t in rec.pred_tokens],
            "ref_text": rec.ref_text,
            "pred_text": rec.pred_text,
        }


@dataclass
class DetailedEvalReport:
    """TER/CER + token 別エラー + confusion matrix + サンプル別詳細.

    既存の ``EvalReport`` を内部に保持するので、従来の TER/CER 出力は
    ``report`` 経由でそのまま利用できる.
    """

    report: EvalReport = field(default_factory=EvalReport)
    analysis: TokenErrorAnalysis = field(default_factory=TokenErrorAnalysis)
    samples: list[SampleEval] = field(default_factory=list)

    @property
    def overall(self) -> AggregateMetrics:
        """``EvalReport`` と同じ呼び出し方で全体 TER/CER を参照するための委譲."""
        return self.report.overall

    def by_mode(self) -> dict[str, AggregateMetrics]:
        """モード別の集計 (サンプルの mode で分類)."""
        out: dict[str, AggregateMetrics] = {}
        for s in self.samples:
            key = s.mode or "unknown"
            out.setdefault(key, AggregateMetrics()).add(s.record)
        return out

    def add(
        self,
        record: EvalRecord,
        name: str = "",
        mode: str | None = None,
        snr_bin: float | None = None,
        wpm_bin: float | None = None,
        eff_snr_bin: float | None = None,
    ) -> None:
        self.report.add(record, snr_bin=snr_bin, wpm_bin=wpm_bin, eff_snr_bin=eff_snr_bin)
        self.analysis.add_record(record)
        self.samples.append(SampleEval(
            name=name, mode=mode, record=record,
            ops=align_sequences(record.pred_tokens, record.ref_tokens),
        ))

    def to_dict(self) -> dict[str, Any]:
        overall = self.report.overall
        return {
            "overall": {
                "n_samples": overall.n_samples,
                "ter": overall.ter,
                "cer": overall.cer,
                "ref_tokens": overall.total_ref_tokens,
                "token_errors": overall.total_token_errors,
                "ref_chars": overall.total_ref_chars,
                "char_errors": overall.total_char_errors,
            },
            "totals": self.analysis.totals,
            "token_errors": self.analysis.token_errors_to_list(),
            "by_eff_snr": {
                f"{b}": {
                    "n_samples": self.report.by_eff_snr[b].n_samples,
                    "ter": self.report.by_eff_snr[b].ter,
                    "cer": self.report.by_eff_snr[b].cer,
                }
                for b in sorted(self.report.by_eff_snr)
            },
            "by_mode": {
                k: {"n_samples": m.n_samples, "ter": m.ter, "cer": m.cer}
                for k, m in self.by_mode().items()
            },
            "samples": [s.to_dict() for s in self.samples],
        }

    def summary_lines(self, top_n: int = 10) -> list[str]:
        totals = self.analysis.totals
        lines = list(self.report.summary_lines())
        lines.append(
            f"Edits    S={totals['substitutions']:5d}  D={totals['deletions']:5d}  "
            f"I={totals['insertions']:5d}  (ref tokens={totals['ref_tokens']})"
        )
        top = self.analysis.top_errors(limit=top_n)
        if top:
            lines.append(f"Top {len(top)} error tokens:")
            for t in top:
                lines.append(
                    f"  {token_label(t['token_id']):<16} ref={t['ref_count']:4d}  "
                    f"S={t['substituted']:3d} D={t['deleted']:3d} I={t['inserted']:3d}  "
                    f"recall={t['recall'] * 100:5.1f}%"
                )
        return lines


def bin_snr(snr_db: float, step: float = 5.0) -> float:
    """SNR を step dB 単位にビニング."""
    return round(snr_db / step) * step


def bin_wpm(wpm: float, step: float = 5.0) -> float:
    """WPM を step 単位にビニング."""
    return round(wpm / step) * step


__all__ = [
    "AggregateMetrics",
    "DEL_KEY",
    "DetailedEvalReport",
    "EditOp",
    "EditOpKind",
    "EvalRecord",
    "EvalReport",
    "INS_KEY",
    "SampleEval",
    "TokenErrorAnalysis",
    "TokenErrorCounts",
    "align_sequences",
    "bin_snr",
    "bin_wpm",
    "describe_token",
    "error_rate",
    "levenshtein_distance",
    "token_label",
]

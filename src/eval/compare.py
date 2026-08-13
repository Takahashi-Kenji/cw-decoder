"""eval report 同士の比較 (torch 非依存の純関数).

``scripts/eval_model.py`` が出力する report dict を 2 つ受け、改善前後の
TER/CER delta と token recall 改善を算出する。TER は減少が改善なので
``*_delta`` は負が改善を意味する。
"""
from __future__ import annotations

from typing import Any

_SECTIONS = ("synth_val", "keyed_val")


def _delta_metrics(base: dict | None, cur: dict | None) -> dict[str, Any]:
    """overall/mode/bin の TER・CER delta を作る. 片方欠損は 'n/a'."""
    if not base or not cur:
        return {"ter_delta": "n/a", "cer_delta": "n/a"}
    out: dict[str, Any] = {}
    for key in ("ter", "cer"):
        b = base.get(key)
        c = cur.get(key)
        if b is None or c is None:
            out[f"{key}_delta"] = "n/a"
        else:
            out[f"{key}_baseline"] = b
            out[f"{key}_current"] = c
            out[f"{key}_delta"] = c - b
    return out


def _delta_binned(base: dict, cur: dict) -> dict[str, Any]:
    """ビン (eff_snr / mode) ごとの delta. 欠損ビンは 'n/a'."""
    out: dict[str, Any] = {}
    for key in sorted(set(base) | set(cur)):
        out[key] = _delta_metrics(base.get(key), cur.get(key))
    return out


def _token_recall_delta(base: list[dict], cur: list[dict]) -> list[dict]:
    """token recall の改善を大きい順に返す."""
    base_by_id = {t["token_id"]: t for t in base}
    improvements: list[dict] = []
    for t in cur:
        tid = t["token_id"]
        b = base_by_id.get(tid)
        if b is None or "recall" not in b or "recall" not in t:
            continue
        improvements.append({
            "token_id": tid,
            "code": t.get("code"),
            "recall_baseline": b["recall"],
            "recall_current": t["recall"],
            "recall_delta": t["recall"] - b["recall"],
        })
    improvements.sort(key=lambda x: -x["recall_delta"])
    return improvements


def compare_reports(baseline: dict, current: dict) -> dict:
    """2 つの report dict を section ごとに比較する."""
    out: dict[str, Any] = {}
    for section in _SECTIONS:
        b = baseline.get(section)
        c = current.get(section)
        if b is None or c is None:
            continue
        out[section] = {
            "overall": _delta_metrics(b.get("overall"), c.get("overall")),
            "by_eff_snr": _delta_binned(b.get("by_eff_snr", {}), c.get("by_eff_snr", {})),
            "by_mode": _delta_binned(b.get("by_mode", {}), c.get("by_mode", {})),
            "token_recall_improvements": _token_recall_delta(
                b.get("token_errors", []), c.get("token_errors", [])
            ),
        }
    return out


__all__ = ["compare_reports"]

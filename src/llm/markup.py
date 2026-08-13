"""LLM 出力のマーカー ⟦…⟧ を赤文字 HTML に変換する純粋関数 (Qt 非依存).

推測箇所を ⟦…⟧ で囲った LLM 出力を、html.escape した上で
赤 span に変換する. プロサイン <KN> 等の角括弧はタグとして
解釈されないよう必ずエスケープする.
"""
from __future__ import annotations

import html

OPEN_MARK = "⟦"   # ⟦
CLOSE_MARK = "⟧"  # ⟧
_RED = "#cc0000"


def to_html(text: str, highlight: bool = True) -> str:
    """⟦…⟧ マーカー入りテキストを HTML に変換する.

    ``highlight=True`` (既定) ならマーカー内を赤 span にする。
    ``False`` なら色を付けず本文だけを出す (マーカー記号自体は常に出さない)。
    赤が多いと読みにくいという運用上の指摘による切り替え.

    閉じ忘れ (アンバランス) の場合は開きマーカー以降をマーカー内として扱う.
    """
    parts: list[str] = []
    in_mark = False
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        escaped = html.escape("".join(buf))
        if in_mark and highlight:
            parts.append(f'<span style="color:{_RED};">{escaped}</span>')
        else:
            parts.append(escaped)
        buf.clear()

    for ch in text:
        if ch == OPEN_MARK:
            flush()
            in_mark = True
        elif ch == CLOSE_MARK:
            flush()
            in_mark = False
        else:
            buf.append(ch)
    flush()
    return "".join(parts)


__all__ = ["OPEN_MARK", "CLOSE_MARK", "to_html"]

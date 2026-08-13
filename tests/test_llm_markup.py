"""マーカー → 赤 HTML 変換のテスト."""
from src.llm.markup import OPEN_MARK, CLOSE_MARK, to_html


def test_plain_text_is_escaped_black():
    html = to_html("本日は晴天")
    assert "本日は晴天" in html
    assert "color:#cc0000" not in html


def test_marked_span_becomes_red():
    html = to_html(f"こちら {OPEN_MARK}JH0ILL{CLOSE_MARK} です")
    assert '<span style="color:#cc0000;">JH0ILL</span>' in html
    assert OPEN_MARK not in html and CLOSE_MARK not in html


def test_prosign_angle_brackets_are_escaped_not_tags():
    html = to_html("<KN>")
    assert "&lt;KN&gt;" in html
    assert "<KN>" not in html


def test_marked_content_is_also_escaped():
    html = to_html(f"{OPEN_MARK}<KN>{CLOSE_MARK}")
    assert '<span style="color:#cc0000;">&lt;KN&gt;</span>' in html


def test_unbalanced_open_marker_is_stripped_safely():
    # 閉じ忘れは残りを赤として扱い、生マーカーを残さない
    html = to_html(f"abc {OPEN_MARK}def")
    assert OPEN_MARK not in html
    assert "def" in html


def test_highlight_off_hides_colour_but_keeps_text():
    """赤が多いと読みにくいという指摘による切替。マーカー記号自体は出さない."""
    from src.llm.markup import to_html
    html = to_html("こんにちは⟦。天気は晴れです⟧", highlight=False)
    assert "#cc0000" not in html
    assert "⟦" not in html and "⟧" not in html
    assert "天気は晴れです" in html


def test_highlight_on_marks_guesses_red():
    from src.llm.markup import to_html
    html = to_html("こんにちは⟦。天気は晴れです⟧")
    assert "#cc0000" in html

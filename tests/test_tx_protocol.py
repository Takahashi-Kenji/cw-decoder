"""LAN 越し打鍵のプロトコルのテスト."""
from __future__ import annotations

import pytest

from src.tx import protocol
from src.tx.protocol import LineReader, ProtocolError, decode_message, encode_message


def test_往復する() -> None:
    original = protocol.check("CQ DE JA1ABC K", 20.0)
    assert decode_message(encode_message(original)) == original


def test_日本語をそのまま載せる() -> None:
    """``ensure_ascii=False``。ログを人が読めることに価値があるため。"""
    raw = encode_message(protocol.send("{HORE}コンニチハ{RATA}", 20.0))
    assert "コンニチハ".encode() in raw
    assert raw.endswith(b"\n")


def test_1メッセージが1行である() -> None:
    raw = encode_message(protocol.error("busy", "打鍵中です\n2 行目"))
    assert raw.count(b"\n") == 1


def test_JSONでない行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message("{壊れている".encode())


def test_辞書でない行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b'["check"]')


def test_typeの無い行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b'{"text": "CQ"}')


def test_UTF8でない行を撥ねる() -> None:
    with pytest.raises(ProtocolError):
        decode_message(b"\xff\xfe")


def test_doneの理由は省略するとNoneになる() -> None:
    """正常完了では reason 欄は None (理由が無い)."""
    message = protocol.done(
        elements_sent=5,
        aborted=False,
        watchdog_tripped=False,
        seconds=1.0,
        max_error_ms=0.1,
        mean_error_ms=0.05,
    )
    assert message["reason"] is None


def test_doneの理由を指定すると載る() -> None:
    message = protocol.done(
        elements_sent=3,
        aborted=True,
        watchdog_tripped=False,
        seconds=0.5,
        max_error_ms=0.2,
        mean_error_ms=0.1,
        reason="lifeline",
    )
    assert message["reason"] == "lifeline"


def test_errorが追加の欄を載せる() -> None:
    message = protocol.error(
        protocol.CODE_UNSENDABLE, "だめ", unsendable=[{"index": 3, "char": "髙"}]
    )
    assert message["code"] == "unsendable"
    assert message["unsendable"] == [{"index": 3, "char": "髙"}]


def test_LineReaderが行に分ける() -> None:
    reader = LineReader()
    assert reader.feed(b'{"type":"ping"}\n{"type":"stop"}\n') == [
        b'{"type":"ping"}',
        b'{"type":"stop"}',
    ]


def test_LineReaderが途中で切れた行を溜める() -> None:
    """**TCP は境界を保存しない。** 1 メッセージが 2 回の recv に割れる。"""
    reader = LineReader()
    assert reader.feed(b'{"type":"pi') == []
    assert reader.feed(b'ng"}\n') == [b'{"type":"ping"}']


def test_LineReaderが空行を捨てる() -> None:
    reader = LineReader()
    assert reader.feed(b"\n\n") == []


def test_LineReaderが長すぎる行を撥ねる() -> None:
    """壊れた相手や悪意ある相手に無限に溜めさせない."""
    reader = LineReader()
    with pytest.raises(ProtocolError):
        reader.feed(b"x" * (protocol.MAX_LINE_BYTES + 1))


def test_LineReaderが改行込みの長すぎる行を撥ねる() -> None:
    """1 回の feed で改行付きで長すぎる行が来た場合。"""
    reader = LineReader()
    with pytest.raises(ProtocolError):
        reader.feed(b"x" * (protocol.MAX_LINE_BYTES + 1) + b"\n")


def test_LineReaderは上限を超えても次の行を読める() -> None:
    """撥ねたあとも LineReader が壊れず、次の正常な行は読める。"""
    reader = LineReader()
    with pytest.raises(ProtocolError):
        reader.feed(b"x" * (protocol.MAX_LINE_BYTES + 1) + b"\n")
    # バッファはクリアされているはず、次の行は読める
    assert reader.feed(b'{"type":"ping"}\n') == [b'{"type":"ping"}']


def test_LineReaderはちょうど上限の行を通す() -> None:
    """全体で MAX_LINE_BYTES ちょうどになる行は通ること。"""
    reader = LineReader()
    line = b"x" * protocol.MAX_LINE_BYTES
    result = reader.feed(line + b"\n")
    assert result == [line]


@pytest.mark.parametrize(
    "line",
    [
        b'{"type":"send","text":"E","wpm":NaN}',
        b'{"type":"send","text":"E","wpm":Infinity}',
        b'{"type":"send","text":"E","wpm":-Infinity}',
    ],
)
def test_NaNやInfinityを含む行を撥ねる(line: bytes) -> None:
    """JSON の非標準トークン NaN/Infinity/-Infinity を含む行は ProtocolError にする."""
    with pytest.raises(ProtocolError):
        decode_message(line)


@pytest.mark.parametrize(
    "line",
    [
        b'{"type":"send","text":"E","wpm":1e400}',
        b'{"type":"send","text":"E","wpm":-1e400}',
    ],
)
def test_オーバーフローする数値を撥ねる(line: bytes) -> None:
    """JSON 数値リテラルがオーバーフローして inf になる場合を ProtocolError にする."""
    with pytest.raises(ProtocolError):
        decode_message(line)


@pytest.mark.parametrize(
    "line",
    [
        b'{"type":"send","wpm":20}',
        b'{"type":"send","wpm":20.0}',
        b'{"type":"send","wpm":-5}',
        b'{"type":"send","wpm":0}',
    ],
)
def test_通常の数値は通す(line: bytes) -> None:
    """正常な有限数値は decode_message を通す。parse_float フックの正常系確認。"""
    result = decode_message(line)
    assert result["type"] == "send"
    # wpm 値は数値として正しく解析されている
    assert isinstance(result["wpm"], (int, float))

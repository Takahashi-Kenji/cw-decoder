"""対話に入る条件のテスト.

**ここを間違えると、自動化やテストが入力待ちで固まる。** 引数が与えられて
いれば従来どおり黙って起動し、端末が無ければ (ログへのリダイレクト、
サービス起動) 対話に入らないこと。

逆に、exe をダブルクリックした人 = 端末があって引数が無い、のときだけ入る。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    """``scripts/`` の中身をモジュールとして読む (パッケージではないため)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def key_server():
    return _load("cw_key_server")


@pytest.fixture(scope="module")
def audio_send():
    return _load("audio_send")


class TestKeyServer:
    @pytest.mark.parametrize(
        ("argv", "isatty", "expected"),
        [
            ([], True, True),                       # ダブルクリック相当
            ([], False, False),                     # 端末が無い
            (["--port", "COM3"], True, False),      # 指定済み
            (["--dry-run"], True, False),           # シリアルに触らない
            (["--port", "COM3", "--check-lines"], True, False),
        ],
    )
    def test_gate(self, key_server, argv: list[str], isatty: bool, expected: bool) -> None:
        args = key_server.build_parser().parse_args(argv)
        assert key_server.wants_interactive(args, isatty) is expected

    def test_check_lines_without_port_still_asks(self, key_server) -> None:
        """結線確認も実ポートが要るので、番号を聞いてよい."""
        args = key_server.build_parser().parse_args(["--check-lines"])
        assert key_server.wants_interactive(args, True) is True


class TestAudioSend:
    @pytest.mark.parametrize(
        ("argv", "isatty", "expected"),
        [
            ([], True, True),
            ([], False, False),
            (["--device", "13"], True, False),
            (["--list"], True, False),             # 一覧を出して終わるだけ
        ],
    )
    def test_gate(self, audio_send, argv: list[str], isatty: bool, expected: bool) -> None:
        args = audio_send.build_parser().parse_args(argv)
        assert audio_send.wants_interactive(args, isatty) is expected


class TestNamespaceShape:
    """対話は ``args`` を書き換えるので、書き換え先の名前が実在すること."""

    def test_key_server_fields(self, key_server) -> None:
        args: argparse.Namespace = key_server.build_parser().parse_args([])
        for name in ("port", "key_line", "ptt_line"):
            assert hasattr(args, name)

    def test_audio_send_fields(self, audio_send) -> None:
        args: argparse.Namespace = audio_send.build_parser().parse_args([])
        assert hasattr(args, "device")

"""LAN 越し打鍵のプロトコル (1 行 1 メッセージの JSON).

なぜ JSON の行なのか
--------------------
LAN に流すのは**確定したテキスト**である (設計書 §3)。テキストを流すと決めた以上、
ログがそのまま読めて ``netcat`` で叩けることに価値がある。

向きは音声 (``net_audio.py``) と同じ。**無線機 PC が待ち受け、GPU PC が繋ぎに行く。**
運用の癖が音声と揃う。ポートは音声の 45678 と分けて 45679。

暗号化しない。家庭内 LAN を前提とし、外部へ公開するポートには割り当てないこと。

メッセージ
----------
==================  ========  ==========================================
向き                type      意味
==================  ========  ==========================================
CLI → アプリ        hello     版・符号表の指紋・結線の要約
アプリ → CLI        check     **打鍵しない検査**
CLI → アプリ        checked   文字数・要素数・所要秒数
アプリ → CLI        send      打鍵する
CLI → アプリ        done      打鍵の実測
アプリ → CLI        stop      直ちに中止
アプリ → CLI        ping      心綱の拍
CLI → アプリ        error     撥ねた理由
==================  ========  ==========================================
"""
from __future__ import annotations

import json
import math
from typing import Any

PROTOCOL_VERSION = 1

# 音声 (45678) と分ける。両方を同時に動かすため。
DEFAULT_KEY_PORT = 45679

# 1 行の上限。壊れた相手や悪意ある相手に無限に溜めさせないための歯止め。
# 和文の長文でも数 KB に収まる。
MAX_LINE_BYTES = 64 * 1024

# error の code
CODE_UNSENDABLE = "unsendable"      # 符号表に無い文字がある
CODE_SERIAL = "serial"              # ポートを開けない・書けない
CODE_BUSY = "busy"                  # 打鍵中
CODE_BAD_REQUEST = "bad_request"    # 解釈できない要求
CODE_LIFELINE = "lifeline"          # 心綱が切れた
CODE_TOO_LONG = "too_long"          # 長さの上限を超えた
# 打鍵側の想定外の失敗 (バグ)。**シリアルの失敗 (CODE_SERIAL) と混ぜない。**
# 混ぜると「COM の問題」に見えてしまい、運用者が結線を疑って時間を溶かす。
CODE_INTERNAL = "internal"


class ProtocolError(ValueError):
    """受け取った行が解釈できない."""


def encode_message(message: dict[str, Any]) -> bytes:
    """1 メッセージを 1 行のバイト列にする.

    ``ensure_ascii=False`` はログを人が読めるようにするため。
    """
    return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")


def decode_message(line: bytes | str) -> dict[str, Any]:
    """1 行を解釈する.

    Raises:
        ProtocolError: UTF-8 でない、JSON でない、辞書でない、``type`` が無い、
            NaN/Infinity を含む、数値がオーバーフロー (1e400 等) している.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"UTF-8 として読めません: {exc}") from exc

    def _reject_non_finite(value: Any) -> Any:
        """NaN と Infinity を含む行を撥ねる (parse_constant: NaN/Infinity/-Infinity トークン)."""
        raise ProtocolError(f"有限でない数値が含まれています: {value!r}")

    def _parse_float_finite(text: str) -> float:
        """数値リテラルをパースし、オーバーフローしたら撥ねる (parse_float: 1e400 等)."""
        value = float(text)
        if not math.isfinite(value):
            raise ProtocolError(f"数値がオーバーフロー (1e±400 等) しています: {text!r} -> {value!r}")
        return value

    try:
        obj = json.loads(
            line,
            parse_constant=_reject_non_finite,  # type: ignore[arg-type]
            parse_float=_parse_float_finite,
        )
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"JSON として読めません: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"辞書ではありません: {type(obj).__name__}")
    if not isinstance(obj.get("type"), str):
        raise ProtocolError("type がありません")
    return obj


# ---- 組み立て ----
def hello(fingerprint: str, wiring: dict[str, str], dry_run: bool) -> dict[str, Any]:
    return {
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "tokens_fingerprint": fingerprint,
        "wiring": wiring,
        "dry_run": dry_run,
    }


def check(text: str, wpm: float) -> dict[str, Any]:
    return {"type": "check", "text": text, "wpm": float(wpm)}


def send(text: str, wpm: float) -> dict[str, Any]:
    return {"type": "send", "text": text, "wpm": float(wpm)}


def stop() -> dict[str, Any]:
    return {"type": "stop"}


def ping() -> dict[str, Any]:
    return {"type": "ping"}


def checked(chars: int, elements: int, seconds: float) -> dict[str, Any]:
    return {
        "type": "checked",
        "chars": int(chars),
        "elements": int(elements),
        "seconds": round(float(seconds), 3),
    }


def done(
    elements_sent: int,
    aborted: bool,
    watchdog_tripped: bool,
    seconds: float,
    max_error_ms: float,
    mean_error_ms: float,
    reason: str | None = None,
) -> dict[str, Any]:
    """打鍵の実測.

    Args:
        reason: 中止した理由 (``"lifeline"`` / ``"stop"``)。**``aborted`` が真の
            ときだけ意味を持つ。** 正常完了なら常に ``None``。

    ``send`` 1 回につきこの ``done`` を必ず 1 通だけ返す。中止の理由を別の
    メッセージ (``error`` 等) で追送しないのは、クライアントが ``send`` の
    応答を 1 通で待ち切る設計だから。追送すると次の要求への応答が 1 つずつ
    ずれる (Task 6 のレビューで実測して判明した)。
    """
    return {
        "type": "done",
        "elements_sent": int(elements_sent),
        "aborted": bool(aborted),
        "watchdog_tripped": bool(watchdog_tripped),
        "seconds": round(float(seconds), 3),
        "max_error_ms": round(float(max_error_ms), 3),
        "mean_error_ms": round(float(mean_error_ms), 3),
        "reason": reason,
    }


def error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message, **extra}


class LineReader:
    """バイト列から 1 行ずつ取り出す. 端数は次まで溜める.

    **TCP は境界を保存しない。** 1 メッセージが 2 回の ``recv`` に割れることも、
    2 メッセージが 1 回で届くこともある。
    """

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, data: bytes) -> list[bytes]:
        """受け取ったバイト列を足し、揃った行を返す.

        Raises:
            ProtocolError: 1 行が ``MAX_LINE_BYTES`` を超えた.
        """
        self._buffer += data
        if b"\n" not in self._buffer:
            if len(self._buffer) > MAX_LINE_BYTES:
                self._buffer = b""
                raise ProtocolError(f"1 行が長すぎます (> {MAX_LINE_BYTES} バイト)")
            return []
        *lines, self._buffer = self._buffer.split(b"\n")
        # 各行が上限を超えていないか確認
        for line in lines:
            if len(line) > MAX_LINE_BYTES:
                self._buffer = b""
                raise ProtocolError(f"1 行が長すぎます (> {MAX_LINE_BYTES} バイト)")
        return [line for line in lines if line.strip()]


__all__ = [
    "CODE_BAD_REQUEST",
    "CODE_BUSY",
    "CODE_INTERNAL",
    "CODE_LIFELINE",
    "CODE_SERIAL",
    "CODE_TOO_LONG",
    "CODE_UNSENDABLE",
    "DEFAULT_KEY_PORT",
    "MAX_LINE_BYTES",
    "PROTOCOL_VERSION",
    "LineReader",
    "ProtocolError",
    "check",
    "checked",
    "decode_message",
    "done",
    "encode_message",
    "error",
    "hello",
    "ping",
    "send",
    "stop",
]

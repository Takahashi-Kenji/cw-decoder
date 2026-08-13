"""シリアルの DTR / RTS で電鍵と PTT を叩く.

開いた瞬間が危ない
------------------
**多くの USB シリアル変換器は、ポートを開いた瞬間に DTR と RTS を H にする。**
何もしなければ**ポートを開いただけで電鍵が押され、キャリアが出続ける。**

* ``dsrdtr=False`` / ``rtscts=False`` で開き、**開いた直後に両線を落とす**
* **アプリ起動時に開かない。** 送信の直前に開き、終わったら閉じる
* 閉じるときも必ず両線を落としてから閉じる

「落とす」は極性を通す
----------------------
**「落とす」は電気的な L ではなく、電鍵と PTT が「切」になる状態を指す。**
反転結線 (``key_invert=True``) では「キー ON = DTR L」なので、``dtr = False`` を
直に書くと**安全のための後始末が逆にキーを押す。** 落とす処理は必ず
:meth:`SerialKeySink.key` / :meth:`SerialKeySink.ptt` を通し、極性の扱いを
一箇所 (:meth:`SerialKeySink._set_line`) にまとめる。**割り当てていない線
(PTT が ``NONE`` のときの余った線) だけは、極性の概念が無いので素の L にする。**

初回の結線確認は**ダミーロードか無線機の電源を切った状態で**行うこと。

どうにもならない失敗
--------------------
**送信中に USB を抜かれると、ソフトからは線を落とせない。** ポートが消えるため
``finally`` も番犬も届かない。この場合は無線機側で止めるしかない。
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


class SerialKeyError(Exception):
    """シリアル関連の失敗 (ポートが無い・開けない・書けない)."""


@dataclass(frozen=True)
class SerialKeyConfig:
    """結線の設定. **無線機の設定で変わるので固定しない。**"""

    port: str
    key_line: str = "DTR"          # "DTR" | "RTS"
    ptt_line: str = "RTS"          # "DTR" | "RTS" | "NONE"
    key_invert: bool = False
    ptt_invert: bool = False

    @property
    def uses_ptt(self) -> bool:
        return self.ptt_line.upper() != "NONE"

    @property
    def assigned_lines(self) -> frozenset[str]:
        """電鍵か PTT に割り当てている線 (``"DTR"`` / ``"RTS"``).

        ここに入らない線は**何にも繋がっていない**ので、極性を考えず素の L に
        して構わない (``_all_lines_off`` が使う)。
        """
        lines = {self.key_line.upper()}
        if self.uses_ptt:
            lines.add(self.ptt_line.upper())
        return frozenset(lines)

    def validate(self) -> None:
        """設定の矛盾を早く見つける."""
        lines = {"DTR", "RTS"}
        if self.key_line.upper() not in lines:
            raise SerialKeyError(f"電鍵の線が不正です: {self.key_line!r}")
        if self.uses_ptt:
            if self.ptt_line.upper() not in lines:
                raise SerialKeyError(f"PTT の線が不正です: {self.ptt_line!r}")
            if self.ptt_line.upper() == self.key_line.upper():
                raise SerialKeyError(
                    "電鍵と PTT に同じ線を割り当てています。"
                    "打鍵のたびに送受信が切り替わってしまいます"
                )


def list_ports() -> list[tuple[str, str]]:
    """使える COM ポートを (デバイス名, 説明) で返す. 取れなければ空."""
    try:
        from serial.tools import list_ports as _list_ports
    except ImportError:
        return []
    try:
        return [(p.device, p.description or "") for p in _list_ports.comports()]
    except Exception:
        return []


class SerialKeySink:
    """``KeySink`` のシリアル実装.

    ``with`` で使うこと。抜けるときに必ず両線を落として閉じる。
    """

    def __init__(self, config: SerialKeyConfig) -> None:
        config.validate()
        self.config = config
        self._serial = None

    # ---- ライフサイクル ----
    def open(self) -> None:
        """ポートを開き、**直後に両線を落とす**."""
        try:
            import serial
        except ImportError as exc:                   # pragma: no cover
            raise SerialKeyError(
                "pyserial が入っていません。pip install pyserial"
            ) from exc
        try:
            # dsrdtr/rtscts を切らないと、開いた瞬間にハンドシェイクで線が上がる
            self._serial = serial.Serial(
                self.config.port, dsrdtr=False, rtscts=False, timeout=0.1
            )
        except Exception as exc:
            raise SerialKeyError(
                f"{self.config.port} を開けませんでした: {exc}"
            ) from exc
        # **開いた直後に必ず「切」にする。** 変換器によっては既に上がっている
        self._all_lines_off()

    def close(self) -> None:
        """電鍵と PTT を「切」にしてから閉じる."""
        if self._serial is None:
            return
        try:
            self._all_lines_off()
        finally:
            with contextlib.suppress(Exception):
                self._serial.close()
            self._serial = None

    def __enter__(self) -> SerialKeySink:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- KeySink ----
    def key(self, on: bool) -> None:
        self._set_line(self.config.key_line, on, self.config.key_invert)

    def ptt(self, on: bool) -> None:
        if self.config.uses_ptt:
            self._set_line(self.config.ptt_line, on, self.config.ptt_invert)

    # ---- 内部 ----
    def _set_line(self, line: str, on: bool, invert: bool) -> None:
        if self._serial is None:
            raise SerialKeyError("ポートが開いていません")
        level = (not on) if invert else on
        try:
            if line.upper() == "DTR":
                self._serial.dtr = level
            else:
                self._serial.rts = level
        except Exception as exc:
            # USB を抜かれた等。**ここで握り潰さない** — 呼び出し側が止める
            raise SerialKeyError(f"{line} を操作できませんでした: {exc}") from exc

    def _all_lines_off(self) -> None:
        """電鍵と PTT を「切」にする. 失敗しても例外にしない (閉じる途中で呼ぶため).

        **極性を必ず通す。** ここで ``dtr = False`` / ``rts = False`` と直に
        書くと、反転結線 (``key_invert=True`` — キー ON = DTR L) では逆に
        キーを押してしまう。実際に ``open()`` の「必ず落とす」と ``close()`` の
        「落としてから閉じる」が、反転結線でキーを押していた
        (``Keyer`` の ``finally`` がせっかく落とした線を、閉じる処理が
        打ち消していた)。

        割り当てていない線 (PTT が ``NONE`` のときの余った線) だけは、極性の
        概念が無いので素の L にする。**開いた瞬間に H になる変換器があるため、
        触らずに放置はしない。**
        """
        if self._serial is None:
            return
        with contextlib.suppress(Exception):
            self.key(False)
        with contextlib.suppress(Exception):
            self.ptt(False)                          # PTT 無し (NONE) なら何もしない
        for line in ("DTR", "RTS"):
            if line in self.config.assigned_lines:
                continue
            with contextlib.suppress(Exception):
                setattr(self._serial, line.lower(), False)


__all__ = [
    "SerialKeyConfig",
    "SerialKeyError",
    "SerialKeySink",
    "list_ports",
]

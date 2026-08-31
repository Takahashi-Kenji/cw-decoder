"""Turbo HAMLOG/Win へ交信データを渡す (同じ PC で動いている Hamlog の入力欄へ).

仕組み
------
Turbo HAMLOG/Win は他のアプリからの操作を公式に受け付ける。ウィンドウメッセージ
``WM_COPYDATA`` を Hamlog の窓 (ウィンドウクラス ``TThwin``) へ送り、
``COPYDATASTRUCT.dwData`` に**項目番号**、``lpData`` に文字列を入れる。
番号にフラグを足すと送った後の動作を指定できる (``THW_FOCUS`` / ``THW_ENTER``)。

**入力欄に入れるだけで、確定 (ログへの書き込み) はしない。** 2026-08-30 の運用者の
判断である。値を間違えたまま書き込むと Hamlog 側で消す手間がかかるので、
運用者が入力ウィンドウを見てから [Save] を押す。**保存のコマンド (18) は送らない。**

**コールサインだけは ``THW_ENTER`` を付けて送る。** 入力ウィンドウが
「コールサインを入力し Enter を押して下さい」と促しているとおり、Hamlog は
コールサインの確定 (重複チェック、ユーザーリストからの取り込み) を経ないと
交信として扱わない。実機で確かめたところ、Enter 無しで流し込むと表示が
「コールサインを入力し Enter を押して下さい」のままで [Save] が働かず、
Enter を付けると「１ｓｔ－ＱＳＯです」に変わった (2026-08-30)。
**Enter は確定 (保存) ではない** — 運用者が押すのと同じ、欄の確定である。
仕様書の例もコールサインだけ ``THW_FOCUS | THW_ENTER`` を付けている。

番号の根拠 (2026-08-30 に確定)
------------------------------
正本は作者配布の ``Th527api.zip`` に入っている ``HamlogMs.txt``
(https://hamlog.sakura.ne.jp/mou/thwapi.html)。そこには

    1 = コールサインに文字列を送る。
    2 = 日付に文字列を送る。
        ・・1～14まで、Turbo HAMLOG/Winの入力欄の並びのとおり・・
    12 = ＱＴＨ / 13 = Remarks１ / 14 = Remarks２
    16 = 入力バッファをクリア

とあり、**残りは「入力欄の並びのとおり」**としか書かれていない。その並びは
Hamlog のヘルプ (入力環境設定・フォントの項) が名指ししている::

    Ctrl キーを押しながらフォントボタンをクリックすると、
    Date, Time, His, My, Freq, Mode, Code, GL, QSL の項目のフォントを…

コールサインは別扱い (Shift+フォント) なので 1 番。続けて Date(2) Time(3)
His(4) My(5) Freq(6) Mode(7) Code(8) GL(9) QSL(10) と並び、残りが
名前(11) QTH(12) Remarks1(13) Remarks2(14) となって、仕様書の 12/13/14 と
きれいに一致する。**His は相手に送る RST、My はもらった RST** である。

**確認済みの環境**: Turbo HAMLOG/Win Ver5.47c。

* **文字コードは Shift_JIS (cp932)。** 仕様書の例はすべて ANSI 版 API
  (``FindWindowA`` / ``SendMessageA`` / Delphi 3 の ``char``) を使っている。
* **文字列の長さは終端の 0 を含める** (仕様書: ``strlen(buffs) + 1``)。
* **権限。** Vista 以降の UIPI で、送る側と受け取る側の権限が違うと
  ``WM_COPYDATA`` が届かない。cw-decoder と Hamlog は同じ権限で動かすこと
  (片方だけ「管理者として実行」しない)。
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Protocol

# 送った後の動作を指定するフラグ (項目番号に足して使う)。
# **``THW_ENTER`` はコールサインにだけ付ける** (上の docstring 参照)。
THW_ENTER = 0x10000
THW_FOCUS = 0x20000

# 入力ウィンドウの [Save] を押すのと同じ。**このモジュールでは絶対に送らない** —
# 登録するかどうかは運用者が画面を見てから決める (2026-08-30 の運用者の判断)。
SAVE_COMMAND = 18

# 入力欄を空にする。
CLEAR_COMMAND = 16

# 入力欄の全項目を取り出す (このモジュールでは使わないが、番号の由来として残す)。
FETCH_ALL_COMMAND = 115

# 項目名 → Hamlog の項目番号 (上の docstring の根拠のとおり)。
FIELD_COMMANDS: dict[str, int] = {
    "call": 1,
    "date": 2,
    "time": 3,
    "rst_sent": 4,        # His = 相手に送る RST
    "rst_received": 5,    # My  = 相手からもらった RST
    "freq_mhz": 6,
    "mode": 7,
    # 8=Code, 9=GL, 10=QSL は使わない
    "name": 11,
    "qth": 12,
    "remarks1": 13,
    "remarks2": 14,
}

# 入力欄へ入れる順。**コールサインを先に入れる** — Hamlog はコールサインを鍵に
# 自分の記録を引くので、後から入れると引いた値が上書きされ得る。
FIELD_ORDER: tuple[str, ...] = (
    "call", "date", "time", "rst_sent", "rst_received", "freq_mhz", "mode",
    "name", "qth", "remarks1", "remarks2",
)

# Hamlog の窓のクラス名。
WINDOW_CLASS = "TThwin"

# 文字列を渡すときの文字コード (上の docstring 参照。実機で確かめること)。
ENCODING = "cp932"

# 交信記録は JST で残す (PC の時計が別の地域でも変わらないよう固定する)。
JST = _dt.timezone(_dt.timedelta(hours=9))


class HamlogNotRunning(RuntimeError):
    """Hamlog の入力ウィンドウが見つからない."""


@dataclass(frozen=True)
class QsoEntry:
    """Hamlog へ渡す 1 交信分.

    値はすべて**文字列**。Hamlog の入力欄にそのまま入る形で持つ
    (日付は ``yy/mm/dd``、時刻は ``HH:MM``、周波数は MHz の文字列)。
    """

    call: str
    date: str
    time: str
    rst_sent: str
    rst_received: str
    freq_mhz: str
    mode: str = "CW"
    name: str = ""
    qth: str = ""
    remarks1: str = ""
    remarks2: str = ""

    def __post_init__(self) -> None:
        # コールサインは大文字で揃える (Hamlog の記録も大文字)
        object.__setattr__(self, "call", self.call.strip().upper())

    @classmethod
    def at(
        cls,
        moment: _dt.datetime,
        *,
        call: str,
        rst_sent: str,
        rst_received: str,
        freq_mhz: str,
        mode: str = "CW",
        name: str = "",
        qth: str = "",
        remarks1: str = "",
        remarks2: str = "",
    ) -> "QsoEntry":
        """その時刻 (JST に直して) の交信として作る."""
        local = moment.astimezone(JST)
        return cls(
            call=call,
            date=local.strftime("%y/%m/%d"),
            time=local.strftime("%H:%M"),
            rst_sent=rst_sent,
            rst_received=rst_received,
            freq_mhz=freq_mhz,
            mode=mode,
            name=name,
            qth=qth,
            remarks1=remarks1,
            remarks2=remarks2,
        )

    def fields(self) -> list[tuple[int, str]]:
        """(項目番号, 値) を入れる順に返す. **空の値は含めない。**

        名前や住所が分からない交信のほうが多い。空で送ると、Hamlog が自分の
        記録から補った値を消してしまう。
        """
        pairs: list[tuple[int, str]] = []
        for name in FIELD_ORDER:
            value = getattr(self, name).strip()
            if value:
                pairs.append((FIELD_COMMANDS[name], value))
        return pairs


class InputWindow(Protocol):
    """Hamlog の入力ウィンドウ (テストでは代役を渡す)."""

    def clear(self) -> None: ...

    def send_field(self, command: int, text: str) -> None: ...


def register(entry: QsoEntry, window: InputWindow | None) -> None:
    """入力欄へ流し込む. **確定はしない** (運用者が Hamlog 側で Enter を押す).

    Raises:
        HamlogNotRunning: ``window`` が ``None`` のとき。**黙って捨てない** —
            Hamlog が起動していないことを運用者に見せる。
    """
    if window is None:
        raise HamlogNotRunning(
            "Hamlog の入力ウィンドウが見つかりません。Hamlog を起動してください "
            "(cw-decoder と同じ権限で動かす必要があります)。"
        )
    # **前の交信の残りを混ぜない。** 消してから入れる
    window.clear()
    for command, value in entry.fields():
        if command == FIELD_COMMANDS["call"]:
            # **コールサインだけ確定させる。** これを経ないと Hamlog は交信として
            # 扱わず [Save] が働かない (上の docstring)。保存ではない。
            # 確定すると Hamlog がユーザーリストから名前・QTH を補うことがある。
            # こちらの値は**空でなければ**この後で上書きするので、分かっている
            # ものはこちらが優先し、分からないものは Hamlog の補いが残る。
            command |= THW_FOCUS | THW_ENTER
        window.send_field(command, value)


def find_input_window() -> "InputWindow | None":
    """動いている Hamlog の入力ウィンドウを探す. 無ければ ``None``.

    Windows でしか動かない。他の環境では常に ``None`` を返す。
    """
    try:
        return _Win32Window.find()
    except OSError:
        return None


WM_COPYDATA = 0x004A


class _Win32Window:
    """``WM_COPYDATA`` を送る実装. **薄く保つ** (テストは代役で行う).

    **型を明示すること。** ``ctypes`` の既定の戻り値は ``c_int`` (32 ビット) で、
    64 ビット Windows のウィンドウハンドルは切り詰められ得る。ポインタとして
    扱わないと、たまたま下位 32 ビットが同じ別の窓へ送りかねない。
    """

    def __init__(self, hwnd: int) -> None:
        self._hwnd = hwnd

    @classmethod
    def find(cls) -> "_Win32Window | None":
        import ctypes

        user32 = ctypes.windll.user32              # type: ignore[attr-defined]
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        hwnd = user32.FindWindowW(WINDOW_CLASS, None)
        return cls(hwnd) if hwnd else None

    def clear(self) -> None:
        self._send(CLEAR_COMMAND, "")

    def send_field(self, command: int, text: str) -> None:
        self._send(command, text)

    def _send(self, command: int, text: str) -> None:
        import ctypes
        from ctypes import wintypes

        class COPYDATASTRUCT(ctypes.Structure):
            _fields_ = [
                ("dwData", ctypes.c_void_p),
                ("cbData", wintypes.DWORD),
                ("lpData", ctypes.c_void_p),
            ]

        # **終端の 0 を含めた長さを渡す** (仕様書: ``strlen(buffs) + 1``)。
        # ``create_string_buffer`` は終端を足すので ``sizeof`` がそのまま使える
        payload = ctypes.create_string_buffer(text.encode(ENCODING, errors="replace"))
        data = COPYDATASTRUCT()
        data.dwData = ctypes.c_void_p(command)
        data.cbData = ctypes.sizeof(payload)
        data.lpData = ctypes.cast(payload, ctypes.c_void_p)
        user32 = ctypes.windll.user32              # type: ignore[attr-defined]
        user32.SendMessageW.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p
        ]
        user32.SendMessageW.restype = ctypes.c_void_p
        user32.SendMessageW(
            ctypes.c_void_p(self._hwnd), WM_COPYDATA, None, ctypes.byref(data)
        )


__all__ = [
    "CLEAR_COMMAND",
    "ENCODING",
    "FETCH_ALL_COMMAND",
    "FIELD_COMMANDS",
    "FIELD_ORDER",
    "HamlogNotRunning",
    "QsoEntry",
    "THW_ENTER",
    "THW_FOCUS",
    "WINDOW_CLASS",
    "find_input_window",
    "register",
]

"""交信ダイアログ (送信と、Hamlog への交信データ受け渡し).

**この PC には COM ポートが無い。** 打鍵は無線機を繋いだ PC の CLI が行い、
ここは確定したテキストを渡すだけ (``src/tx/net_key.py``)。

関門
----
**確認するまで [送信] は押せない。** ``check`` を打鍵側へ投げ、打鍵せずに
符号化だけさせる。それが返って初めて次の 3 つが確定する:

1. 打鍵側が生きていて繋がっている
2. そのテキストが**打鍵側の符号表**で通る
3. 何秒間 電波が出るのか

**編集したら [送信] は無効に戻る。** 確認していない文字列は送れない。
**速度を変えても無効に戻る** — 3 番目「何秒間 電波が出るのか」は速度で変わる
(20 WPM で確認した 6.3 秒は、5 WPM では 4 倍の長さになる)。

閉じ方は 1 つではない
--------------------
``QDialog`` は **Esc で ``reject()`` が呼ばれ、``closeEvent`` は呼ばれない。**
後片付けを ``closeEvent`` にだけ置くと、Esc で画面が消えても打鍵は最後まで
続き、接続と 3 秒タイマも生き残る (次に開いたとき自分自身の古い接続に
``busy`` で撥ねられ、**「別の運用者が使用中です」という嘘の理由**が出る)。
後片付けは :meth:`TxDialog.shutdown` に集め、``closeEvent`` / ``reject`` /
``finished`` のどれからでも通す。**送信中の Esc は閉じずに [中止] を促す** —
運用者にとって唯一のソフト中止手段を画面ごと消さないため。

試聴は作らない (設計書 §6)。運用者は無線機の前に座っており、実際の送信は
無線機自身のサイドトーンで聞こえる。

中止の理由
----------
``send`` が中止で終わったとき、``SendResult.reason`` が
``"stop"`` (運用者が [中止] を押した) か ``"lifeline"`` (拍が途絶えた =
LAN が止まった) かを画面に出し分ける。**運用者にとって「自分が止めた」と
「LAN が切れた」は全く違う意味を持つ。** LAN 切断は繋がっている体で
使い続けると危ういので、その場で接続を落として待機中の自動繋ぎ直しに任せる。

busy の見分け
--------------
``connect()`` は打鍵側が別の運用者と繋がっているとき ``NetKeyRejected``
(``code == "busy"``) を投げる。**``NetKeyRejected`` は ``NetKeyError`` の
派生なので、先に捕まえること。** 汎用の「繋がりません」文言に埋もれさせない。
"""
from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.infer.net_audio import parse_endpoint
from src.infer.settings import AppSettings
from src.tokens.morse_tokens import DisplayMode
from src.tx.encoder import HORE, find_unsendable, needs_japanese_wrap, wrap_japanese
from src.tx.net_key import (
    Hello,
    NetKeyClient,
    NetKeyError,
    NetKeyRejected,
    SendResult,
)
from src.tx.profile import OperatorProfile, load_profile
from src.tx.protocol import DEFAULT_KEY_PORT
from src.tx.hamlog import (
    HamlogNotRunning,
    QsoEntry,
    find_input_window,
    register as hamlog_register,
)
from src.tx.qso_fields import extract_fields, normalise_rst
from src.tx.reading import to_sendable_kana
from src.tx.templates import (
    DEFAULT_TEMPLATES_PATH,
    fill,
    load_templates,
    profile_values,
    templates_for_mode,
)

# 並べる送信欄の数 (2026-08-30 の運用者の指示で 4)。**交信中に返信を組み立てる
# 時間が無い**ので、先に何通か書いて全部 [確認] まで通しておき、相手の信号に
# 合う 1 通だけを [送信] で出す。**打鍵器は 1 台なので同時に送れるのは 1 通。**
PANEL_COUNT = 4

# 待機中に繋ぎ直す間隔 (秒)。打鍵側を後から起こしても繋がるようにするため。
RETRY_INTERVAL_S = 3.0

# 送信ワーカーが終わるのを待つ上限 (ミリ秒)。ダイアログを閉じるときに使う。
# 打鍵側の応答は ping 間隔 (0.25 秒) より粗くはならない設計なので十分な余裕を見る。
_WORKER_WAIT_MS = 5000

# `refresh_kana` が出す「送れない文字」警告の先頭一致。**この警告自身が出した
# ものかどうかを ``status_label`` の現在の文言で判定するために使う** (接続結果
# や確認結果など他の文言を無条件クリアで巻き込まないため)。
# **打鍵側 (``src/tx/key_server.py`` の ``prepare``) が出す拒否文言は
# 「送信できない文字が含まれています」で、こちらは「あります」。** 語尾が違う
# ので文字列としては別物であり、どちらか一方の文言を変えても他方は追従しない。
# ここで判定に使っているのは画面側 (下の ``setText`` ) の文言だけである。
_UNSENDABLE_PREFIX = "送信できない文字があります"

# 「和文が無いのに囲んでいる」警告の先頭一致。**`apply_template` は中身を見て
# 自動で ``wrap_check`` を設定するので安全だが、運用者が ``japanese_edit`` に
# 直接打つ経路 (``refresh_kana``) は ``wrap_check.isChecked()`` をそのまま
# 使うだけだった。** チェックを入れたまま ``「FT991」`` のように和文の無い本文を
# 直接打つと `{HORE}「FT991」{RATA}` が**警告なしで**できる。中身は欧文として
# 符号化できてしまうので「送信できない文字」にはならず、**送れるのに化ける**
# という一番気づきにくい壊れ方をする (2026-08-12 の最終レビューで指摘。
# ``needs_japanese_wrap`` 自体の退行は直したが、``apply_template`` を使わない
# 経路にはそもそも判定が無かった)。**チェックボックスは運用者が明示的に
# 操作するものなので黙って無視せず、この組み合わせのときだけ警告する。**
_NEEDLESS_WRAP_PREFIX = "和文がありません"

# ``refresh_kana`` が本文から作る警告の先頭一致。ここに載っている文言だけを
# 自動で消してよい (接続結果・確認結果・中止理由は消さない)。
_TEXT_WARNING_PREFIXES = (_UNSENDABLE_PREFIX, _NEEDLESS_WRAP_PREFIX)


class _StatusLabel(QLabel):
    """状態を出す行. **文言で画面の大きさを動かさない。**

    折り返す ``QLabel`` は文言によってレイアウトの下限を押し上げるので、素朴に
    置くと窓ごと伸びる。高さは行数で決め打ちし、横は縮められるようにする
    (``Ignored``)。**溢れた文言はツールチップで読める。**
    """

    def __init__(self, text: str = "", *, lines: int = 1, parent=None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.fontMetrics().lineSpacing() * lines + 2)
        self.setToolTip(text)

    def setText(self, text: str) -> None:       # noqa: N802
        super().setText(text)
        self.setToolTip(text)

    def clear(self) -> None:
        self.setText("")


class TxMessagePanel(QGroupBox):
    """送信文 1 通分の欄. **``PANEL_COUNT`` 個並べて、そのうち 1 つを送る。**

    持つのは「その 1 通のもの」だけ — 本文・囲み・送信される文字・その欄の
    ボタン・その欄の状態表示、そして**確認が通った文字列**である。
    接続・速度・相手コール・型は 1 回の交信に 1 つなので :class:`TxDialog` が持つ。

    **判断もここには置かない。** カナ変換・関門・打鍵側とのやりとりは
    ``TxDialog`` の同じメソッドが欄を引数に取って行う。積み上げてきた歯止め
    (確認の閉じ直し、囲みの警告、送信中の保護) を欄ごとに書き写すと、必ず
    どれか 1 つが古いまま取り残される。
    """

    def __init__(self, number: int, parent=None) -> None:
        super().__init__(f"送信文 {number}", parent)
        # **確認が通った文字列は欄ごとに持つ。** 別の欄を書き直しただけで
        # この欄の関門が閉じては「先に全部確認しておく」使い方が成り立たない
        self._confirmed_text: str | None = None
        # 型を入れる直前の (本文, 囲みの状態)。``[元に戻す]`` 用 (欄ごと)
        self._state_before_template: tuple[str, bool] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # **箱の高さは固定する。** ``QPlainTextEdit`` の既定の縦の方針は
        # ``Expanding`` で、状態表示に文字が入って ``updateGeometry()`` が走った
        # 拍子にスクロール領域の中身が組み直され、余った高さが箱に配られる。
        # 実際に「[確認] を押すと本文の箱が 3 倍・カナの箱が 2 倍になり、
        # 何も入っていない欄まで一様に伸びる」という形で表面化した
        # (2026-08-30 の運用者の報告。実測 52→192 / 40→90)。
        # **行数で決める** — 表示倍率 (DPI) や字の大きさが変わっても崩れない。
        line = self.fontMetrics().lineSpacing()
        self.japanese_edit = QPlainTextEdit()
        self.japanese_edit.setPlaceholderText("日本語 (漢字かな交じりで可)")
        self.japanese_edit.setFixedHeight(line * 3 + 14)
        self.japanese_edit.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.japanese_edit)

        row = QHBoxLayout()
        self.wrap_check = QCheckBox("和文をホレ/ラタで囲む")
        # **既定はオフ (使うときだけチェック)。** 2026-08-30 の運用者の指示。
        # 運用者は `JH0ILL ホレ JH0ILL …` のようにホレをカナで本文に打つので、
        # 自動で囲むと二重になる。`apply_template` も自動でオンにはしない
        self.wrap_check.setChecked(False)
        row.addWidget(self.wrap_check)
        row.addStretch(1)
        # **1 回の交信で何度も打ち直す。** 全選択して消すのは手数が多い。
        # 消すのは**この欄の本文だけ** — 相手コール・RST・天気は共有で、
        # 他の欄に用意した文も巻き込まない
        self.clear_btn = QPushButton("クリア")
        self.clear_btn.setToolTip("この欄の本文だけを空にします")
        row.addWidget(self.clear_btn)
        self.check_btn = QPushButton("確認")
        self.check_btn.setToolTip("打鍵せずに、この欄の文が送れるか打鍵側へ問い合わせます")
        row.addWidget(self.check_btn)
        self.send_btn = QPushButton("送信")
        row.addWidget(self.send_btn)
        # **[中止] は送信の右隣** (2026-08-30 の運用者の指示)。押した欄がどれでも
        # 同じ 1 通を止める — 打鍵しているのは常に 1 通だけなので、
        # 「どの中止を押せばいいのか」を考えずに済む
        self.stop_btn = QPushButton("中止")
        self.stop_btn.setToolTip("いま打鍵している 1 通を止めます (どの欄の [中止] でも同じです)")
        row.addWidget(self.stop_btn)
        layout.addLayout(row)

        self.kana_view = QPlainTextEdit()
        self.kana_view.setReadOnly(True)
        self.kana_view.setPlaceholderText("送信される文字")
        self.kana_view.setFixedHeight(line * 2 + 14)
        self.kana_view.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.kana_view)

        # **その欄の警告と確認結果はその欄に出す。** 共有の 1 行に出すと、
        # 別の欄を触った瞬間に上書きされて消え、送れない文字を抱えた欄が
        # 黙ってそこに残る (4 つ並べて初めて起きる壊れ方)
        self.status_label = _StatusLabel("")
        layout.addWidget(self.status_label)

        # **欄そのものも伸びない。** 余った高さは欄の下の余白へ行く
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)


class _SendWorker(QThread):
    """打鍵を待つあいだ画面を固めないためのスレッド.

    ``NetKeyClient.send`` は完了まで戻らず、そのあいだ拍を打ち続ける。
    """

    finished_ok = Signal(object)      # SendResult
    failed = Signal(str)

    def __init__(self, client: NetKeyClient, text: str, wpm: float) -> None:
        super().__init__()
        self._client = client
        self._text = text
        self._wpm = wpm

    def run(self) -> None:
        try:
            result = self._client.send(self._text, self._wpm)
        except NetKeyError as exc:
            self.failed.emit(str(exc))
        else:
            self.finished_ok.emit(result)


class _ConnectWorker(QThread):
    """繋ぐあいだ画面を固めないためのスレッド.

    ``NetKeyClient.connect`` は相手が応答するまで戻らない。**実測 (2026-08-31)**:
    打鍵サーバが動いていない 2.06 秒 / 相手の PC ごと落ちている 5.01 秒
    (``connect_timeout_s`` の 5 秒で切れる) / 繋がるが名乗りを返さない 5.01 秒。

    これを 3 秒おきの ``retry_tick`` から GUI スレッドで呼んでいたため、
    **打鍵側を起こしていないあいだ受信の画面がほとんど止まっていた**
    (運用者の報告)。デコード自体はワーカースレッドで生きているが、画面が
    更新されないので「受信もデコードも止まった」ように見える。
    """

    connected = Signal(object, object)      # (client, Hello)
    failed = Signal(object, object)         # (client, 例外)

    def __init__(self, client: NetKeyClient) -> None:
        super().__init__()
        self._client = client

    def run(self) -> None:
        try:
            hello = self._client.connect()
        except NetKeyError as exc:          # NetKeyRejected (busy 等) も含む
            self.failed.emit(self._client, exc)
        else:
            self.connected.emit(self._client, hello)

    def close_client(self) -> None:
        """受け取り手が居なくなったときに、繋がってしまった接続を捨てる.

        閉じないと打鍵側は「使用中」のまま残り、次に開いたときに**自分自身の
        古い接続**に busy で撥ねられる (``TxDialog.shutdown`` と同じ理由)。
        """
        self._client.close()


class TxDialog(QDialog):
    """交信ダイアログ."""

    def __init__(
        self,
        settings: AppSettings,
        profile: OperatorProfile | None = None,
        client_factory: Callable[..., NetKeyClient] = NetKeyClient,
        parent=None,
        *,
        received_text: str = "",
        mode: DisplayMode = "european",
        templates_path: Path | str = DEFAULT_TEMPLATES_PATH,
        received_wpm: float | None = None,
        profile_path: Path | str | None = None,
        hamlog_window_factory: Callable[[], object | None] = find_input_window,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("交信")
        self._settings = settings
        self._profile = profile if profile is not None else load_profile()
        self._client_factory = client_factory
        # Hamlog の入力ウィンドウの探し方。**テストは代役を渡すこと**
        # (本番は同じ PC で動いている Hamlog を探す)。
        self._hamlog_window_factory = hamlog_window_factory
        self._client: NetKeyClient | None = None
        self._worker: _SendWorker | None = None
        # 繋ぎに行っているスレッド。**同時に 1 本だけ** (打鍵側は同時 1 接続
        # しか受けない)。終わったら必ず None に戻す。
        self._connect_worker: _ConnectWorker | None = None
        # その接続が自動の繋ぎ直しか (失敗を画面に書かない) と、その行き先
        self._connect_quiet = False
        self._connect_endpoint = ""
        # 並べた送信欄と、いま選んでいる欄の番号。**``_build_ui`` より前に置く。**
        # 「確認が通った文字列」は**欄ごと** (``TxMessagePanel``) に移した
        self._panels: list[TxMessagePanel] = []
        self._active_index = 0
        # **送り終えた (文字列, 速度) の組。** 一度打鍵側が「この速度で
        # 送れる」と答え、実際に最後まで送れたものは、確認を押し直さずに
        # 送れるようにする (運用者の要望、2026-08-12)。交信中に同じ文を
        # 送り直すたびに確認の往復を待つのが、実運用で一番効く無駄だった。
        # **画面を閉じれば忘れる。** 古い確認結果で送る事故を残さないため。
        self._sent_ok: set[tuple[str, float]] = set()
        # **送信ワーカーへ渡した (欄, 文字列, 速度) の組。** ``run_send`` が
        # ワーカーを作った時点で確定させ、``_on_sent`` はこれを使う。完了時に
        # ``wpm_spin.value()`` を読み直すと、送信中に運用者が速度を変えたときに
        # 実際に送った速度と違う値を「送れた」記録として覚えてしまう。
        self._send_pending: tuple[str, float] | None = None
        # **どの欄から送ったか。** 送り終えた欄の確認済みの印だけを落とすため
        # (他の欄の印まで落とすと、用意しておいた文が送れなくなる)。
        self._send_panel: "TxMessagePanel | None" = None
        # **画面が今表示しているモード** (``auto`` を含む)。設定ファイルの
        # ``mode`` ではない — あれは画面を閉じるときにしか書き戻されないので、
        # 画面が和文でも設定が欧文なら和文の型が消えていた
        self._mode: DisplayMode = mode
        self._templates_path = Path(templates_path)
        # 受信信号から測った速度。**開いた時点の値で固定する。**
        # 開いている間も動かすと、押そうとした瞬間に値が変わる。
        self._received_wpm = received_wpm
        # 経歴の保存先。**テストは必ず一時パスを渡すこと**
        # (`None` なら `~/.cw-decorder/operator.json`)。
        self._profile_path = Path(profile_path) if profile_path else None
        # そのモードで使える型だけを持つ (一覧の並びと同じ順)。
        # ``auto`` では両方の型が残る (``templates_for_mode`` 参照)
        self._templates = templates_for_mode(load_templates(self._templates_path), mode)

        self._build_ui()
        self._fit_to_content()
        self._fill_from_received(received_text)
        self._update_buttons()

        # **待機中は自動で繋ぎ直す** (設計書 §8.3)
        self._retry_timer = QTimer(self)
        self._retry_timer.setInterval(int(RETRY_INTERVAL_S * 1000))
        self._retry_timer.timeout.connect(self.retry_tick)
        self._retry_timer.start()

        # **どの閉じ方でも後片付けを通す。** ``accept()``/``reject()``/``done()``
        # のすべてが ``finished`` を出す (``closeEvent`` は Esc では呼ばれない)
        self.finished.connect(lambda _result: self.shutdown())

    # ---- 画面 ----
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("打鍵側:"))
        self.endpoint_edit = QLineEdit(self._settings.tx_endpoint)
        self.endpoint_edit.setPlaceholderText("192.168.0.10:45679")
        top.addWidget(self.endpoint_edit, 1)
        self.connect_btn = QPushButton("接続")
        # **``clicked`` は ``checked: bool`` を渡す。** そのまま繋ぐと第 1 引数の
        # ``quiet`` に入る。引数の食い違いは過去にこのリポジトリで実際に踏んでいる
        self.connect_btn.clicked.connect(lambda: self.begin_connect())
        top.addWidget(self.connect_btn)
        top.addWidget(QLabel("速度:"))
        self.wpm_spin = QDoubleSpinBox()
        self.wpm_spin.setRange(5.0, 40.0)
        self.wpm_spin.setSingleStep(1.0)
        self.wpm_spin.setValue(self._settings.tx_wpm)
        self.wpm_spin.setSuffix(" WPM")
        self.wpm_spin.valueChanged.connect(self.on_wpm_changed)
        top.addWidget(self.wpm_spin)
        # **受信の速度に合わせる。** 勝手に追従はしない — 送信直前に値が動くと
        # 確認をやり直すことになり、押そうとした瞬間に速度が変わる。
        # 押したときだけ入る (押せば `on_wpm_changed` が確認を閉じ直す)。
        self.match_wpm_btn = QPushButton("受信に合わせる")
        if self._received_wpm is None:
            self.match_wpm_btn.setEnabled(False)
            self.match_wpm_btn.setToolTip("受信信号から速度を測れていません")
        else:
            self.match_wpm_btn.setToolTip(
                f"受信 {self._received_wpm:.1f} WPM に合わせる (だいたいの目安です)"
            )
        self.match_wpm_btn.clicked.connect(lambda: self.match_received_wpm())
        top.addWidget(self.match_wpm_btn)
        layout.addLayout(top)

        fields = QHBoxLayout()
        fields.addWidget(QLabel("相手:"))
        self.their_call_edit = QLineEdit()
        self.their_call_edit.setPlaceholderText("JA1ABC")
        fields.addWidget(self.their_call_edit, 1)
        fields.addWidget(QLabel("相手名前:"))
        self.their_name_edit = QLineEdit()
        fields.addWidget(self.their_name_edit, 1)
        fields.addWidget(QLabel("送る RST:"))
        self.rst_edit = QLineEdit("599")
        self.rst_edit.setMaxLength(3)
        fields.addWidget(self.rst_edit)
        # **もらった RST は別の欄。** Hamlog は送った側 (RSTs) と
        # もらった側 (RSTr) を別々に記録する
        # **Hamlog の言い方では「送る RST」= His、「もらった RST」= My。**
        # 画面では紛れないよう「送る/もらった」と書く
        fields.addWidget(QLabel("もらった RST:"))
        self.received_rst_edit = QLineEdit("599")
        self.received_rst_edit.setMaxLength(3)
        fields.addWidget(self.received_rst_edit)
        layout.addLayout(fields)

        # **交信記録 (Hamlog) 用の欄。** 送信そのものには使わない。
        # 周波数は**保存しない** — 前回の値が残っていると、band を変えたのに
        # 気づかず違う周波数で記録してしまう (天気の欄と同じ考え方)
        log_row = QHBoxLayout()
        log_row.addWidget(QLabel("周波数:"))
        self.freq_edit = QLineEdit()
        self.freq_edit.setPlaceholderText("7.026")
        self.freq_edit.setMaximumWidth(90)
        log_row.addWidget(self.freq_edit)
        log_row.addWidget(QLabel("MHz"))
        log_row.addWidget(QLabel("住所:"))
        self.qth_edit = QLineEdit()
        self.qth_edit.setPlaceholderText("神奈川県横浜市")
        log_row.addWidget(self.qth_edit, 1)
        # **入力欄に入れるだけ。** 確定 (ログへの書き込み) は Hamlog 側で
        # 運用者が Enter を押す (2026-08-30 の運用者の判断)
        self.hamlog_btn = QPushButton("hamlog登録")
        self.hamlog_btn.setToolTip(
            "同じ PC で動いている Hamlog の入力欄へ交信データを入れます "
            "(登録の確定は Hamlog 側で Enter を押してください)"
        )
        self.hamlog_btn.clicked.connect(lambda: self.register_to_hamlog())
        log_row.addWidget(self.hamlog_btn)
        layout.addLayout(log_row)

        # **備考は別の行。** 1 行に詰めると欄が 60 px ほどになり読めない
        remarks_row = QHBoxLayout()
        remarks_row.addWidget(QLabel("備考1:"))
        self.remarks1_edit = QLineEdit()
        self.remarks1_edit.setPlaceholderText("Hamlog の Remarks1 へ")
        remarks_row.addWidget(self.remarks1_edit, 1)
        remarks_row.addWidget(QLabel("備考2:"))
        self.remarks2_edit = QLineEdit()
        self.remarks2_edit.setPlaceholderText("Remarks2 へ")
        remarks_row.addWidget(self.remarks2_edit, 1)
        layout.addLayout(remarks_row)

        # **天気と気温は経歴ではなくここに置く** (設計書 §5)。相手コールや RST と
        # 同じ「その交信のもの」であり、経歴に入れると運用のたびに経歴の画面を
        # 開いて書き換えることになる。
        # **既定は空。** 前回の値が残っていると、書き忘れたまま嘘の天気を送る。
        # 空なら送信文に `?` が出るので、そこで気づける。
        weather = QHBoxLayout()
        weather.addWidget(QLabel("天気:"))
        self.weather_edit = QLineEdit()
        self.weather_edit.setPlaceholderText("ハレ")
        weather.addWidget(self.weather_edit, 1)
        weather.addWidget(QLabel("気温:"))
        self.temp_edit = QLineEdit()
        # **数字で書く** (`20`)。数字は両方の符号表にあるので和文でも通る
        self.temp_edit.setPlaceholderText("20")
        self.temp_edit.setMaxLength(4)
        weather.addWidget(self.temp_edit)
        # **気温の右の空きに置く** (2026-08-30 の運用者の指示)。天気の欄が
        # 伸びるので、ボタンは行の右端に寄る
        self.clear_qso_btn = QPushButton("交信欄クリア")
        self.clear_qso_btn.setToolTip(
            "相手コール・相手名前・住所・備考・天気・気温を空にします\n"
            "(RST と周波数は残ります。送信文は各欄の [クリア] で消してください)"
        )
        self.clear_qso_btn.clicked.connect(lambda: self.clear_qso_fields())
        weather.addWidget(self.clear_qso_btn)
        layout.addLayout(weather)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("型:"))
        self.template_combo = QComboBox()
        for template in self._templates:
            self.template_combo.addItem(template.name)
        picker.addWidget(self.template_combo, 1)
        # **選んだ瞬間に本文へ入れる** (運用者の要望、2026-08-12)。交信中に
        # 送信文を作るのは時間の勝負で、ボタンを押す 1 手間が重い。
        # ``[型を使う]`` は残す — 同じ型をもう一度差し込み直したいとき
        # (相手コールを入れ直した後など) に選び直せないため。
        self.template_combo.currentIndexChanged.connect(self._on_template_selected)
        self.use_template_btn = QPushButton("型を使う")
        # **``clicked`` は ``checked: bool`` を渡す。** 引数の食い違いは
        # 過去にこのリポジトリで実際に踏んでいる
        self.use_template_btn.clicked.connect(lambda: self.apply_template())
        picker.addWidget(self.use_template_btn)
        # **消してしまった本文への逃げ道。** 選ぶだけで入れ替わる以上、
        # 手で書いた内容を事故で失う経路ができる
        self.undo_template_btn = QPushButton("元に戻す")
        self.undo_template_btn.setToolTip("型を入れる直前の本文に戻します")
        self.undo_template_btn.setEnabled(False)
        self.undo_template_btn.clicked.connect(lambda: self.undo_template())
        picker.addWidget(self.undo_template_btn)
        # **経歴が空だと型が 1 つも実用にならない** (`{自局コール}` `{名前}` が
        # 埋まらず `?` になる)。ここから書けるようにする。
        self.profile_btn = QPushButton("経歴…")
        self.profile_btn.setToolTip("自局の情報 (コールサイン・名前・QTH・設備) を編集します")
        self.profile_btn.clicked.connect(lambda: self.open_profile_dialog())
        picker.addWidget(self.profile_btn)
        # 型も JSON を手で書くしかなかった (経歴と同じ状態)
        self.edit_templates_btn = QPushButton("型の編集…")
        self.edit_templates_btn.setToolTip("返信の型を追加・編集・並べ替えします")
        self.edit_templates_btn.clicked.connect(lambda: self.open_template_dialog())
        picker.addWidget(self.edit_templates_btn)
        layout.addLayout(picker)

        # **送信欄を ``PANEL_COUNT`` 個並べる。** 画面が狭い PC でも全部に手が
        # 届くよう、欄の列だけを縦にスクロールさせる (上の共有の欄と下の
        # [中止] は常に見えたままにする)。
        panel_host = QWidget()
        panel_column = QVBoxLayout(panel_host)
        panel_column.setContentsMargins(0, 0, 0, 0)
        for number in range(1, PANEL_COUNT + 1):
            panel = TxMessagePanel(number)
            # **``clicked`` は ``checked: bool`` を渡す。** 欄を引数に取る
            # メソッドへ直に繋ぐと、その ``bool`` が欄の引数に入る
            # (引数の食い違いは過去にこのリポジトリで実際に踏んでいる)。
            panel.japanese_edit.textChanged.connect(
                lambda p=panel: self.refresh_kana(p)
            )
            panel.wrap_check.toggled.connect(
                lambda _checked=False, p=panel: self.refresh_kana(p)
            )
            panel.clear_btn.clicked.connect(
                lambda _checked=False, p=panel: self.clear_text(p)
            )
            panel.check_btn.clicked.connect(
                lambda _checked=False, p=panel: self.run_check(p)
            )
            panel.send_btn.clicked.connect(
                lambda _checked=False, p=panel: self.run_send(p)
            )
            # **どの欄の [中止] も同じ 1 通を止める** (運用者の指示)
            panel.stop_btn.clicked.connect(lambda _checked=False: self.run_stop())
            # 本文に触った欄を「選んでいる欄」にする (型はそこへ入る)
            panel.japanese_edit.installEventFilter(self)
            self._panels.append(panel)
            panel_column.addWidget(panel)
        panel_column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel_host)
        layout.addWidget(scroll, 1)
        # 開くときの高さを決めるために覚えておく (``_fit_to_content``)
        self._panel_host = panel_host
        self._panel_scroll = scroll

        # **共有の 1 行。** 接続の結果と送信の結果を出す。本文から分かる警告と
        # 確認の結果は**欄ごとの行**に出る (``TxMessagePanel.status_label``)。
        # 接続時の「符号表が両 PC で違います」は改行を含む 2 行なので 2 行取る
        self.status_label = _StatusLabel("未接続", lines=2)
        layout.addWidget(self.status_label)

    def _fit_to_content(self) -> None:
        """送信欄が全部見える高さで開く. **画面からはみ出さない範囲で。**

        ``QScrollArea`` の ``sizeHint`` は頭打ちになる (Qt の仕様) ので、
        中身 (欄の列) の ``sizeHint`` から必要な高さを足し直す。足りなければ
        スクロールするので、狭い画面でも全部に手が届く。

        **開く前に決める。** 出してから広げ直すと、開いた瞬間に窓が跳ねる。
        """
        base = self.sizeHint().height() - self._panel_scroll.sizeHint().height()
        want = base + self._panel_host.sizeHint().height() + 8
        screen = QApplication.primaryScreen()
        if screen is not None:
            want = min(want, screen.availableGeometry().height() - 60)
        self.resize(600, max(want, 400))

    # ---- 欄 ----
    @property
    def panels(self) -> tuple["TxMessagePanel", ...]:
        """並べた送信欄 (上から順)."""
        return tuple(self._panels)

    @property
    def active_panel(self) -> "TxMessagePanel":
        """いま選んでいる欄. 型と ``[元に戻す]`` はここに効く."""
        return self._panels[self._active_index]

    def set_active_panel(self, index: int) -> None:
        """選んでいる欄を変える (本文に触ると自動で変わる)."""
        if not 0 <= index < len(self._panels):
            return
        self._active_index = index
        self._update_buttons()

    def eventFilter(self, watched, event) -> bool:      # noqa: N802
        """本文に入力の焦点が来た欄を「選んでいる欄」にする.

        **クリックしてから型を選ぶ**、が自然な手順になる。
        """
        if event.type() == QEvent.Type.FocusIn:
            for index, panel in enumerate(self._panels):
                if watched is panel.japanese_edit:
                    self.set_active_panel(index)
                    break
        return super().eventFilter(watched, event)

    # 既存の呼び名は**選んでいる欄**を指す。積み上げてきた関門のテストを
    # そのまま回帰網として使えるようにするための約束である。
    @property
    def japanese_edit(self) -> QPlainTextEdit:
        return self.active_panel.japanese_edit

    @property
    def wrap_check(self) -> QCheckBox:
        return self.active_panel.wrap_check

    @property
    def kana_view(self) -> QPlainTextEdit:
        return self.active_panel.kana_view

    @property
    def clear_btn(self) -> QPushButton:
        return self.active_panel.clear_btn

    @property
    def check_btn(self) -> QPushButton:
        return self.active_panel.check_btn

    @property
    def send_btn(self) -> QPushButton:
        return self.active_panel.send_btn

    @property
    def stop_btn(self) -> QPushButton:
        return self.active_panel.stop_btn

    @property
    def _confirmed_text(self) -> str | None:
        """選んでいる欄の「確認が通った文字列」.

        **中の処理はこれを使わず ``panel._confirmed_text`` を直接読むこと。**
        欄を 4 つにした今、どの欄の話なのかを暗黙にしてはいけない。
        ここは 1 欄だった頃からの関門テストを生かしておくための橋である。
        """
        return self.active_panel._confirmed_text

    @_confirmed_text.setter
    def _confirmed_text(self, value: str | None) -> None:
        self.active_panel._confirmed_text = value

    def status_text(self) -> str:
        """共有の 1 行と、全欄の 1 行を繋げたもの.

        画面では別々の行に出ている (接続の結果は共有、送れない文字の警告は
        その欄)。「どこかに出ているか」を見たいときはここを読む。
        """
        lines = [self.status_label.text()]
        lines.extend(panel.status_label.text() for panel in self._panels)
        return "\n".join(line for line in lines if line)

    # ---- 文字 ----
    def wire_text(self, panel: "TxMessagePanel | None" = None) -> str:
        """LAN に流す確定テキスト (既定は**選んでいる欄**)."""
        return (panel or self.active_panel).kana_view.toPlainText().strip()

    def refresh_kana(self, panel: "TxMessagePanel | None" = None) -> None:
        """日本語をカタカナに直し、**関門を閉じ直す**.

        送信できるかの判定は **``encoder.find_unsendable`` を使う**。
        ``reading`` 側の ``bad_chars`` は和文表だけで照合しており、
        **コールサインを含む文が必ず赤くなっていた** (設計書 §3)。
        打鍵側 (``key_server.prepare``) と同じ規則で判定しないと、
        画面と実際の可否が食い違う。
        """
        panel = panel or self.active_panel
        source = panel.japanese_edit.toPlainText()
        result = to_sendable_kana(source, self._profile)
        wrap_on = panel.wrap_check.isChecked()
        text = wrap_japanese(result.text) if wrap_on else result.text
        panel.kana_view.setPlainText(text)
        # **書き直した欄の関門だけを閉じる。** 全部閉じると、先に確認して
        # おいた他の文まで送れなくなり、4 つ並べた意味が無くなる
        panel._confirmed_text = None
        self._show_text_warnings(panel, text, wrap_on=wrap_on, unwrapped=result.text)
        self._update_buttons()

    def _show_text_warnings(
        self, panel: "TxMessagePanel", wire_text: str, *, wrap_on: bool, unwrapped: str
    ) -> None:
        """本文から分かる警告を出す。**直ったら消え、複数起きたら全部出す。**

        警告は 2 つある — 「送信できない文字」「和文が無いのに囲んでいる」。

        **どちらも本文から作り直す。** 消えない警告は、直したのに直っていない
        と思わせる (2026-08-11 レビュー Minor 1 で実際に踏んだ)。

        「埋まっていない欄」の警告は**廃止した**。差し込みで空の値は ``?`` に
        なるので埋め残しが起きない (設計書 §2.2)。運用者は送信文に出た ``?``
        を見て、必要なら直してから確認・送信する。

        **片方を上書きしない。** 以前は後から書いたほうだけが見え、
        実際には送れない文字があるのに「埋まっていない欄」しか出ない、という
        状態になり得た (同 Minor 2)。改行で並べて全部見せる。

        **「和文が無いのに囲んでいる」は ``apply_template`` を使わず
        ``japanese_edit`` に直接打つ経路のためのもの。** ``wrap_check`` は
        運用者が明示的に操作するチェックボックスであり、入れたまま
        和文の無い本文 (``「FT991」`` など) を直接打つと `{HORE}「FT991」{RATA}`
        が**警告なしで**できてしまう。中身は欧文として符号化できるので
        「送信できない文字」にはならず、**送れるのに化ける**という一番
        気づきにくい壊れ方をする (2026-08-12 の最終レビューで指摘。
        ``apply_template`` 側は中身を見て自動で ``wrap_check`` を設定するので
        対象外)。本文が空のとき (ダイアログを開いた直後など) にまで警告が
        出ると邪魔なので、``unwrapped`` に何か書かれているときだけ見る。

        **``{HORE}`` が既に書かれているときも見ない。** ``needs_japanese_wrap``
        は「和文が無い」ときと「既に囲んである」ときの**両方で偽を返す**。
        後者を前者と取り違えると、手で ``{HORE}コンニチハ{RATA}`` と打った
        運用者に「和文がありません」と言うことになる (``wrap_japanese`` は
        二重に囲まないので電波は正しい)。**唯一の歯止めである警告を
        無意味に鳴らすと、次に本当に鳴ったとき無視される。**
        ``apply_template`` (``HORE not in filled``) と同じ判定である。

        消してよいのはこの関数が出した文言だけである。無条件に消すと、接続結果・
        確認結果 (「38 文字 / 34.2 秒」等)・中止理由まで巻き込む。
        """
        warnings: list[str] = []
        bad = find_unsendable(wire_text)
        if bad:
            warnings.append(f"{_UNSENDABLE_PREFIX}: " + "".join(b.char for b in bad))
        if (
            wrap_on
            and unwrapped.strip()
            and HORE not in unwrapped
            and not needs_japanese_wrap(unwrapped)
        ):
            warnings.append(
                f"{_NEEDLESS_WRAP_PREFIX}: このまま囲むと相手のデコーダが和文に切り替わり、"
                "欧文が読めなくなります。「和文をホレ/ラタで囲む」を外してください。"
            )
        if warnings:
            panel.status_label.setText("\n".join(warnings))
        elif panel.status_label.text().startswith(_TEXT_WARNING_PREFIXES):
            panel.status_label.clear()

    def _fill_from_received(self, received_text: str) -> None:
        """受信テキストから拾えた欄を入れる. **拾えなければ空のまま。**

        自局コールは経歴が 1 つだけ持つ (和文の交信でも欧文で送るため)。
        以前は ``display``/``reading`` の 2 通りがあり、読みを入れていると
        ``ジェイキュー`` が「自局コール」として渡って受信文中の自分のコールを
        除外できず、**自分のコールが「相手」欄に入っていた**
        (2026-08-11 レビュー I3)。**1 値になったので起こりようがない。**
        """
        if not received_text:
            return
        found = extract_fields(received_text, self._profile.callsign)
        self.their_call_edit.setText(found.their_call)
        self.their_name_edit.setText(found.their_name)

    def field_values(self, mode: str) -> dict[str, str]:
        """型に差し込む値 (経歴 + 画面の欄).

        **経歴の値は型のモードで変わる** — 和文の型は和文用、欧文と ``any``
        の型は欧文用 (:func:`profile_values`)。

        **空の値は ``?`` になる** (:func:`fill`)。止めも通知もしない。
        """
        values = profile_values(self._profile, mode)
        values["相手コール"] = self.their_call_edit.text().strip()
        values["相手名前"] = self.their_name_edit.text().strip()
        values["RST"] = self.rst_edit.text().strip()
        # **その交信のもの。経歴には置かない** (設計書 §5)。
        # 既定は空 — 前回の値が残っていると、書き忘れたまま嘘の天気を送る
        values["天気"] = self.weather_edit.text().strip()
        values["気温"] = self.temp_edit.text().strip()
        # **空の値も落とさない。** `fill` が `?` に倒す。ここで落とすと
        # 「値が無い」と「欄そのものを渡していない」の区別が消える
        return values

    # ---- 交信の欄 ----
    #: 欄の名前 → その欄のウィジェット名。**主画面はこの名前で呼ぶ**
    #: (``set_qso_field``)。画面の作りを外から触らせないための入口である。
    QSO_FIELDS: dict[str, str] = {
        "their_call": "their_call_edit",
        "their_name": "their_name_edit",
        "qth": "qth_edit",
        "rst_sent": "rst_edit",
        "rst_received": "received_rst_edit",
        "freq_mhz": "freq_edit",
        "remarks1": "remarks1_edit",
        "remarks2": "remarks2_edit",
    }

    #: 欄の名前 → 画面に出す呼び名 (右クリックのメニューが使う。**並ぶ順**)。
    QSO_FIELD_LABELS: dict[str, str] = {
        "their_call": "相手コール",
        "their_name": "相手名前",
        "qth": "住所",
        "rst_sent": "送る RST",
        "rst_received": "もらった RST",
    }

    #: ``[交信欄クリア]`` で**消さない**欄。RST と周波数は、同じバンド・同じ
    #: 運用のあいだ変わらないので残す (2026-08-30 の運用者の指示)。
    QSO_KEEP_ON_CLEAR: tuple[str, ...] = ("rst_sent", "rst_received", "freq_mhz")

    def clear_qso_fields(self) -> None:
        """相手ごとに変わる欄を空にする. **RST と周波数は残す。**

        次の相手に移るときに使う。消えるのは相手コール・相手名前・住所・
        備考1・備考2・天気・気温。

        **送信文の欄は触らない。** 用意しておいた文を巻き添えにしないため
        (送信文には欄ごとの ``[クリア]`` がある)。
        """
        for field, widget_name in self.QSO_FIELDS.items():
            if field in self.QSO_KEEP_ON_CLEAR:
                continue
            getattr(self, widget_name).clear()
        # 天気・気温は Hamlog へ渡す欄ではない (型に差し込む値) ので別に消す
        self.weather_edit.clear()
        self.temp_edit.clear()

    def set_qso_field(self, field: str, value: str) -> None:
        """交信の欄に値を入れる. **名前で指すこと** (``QSO_FIELDS``).

        受信文から拾った文字をそのまま入れられるよう、ここで軽く整える:

        * 相手コール — 前後の空白を落として大文字に
        * RST — RST らしい塊を探し、**略号数字を数字に直す**
          (``UR 579`` → ``579``、``5NN`` → ``599``)。
          RST らしい塊が無ければ**空にする** (当てにならない値を入れない)
        * それ以外 — 前後の空白を落とすだけ

        知らない名前は黙って無視する (呼び出し側の綴り間違いで落とさない)。
        """
        widget_name = self.QSO_FIELDS.get(field)
        if widget_name is None:
            return
        text = value.strip()
        if field == "their_call":
            text = text.upper()
        elif field in ("rst_sent", "rst_received"):
            text = normalise_rst(text)
        getattr(self, widget_name).setText(text)

    # ---- 交信記録 (Hamlog) ----
    def hamlog_entry(self) -> QsoEntry:
        """いま画面にある値から、Hamlog へ渡す 1 交信分を作る.

        **日時は今 (JST)。** 交信の途中で何度押しても、押した時刻が入る。
        モードは ``CW`` で固定 (このアプリが扱うのは CW だけ)。
        """
        return QsoEntry.at(
            _dt.datetime.now(_dt.timezone.utc),
            call=self.their_call_edit.text().strip(),
            rst_sent=self.rst_edit.text().strip(),
            rst_received=self.received_rst_edit.text().strip(),
            freq_mhz=self.freq_edit.text().strip(),
            name=self.their_name_edit.text().strip(),
            qth=self.qth_edit.text().strip(),
            remarks1=self.remarks1_edit.text().strip(),
            remarks2=self.remarks2_edit.text().strip(),
        )

    def register_to_hamlog(self) -> None:
        """Hamlog の入力欄へ交信データを入れる. **確定はしない。**

        確定 (ログへの書き込み) は運用者が Hamlog 側で Enter を押す。値を
        間違えたまま書き込むと Hamlog 側で消す手間がかかるためである
        (2026-08-30 の運用者の判断)。

        **相手のコールサインが無いときは何もしない。** コールサインの無い
        交信記録は意味を持たない。
        """
        entry = self.hamlog_entry()
        if not entry.call:
            self.status_label.setText(
                "相手のコールサインが空です。「相手:」欄を埋めてから押してください。"
            )
            return
        try:
            hamlog_register(entry, self._hamlog_window_factory())
        except HamlogNotRunning as exc:
            self.status_label.setText(str(exc))
            return
        except OSError as exc:                      # 窓は在るが送れなかった
            self.status_label.setText(f"Hamlog へ渡せませんでした: {exc}")
            return
        self.status_label.setText(
            f"Hamlog の入力欄に入れました — {entry.call} {entry.date} {entry.time} "
            f"{entry.freq_mhz or '(周波数なし)'} MHz {entry.mode}。"
            "内容を確かめて Hamlog 側で Enter を押すと登録されます。"
        )

    def clear_text(self, panel: "TxMessagePanel | None" = None) -> None:
        """その欄の日本語ボックスを空にする.

        **消すのはその欄の本文だけ。** 相手コール・相手名前・RST・天気・気温は
        交信のあいだ変わらないので、巻き込むと打ち直しになる。
        **他の欄に用意した文も巻き込まない** (先に何通か書いておくため)。

        空にすれば ``textChanged`` → ``refresh_kana`` が走り、**確認は
        やり直しになる** (送信される文字も空になるので ``[確認]`` も押せない)。

        **送信中は押せない** (``_update_buttons``)。打鍵している最中に本文が
        消えると、何を送っているのか画面から分からなくなる。
        """
        (panel or self.active_panel).japanese_edit.clear()

    def open_profile_dialog(self) -> None:
        """経歴の編集画面を開き、**閉じたら読み直す**.

        書いてすぐ型に反映されないと確かめようがない。読み直すだけで
        日本語ボックスは触らない — 既に書いた本文を勝手に作り直さない。
        """
        from src.app.profile_dialog import ProfileDialog

        dialog = ProfileDialog(parent=self, path=self._profile_path)
        dialog.exec()
        self._profile = (
            load_profile(self._profile_path)
            if self._profile_path is not None
            else load_profile()
        )

    def open_template_dialog(self) -> None:
        """型の編集画面を開き、**閉じたら一覧を作り直す**.

        書いてすぐ使えないと確かめようがない。**いま選んでいる型は保てない**
        ので (並べ替えや削除で位置が変わる)、先頭に戻す。
        日本語ボックスは触らない — 既に書いた本文を勝手に作り直さない。
        """
        from src.app.template_dialog import TemplateDialog

        dialog = TemplateDialog(
            parent=self,
            path=self._templates_path,
            profile=self._profile,
        )
        dialog.exec()
        self._templates = templates_for_mode(
            load_templates(self._templates_path), self._mode
        )
        # **作り直しの間は選択の信号を止める。** 止めないと ``clear()`` と
        # ``addItem()`` が ``currentIndexChanged`` を出し、本文が勝手に
        # 入れ替わる (選択で適用するようにした副作用)
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        for template in self._templates:
            self.template_combo.addItem(template.name)
        self.template_combo.blockSignals(False)

    def _on_template_selected(self, _index: int) -> None:
        """一覧で選ばれたら、そのまま本文へ入れる.

        **画面を組み立てている途中には呼ばれない** (``currentIndexChanged``
        を繋ぐのは一覧を作った後)。作り直し中の誤爆を防ぐため、
        ``open_template_dialog`` は繋ぎ直す前に信号を止める。
        """
        self.apply_template()

    def undo_template(self) -> None:
        """型を入れる直前の本文と ``wrap_check`` の状態に戻す. **一度だけ。**

        **本文だけでは足りない。** ``apply_template`` は中身を見て
        ``wrap_check`` も書き換えるので、本文だけ戻すと和文の本文なのに
        囲み OFF のまま残り、無囲みの和文がそのまま送信されて相手のデコーダで
        化ける (2026-08-12 最終レビュー)。先に ``wrap_check`` を戻してから
        本文を戻す — 逆にすると本文の ``setPlainText`` が起こす
        ``refresh_kana`` が一瞬古い ``wrap_check`` のままの警告を出しかねない。
        """
        panel = self.active_panel
        if panel._state_before_template is None:
            return
        text, wrap_checked = panel._state_before_template
        panel._state_before_template = None
        panel.wrap_check.setChecked(wrap_checked)
        panel.japanese_edit.setPlainText(text)
        self.undo_template_btn.setEnabled(False)

    def apply_template(self) -> None:
        """選んだ型に欄を差し込み、日本語ボックスへ入れる.

        **欄を差し込んでから入れる。** 逆にすると ``{相手コール}`` が
        カナ変換器を通って壊れる (設計書 §6.1)。

        **囲むかどうかは中身で決める。ただし自動でオンにはしない** (既定は
        オフ、使うときだけチェック。2026-08-30 の運用者の指示)。チェックが
        入ったまま欧文の型を流すと中の欧文が丸ごと ``{HORE}``/``{RATA}`` に
        囲まれ、「送信できない文字」として弾かれる (``find_unsendable`` は
        打鍵側 ``key_server.prepare`` と同じ関数なので、確認を押しても実機側で
        弾かれる)。逆に**型の ``mode`` で決めると ``any`` の型に和文を書いた
        ときに囲みが外れ、符号としては通るので警告も出ないまま、受信側が
        モードを切り替えられず化けて届く** (2026-08-11 レビュー I5)。
        判定は :func:`~src.tx.encoder.needs_japanese_wrap` に任せる。

        **判定は変換後の文で行う。** 型は漢字かな交じりで書けるので、
        変換前の本文を見ても和文かどうかは分からない (``こんにちは`` は
        和文表にも欧文表にも無い)。ここで 1 回余分に変換するが、
        ``to_sendable_kana`` は純粋関数で短い本文なら軽い。

        ``{HORE}`` を自前で持つ型は「囲まない」にする (中身は既に囲まれている)。
        **ここで触るのは型を適用した瞬間だけ。** 以後は運用者が
        ``wrap_check`` を手で切り替えれば、その操作が優先される。

        埋まっていない欄の警告は :meth:`_show_text_warnings` が出す
        (``setPlainText`` → ``refresh_kana`` の経路)。**ここで一度だけ書くと、
        運用者が欄を手で埋めても消えない。**
        """
        if not self._templates:
            # **無言で終わらない** (設計書 §8)。I1 と重なると「型が消えた上に
            # 押しても何も起きない」になり、運用者は壊れたと思う
            self.status_label.setText(
                f"このモードの型がありません ({self._templates_path})"
            )
            return
        index = self.template_combo.currentIndex()
        if index < 0 or index >= len(self._templates):
            return
        template = self._templates[index]
        filled = fill(template.text, self.field_values(template.mode))
        converted = to_sendable_kana(filled, self._profile).text
        # **入れる前の (本文, wrap_check の状態) を覚えておく** (`[元に戻す]`)。
        # 本文が空なら覚えない — 戻す意味が無いのにボタンが押せると紛らわしい。
        # **直後の ``setChecked`` より前に読む。** ここで捕まえておかないと
        # 「元に戻す」が本文だけ戻し、和文の本文なのに囲み OFF のまま残る
        # (送れるのに化ける)。
        panel = self.active_panel
        previous = panel.japanese_edit.toPlainText()
        panel._state_before_template = (
            (previous, panel.wrap_check.isChecked()) if previous else None
        )
        self.undo_template_btn.setEnabled(bool(previous))
        # setChecked は値が変わったときだけ toggled (→ refresh_kana) を
        # 起こす。その後の setPlainText でも textChanged (→ refresh_kana)
        # が起きるので、多くても 2 回で確定する (どちらも副作用は無い)。
        # **自動でオンにはしない** (既定オフ、使うときだけチェック。2026-08-30)。
        # チェックが入ったまま欧文の型を入れたときだけ外す (欧文を丸ごと囲んで
        # 「送信できない」にする経路を塞ぐ歯止めは残す)
        panel.wrap_check.setChecked(
            panel.wrap_check.isChecked()
            and HORE not in filled
            and needs_japanese_wrap(converted)
        )
        # setPlainText が textChanged を起こし refresh_kana が走る
        # (関門もそこで閉じ直り、警告もそこで出る)
        panel.japanese_edit.setPlainText(filled)

    def match_received_wpm(self) -> None:
        """送信の速度を、受信信号から測った速度に合わせる.

        **整数に丸める。** 測定は「だいたいの速さ」であって精密な値ではなく
        (``src/infer/wpm.py``)、`18.3 WPM` と出しても精度の裏付けが無い。

        値を入れれば ``on_wpm_changed`` が走り、**確認はやり直しになる**。
        速度は「何秒間 電波が出るのか」を変えるので、これは正しい。
        """
        if self._received_wpm is None:
            return
        self.wpm_spin.setValue(float(round(self._received_wpm)))

    def on_wpm_changed(self, value: float) -> None:
        """速度が変わったら**関門を閉じ直し**、設定に書き戻す.

        確認が答えるのは「打鍵側が生きているか」「その文字列が通るか」に加えて
        「**何秒間 電波が出るのか**」である。速度はその 3 番目を直接変えるので、
        テキストを変えたときと同じく確認をやり直させる (20 WPM で確認した
        6.3 秒が、5 WPM では 4 倍の長さの電波になる)。

        設定への書き戻しをここで行うのは、速度が毎回入れ直しになるのを避ける
        ため。実際の保存は ``main_window`` がダイアログを閉じた後に行う。
        """
        self._settings.tx_wpm = float(value)
        self._confirmed_text = None
        self._update_buttons()

    def can_send(self, panel: "TxMessagePanel | None" = None) -> bool:
        """送れるか. **確認が通っているか、前に送り終えたものと同じなら送れる。**

        関門を外したわけではない。打鍵側が「その文字列はこの速度で送れる」と
        答えた事実を覚えておき、**まったく同じ文字列・同じ速度**のときだけ
        確認を省く。速度が変われば秒数が変わるので覚え直す
        (20 WPM の 6.3 秒は 5 WPM では 4 倍の長さの電波になる)。

        **打鍵側が居なければ送れない**のは変わらない。
        """
        panel = panel or self.active_panel
        text = self.wire_text(panel)
        if not text or self._client is None:
            return False
        return (
            panel._confirmed_text == text
            or (text, self.wpm_spin.value()) in self._sent_ok
        )

    def _forget_sent_texts(self) -> None:
        """覚えていた「送り終えた」記録を捨てる.

        **打鍵側が変わったら当てにならない。** 別の PC・別の符号表かもしれない。
        """
        self._sent_ok.clear()

    # ---- 操作 ----
    def retry_tick(self) -> None:
        """**待機中は自動で繋ぎ直す** (設計書 §8.3).

        打鍵側を後から起こしても繋がる。**送信中は繋ぎ直さない** — 何がどこまで
        出たのか分からない状態で電波を出さないため。
        """
        if self._client is not None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        if not self.endpoint_edit.text().strip():
            return
        self.begin_connect(quiet=True)

    def begin_connect(self, quiet: bool = False) -> None:
        """打鍵側へ繋ぐ. **待たない** — 実際に繋ぐのは別スレッド (:class:`_ConnectWorker`).

        **運用の経路はこちら。** GUI スレッドで ``connect()`` を待つと、打鍵側が
        応答しないあいだ画面がまるごと止まる (実測 2〜5 秒。3 秒おきの
        ``retry_tick`` から呼んでいたので、打鍵サーバを立てていない交信では
        受信の画面がほぼ止まりっぱなしになっていた。2026-08-31 運用者の報告)。

        Args:
            quiet: 真なら失敗しても画面に書かない (自動の繋ぎ直しから呼ぶため。
                3 秒おきに赤い文字が書き換わると読めない)。
        """
        if self._connect_worker is not None and self._connect_worker.isRunning():
            # **繋ぎに行っている最中は重ねない。** 打鍵側は同時 1 接続しか
            # 受けず、2 本目は自分自身の 1 本目に busy で撥ねられる。
            return
        prepared = self._prepare_connect(quiet)
        if prepared is None:
            return
        client, endpoint = prepared
        worker = _ConnectWorker(client)
        worker.connected.connect(self._on_connect_ok)
        worker.failed.connect(self._on_connect_failed)
        self._connect_worker = worker
        self._connect_quiet = quiet
        self._connect_endpoint = endpoint
        if not quiet:
            self.status_label.setText(f"繋いでいます… ({endpoint})")
        self._update_buttons()
        worker.start()

    def connect_to_keyer(self, quiet: bool = False) -> None:
        """打鍵側へ繋ぐ. **繋ぎ終わるまで戻らない (同期)。**

        **GUI スレッドから直に呼ばないこと** — 画面が止まる。運用の経路は
        :meth:`begin_connect` である。ここは「繋がった状態」を待って作りたい
        場面 (テスト) のために残してある。結果の扱いは別スレッド経由と同じ
        ``_on_connect_ok`` / ``_on_connect_failed`` を通す。
        """
        prepared = self._prepare_connect(quiet)
        if prepared is None:
            return
        client, endpoint = prepared
        self._connect_quiet = quiet
        self._connect_endpoint = endpoint
        try:
            hello = client.connect()
        except NetKeyError as exc:          # NetKeyRejected (busy 等) も含む
            self._on_connect_failed(client, exc)
            return
        self._on_connect_ok(client, hello)

    def _prepare_connect(self, quiet: bool) -> tuple[NetKeyClient, str] | None:
        """行き先を確かめ、古い接続を畳んで、新しいクライアントを作る.

        Returns:
            ``(client, endpoint)``。行き先が読めないなら ``None``。
        """
        endpoint = self.endpoint_edit.text().strip()
        if not endpoint:
            if not quiet:
                self.status_label.setText("打鍵側の host:port を入れてください。")
            self._update_buttons()
            return None
        try:
            host, port = parse_endpoint(endpoint, default_port=DEFAULT_KEY_PORT)
        except ValueError as exc:
            if not quiet:
                self.status_label.setText(f"打鍵側の指定が読めません: {exc}")
            self._update_buttons()
            return None

        if self._client is not None:
            # **繋ぎ直す前に必ず古い接続を閉じる。** 打鍵側は同時 1 接続しか
            # 受けず、しかも待機中には期限を掛けない (key_server.py)。ここで
            # 閉じずに新しい接続を作ると、古い接続が「使用中」のまま永久に
            # 残り、以後の再接続がすべて busy で撥ねられる (打鍵側 CLI の
            # 再起動でしか回復しない)。エンドポイントを変えて繋ぎ直したい、
            # という運用者の意図もこれで満たせる (常に古い接続を畳んでから
            # 新しい接続を作る)。送信中はそもそも [接続] を押せない
            # (``_update_buttons`` で無効化) のでここに来ない。
            self._client.close()
            self._client = None

        return self._client_factory(host, port), endpoint

    def _on_connect_failed(self, client: NetKeyClient, exc: Exception) -> None:
        """繋げなかった. **同期・別スレッドのどちらの経路もここを通る。**"""
        if self._connect_worker is not None and not self._connect_worker.isRunning():
            self._connect_worker = None
        # **繋ぎかけを畳んでおく。** 名乗りが来ないまま抜けた接続を残すと、
        # 打鍵側から見て「使用中」のままになる。
        client.close()
        self._client = None
        if not self._connect_quiet:
            if isinstance(exc, NetKeyRejected) and exc.code == "busy":
                # **``NetKeyError`` の一般文言に埋もれさせない。** busy は
                # 理由が分かる文言にする。
                self.status_label.setText(
                    f"打鍵側は今、別の運用者が使用中です。しばらく待って再接続してください。({exc})"
                )
            else:
                self.status_label.setText(str(exc))
        self._update_buttons()

    def _on_connect_ok(self, client: NetKeyClient, hello: Hello) -> None:
        """繋がった. **同期・別スレッドのどちらの経路もここを通る。**"""
        if self._connect_worker is not None and not self._connect_worker.isRunning():
            self._connect_worker = None
        self._client = client
        self._settings.tx_endpoint = self._connect_endpoint or self.endpoint_edit.text().strip()
        message = f"接続しました — {hello.describe_wiring()}"
        if not hello.fingerprint_matches:
            # **静かな食い違いを見える警告にする** (設計書 §2.1)
            message += "\n**警告: 符号表が両 PC で違います。** リポジトリを揃えてください。"
            # **「送り終えた」記録も当てにならない。** 別の PC・別の符号表かも
            # しれないので、覚えていた記録で確認を飛ばして送らせない
            # (2026-08-13 最終レビュー Critical 1)。
            self._forget_sent_texts()
        self.status_label.setText(message)
        # **全部の欄の確認を落とす。** 別の PC・別の符号表かもしれないので、
        # 1 番目だけ落とすのでは足りない
        for panel in self._panels:
            panel._confirmed_text = None
        self._update_buttons()

    def run_check(self, panel: "TxMessagePanel | None" = None) -> None:
        """**打鍵しない検査。** これが通って初めてその欄が送れる.

        欄ごとに押せる。**先に 4 通とも確認しておき**、相手の信号に合う 1 通を
        [送信] で出す、というのが狙いの使い方である (2026-08-30)。
        """
        panel = panel or self.active_panel
        if self._client is None:
            return
        text = self.wire_text(panel)
        if not text:
            return
        try:
            result = self._client.check(text, self.wpm_spin.value())
        except NetKeyRejected as exc:
            panel._confirmed_text = None
            # **撥ねられたら「送り終えた」記録も当てにならない。** 打鍵側が
            # 「もう通らない」と言っているのに、以前送れた記録だけで [送信] を
            # 有効なままにしない (2026-08-13 最終レビュー Important 4)。
            self._sent_ok.discard((text, self.wpm_spin.value()))
            detail = "".join(bad["char"] for bad in exc.unsendable)
            panel.status_label.setText(f"{exc}: {detail}" if detail else str(exc))
        except NetKeyError as exc:
            self._client = None
            panel._confirmed_text = None
            panel.status_label.setText(str(exc))
        else:
            panel._confirmed_text = text
            panel.status_label.setText(
                f"確認しました — {result.chars} 文字 / {result.elements} 要素 / {result.seconds:.1f} 秒"
            )
        self._update_buttons()

    def run_send(self, panel: "TxMessagePanel | None" = None) -> None:
        """その欄の文を打鍵側へ送る.

        **打鍵器は 1 台。** 送信中は全部の欄の [送信]/[確認]/[クリア] が
        無効になる (``_update_buttons``) ので、2 通目が重なることはない。
        """
        panel = panel or self.active_panel
        if not self.can_send(panel) or self._client is None:
            return
        self.status_label.setText("送信中…")
        text = self.wire_text(panel)
        wpm = self.wpm_spin.value()
        # **ここで確定させる。** ``_on_sent`` が完了時に ``wpm_spin.value()`` を
        # 読み直すと、送信中に運用者が速度を変えたときに実際に送った速度と
        # 違う値を「送れた」記録にしてしまう (2026-08-13 最終レビュー Minor 6)。
        self._send_pending = (text, wpm)
        self._send_panel = panel
        self._worker = _SendWorker(self._client, text, wpm)
        self._worker.finished_ok.connect(self._on_sent)
        self._worker.failed.connect(self._on_send_failed)
        self._worker.start()
        self._update_buttons()

    def run_stop(self) -> None:
        if self._client is not None:
            self._client.stop()

    def _on_sent(self, result: SendResult) -> None:
        self._worker = None
        sent_pair = self._send_pending
        sent_panel = self._send_panel or self.active_panel
        self._send_pending = None
        self._send_panel = None
        # **確認済みの印は落とす** (編集したら送れない、を保つため)。
        # 代わりに「送り終えた」ほうへ移す。**落とすのは送った欄だけ** —
        # 他の欄の印まで落とすと、用意しておいた文が送れなくなる
        sent_panel._confirmed_text = None
        if not result.aborted and sent_pair is not None:
            # **最後まで送れたものだけ覚える。** 途中で止めたものは
            # 「送れた」とは言えない。**中止しても、それより前に完了した
            # 記録は消さない** — 打鍵側のお墨付きは今回の中止で無効に
            # なるわけではない (意図的、2026-08-13 最終レビュー Minor 7)。
            self._sent_ok.add(sent_pair)
        if result.aborted:
            if result.reason == "stop":
                # **運用者自身が止めた。** 接続は生きているので繋ぎ直さない。
                self.status_label.setText(
                    f"中止しました (運用者による中止) — {result.elements_sent} 要素まで送信"
                )
            elif result.reason == "lifeline":
                # **LAN が止まった。** 「自分が止めた」とは全く違う意味を持つので
                # 文言を分ける。何がどこまで出たか分からない状態を続けないよう
                # 接続を落とし、待機中の自動繋ぎ直しに任せる。
                self.status_label.setText(
                    f"中断しました — 打鍵側との通信が途切れました "
                    f"({result.elements_sent} 要素まで送信)"
                )
                if self._client is not None:
                    self._client.close()
                    self._client = None
            else:
                self.status_label.setText(f"中止しました ({result.elements_sent} 要素まで送信)")
        else:
            self.status_label.setText(
                f"送信しました — {result.elements_sent} 要素 / {result.seconds:.1f} 秒 / "
                f"ずれ 最大 {result.max_error_ms:.1f} ms"
            )
        self._update_buttons()

    def _on_send_failed(self, message: str) -> None:
        self._worker = None
        self._client = None
        self._send_pending = None
        self._send_panel = None
        # **打鍵側が居なくなった。** どの欄の確認結果も当てにならない
        for panel in self._panels:
            panel._confirmed_text = None
        self.status_label.setText(message)
        self._update_buttons()

    # ---- 有効・無効 ----
    def _update_buttons(self) -> None:
        sending = self._worker is not None and self._worker.isRunning()
        connected = self._client is not None
        # **繋ぎに行っている最中も押させない。** 押せてしまうと 2 本目の接続を
        # 作り、自分自身の 1 本目に busy で撥ねられる (``begin_connect`` 参照)。
        connecting = self._connect_worker is not None and self._connect_worker.isRunning()
        self.connect_btn.setEnabled(not sending and not connecting)
        # **送信中はどの欄も触らせない。** 打鍵器は 1 台なので 2 通目を
        # 重ねて出させないこと、そして打鍵中に本文が消えると、いま何が電波に
        # 出ているのか画面から分からなくなること、の 2 つの理由がある
        for panel in self._panels:
            panel.clear_btn.setEnabled(
                not sending and bool(panel.japanese_edit.toPlainText())
            )
            panel.check_btn.setEnabled(
                connected and not sending and bool(self.wire_text(panel))
            )
            panel.send_btn.setEnabled(self.can_send(panel) and not sending)
            # **どの欄の [中止] も同時に効く** (押した欄がどれでも同じ 1 通を止める)
            panel.stop_btn.setEnabled(sending)
        # ``[元に戻す]`` は**選んでいる欄**に効く (欄を変えると押せる/押せないも変わる)
        self.undo_template_btn.setEnabled(
            self.active_panel._state_before_template is not None
        )

    # ---- 後片付け ----
    def is_sending(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def shutdown(self) -> None:
        """接続・3 秒タイマ・送信スレッドを畳む. **何度呼んでも安全。**

        ``closeEvent`` / ``reject`` / ``finished`` のどれからでもここを通す。
        **``closeEvent`` にだけ置くと Esc で素通りする** (モジュールの docstring
        参照)。``main_window`` もダイアログを捨てる前にここを呼ぶ。
        """
        self._retry_timer.stop()
        connect_worker, self._connect_worker = self._connect_worker, None
        if connect_worker is not None:
            # **繋ぎに行っている最中に畳まれた。** 結果を受け取る先はもう無いので
            # シグナルを外し、スレッドが終わるのを待ってから、繋がってしまった
            # 接続をここで閉じる。閉じないと打鍵側は「使用中」のまま残り、次に
            # 開いたときに**自分自身の古い接続**に busy で撥ねられる。
            try:
                connect_worker.connected.disconnect()
                connect_worker.failed.disconnect()
            except (RuntimeError, TypeError):       # 既に外れている
                pass
            connect_worker.wait(_WORKER_WAIT_MS)
            connect_worker.close_client()
        if self.is_sending():
            # **送信中に閉じられても、スレッドを残さない。** [中止] と同じ経路で
            # 打鍵側へ停止を伝え、スレッドが実際に終わるのを待ってから閉じる。
            self.run_stop()
            self._worker.wait(_WORKER_WAIT_MS)     # type: ignore[union-attr]
        if self._worker is not None and not self._worker.isRunning():
            # **走っている QThread の参照は手放さない** (破棄すると落ちる)。
            # 待っても終わらなかったときだけ、持ったままにする
            self._worker = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def reject(self) -> None:
        """Esc の経路. **送信中は閉じない。**

        ここを素通りさせると、画面が消えるのに打鍵は最後まで続き、運用者が
        持っていた唯一のソフト中止手段 ([中止] ボタン) が画面ごと消える。
        """
        if self.is_sending():
            self.status_label.setText(
                "送信中です。止めるときは [中止] を押してください。"
            )
            return
        self.shutdown()
        super().reject()

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


__all__ = ["TxDialog"]

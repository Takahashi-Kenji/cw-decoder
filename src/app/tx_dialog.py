"""送信ダイアログ.

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

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from src.infer.net_audio import parse_endpoint
from src.infer.settings import AppSettings
from src.tokens.morse_tokens import DisplayMode
from src.tx.encoder import HORE, find_unsendable, needs_japanese_wrap, wrap_japanese
from src.tx.net_key import (
    NetKeyClient,
    NetKeyError,
    NetKeyRejected,
    SendResult,
)
from src.tx.profile import OperatorProfile, load_profile
from src.tx.protocol import DEFAULT_KEY_PORT
from src.tx.qso_fields import extract_fields
from src.tx.reading import to_sendable_kana
from src.tx.templates import (
    DEFAULT_TEMPLATES_PATH,
    fill,
    load_templates,
    profile_values,
    templates_for_mode,
)

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
# 使うだけだった。** 既定がオンなので、``「FT991」`` のように和文の無い本文を
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


class TxDialog(QDialog):
    """送信ダイアログ."""

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
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("送信")
        self._settings = settings
        self._profile = profile if profile is not None else load_profile()
        self._client_factory = client_factory
        self._client: NetKeyClient | None = None
        self._worker: _SendWorker | None = None
        # **確認が通った文字列。** これと今の文字列が一致するときだけ送れる
        self._confirmed_text: str | None = None
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
        self.connect_btn.clicked.connect(lambda: self.connect_to_keyer())
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
        layout.addLayout(fields)

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
        weather.addStretch(1)
        layout.addLayout(weather)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("型:"))
        self.template_combo = QComboBox()
        for template in self._templates:
            self.template_combo.addItem(template.name)
        picker.addWidget(self.template_combo, 1)
        self.use_template_btn = QPushButton("型を使う")
        # **``clicked`` は ``checked: bool`` を渡す。** 引数の食い違いは
        # 過去にこのリポジトリで実際に踏んでいる
        self.use_template_btn.clicked.connect(lambda: self.apply_template())
        picker.addWidget(self.use_template_btn)
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

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("日本語 (漢字かな交じりで可):"))
        source_row.addStretch(1)
        # **1 回の交信で何度も打ち直す。** 全選択して消すのは手数が多い。
        # 消すのは本文だけ — 相手コール・RST・天気は交信のあいだ変わらないので
        # 巻き込むと打ち直しになる。
        self.clear_btn = QPushButton("クリア")
        self.clear_btn.setToolTip("日本語ボックスを空にします (相手・RST・天気はそのまま)")
        self.clear_btn.clicked.connect(lambda: self.clear_text())
        source_row.addWidget(self.clear_btn)
        layout.addLayout(source_row)

        self.japanese_edit = QPlainTextEdit()
        self.japanese_edit.textChanged.connect(self.refresh_kana)
        layout.addWidget(self.japanese_edit)

        self.wrap_check = QCheckBox("和文をホレ/ラタで囲む")
        self.wrap_check.setChecked(True)
        self.wrap_check.toggled.connect(self.refresh_kana)
        layout.addWidget(self.wrap_check)

        layout.addWidget(QLabel("送信される文字:"))
        self.kana_view = QPlainTextEdit()
        self.kana_view.setReadOnly(True)
        layout.addWidget(self.kana_view)

        self.status_label = QLabel("未接続")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.check_btn = QPushButton("確認")
        self.check_btn.clicked.connect(self.run_check)
        buttons.addWidget(self.check_btn)
        buttons.addStretch(1)
        self.send_btn = QPushButton("送信")
        self.send_btn.clicked.connect(self.run_send)
        buttons.addWidget(self.send_btn)
        self.stop_btn = QPushButton("中止")
        self.stop_btn.clicked.connect(self.run_stop)
        buttons.addWidget(self.stop_btn)
        layout.addLayout(buttons)

    # ---- 文字 ----
    def wire_text(self) -> str:
        """LAN に流す確定テキスト."""
        return self.kana_view.toPlainText().strip()

    def refresh_kana(self) -> None:
        """日本語をカタカナに直し、**関門を閉じ直す**.

        送信できるかの判定は **``encoder.find_unsendable`` を使う**。
        ``reading`` 側の ``bad_chars`` は和文表だけで照合しており、
        **コールサインを含む文が必ず赤くなっていた** (設計書 §3)。
        打鍵側 (``key_server.prepare``) と同じ規則で判定しないと、
        画面と実際の可否が食い違う。
        """
        source = self.japanese_edit.toPlainText()
        result = to_sendable_kana(source, self._profile)
        wrap_on = self.wrap_check.isChecked()
        text = wrap_japanese(result.text) if wrap_on else result.text
        self.kana_view.setPlainText(text)
        self._confirmed_text = None            # 編集したら確認をやり直す
        self._show_text_warnings(text, wrap_on=wrap_on, unwrapped=result.text)
        self._update_buttons()

    def _show_text_warnings(
        self, wire_text: str, *, wrap_on: bool, unwrapped: str
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
        運用者が明示的に操作するチェックボックスであり、既定がオンなので、
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
            self.status_label.setText("\n".join(warnings))
        elif self.status_label.text().startswith(_TEXT_WARNING_PREFIXES):
            self.status_label.clear()

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

    def clear_text(self) -> None:
        """日本語ボックスを空にする.

        **消すのは本文だけ。** 相手コール・相手名前・RST・天気・気温は
        交信のあいだ変わらないので、巻き込むと打ち直しになる。

        空にすれば ``textChanged`` → ``refresh_kana`` が走り、**確認は
        やり直しになる** (送信される文字も空になるので ``[確認]`` も押せない)。

        **送信中は押せない** (``_update_buttons``)。打鍵している最中に本文が
        消えると、何を送っているのか画面から分からなくなる。
        """
        self.japanese_edit.clear()

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
        self.template_combo.clear()
        for template in self._templates:
            self.template_combo.addItem(template.name)

    def apply_template(self) -> None:
        """選んだ型に欄を差し込み、日本語ボックスへ入れる.

        **欄を差し込んでから入れる。** 逆にすると ``{相手コール}`` が
        カナ変換器を通って壊れる (設計書 §6.1)。

        **囲むかどうかは中身で決める。** ``wrap_check`` の既定はオンなので、
        合わせずに欧文の型を流すと中の欧文が丸ごと ``{HORE}``/``{RATA}`` に
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
        # setChecked は値が変わったときだけ toggled (→ refresh_kana) を
        # 起こす。その後の setPlainText でも textChanged (→ refresh_kana)
        # が起きるので、多くても 2 回で確定する (どちらも副作用は無い)。
        self.wrap_check.setChecked(HORE not in filled and needs_japanese_wrap(converted))
        # setPlainText が textChanged を起こし refresh_kana が走る
        # (関門もそこで閉じ直り、警告もそこで出る)
        self.japanese_edit.setPlainText(filled)

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

    def can_send(self) -> bool:
        """**確認が通った文字列と今の文字列が一致しているか。**"""
        text = self.wire_text()
        return bool(text) and self._confirmed_text == text and self._client is not None

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
        self.connect_to_keyer(quiet=True)

    def connect_to_keyer(self, quiet: bool = False) -> None:
        """打鍵側へ繋ぐ.

        Args:
            quiet: 真なら失敗しても画面に書かない (自動の繋ぎ直しから呼ぶため。
                3 秒おきに赤い文字が書き換わると読めない)。
        """
        endpoint = self.endpoint_edit.text().strip()
        if not endpoint:
            if not quiet:
                self.status_label.setText("打鍵側の host:port を入れてください。")
            self._update_buttons()
            return
        try:
            host, port = parse_endpoint(endpoint, default_port=DEFAULT_KEY_PORT)
        except ValueError as exc:
            if not quiet:
                self.status_label.setText(f"打鍵側の指定が読めません: {exc}")
            self._update_buttons()
            return

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

        client = self._client_factory(host, port)
        try:
            hello = client.connect()
        except NetKeyRejected as exc:
            # **``NetKeyError`` より先に捕まえる。** busy は「繋がらない」の
            # 一般文言に埋もれさせず、理由が分かる文言にする。
            self._client = None
            if not quiet:
                if exc.code == "busy":
                    self.status_label.setText(
                        f"打鍵側は今、別の運用者が使用中です。しばらく待って再接続してください。({exc})"
                    )
                else:
                    self.status_label.setText(str(exc))
            self._update_buttons()
            return
        except NetKeyError as exc:
            self._client = None
            if not quiet:
                self.status_label.setText(str(exc))
            self._update_buttons()
            return

        self._client = client
        self._settings.tx_endpoint = endpoint
        message = f"接続しました — {hello.describe_wiring()}"
        if not hello.fingerprint_matches:
            # **静かな食い違いを見える警告にする** (設計書 §2.1)
            message += "\n**警告: 符号表が両 PC で違います。** リポジトリを揃えてください。"
        self.status_label.setText(message)
        self._confirmed_text = None
        self._update_buttons()

    def run_check(self) -> None:
        """**打鍵しない検査。** これが通って初めて送れる."""
        if self._client is None:
            return
        text = self.wire_text()
        if not text:
            return
        try:
            result = self._client.check(text, self.wpm_spin.value())
        except NetKeyRejected as exc:
            self._confirmed_text = None
            detail = "".join(bad["char"] for bad in exc.unsendable)
            self.status_label.setText(f"{exc}: {detail}" if detail else str(exc))
        except NetKeyError as exc:
            self._client = None
            self._confirmed_text = None
            self.status_label.setText(str(exc))
        else:
            self._confirmed_text = text
            self.status_label.setText(
                f"確認しました — {result.chars} 文字 / {result.elements} 要素 / {result.seconds:.1f} 秒"
            )
        self._update_buttons()

    def run_send(self) -> None:
        if not self.can_send() or self._client is None:
            return
        self.status_label.setText("送信中…")
        self._worker = _SendWorker(self._client, self.wire_text(), self.wpm_spin.value())
        self._worker.finished_ok.connect(self._on_sent)
        self._worker.failed.connect(self._on_send_failed)
        self._worker.start()
        self._update_buttons()

    def run_stop(self) -> None:
        if self._client is not None:
            self._client.stop()

    def _on_sent(self, result: SendResult) -> None:
        self._worker = None
        # **送ったものは確認済みでなくなる。** 同じ文を続けて送るときも確認から
        self._confirmed_text = None
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
        self._confirmed_text = None
        self.status_label.setText(message)
        self._update_buttons()

    # ---- 有効・無効 ----
    def _update_buttons(self) -> None:
        sending = self._worker is not None and self._worker.isRunning()
        connected = self._client is not None
        self.connect_btn.setEnabled(not sending)
        # **送信中は消させない。** 打鍵中に本文が消えると、いま何が電波に
        # 出ているのか画面から分からなくなる
        self.clear_btn.setEnabled(not sending and bool(self.japanese_edit.toPlainText()))
        self.check_btn.setEnabled(connected and not sending and bool(self.wire_text()))
        self.send_btn.setEnabled(self.can_send() and not sending)
        self.stop_btn.setEnabled(sending)

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

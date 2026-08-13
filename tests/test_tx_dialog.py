"""送信ダイアログのテスト.

**経歴は必ず明示的に渡すこと。** ``profile=None`` は本番の既定であり
``~/.cw-decorder/operator.json`` を読みに行く。テストで使うと**利用者の
実ファイルに結果が左右される** — 実際に、運用者が経歴を書いた日に
``? DE ?`` を期待していたテストが ``? DE JH0ILL`` になって落ちた
(2026-08-12)。空の経歴が要るなら ``OperatorProfile()`` を渡す。

**関門を機械で確かめる。** 「確認するまで [送信] を押せない」は運用者が
決めた安全の要であり、人手の確認に任せない。

pykakasi 抜きで日本語を試す
----------------------------
この環境には ``pykakasi`` が入っていない (``tests/test_tx_reading.py`` も同じ
理由で一部失敗している)。``src.tx.reading._kana_words`` は呼ぶたびに
``import pykakasi`` するため、日本語 (ひらがな・漢字) を含む文字列を
``to_sendable_kana`` に通すテストは軒並み ``ModuleNotFoundError`` になる。

``src/tx/reading.py`` はこのタスクで変更禁止なので、テスト側で
``_kana_words`` をひらがな→カタカナの単純な Unicode オフセット
(``+0x60``) に差し替える。ここで使う文はすべてひらがなのみ (漢字なし) で
選んでおり、このオフセットで実際の変換と同じ結果になる。
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from src.infer.settings import AppSettings
from src.tx.net_key import CheckResult, Hello, NetKeyRejected, SendResult
from src.tx.profile import BilingualField, OperatorProfile
from src.tx.templates import ReplyTemplate, save_templates


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _stub_pykakasi(monkeypatch):
    """pykakasi の代わりにひらがな→カタカナのオフセット変換を使う (上記docstring参照)."""

    def _fake_kana_words(text: str) -> list[str]:
        converted = "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)
        return [converted] if converted.strip() else []

    monkeypatch.setattr("src.tx.reading._kana_words", _fake_kana_words)


class FakeClient:
    """打鍵側のふり. **本物のソケットを使わない** (画面の関門だけを見る)."""

    def __init__(self, host: str, port: int = 45679, **kwargs: object) -> None:
        self.host = host
        self.port = port
        self.checked: list[tuple[str, float]] = []
        self.sent: list[tuple[str, float]] = []
        self.stopped = False
        self.closed = False
        self.reject_with: NetKeyRejected | None = None
        # connect() 専用の撥ね。check()/send() 用の reject_with とは独立させる
        # (connect() は生成直後に呼ばれるので、生成後に仕込む reject_with では
        # 間に合わない)。
        self.connect_reject: NetKeyRejected | None = None

    def connect(self) -> Hello:
        if self.connect_reject is not None:
            raise self.connect_reject
        from src.tx.fingerprint import tokens_fingerprint

        return Hello(1, tokens_fingerprint(), {"port": "COM3", "key": "DTR", "ptt": "RTS"}, False)

    def close(self) -> None:
        self.closed = True

    def check(self, text: str, wpm: float) -> CheckResult:
        if self.reject_with is not None:
            raise self.reject_with
        self.checked.append((text, wpm))
        return CheckResult(chars=len(text), elements=100, seconds=12.3)

    def send(self, text: str, wpm: float) -> SendResult:
        self.sent.append((text, wpm))
        return SendResult(100, False, False, 12.3, 1.0, 0.3)

    def stop(self) -> None:
        self.stopped = True


class SlowClient(FakeClient):
    """止める (か疑似 LAN 切断) まで打ち続けるふり.

    ``closeEvent`` が送信スレッドを残さず後始末することを見るために使う。
    """

    def __init__(self, host: str, port: int = 45679, **kwargs: object) -> None:
        super().__init__(host, port, **kwargs)
        self._release = threading.Event()

    def send(self, text: str, wpm: float) -> SendResult:
        self._release.wait(timeout=2.0)
        self.sent.append((text, wpm))
        return SendResult(50, True, False, 1.0, 1.0, 0.3, "stop")

    def stop(self) -> None:
        self.stopped = True
        self._release.set()


# ``build()``/``build_with_factory()`` (どちらも fixture ではないプレーン関数) が
# 使う隔離ディレクトリ。モジュール単位の autouse フィクスチャ (下) が
# セットアップ時にここへ入れ、モジュールのテストがすべて終わったら消して None に戻す。
_isolated_templates_dir: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _isolated_templates_dir_fixture():
    """``build()`` 系が使う隔離ディレクトリの寿命を管理する.

    **``tempfile.mkdtemp()`` は自動で消えない。** 呼び出しのたびに
    ``%TEMP%`` へディレクトリが残り、テストを走らせるたびに無制限に増える
    (2026-08-11 レビューで実測: 1 回の実行で 31 個増加、環境には過去分が
    281 個溜まっていた)。``tempfile.TemporaryDirectory`` はコンテキストを
    抜けると自動で消えるので、モジュール単位の ``autouse`` フィクスチャに
    包み、このファイルのテストが動いている間だけ生かす。

    ``build()``/``build_with_factory()`` は fixture ではないプレーン関数で
    32 箇所から呼ばれており、``tmp_path`` を全呼び出し元に配線するより、
    モジュールで 1 つだけ隔離ディレクトリを作って使い回すほうが変更が
    小さい。型ファイルの中身を書くテストは別途 ``tmp_path`` を直接使う
    (``build_with_templates`` 等) ので、ここで使い回しても衝突しない。
    """
    global _isolated_templates_dir
    with tempfile.TemporaryDirectory(prefix="cw-decoder-test-") as tmp:
        _isolated_templates_dir = Path(tmp)
        yield
    _isolated_templates_dir = None


def _isolated_templates_path() -> Path:
    """存在しない一時パスを返す (型を使わないテスト用).

    ``templates_path`` の既定は運用者の実ファイル (``~/.cw-decorder/templates.json``)
    を指す。型を検証しないテストがこれを渡さずに ``TxDialog`` を作ると、
    実行環境のホームディレクトリの中身にテスト結果が依存してしまう
    (``src/tx/templates.py`` 自身が「テストは必ず一時パスを渡すこと」と
    明記している原則に反する。2026-08-11 レビューで指摘)。

    ``load_templates`` は存在しないパスを渡されると即座に空リストを返すので、
    実際にファイルを作る必要は無い。ディレクトリ自体は上の autouse
    フィクスチャが後片付けする。
    """
    assert _isolated_templates_dir is not None, "_isolated_templates_dir_fixture が未セットアップ"
    return _isolated_templates_dir / "templates.json"


def build(qapp, **overrides):
    from src.app.tx_dialog import TxDialog

    # **`**overrides` を後に置く。** 先に置くと tx_endpoint が二重に渡り
    # TypeError になる
    settings = AppSettings(**{"tx_endpoint": "127.0.0.1:45679", "tx_wpm": 20.0, **overrides})
    clients: list[FakeClient] = []

    def factory(host, port=45679, **kwargs):
        client = FakeClient(host, port, **kwargs)
        clients.append(client)
        return client

    dialog = TxDialog(
        settings, profile=OperatorProfile(), client_factory=factory, templates_path=_isolated_templates_path()
    )
    return dialog, clients


def build_with_factory(qapp, factory, **overrides):
    """クライアントの型を差し替えたいテスト用 (busy 撥ね・低速クライアント等)."""
    from src.app.tx_dialog import TxDialog

    settings = AppSettings(**{"tx_endpoint": "127.0.0.1:45679", "tx_wpm": 20.0, **overrides})
    return TxDialog(
        settings, profile=OperatorProfile(), client_factory=factory, templates_path=_isolated_templates_path()
    )


def _isolate_operator_files(monkeypatch) -> None:
    """``main_window`` 経由で ``TxDialog`` を作るテストを実ファイルから切り離す.

    ``main_window._open_tx_dialog`` は ``templates_path`` も ``profile`` も
    渡さない (運用ではそれが正しい) ので、既定のまま作ると
    ``~/.cw-decorder/`` の**利用者の実ファイル**を読む。テスト結果が実行環境の
    ホームディレクトリの中身に左右されるので、読み込み関数ごと差し替える
    (``src/tx/templates.py`` が明記している「利用者の実ファイルを壊さない・
    見ない」の約束)。
    """
    monkeypatch.setattr("src.app.tx_dialog.load_templates", lambda path: [])
    monkeypatch.setattr("src.app.tx_dialog.load_profile", lambda *a, **kw: OperatorProfile())


def wait_for_worker(dialog) -> None:
    """送信ワーカーの完了を待ち、Qt のシグナル配送のためイベントを回す."""
    worker = dialog._worker
    assert worker is not None
    worker.wait(2000)
    QApplication.processEvents()


def test_日本語がカタカナに直る(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    assert "コンニチハ" in dialog.kana_view.toPlainText()


def test_和文がホレとラタで囲まれる(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.wrap_check.setChecked(True)
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    text = dialog.wire_text()
    assert text.startswith("{HORE}")
    assert text.endswith("{RATA}")


def test_確認する前は送信を押せない(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    assert dialog.can_send() is False
    assert dialog.send_btn.isEnabled() is False


def test_確認すると送信を押せる(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    assert clients[0].checked
    assert dialog.can_send() is True
    assert dialog.send_btn.isEnabled() is True


def test_編集すると送信が無効に戻る(qapp) -> None:
    """**確認していない文字列は送れない。**"""
    dialog, _ = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    assert dialog.can_send() is True

    dialog.japanese_edit.setPlainText("こんばんは")
    dialog.refresh_kana()
    assert dialog.can_send() is False
    assert dialog.send_btn.isEnabled() is False


def test_ホレラタの切替も確認を無効にする(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    dialog.wrap_check.setChecked(not dialog.wrap_check.isChecked())
    dialog.refresh_kana()
    assert dialog.can_send() is False


def test_送れない文字があると送信できない(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    clients[0].reject_with = NetKeyRejected(
        "unsendable", "送信できない文字", [{"index": 0, "char": "髙"}]
    )
    dialog.japanese_edit.setPlainText("髙")
    dialog.refresh_kana()
    dialog.run_check()
    assert dialog.can_send() is False
    assert "髙" in dialog.status_label.text()


def test_確認で撥ねられたら送り終えた記録も消える(qapp) -> None:
    """**Important 4 (2026-08-13 最終レビュー): 撥ねられたら記録も捨てる。**

    以前送れた記録だけを頼りに `[送信]` を有効なままにすると、打鍵側が
    「もう通らない」と言っているのに気づけない。撥ねられたら
    ``_sent_ok`` からもその (文字列, 速度) を消す。
    """
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    dialog._send_pending = (dialog.wire_text(), dialog.wpm_spin.value())
    dialog._on_sent(SendResult(100, False, False, 12.3, 1.0, 0.3))
    assert dialog.can_send() is True

    clients[0].reject_with = NetKeyRejected(
        "unsendable", "送信できない文字", [{"index": 0, "char": "こ"}]
    )
    dialog.run_check()

    assert dialog.can_send() is False


def test_未接続なら確認も送信も押せない(qapp) -> None:
    dialog, _ = build(qapp)
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    assert dialog.check_btn.isEnabled() is False
    assert dialog.send_btn.isEnabled() is False


def test_送信先が空なら繋ぎに行かない(qapp) -> None:
    dialog, clients = build(qapp, tx_endpoint="")
    dialog.connect_to_keyer()
    assert clients == []
    assert dialog.check_btn.isEnabled() is False


def test_待機中は自動で繋ぎ直す(qapp) -> None:
    """打鍵側を後から起こしても繋がる (設計書 §8.3)."""
    dialog, clients = build(qapp)
    assert clients == []
    dialog.retry_tick()
    assert len(clients) == 1


def test_繋がっていれば繋ぎ直さない(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    assert len(clients) == 1
    dialog.retry_tick()
    assert len(clients) == 1


def test_送信先が空なら自動でも繋ぎに行かない(qapp) -> None:
    dialog, clients = build(qapp, tx_endpoint="")
    dialog.retry_tick()
    assert clients == []


# ---- ここから: brief より新しい決定事項の検証 ----


def test_busyで撥ねられると理由が分かる文言が出る(qapp) -> None:
    """打鍵側が別の運用者と繋がっているとき、汎用エラーと区別できる文言を出す.

    ``NetKeyRejected`` の例外メッセージ自体に「使用中」「他の」という語を
    **含めない** (テスト文言をこう選ぶ)。そうしないと、busy 専用の分岐を
    削除して汎用 ``except NetKeyError`` の ``str(exc)`` だけにしても
    このアサーションが偶然通ってしまい、レビューで指摘された「壊しても
    通る」空洞テストになる。ここで見る文言 (``しばらく待って再接続して
    ください``) は busy 専用分岐でしか書かれない。
    """

    class BusyClient(FakeClient):
        def connect(self) -> Hello:
            raise NetKeyRejected("busy", "reject: already connected")

    dialog = build_with_factory(qapp, BusyClient)
    dialog.connect_to_keyer()
    assert "しばらく待って再接続してください" in dialog.status_label.text()
    assert dialog.check_btn.isEnabled() is False
    assert dialog.send_btn.isEnabled() is False


def test_busy以外の接続エラーは汎用文言になる(qapp) -> None:
    """``NetKeyRejected`` (busy) と汎用 ``NetKeyError`` を混同しない."""
    from src.tx.net_key import NetKeyError

    class BrokenClient(FakeClient):
        def connect(self) -> Hello:
            raise NetKeyError("打鍵側に繋がりません (テスト)")

    dialog = build_with_factory(qapp, BrokenClient)
    dialog.connect_to_keyer()
    assert "しばらく待って再接続してください" not in dialog.status_label.text()
    assert "繋がりません" in dialog.status_label.text()


def test_busy以外のNetKeyRejectedは撥ねられた理由がそのまま出る(qapp) -> None:
    """``code`` が ``busy`` でなければ busy 専用文言は使わない (else 分岐)."""

    class OtherRejectClient(FakeClient):
        def connect(self) -> Hello:
            raise NetKeyRejected("bad_request", "reject: malformed request")

    dialog = build_with_factory(qapp, OtherRejectClient)
    dialog.connect_to_keyer()
    assert "しばらく待って再接続してください" not in dialog.status_label.text()
    assert "reject: malformed request" in dialog.status_label.text()


def test_接続を繰り返すと古い接続を閉じてから繋ぎ直す(qapp) -> None:
    """**Critical:** [接続] を連打しても自己ロックしない.

    打鍵側は同時 1 接続しか受けず、待機中には期限を掛けない (key_server.py)。
    古い接続を閉じずに新しい接続を作ると、古い接続が「使用中」のまま永久に
    残り、以後の接続がすべて busy で撥ねられる (打鍵側 CLI の再起動でしか
    回復しない)。
    """
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    assert len(clients) == 1
    assert clients[0].closed is False

    dialog.connect_to_keyer()
    assert len(clients) == 2
    assert clients[0].closed is True            # 古い接続を畳んでから繋ぎ直す
    assert clients[1].closed is False
    assert dialog._client is clients[1]


def test_符号表の指紋が食い違うと警告が出る(qapp) -> None:
    """設計書 §2.1: 静かな食い違いを見える警告にする."""

    class MismatchedClient(FakeClient):
        def connect(self) -> Hello:
            return Hello(
                1,
                "this-does-not-match-the-real-fingerprint",
                {"port": "COM3", "key": "DTR", "ptt": "RTS"},
                False,
            )

    dialog = build_with_factory(qapp, MismatchedClient)
    dialog.connect_to_keyer()
    assert "符号表が両 PC で違います" in dialog.status_label.text()


def test_中止理由_運用者による停止と分かる(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()

    clients[0].send = lambda text, wpm: SendResult(30, True, False, 5.0, 1.0, 0.3, "stop")
    dialog.run_send()
    wait_for_worker(dialog)

    status = dialog.status_label.text()
    # 運用者自身が止めたと分かる、この分岐でしか出ない文言であること
    # (``reason`` の分岐そのものを削っても通ってしまう空洞テストにしない)
    assert "運用者による中止" in status
    assert "打鍵側との通信が途切れました" not in status
    assert dialog._client is not None          # 運用者の中止では接続を切らない
    assert dialog.can_send() is False           # 送った以上、同じ文は確認からやり直す


def test_中止理由_LAN切断だと分かり接続が切れる(qapp) -> None:
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()

    clients[0].send = lambda text, wpm: SendResult(10, True, False, 2.0, 1.0, 0.3, "lifeline")
    dialog.run_send()
    wait_for_worker(dialog)

    status = dialog.status_label.text()
    # LAN が止まったと分かる、この分岐でしか出ない文言であること
    assert "打鍵側との通信が途切れました" in status
    assert "運用者による中止" not in status
    # **LAN が止まった扱い。** 送信中は繋ぎ直さない設計なので、送信が終わった
    # 今は繋ぎ直しに回すため接続を落とす
    assert dialog._client is None
    assert dialog.check_btn.isEnabled() is False
    assert dialog.send_btn.isEnabled() is False


def test_中止理由が無ければ汎用文言になる(qapp) -> None:
    """``reason`` が ``stop``/``lifeline`` のどちらでもない (想定外) 場合の保険."""
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()

    clients[0].send = lambda text, wpm: SendResult(5, True, False, 1.0, 1.0, 0.3, None)
    dialog.run_send()
    wait_for_worker(dialog)

    status = dialog.status_label.text()
    assert "運用者による中止" not in status
    assert "打鍵側との通信が途切れました" not in status
    assert "中止しました" in status
    assert dialog._client is not None           # 汎用中止では接続を切らない


def test_送信が終わっても同じ文字列なら送り直せる(qapp) -> None:
    """**2026-08-12 に振る舞いを変えた。**

    以前は「送信後は確認からやり直す」だった。交信中に同じ文を送り直す
    たびに確認の往復を待つのが、実運用で一番効く無駄だったため
    (運用者の要望)。**関門を外したのではない** — 打鍵側が「その文字列は
    この速度で送れる」と答えた事実を覚えているだけで、文字列か速度が
    変われば確認からやり直す (下の 2 つのテスト)。
    """
    dialog, _clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    assert dialog.can_send() is True

    dialog.run_send()
    wait_for_worker(dialog)

    assert dialog.can_send() is True
    assert dialog.send_btn.isEnabled() is True


def test_速度を変えると確認が無効に戻る(qapp) -> None:
    """関門の 3 番目「**何秒間 電波が出るのか**」は速度で変わる.

    20 WPM で確認した 6.3 秒は、5 WPM では 4 倍の長さの電波になる。
    以前は ``can_send()`` がテキストしか見ておらず、速度だけ変えても
    [送信] が有効なままだった。
    """
    dialog, _clients = build(qapp)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    assert dialog.can_send() is True

    dialog.wpm_spin.setValue(5.0)

    assert dialog.can_send() is False
    assert dialog.send_btn.isEnabled() is False


def test_速度が設定に書き戻される(qapp) -> None:
    """**毎回入れ直しにしない。** 設定の保存は main_window が閉じた後に行う."""
    dialog, _clients = build(qapp)
    dialog.wpm_spin.setValue(25.0)
    assert dialog._settings.tx_wpm == pytest.approx(25.0)


def test_Escで閉じると接続もタイマも畳まれる(qapp) -> None:
    """**``QDialog`` は Esc で ``reject()``。``closeEvent`` は呼ばれない。**

    後片付けが ``closeEvent`` にしか無かったため、Esc で画面が消えても接続と
    3 秒タイマが生き残り、次に開くと**自分自身の古い接続**に busy で撥ねられて
    「別の運用者が使用中です」という嘘の理由が出ていた。
    """
    dialog, clients = build(qapp)
    dialog.connect_to_keyer()
    assert dialog._client is not None

    dialog.reject()                       # Esc と同じ経路

    assert dialog._client is None
    assert clients[0].closed is True
    assert dialog._retry_timer.isActive() is False


def test_送信中のEscでは閉じず中止を促す(qapp) -> None:
    """**画面ごと消してはいけない。** [中止] が運用者の唯一のソフト中止手段."""
    dialog = build_with_factory(qapp, SlowClient)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    dialog.run_send()
    assert dialog._worker is not None and dialog._worker.isRunning()

    dialog.reject()                       # Esc

    assert dialog._worker.isRunning()     # 打鍵は続いている
    assert dialog._client is not None     # 接続も生きている
    assert dialog.stop_btn.isEnabled() is True
    assert "中止" in dialog.status_label.text()

    dialog.run_stop()                     # 後始末
    wait_for_worker(dialog)


def test_送信ダイアログはメイン画面を固めない(qapp, monkeypatch, tmp_path) -> None:
    """**開いている間もメイン画面を操作できる** (運用者の要望、2026-08-12).

    受信を見ながら返信を書けること、送信中でもデコードを止められることが目的。
    ``exec()`` はメイン画面を固めるので使わない。
    """
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine

    _isolate_operator_files(monkeypatch)
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(tx_endpoint="127.0.0.1:45679"),
        config_path=tmp_path / "settings.json",
    )
    try:
        window._open_tx_dialog()
        dialog = window._tx_dialog

        assert dialog is not None
        assert dialog.isModal() is False
        assert window.isEnabled() is True          # 固まっていない
    finally:
        window.close()


def test_送信ダイアログの二枚目は開かない(qapp, monkeypatch, tmp_path) -> None:
    """**打鍵側は 1 つしか繋がない。**

    画面が固まらなくなった以上 [送信…] は何度でも押せる。二枚目を開くと、
    後から開いたほうが**自分自身の接続**に busy で撥ねられる。
    """
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine

    _isolate_operator_files(monkeypatch)
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(tx_endpoint="127.0.0.1:45679"),
        config_path=tmp_path / "settings.json",
    )
    try:
        window._open_tx_dialog()
        first = window._tx_dialog
        window._open_tx_dialog()

        assert window._tx_dialog is first
    finally:
        window.close()


def test_送信ダイアログを閉じると接続もタイマも残らない(qapp, monkeypatch, tmp_path) -> None:
    """畳まないと、閉じた後も接続と 3 秒タイマが生き続け、次に開いた
    ダイアログが自分自身の古い接続に busy で撥ねられる。

    **後片付けは ``finished`` に繋いである。** ``show()`` の後ろに書くと
    開いた瞬間に走ってしまう。
    """
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine

    _isolate_operator_files(monkeypatch)
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(tx_endpoint="127.0.0.1:45679"),
        config_path=tmp_path / "settings.json",
    )
    try:
        window._open_tx_dialog()
        dialog = window._tx_dialog
        assert dialog._retry_timer.isActive() is True     # 開いている間は動く

        dialog.reject()                                    # Esc で閉じる

        assert window._tx_dialog is None
        assert dialog._retry_timer.isActive() is False
        assert dialog._client is None
    finally:
        window.close()


def test_開いた直後に畳んでいない(qapp, monkeypatch, tmp_path) -> None:
    """``exec()`` の後ろの後片付けをそのまま ``show()`` の後ろに残すと、
    **開いた瞬間に接続もタイマも畳んでしまう。**
    """
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine

    _isolate_operator_files(monkeypatch)
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(tx_endpoint="127.0.0.1:45679"),
        config_path=tmp_path / "settings.json",
    )
    try:
        window._open_tx_dialog()
        assert window._tx_dialog is not None
        assert window._tx_dialog._retry_timer.isActive() is True
    finally:
        window.close()


def test_メイン画面を閉じたら送信ダイアログも畳む(qapp, monkeypatch, tmp_path) -> None:
    """**置き去りにしない。** モードレスなので、メイン画面だけ閉じると
    打鍵の接続と 3 秒タイマが残る。
    """
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine

    _isolate_operator_files(monkeypatch)
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(tx_endpoint="127.0.0.1:45679"),
        config_path=tmp_path / "settings.json",
    )
    window._open_tx_dialog()
    dialog = window._tx_dialog

    window.close()

    assert dialog._retry_timer.isActive() is False
    assert dialog._client is None


def test_送信ダイアログには画面の今のモードを渡す(qapp, monkeypatch, tmp_path) -> None:
    """**I1 の後半: 設定の ``mode`` は画面を閉じるときにしか書き戻らない。**

    ``self._settings.mode`` を渡していたため、画面を和文にしていても設定が
    欧文のままなら和文の型が一覧から消えていた (2026-08-11 レビュー)。
    """
    from src.app.main_window import CWDecoderWindow
    from src.infer.engine import InferenceEngine

    _isolate_operator_files(monkeypatch)
    window = CWDecoderWindow(
        InferenceEngine.untrained(device="cpu"),
        AppSettings(mode="european", tx_endpoint="127.0.0.1:45679"),
        config_path=tmp_path / "settings.json",
    )
    try:
        window.mode_combo.setCurrentIndex(1)          # 画面だけ和文にする
        assert window._settings.mode == "european"    # 設定はまだ欧文のまま
        window._open_tx_dialog()
        assert window._tx_dialog._mode == "japanese"
        window._tx_dialog.reject()                    # 二枚目を作れるよう閉じる

        window.mode_combo.setCurrentIndex(2)          # 自動
        window._open_tx_dialog()
        assert window._tx_dialog._mode == "auto"
    finally:
        window.close()



def test_ダイアログを閉じても送信スレッドが残らない(qapp) -> None:
    """送信中に閉じられても、スレッドを残さず打鍵側へ停止を伝える."""
    dialog = build_with_factory(qapp, SlowClient)
    dialog.connect_to_keyer()
    dialog.japanese_edit.setPlainText("こんにちは")
    dialog.refresh_kana()
    dialog.run_check()
    dialog.run_send()
    assert dialog._worker is not None
    assert dialog._worker.isRunning()

    dialog.close()

    assert dialog._worker is None or not dialog._worker.isRunning()


class TestSendableCheck:
    """送信できない文字の判定.

    **画面は打鍵側と同じ規則で判定しなければならない。** 以前は
    ``reading.find_bad_chars`` (和文表だけで照合) を使っており、
    **コールサインを含む文が必ず赤くなっていた** (2026-08-11 に発覚)。
    定型交換は必ずコールサインを含むので、これでは使えない。
    """

    def test_コールサインを含む和文が赤くならない(self, qapp) -> None:
        dialog, _ = build(qapp)
        dialog.wrap_check.setChecked(False)
        dialog.japanese_edit.setPlainText("JA1ABC DE JH0ILL {HORE}コンニチハ{RATA} K")
        dialog.refresh_kana()
        assert "送信できない" not in dialog.status_label.text()

    def test_欧文だけの文が赤くならない(self, qapp) -> None:
        dialog, _ = build(qapp)
        dialog.wrap_check.setChecked(False)
        dialog.japanese_edit.setPlainText("CQ CQ DE JH0ILL K")
        dialog.refresh_kana()
        assert "送信できない" not in dialog.status_label.text()

    def test_本当に送れない文字は赤くなる(self, qapp) -> None:
        """**歯止めを外さない。** 符号表に無い文字は今までどおり弾く."""
        dialog, _ = build(qapp)
        dialog.japanese_edit.setPlainText("コンニチハ+")
        dialog.refresh_kana()
        assert "送信できない" in dialog.status_label.text()
        assert "+" in dialog.status_label.text()

    def test_直すと警告が消える(self, qapp) -> None:
        """**直したのに警告が残ると、運用者は「まだ送れない」と誤解する.**

        2026-08-11 レビューで指摘。以前は ``if bad:`` のときだけ書き換えて
        おり、送れない文字を消しても警告が残ったままだった。
        """
        dialog, _ = build(qapp)
        dialog.japanese_edit.setPlainText("コンニチハ+")
        dialog.refresh_kana()
        assert "送信できない" in dialog.status_label.text()

        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.refresh_kana()
        assert "送信できない" not in dialog.status_label.text()

    def test_他の表示を消さない(self, qapp) -> None:
        """**肝心な歯止め。** 警告を消すとき、接続結果などの別の文言を巻き込まない.

        無条件に ``status_label.clear()`` すると、``refresh_kana`` を呼ぶ
        たびに接続結果・確認結果・中止理由といった他の表示まで消えてしまう。
        この警告自身が出した文言のときだけ消さなければならない。
        """
        dialog, _ = build(qapp)
        dialog.wrap_check.setChecked(False)     # 欧文だけの文をホレ/ラタで囲ませない
        dialog.connect_to_keyer()
        message = dialog.status_label.text()
        assert message.startswith("接続しました")

        dialog.japanese_edit.setPlainText("CQ CQ DE JH0ILL K")
        dialog.refresh_kana()
        assert dialog.status_label.text() == message


# ---- ここから: Task 5 (型と欄) ----


class TestTemplates:
    def test_モードに合う型だけ一覧に出る(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(
            [
                ReplyTemplate(name="欧文", mode="european", text="CQ DE {自局コール} K"),
                ReplyTemplate(name="和文", mode="japanese", text="{HORE}コンニチハ{RATA}"),
                ReplyTemplate(name="どちらでも", mode="any", text="K"),
            ],
            path,
        )
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="european")
        names = [dialog.template_combo.itemText(i) for i in range(dialog.template_combo.count())]
        assert names == ["欧文", "どちらでも"]

    def test_型を使うと日本語ボックスに入る(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="CQ", mode="european", text="CQ DE {自局コール} K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        profile = OperatorProfile(callsign="JH0ILL")
        dialog = TxDialog(settings, profile=profile, templates_path=path, mode="european")
        dialog.apply_template()
        assert dialog.japanese_edit.toPlainText() == "CQ DE JH0ILL K"

    def test_相手コールの欄が差し込まれる(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="応答", mode="european", text="{相手コール} DE {自局コール} K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        profile = OperatorProfile(callsign="JH0ILL")
        dialog = TxDialog(
            settings, profile=profile, templates_path=path, mode="european",
            received_text="JH0ILL DE JA1ABC K",
        )
        assert dialog.their_call_edit.text() == "JA1ABC"
        dialog.apply_template()
        assert dialog.japanese_edit.toPlainText() == "JA1ABC DE JH0ILL K"

    def test_埋まらない欄は疑問符になる(self, qapp, tmp_path) -> None:
        """**止めない。通知もしない。送信文に ``?`` が見える** (設計書 §2.2).

        以前は「埋まっていない欄があります」で送信を止めていた。運用者は
        送信文に出た ``?`` を見て、必要なら直してから確認・送信する。
        ``[確認]`` を押すまで送れない関門そのものは変わらない。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="応答", mode="european", text="{相手コール} DE {自局コール} K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="european")
        dialog.apply_template()

        assert dialog.japanese_edit.toPlainText() == "? DE ? K"
        # **送れる文字である** (`?` は両方の符号表にある) ので警告は出ない
        assert "送信できない文字" not in dialog.status_label.text()
        assert "埋まっていない" not in dialog.status_label.text()

    def test_RSTの既定は599(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")
        assert dialog.rst_edit.text() == "599"

    def test_型を使うと関門が閉じ直る(self, qapp, tmp_path) -> None:
        """**確認していない文字列は送れない。**"""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="CQ", mode="european", text="CQ DE JH0ILL K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        clients: list[FakeClient] = []

        def factory(host, port=45679, **kw):
            client = FakeClient(host, port, **kw)
            clients.append(client)
            return client

        dialog = TxDialog(
            settings, profile=OperatorProfile(), client_factory=factory, templates_path=path, mode="european"
        )
        dialog.connect_to_keyer()
        dialog.japanese_edit.setPlainText("CQ TEST")
        dialog.refresh_kana()
        dialog.run_check()
        assert dialog.can_send() is True
        dialog.apply_template()
        assert dialog.can_send() is False

    def test_型が無いときは理由を出す(self, qapp, tmp_path) -> None:
        """**無言で終わらない** (設計書 §8「このモードの型がありません」).

        以前は ``return`` するだけで、押しても何も起きなかった。運用者には
        壊れているのか型が無いのか区別が付かない (2026-08-11 レビュー I2)。
        """
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")
        assert dialog.template_combo.count() == 0

        dialog.apply_template()                      # 落ちないこと

        assert dialog.japanese_edit.toPlainText() == ""
        assert "このモードの型がありません" in dialog.status_label.text()

    # ---- 2026-08-11 レビュー: 欧文の型が既定の囲みで送れなくなる ----

    def test_欧文の型は囲まれない(self, qapp, tmp_path) -> None:
        """**肝心。** ``wrap_check`` の既定 (オン) のまま欧文の型を流すと、
        中の欧文がまるごと ``{HORE}``/``{RATA}`` に囲まれ「送信できない」に
        なる (CQ を出すという一番基本の操作が既定状態で使えなくなっていた)。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="CQ", mode="european", text="CQ CQ DE {自局コール} K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        profile = OperatorProfile(callsign="JH0ILL")
        dialog = TxDialog(settings, profile=profile, templates_path=path, mode="european")
        assert dialog.wrap_check.isChecked() is True     # 既定はオン

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is False
        assert "{HORE}" not in dialog.wire_text()
        assert "{RATA}" not in dialog.wire_text()
        assert "送信できない" not in dialog.status_label.text()

    def test_和文の型は囲まれる(self, qapp, tmp_path) -> None:
        """和文の型 (マーカー無しのカタカナ本文) を使うと囲まれる."""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="挨拶", mode="japanese", text="コンニチハ")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="japanese")
        dialog.wrap_check.setChecked(False)               # 手で切り替えていても

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is True
        text = dialog.wire_text()
        assert text.startswith("{HORE}")
        assert text.endswith("{RATA}")

    def test_マーカーを自前で持つ和文の型が二重に囲まれない(self, qapp, tmp_path) -> None:
        """型に ``{HORE}`` があるとき ``wire_text()`` に ``{HORE}`` は 1 つだけ.

        ``wrap_japanese`` は既に ``{HORE}`` があれば何もしないので安全
        (``src/tx/encoder.py``)。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(
            [ReplyTemplate(name="挨拶", mode="japanese", text="{HORE}コンニチハ{RATA}")], path
        )
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="japanese")

        dialog.apply_template()

        text = dialog.wire_text()
        assert text.count("{HORE}") == 1
        assert text.count("{RATA}") == 1

    def test_型のモードで経歴の値が変わる(self, qapp, tmp_path) -> None:
        """``field_values`` に渡すモードは型の ``mode`` である (呼び出し側の固定値ではない).

        経歴の表示形と読みを違えておかないと、``field_values("european")`` に
        固定しても気づけない空洞テストになる (欧文は表示形、和文は読み)。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(
            [
                ReplyTemplate(name="欧文", mode="european", text="DE {名前} K"),
                ReplyTemplate(name="和文", mode="japanese", text="{名前}デス"),
            ],
            path,
        )
        # **欧文用と和文用は独立した値である** (読みではない)
        profile = OperatorProfile(name=BilingualField(european="TARO", japanese="タロウ"))
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")

        eu_dialog = TxDialog(settings, profile=profile, templates_path=path, mode="european")
        eu_dialog.apply_template()
        assert eu_dialog.japanese_edit.toPlainText() == "DE TARO K"

        ja_dialog = TxDialog(settings, profile=profile, templates_path=path, mode="japanese")
        ja_dialog.apply_template()
        assert ja_dialog.japanese_edit.toPlainText() == "タロウデス"

    # ---- 2026-08-11 最終レビュー ----

    def test_自動モードでは型が消えない(self, qapp, tmp_path) -> None:
        """**I1: ``auto`` が主力の運用モードである。**

        以前は ``auto`` が未知のモード扱いで、``any`` 以外の型がすべて一覧から
        消えていた (例文なら 10 個中 9 個)。しかも理由が出なかった。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(
            [
                ReplyTemplate(name="欧文", mode="european", text="CQ DE JH0ILL K"),
                ReplyTemplate(name="和文", mode="japanese", text="{HORE}コンニチハ{RATA}"),
                ReplyTemplate(name="どちらでも", mode="any", text="K"),
            ],
            path,
        )
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="auto")
        names = [dialog.template_combo.itemText(i) for i in range(dialog.template_combo.count())]
        assert names == ["欧文", "和文", "どちらでも"]

    def test_自局コールで自分を除外する(self, qapp, tmp_path) -> None:
        """**I3 の回帰テスト。** 自分のコールが「相手」欄に入らないこと.

        以前は経歴のコールが ``display``/``reading`` の 2 通りを持ち、読みを
        入れていると ``ジェイキュー`` が自局コールとして渡って自分を除外
        できなかった。**コールサインを 1 値にしたので起こりようがない**
        (和文の交信でも欧文で送るため、分ける意味が無い)。

        ``DE`` が拾えなかった受信文で確かめる。``DE`` があるときは
        ``qso_fields`` がそちらを手掛かりにするので、自局コールが使われるのは
        **``DE`` が落ちたときだけ**である (実運用ではよく落ちる)。
        """
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        profile = OperatorProfile(callsign="JH0ILL")
        dialog = TxDialog(
            settings,
            profile=profile,
            templates_path=tmp_path / "なし.json",
            received_text="JH0ILL JA1ABC PSE K",
        )
        assert dialog.their_call_edit.text() == "JA1ABC"

    def test_どちらでもの型に和文を書くと囲まれる(self, qapp, tmp_path) -> None:
        """**I5: この機能で唯一の「警告なしで誤った電波が出る」経路。**

        囲みを型の ``mode`` で決めていたため、``any`` の型に和文本文を書くと
        囲みが外れた。和文の符号は作れるので警告も出ないが、受信側は
        モードを切り替えられず**化けたまま届く**。囲みは中身で決める。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="お礼", mode="any", text="アリガトウ")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="auto")

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is True
        text = dialog.wire_text()
        assert text.startswith("{HORE}")
        assert text.endswith("{RATA}")

    def test_どちらでもの型が欧文なら囲まれない(self, qapp, tmp_path) -> None:
        """中身で決めるので、``any`` でも欧文だけの本文は囲まない (I5 の対)."""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="再送", mode="any", text="PSE AGN AGN K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="auto")

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is False
        assert "{HORE}" not in dialog.wire_text()
        assert "送信できない" not in dialog.status_label.text()

    def test_和文の型を漢字かな交じりで書いても囲まれる(self, qapp, tmp_path) -> None:
        """判定は**変換後**の文で行う (型は漢字かな交じりで書いてよい)."""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="挨拶", mode="any", text="こんにちは")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="auto")

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is True
        assert dialog.wire_text().startswith("{HORE}")

    def test_欄を手で埋めれば疑問符が消える(self, qapp, tmp_path) -> None:
        """埋め残しは ``?`` として見え、直せば消える (本文から毎回作り直す)."""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="応答", mode="european", text="{相手コール} DE JH0ILL K")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="european")

        dialog.apply_template()
        assert "?" in dialog.japanese_edit.toPlainText()

        dialog.japanese_edit.setPlainText("JA1ABC DE JH0ILL K")

        assert "?" not in dialog.wire_text()
        assert dialog.status_label.text() in ("", "未接続")

    def test_手で書いた欄は送れない文字として見える(self, qapp, tmp_path) -> None:
        """**Minor 2 の名残。** 差し込みを通らない ``{…}`` は黙って通さない."""
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")
        dialog.wrap_check.setChecked(False)

        dialog.japanese_edit.setPlainText("{相手コール} DE JH0ILL #")

        # 手で書いた `{相手コール}` は差し込みを通らないので、そのまま
        # 「送信できない文字」として見える (`{` `}` は符号表に無い)
        status = dialog.status_label.text()
        assert "送信できない文字があります" in status
        assert "#" in status
        assert "{" in status

    # ---- 2026-08-12 最終レビュー: 欧文区間 「…」 ----
    #
    # `encoder.needs_japanese_wrap` が `_initial_mode` と違って `「…」` の
    # 中身を取り除かずにモードを判定していたため、`」` を「和文にしか無い
    # 文字」と誤認し、欧文だけの本文にもホレ・ラタが静かに付いていた
    # (I5 と同じ「警告なしで誤った電波が出る」経路の再発)。
    # `tests/test_tx_encoder.py` の単体テストは `needs_japanese_wrap` を
    # 直接呼ばず、`apply_template` → `wrap_check` を経由するこの継ぎ目を
    # 通らないので検知できなかった。

    def test_欧文区間だけの本文にホレが付かない(self, qapp, tmp_path) -> None:
        """`「FT991」` だけの本文にホレ・ラタが付かないこと (`apply_template` 経路).

        以前は `」` を和文の手掛かりと誤認し、`{HORE}「FT991」{RATA}` が
        できていた。受信側はホレで和文モードに入ってから欧文の符号を
        受け取ることになり、警告も出ないまま化けた電波が出る。

        **これは `apply_template` (型を使う経路) だけの確認である。**
        `apply_template` は中身を見て `wrap_check` を自動設定するので
        安全だが、`japanese_edit` に直接打つ経路 (`refresh_kana`) は
        `wrap_check.isChecked()` をそのまま使うだけで、この経路には
        判定が無かった。それが今回の見逃しの原因だったので、打つ経路は
        :class:`TestNeedlessWrapWarning` で別途確かめる。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="リグ", mode="european", text="「FT991」")], path)
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="european")

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is False
        assert "{HORE}" not in dialog.wire_text()
        assert "{RATA}" not in dialog.wire_text()

    def test_欧文区間を含む欧文の本文が送れる(self, qapp, tmp_path) -> None:
        """`RIG 「FT991」 ANT 「DP」 K` が警告なしで送れること."""
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(
            [ReplyTemplate(name="欧文", mode="european", text="RIG 「FT991」 ANT 「DP」 K")], path
        )
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="european")

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is False
        assert "{HORE}" not in dialog.wire_text()
        assert "{RATA}" not in dialog.wire_text()
        assert "送信できない" not in dialog.status_label.text()

    def test_和文に欧文区間があればホレが付く(self, qapp, tmp_path) -> None:
        """``コチラノ リグ ハ 「FT991」`` にはホレが付くこと (従来の振る舞いを壊さない).

        `「FT991」` を除いても本文にカタカナが残っているので、和文として
        ホレ・ラタで囲む従来どおりの判定が働く。
        """
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(
            [ReplyTemplate(name="和文", mode="any", text="コチラノ リグ ハ 「FT991」")], path
        )
        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=path, mode="auto")

        dialog.apply_template()

        assert dialog.wrap_check.isChecked() is True
        text = dialog.wire_text()
        assert text.startswith("{HORE}")
        assert text.endswith("{RATA}")


class TestNeedlessWrapWarning:
    """**2026-08-12 最終レビュー (2 回目): 打つ経路には判定が無かった.**

    `TestTemplates` の `「…」` 系テストは `apply_template` 経由だけを確かめて
    いた。`apply_template` は中身を見て `wrap_check` を自動設定するので
    そもそも問題にならないが、**運用者が `japanese_edit` に直接打つ経路
    (`refresh_kana`) は `wrap_check.isChecked()` をそのまま使うだけ**で、
    既定がオンなので `「FT991」` のように和文の無い本文を直接打つと
    `{HORE}「FT991」{RATA}` が**警告なしで**できていた。中身は欧文として
    符号化できるので「送信できない文字」にはならず、**送れるのに化ける**
    という一番気づきにくい壊れ方をする。

    **チェックボックスは運用者が明示的に操作するものなので、黙って無視せず
    警告する** (自動でチェックを外したりはしない)。
    """

    def test_和文が無いのに囲むと警告が出る(self, qapp, tmp_path) -> None:
        """`wrap_check` がオンのまま `「FT991」` を打つと警告が出ること.

        **打つ経路** (`japanese_edit.setPlainText` + `refresh_kana`) で
        確かめる。`apply_template` を経由しない。
        """
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")
        assert dialog.wrap_check.isChecked() is True     # 既定はオン

        dialog.japanese_edit.setPlainText("「FT991」")
        dialog.refresh_kana()

        assert "{HORE}" in dialog.wire_text()    # 既定のチェックは黙って無視しない
        assert "和文がありません" in dialog.status_label.text()

    def test_和文があれば警告は出ない(self, qapp, tmp_path) -> None:
        """``コチラノ リグ ハ 「FT991」`` では警告が出ないこと (打つ経路)."""
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")

        dialog.japanese_edit.setPlainText("コチラノ リグ ハ 「FT991」")
        dialog.refresh_kana()

        assert "{HORE}" in dialog.wire_text()
        assert "和文がありません" not in dialog.status_label.text()

    def test_手で囲んだ本文には警告が出ない(self, qapp, tmp_path) -> None:
        """本文に ``{HORE}…{RATA}`` と手で書いてあるときは鳴らないこと.

        **2026-08-12 再レビューが見つけた誤警告。** ``needs_japanese_wrap`` は
        「和文が無い」ときと「既に囲んである」ときの**両方で偽を返す**ので、
        後者を前者と取り違えて「和文がありません」と言っていた。
        ``wrap_japanese`` は二重に囲まないので**電波は正しい**。
        **唯一の歯止めである警告を無意味に鳴らしてはいけない。**
        """
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")
        assert dialog.wrap_check.isChecked() is True     # 既定はオンのまま

        dialog.japanese_edit.setPlainText("{HORE}コンニチハ{RATA}")
        dialog.refresh_kana()

        assert "和文がありません" not in dialog.status_label.text()
        # 二重には囲まない (電波は正しい)
        assert dialog.wire_text().count("{HORE}") == 1

    def test_囲みを外せば警告が消える(self, qapp, tmp_path) -> None:
        """`wrap_check` を外すと「和文がありません」警告が消えること."""
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")
        dialog.japanese_edit.setPlainText("「FT991」")
        dialog.refresh_kana()
        assert "和文がありません" in dialog.status_label.text()

        dialog.wrap_check.setChecked(False)

        assert "{HORE}" not in dialog.wire_text()
        assert "和文がありません" not in dialog.status_label.text()

    def test_空の本文では警告が出ない(self, qapp, tmp_path) -> None:
        """ダイアログを開いた直後 (本文が空、`wrap_check` は既定でオン) に警告が出ないこと.

        ``needs_japanese_wrap("")`` は偽を返すので、素朴に実装すると
        本文が空のときにまで「和文がありません」が出てしまう。
        """
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json")

        assert "和文がありません" not in dialog.status_label.text()


class TestMatchReceivedWpm:
    """「受信に合わせる」ボタン (運用者の要望、2026-08-12).

    **勝手に追従はしない。** 送信直前に速度が動くと確認をやり直すことになり、
    押そうとした瞬間に値が変わる。押したときだけ入る。
    """

    def test_押すと受信の速度が入る(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679", tx_wpm=25.0)
        dialog = TxDialog(
            settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json",
            received_wpm=18.4,
        )
        assert dialog.wpm_spin.value() == pytest.approx(25.0)

        dialog.match_wpm_btn.click()

        # **整数に丸める。** 測定は「だいたいの速さ」で 18.4 の精度は無い
        assert dialog.wpm_spin.value() == pytest.approx(18.0)

    def test_測れていなければ押せない(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679")
        dialog = TxDialog(
            settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json",
            received_wpm=None,
        )
        assert dialog.match_wpm_btn.isEnabled() is False
        before = dialog.wpm_spin.value()
        dialog.match_received_wpm()            # 呼ばれても何も起きないこと
        assert dialog.wpm_spin.value() == pytest.approx(before)

    def test_押すと確認がやり直しになる(self, qapp, tmp_path) -> None:
        """速度は「何秒間 電波が出るのか」を変えるので、確認は無効に戻る."""
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679", tx_wpm=25.0)
        dialog = TxDialog(
            settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json",
            received_wpm=18.4,
        )
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.refresh_kana()
        dialog._confirmed_text = dialog.wire_text()      # 確認が通った状態を作る

        dialog.match_wpm_btn.click()

        assert dialog._confirmed_text is None

    def test_同じ値なら確認は保たれる(self, qapp, tmp_path) -> None:
        """既に同じ速度なら ``valueChanged`` が飛ばず、確認も落ちない."""
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679", tx_wpm=18.0)
        dialog = TxDialog(
            settings, profile=OperatorProfile(), templates_path=tmp_path / "なし.json",
            received_wpm=18.4,
        )
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.refresh_kana()
        dialog._confirmed_text = dialog.wire_text()

        dialog.match_wpm_btn.click()

        assert dialog.wpm_spin.value() == pytest.approx(18.0)
        assert dialog._confirmed_text is not None


class TestClearButton:
    """日本語ボックスを空にするボタン (運用者の要望、2026-08-12).

    1 回の交信で何度も打ち直すので、全選択して消すのは手数が多い。
    """

    def _dialog(self, tmp_path):
        from src.app.tx_dialog import TxDialog

        return TxDialog(
            AppSettings(tx_endpoint="127.0.0.1:45679"),
            profile=OperatorProfile(),
            templates_path=tmp_path / "なし.json",
        )

    def test_押すと本文が消える(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path)
        dialog.japanese_edit.setPlainText("コンニチハ")
        assert dialog.wire_text() != ""

        dialog.clear_btn.click()

        assert dialog.japanese_edit.toPlainText() == ""
        assert dialog.wire_text() == ""

    def test_相手や_RST_は消さない(self, qapp, tmp_path) -> None:
        """**交信のあいだ変わらないものを巻き込まない** (打ち直しになる)."""
        dialog = self._dialog(tmp_path)
        dialog.their_call_edit.setText("JA1ABC")
        dialog.their_name_edit.setText("タロウ")
        dialog.rst_edit.setText("579")
        dialog.weather_edit.setText("ハレ")
        dialog.temp_edit.setText("20")
        dialog.japanese_edit.setPlainText("コンニチハ")

        dialog.clear_btn.click()

        assert dialog.their_call_edit.text() == "JA1ABC"
        assert dialog.their_name_edit.text() == "タロウ"
        assert dialog.rst_edit.text() == "579"
        assert dialog.weather_edit.text() == "ハレ"
        assert dialog.temp_edit.text() == "20"

    def test_消すと確認がやり直しになる(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog._confirmed_text = dialog.wire_text()

        dialog.clear_btn.click()

        assert dialog._confirmed_text is None
        assert dialog.can_send() is False

    def test_本文が空なら押せない(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path)
        assert dialog.clear_btn.isEnabled() is False

        dialog.japanese_edit.setPlainText("コンニチハ")

        assert dialog.clear_btn.isEnabled() is True

    def test_送信中は押せない(self, qapp, tmp_path) -> None:
        """**打鍵中に本文が消えると、何を送っているのか分からなくなる.**"""
        dialog = self._dialog(tmp_path)
        dialog.japanese_edit.setPlainText("コンニチハ")
        assert dialog.clear_btn.isEnabled() is True

        class _Running:
            def isRunning(self) -> bool:
                return True

        dialog._worker = _Running()
        dialog._update_buttons()

        assert dialog.clear_btn.isEnabled() is False
        dialog._worker = None


def _ok_result(*, aborted: bool = False, reason=None) -> SendResult:
    """打鍵側からの応答のふり."""
    return SendResult(
        elements_sent=10, aborted=aborted, watchdog_tripped=False,
        seconds=1.0, max_error_ms=0.1, mean_error_ms=0.05, reason=reason,
    )


def _simulate_send(dialog, **kw) -> None:
    """``run_send`` を経由せず ``_on_sent`` を直接呼ぶテスト用.

    本番は ``run_send`` がワーカー生成時に ``_send_pending`` (送った文字列と
    速度の組) を確定させ、``_on_sent`` はそれを使う (2026-08-13 最終レビュー
    Minor 6)。ここでは非同期ワーカーを起こさずに完了だけを模したいので、
    呼び出し時点の文字列・速度をそのまま ``_send_pending`` にセットしてから
    ``_on_sent`` を呼ぶ — ``run_send`` を直前に呼んだのと同じ状態を作る。
    """
    dialog._send_pending = (dialog.wire_text(), dialog.wpm_spin.value())
    dialog._on_sent(_ok_result(**kw))


class TestSentTextsNeedNoRecheck:
    """**一度送ったものは確認を押し直さなくてよい** (運用者の要望、2026-08-12).

    交信中に送信文を作り直すのは手間が大きい。同じ文を送り直すたびに
    ``[確認]`` の往復を待つのは、実運用で一番効く無駄だった。

    **関門を外すわけではない。** 打鍵側が「その文字列はこの速度で送れる」と
    答えた事実を覚えておき、**まったく同じ文字列・同じ速度**のときだけ
    ``[送信]`` を直接押せるようにする。速度が変われば秒数が変わるので覚え直す。
    """

    def _dialog(self, tmp_path, **kw):
        from src.app.tx_dialog import TxDialog

        settings = AppSettings(tx_endpoint="127.0.0.1:45679", tx_wpm=20.0)
        return TxDialog(
            settings, profile=OperatorProfile(),
            templates_path=tmp_path / "なし.json", **kw,
        )

    @staticmethod
    def _connect(dialog):
        """接続済みにして、確認を通した状態にする."""
        client = FakeClient("127.0.0.1")
        dialog._client = client
        return client

    def test_送信し終えたら確認なしで送れる(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path)
        self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        assert dialog.can_send() is True

        _simulate_send(dialog)

        assert dialog.can_send() is True          # 押し直さなくてよい

    def test_中止したものは覚えない(self, qapp, tmp_path) -> None:
        """**途中で止めたものは「送れた」とは言えない。**"""
        dialog = self._dialog(tmp_path)
        self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()

        _simulate_send(dialog, aborted=True, reason="stop")

        assert dialog.can_send() is False

    def test_別の文は確認が要る(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path)
        self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        _simulate_send(dialog)

        dialog.japanese_edit.setPlainText("サヨウナラ")

        assert dialog.can_send() is False

    def test_同じ文に戻せばまた送れる(self, qapp, tmp_path) -> None:
        """**これが目的。** 型を選び直しても確認の往復が要らない."""
        dialog = self._dialog(tmp_path)
        self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        _simulate_send(dialog)
        dialog.japanese_edit.setPlainText("サヨウナラ")
        assert dialog.can_send() is False

        dialog.japanese_edit.setPlainText("コンニチハ")

        assert dialog.can_send() is True

    def test_速度を変えたら確認し直す(self, qapp, tmp_path) -> None:
        """**秒数が変わる。** 20 WPM で確認した 6.3 秒は 5 WPM では 4 倍になる."""
        dialog = self._dialog(tmp_path)
        self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        _simulate_send(dialog)

        dialog.wpm_spin.setValue(25.0)

        assert dialog.can_send() is False

    def test_速度を戻せばまた送れる(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path)
        self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        _simulate_send(dialog)
        dialog.wpm_spin.setValue(25.0)
        assert dialog.can_send() is False

        dialog.wpm_spin.setValue(20.0)

        assert dialog.can_send() is True

    def test_繋がっていなければ送れない(self, qapp, tmp_path) -> None:
        """**覚えていても、打鍵側が居なければ送れない。**"""
        dialog = self._dialog(tmp_path)
        self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        _simulate_send(dialog)

        dialog._client = None

        assert dialog.can_send() is False

    def test_確定済みの記録は中止した再送では消えない(self, qapp, tmp_path) -> None:
        """**中止は打鍵側のお墨付きを無効にしない** (2026-08-13 最終レビュー Minor 7).

        一度最後まで送れた記録は、そのあとの再送を運用者が `[中止]` しても
        残る。中止したのは「今回の再送」であって、前回の完了そのものが
        取り消されるわけではない — このまま送れば通ることに変わりはない。
        """
        dialog = self._dialog(tmp_path)
        client = self._connect(dialog)
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        _simulate_send(dialog)
        assert dialog.can_send() is True

        client.send = lambda text, wpm: SendResult(30, True, False, 5.0, 1.0, 0.3, "stop")
        dialog.run_send()
        wait_for_worker(dialog)

        assert dialog.can_send() is True

    def test_符号表が違う相手に繋ぎ直したら忘れる(self, qapp, tmp_path) -> None:
        """**打鍵側が変わったら、覚えていた確認は当てにならない。**

        以前は ``_forget_sent_texts()`` を直接呼ぶだけの空洞テストだった。
        ``connect_to_keyer`` 自身が指紋の食い違いを見て記録を捨てることを、
        実際にその経路を通して確かめる (2026-08-13 最終レビュー Critical 2)。
        """
        dialog = self._dialog(tmp_path, client_factory=FakeClient)
        dialog.connect_to_keyer()
        dialog.japanese_edit.setPlainText("コンニチハ")
        dialog.run_check()
        _simulate_send(dialog)
        assert dialog.can_send() is True

        class MismatchedClient(FakeClient):
            def connect(self) -> Hello:
                return Hello(
                    1,
                    "this-does-not-match-the-real-fingerprint",
                    {"port": "COM3", "key": "DTR", "ptt": "RTS"},
                    False,
                )

        dialog._client_factory = MismatchedClient
        dialog.connect_to_keyer()

        assert dialog.can_send() is False


class TestTemplateAppliesOnSelect:
    """**型を選んだ瞬間に本文へ入れる** (運用者の要望、2026-08-12).

    交信中に送信文を作るのは時間の勝負で、``[型を使う]`` を押す 1 手間が重い。
    選ぶだけで入るようにする。**手で書いた内容は消えるので、直前に戻す
    ボタンを添える。**
    """

    def _dialog(self, tmp_path, templates):
        from src.app.tx_dialog import TxDialog

        path = tmp_path / "templates.json"
        save_templates(templates, path)
        return TxDialog(
            AppSettings(tx_endpoint="127.0.0.1:45679"),
            profile=OperatorProfile(), templates_path=path, mode="european",
        )

    @staticmethod
    def _two():
        return [
            ReplyTemplate(name="CQ", mode="european", text="CQ CQ DE JH0ILL K"),
            ReplyTemplate(name="締め", mode="european", text="TU 73 GB"),
        ]

    def test_選ぶだけで本文が入れ替わる(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path, self._two())
        # **開いただけでは入れない。** 勝手に本文が埋まると驚く
        assert dialog.japanese_edit.toPlainText() == ""

        dialog.template_combo.setCurrentIndex(1)

        assert dialog.japanese_edit.toPlainText() == "TU 73 GB"

    def test_開いたときの型は_型を使う_で入れる(self, qapp, tmp_path) -> None:
        """先頭の型は選び直せない (既に選ばれている) のでボタンが要る."""
        dialog = self._dialog(tmp_path, self._two())

        dialog.use_template_btn.click()

        assert dialog.japanese_edit.toPlainText() == "CQ CQ DE JH0ILL K"

    def test_元に戻せる(self, qapp, tmp_path) -> None:
        """**手で書いた内容を消してしまったときの逃げ道。**"""
        dialog = self._dialog(tmp_path, self._two())
        dialog.japanese_edit.setPlainText("テガキ ノ ナイヨウ")

        dialog.template_combo.setCurrentIndex(1)
        assert dialog.japanese_edit.toPlainText() == "TU 73 GB"

        dialog.undo_template_btn.click()

        assert dialog.japanese_edit.toPlainText() == "テガキ ノ ナイヨウ"

    def test_元に戻すとホレラタの囲みも戻る(self, qapp, tmp_path) -> None:
        """**Important 3 (2026-08-13 最終レビュー): 囲みも一緒に戻す。**

        ``apply_template`` は中身を見て ``wrap_check`` も書き換える。以前の
        ``undo_template`` は本文しか戻さなかったので、和文の本文 + 囲み ON
        の状態から欧文の型を選ぶと囲みが OFF になり、そこで元に戻しても
        本文は和文に戻るのに囲みが OFF のまま残っていた。無囲みの和文は
        そのまま送信されて相手のデコーダで化ける (「送れるのに化ける」)。
        """
        dialog = self._dialog(tmp_path, self._two())
        dialog.wrap_check.setChecked(True)
        dialog.japanese_edit.setPlainText("コンニチハ")
        assert dialog.wrap_check.isChecked() is True

        dialog.template_combo.setCurrentIndex(1)      # 欧文の型 → 囲み OFF になる
        assert dialog.wrap_check.isChecked() is False

        dialog.undo_template_btn.click()

        assert dialog.japanese_edit.toPlainText() == "コンニチハ"
        assert dialog.wrap_check.isChecked() is True

    def test_戻すボタンは何も消していなければ押せない(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path, self._two())
        dialog.japanese_edit.setPlainText("")

        dialog.template_combo.setCurrentIndex(1)

        assert dialog.undo_template_btn.isEnabled() is False

    def test_二度戻さない(self, qapp, tmp_path) -> None:
        """戻したら記憶は消える (押すたびに古い内容が甦らない)."""
        dialog = self._dialog(tmp_path, self._two())
        dialog.japanese_edit.setPlainText("テガキ")
        dialog.template_combo.setCurrentIndex(1)

        dialog.undo_template_btn.click()

        assert dialog.undo_template_btn.isEnabled() is False

    def test_型が無くても落ちない(self, qapp, tmp_path) -> None:
        dialog = self._dialog(tmp_path, [])
        assert dialog.template_combo.count() == 0
        assert dialog.japanese_edit.toPlainText() == ""

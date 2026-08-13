"""経歴の編集画面.

**利用者の実ファイルを絶対に触らないこと。** どのテストも ``tmp_path`` を渡す
(過去に UI のスモークテストが利用者の実設定を書き戻していた)。
"""
from __future__ import annotations

import json

import pytest

from src.app.profile_dialog import ProfileDialog, unsendable_in_value
from src.tx.profile import BilingualField, OperatorProfile, load_profile, save_profile


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _dialog(tmp_path, profile: OperatorProfile | None = None) -> ProfileDialog:
    return ProfileDialog(profile=profile, path=tmp_path / "operator.json")


class TestUnsendableInValue:
    """**モードを固定して調べる。** 値の中身から推測させてはいけない.

    欧文用の欄にカタカナを書いたら「和文だから送れる」ではなく
    「欧文で送れない」と言わなければならない。それがまさに見つけたい誤りである。
    """

    @pytest.mark.parametrize(
        ("value", "mode", "sendable"),
        [
            ("TARO", "european", True),
            ("TARO", "japanese", False),      # 和文の中に裸の欧文は置けない
            ("タロウ", "japanese", True),
            ("タロウ", "european", False),
            ("「FT991」", "japanese", True),     # 欧文区間なので和文でも送れる
            ("「FT991」", "european", True),     # 欧文では括弧が落ちる
            ("50W", "european", True),
            ("50W", "japanese", False),        # W は和文表に無い
            ("JH0ILL", "european", True),
        ],
    )
    def test_モードごとの可否(self, value: str, mode: str, sendable: bool) -> None:
        assert (unsendable_in_value(value, mode) == "") is sendable

    def test_実際に通る道を通す(self) -> None:
        """**生の値を見ると誤判定する** (設計書 §4.2).

        小書きの ``ュ`` は符号表に無いが、``to_sendable_kana`` が ``ユ`` に
        倒すので実際には送れる。
        """
        assert unsendable_in_value("ニジュウド", "japanese") == ""

    def test_漢字は送れないと分かる(self) -> None:
        """欧文用の欄に漢字を書いたら気づけること."""
        assert unsendable_in_value("佐藤", "european") != ""

    def test_空は送れる扱い(self) -> None:
        """空欄は未記入であって誤りではない (差し込みで ``?`` になる)."""
        assert unsendable_in_value("", "european") == ""
        assert unsendable_in_value("", "japanese") == ""


class TestRoundTrip:
    def test_書いて保存して読み直せる(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.callsign_edit.setText("JH0ILL")
        d.field_edits["name"][0].setText("TARO")
        d.field_edits["name"][1].setText("タロウ")
        d.field_edits["rig"][0].setText("FT991")
        d.field_edits["rig"][1].setText("「FT991」")
        d.notes_edit.setPlainText("和文中心")
        d.add_dictionary_row("東京", "トウキョウ")

        d.save()

        loaded = load_profile(tmp_path / "operator.json")
        assert loaded.callsign == "JH0ILL"
        assert loaded.name.european == "TARO"
        assert loaded.name.japanese == "タロウ"
        assert loaded.rig.japanese == "「FT991」"
        assert loaded.notes == "和文中心"
        assert loaded.reading_dictionary == {"東京": "トウキョウ"}

    def test_既存の経歴を開くと欄が埋まっている(self, qapp, tmp_path) -> None:
        path = tmp_path / "operator.json"
        save_profile(
            OperatorProfile(
                callsign="JH0ILL",
                qth=BilingualField("YOKOHAMA", "ヨコハマシ"),
                reading_dictionary={"東京": "トウキョウ"},
            ),
            path,
        )

        d = ProfileDialog(path=path)

        assert d.callsign_edit.text() == "JH0ILL"
        assert d.field_edits["qth"][0].text() == "YOKOHAMA"
        assert d.field_edits["qth"][1].text() == "ヨコハマシ"
        assert d.dictionary() == {"東京": "トウキョウ"}

    def test_前後の空白は落とす(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.callsign_edit.setText("  JH0ILL  ")
        d.field_edits["name"][0].setText(" TARO ")
        assert d.profile().callsign == "JH0ILL"
        assert d.profile().name.european == "TARO"

    def test_人が読める_utf8_で書く(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.field_edits["name"][1].setText("タロウ")
        d.save()
        text = (tmp_path / "operator.json").read_text(encoding="utf-8")
        assert "タロウ" in text
        assert json.loads(text)["name"]["japanese"] == "タロウ"


class TestDictionaryTable:
    def test_行を足せる(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.add_row_btn.click()
        assert d.dict_table.rowCount() == 1

    def test_行を消せる(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.add_dictionary_row("東京", "トウキョウ")
        d.add_dictionary_row("横浜", "ヨコハマ")
        d.dict_table.setCurrentCell(0, 0)

        d.remove_row_btn.click()

        assert d.dictionary() == {"横浜": "ヨコハマ"}

    def test_選んでいなければ最後の行を消す(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.add_dictionary_row("東京", "トウキョウ")
        d.add_dictionary_row("横浜", "ヨコハマ")
        d.dict_table.setCurrentCell(-1, -1)

        d.remove_dictionary_row()

        assert d.dictionary() == {"東京": "トウキョウ"}

    def test_空の行は捨てる(self, qapp, tmp_path) -> None:
        """**語が空の行は保存しない** (設計書 §6)."""
        d = _dialog(tmp_path)
        d.add_dictionary_row("", "トウキョウ")
        d.add_dictionary_row("東京", "トウキョウ")
        assert d.dictionary() == {"東京": "トウキョウ"}

    def test_行が空でも落ちない(self, qapp, tmp_path) -> None:
        """``insertRow`` 直後の未設定セルは ``None`` になる."""
        d = _dialog(tmp_path)
        d.dict_table.insertRow(0)
        assert d.dictionary() == {}


class TestWarnings:
    """**警告は止めるためではなく、見せるためのもの。**"""

    def test_和文の欄に欧文を書くと警告(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.field_edits["name"][1].setText("TARO")
        assert any("和文で送れません" in w and "名前" in w for w in d.warnings())

    def test_欧文の欄に漢字を書くと警告(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.field_edits["name"][0].setText("佐藤")
        assert any("欧文で送れません" in w and "名前" in w for w in d.warnings())

    def test_括弧に入れれば警告は出ない(self, qapp, tmp_path) -> None:
        """``「FT991」`` は和文の中でも送れる (欧文区間)."""
        d = _dialog(tmp_path)
        d.field_edits["rig"][0].setText("FT991")
        d.field_edits["rig"][1].setText("「FT991」")
        assert d.warnings() == []

    def test_括弧を忘れると警告(self, qapp, tmp_path) -> None:
        """**これが一番よくある書き間違いである.**"""
        d = _dialog(tmp_path)
        d.field_edits["rig"][1].setText("FT991")
        assert any("和文で送れません" in w and "リグ" in w for w in d.warnings())

    def test_自局コールは欧文で見る(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.callsign_edit.setText("ジェイキュー")
        assert any("自局コール" in w for w in d.warnings())

    def test_読み辞書の欧文の語を警告する(self, qapp, tmp_path) -> None:
        """**読み辞書は本文全体に当たる** (経歴の欄と違って避けようがない)."""
        d = _dialog(tmp_path)
        d.add_dictionary_row("TARO", "タロウ")
        assert any("読み辞書に欧文の語があります" in w for w in d.warnings())

    def test_和文の語なら警告しない(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.add_dictionary_row("東京", "トウキョウ")
        assert d.warnings() == []

    def test_空の経歴では警告しない(self, qapp, tmp_path) -> None:
        """書き始める前から赤くしない."""
        assert _dialog(tmp_path).warnings() == []

    def test_警告があっても保存する(self, qapp, tmp_path) -> None:
        """**書きかけを保存できないと育てるのが苦痛になる.**"""
        d = _dialog(tmp_path)
        d.callsign_edit.setText("JH0ILL")
        d.field_edits["name"][1].setText("TARO")      # 和文の欄に欧文

        d.save()

        assert load_profile(tmp_path / "operator.json").callsign == "JH0ILL"
        assert "保存しました" in d.status_label.text()
        assert "和文で送れません" in d.status_label.text()

    def test_問題が無ければ保存の報告だけ(self, qapp, tmp_path) -> None:
        d = _dialog(tmp_path)
        d.callsign_edit.setText("JH0ILL")
        d.save()
        assert d.status_label.text() == "保存しました"

    def test_保存できなければ黙らせない(self, qapp, tmp_path, monkeypatch) -> None:
        """**書いたつもりで消えるのが一番困る.**"""
        d = _dialog(tmp_path)
        d.callsign_edit.setText("JH0ILL")

        import src.app.profile_dialog as module

        def boom(*args, **kwargs):
            raise OSError("書き込み失敗")

        monkeypatch.setattr(module, "save_profile", boom)
        d.save()

        assert "保存できませんでした" in d.status_label.text()


class TestOpenedFromTxDialog:
    """送信ダイアログとの繋ぎ目.

    **単体で通っても層をまたぐと壊れる。** このリポジトリで繰り返し踏んでいる。
    """

    def test_経歴ボタンがある(self, qapp, tmp_path) -> None:
        from src.app.tx_dialog import TxDialog
        from src.infer.settings import AppSettings

        d = TxDialog(
            AppSettings(tx_endpoint="127.0.0.1:45679"),
            profile=OperatorProfile(),
            templates_path=tmp_path / "なし.json",
            profile_path=tmp_path / "operator.json",
        )
        assert d.profile_btn.isEnabled()

    def test_閉じたら読み直す(self, qapp, tmp_path, monkeypatch) -> None:
        """**書いてすぐ型に反映されないと確かめようがない.**"""
        from src.app.tx_dialog import TxDialog
        from src.infer.settings import AppSettings

        path = tmp_path / "operator.json"
        d = TxDialog(
            AppSettings(tx_endpoint="127.0.0.1:45679"),
            profile=OperatorProfile(),
            templates_path=tmp_path / "なし.json",
            profile_path=path,
        )
        assert d._profile.callsign == ""

        # 経歴の画面が開いて、書いて、閉じたことにする
        save_profile(OperatorProfile(callsign="JH0ILL"), path)
        monkeypatch.setattr(
            "src.app.profile_dialog.ProfileDialog.exec", lambda self: 0
        )
        d.open_profile_dialog()

        assert d._profile.callsign == "JH0ILL"

    def test_書いた経歴が型に入る(self, qapp, tmp_path, monkeypatch) -> None:
        """繋ぎ目の本番: 経歴 → 差し込み → 送信文."""
        from src.app.tx_dialog import TxDialog
        from src.infer.settings import AppSettings
        from src.tx.templates import ReplyTemplate, save_templates

        templates = tmp_path / "templates.json"
        save_templates(
            [ReplyTemplate(name="CQ", mode="european", text="CQ DE {自局コール} K")],
            templates,
        )
        path = tmp_path / "operator.json"
        d = TxDialog(
            AppSettings(tx_endpoint="127.0.0.1:45679"),
            profile=OperatorProfile(),
            templates_path=templates,
            profile_path=path,
            mode="european",
        )
        d.apply_template()
        assert d.japanese_edit.toPlainText() == "CQ DE ? K"      # まだ空

        save_profile(OperatorProfile(callsign="JH0ILL"), path)
        monkeypatch.setattr(
            "src.app.profile_dialog.ProfileDialog.exec", lambda self: 0
        )
        d.open_profile_dialog()
        d.apply_template()

        assert d.japanese_edit.toPlainText() == "CQ DE JH0ILL K"

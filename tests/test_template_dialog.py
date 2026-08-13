"""返信の型の編集画面.

**利用者の実ファイルを絶対に触らないこと。** どのテストも ``tmp_path`` と
明示的な経歴を渡す (``profile`` を省くと ``~/.cw-decorder/operator.json`` を
読みに行き、結果が運用者の環境に左右される。実際に踏んだ)。
"""
from __future__ import annotations

import pytest

from src.app.template_dialog import MODE_CHOICES, TemplateDialog, unknown_placeholders
from src.tx.profile import BilingualField, OperatorProfile
from src.tx.templates import ReplyTemplate, load_templates, save_templates


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def profile() -> OperatorProfile:
    return OperatorProfile(
        callsign="JH0ILL",
        name=BilingualField("TARO", "タロウ"),
        rig=BilingualField("FT991", "「FT991」"),
    )


def _dialog(tmp_path, profile, templates=None) -> TemplateDialog:
    path = tmp_path / "templates.json"
    if templates is not None:
        save_templates(templates, path)
    return TemplateDialog(path=path, profile=profile)


def _three() -> list[ReplyTemplate]:
    return [
        ReplyTemplate(name="CQ", mode="european", text="CQ CQ DE {自局コール} K"),
        ReplyTemplate(name="応答", mode="japanese", text="{HORE}コンニチハ{RATA}"),
        ReplyTemplate(name="締め", mode="any", text="TU 73 {SK}"),
    ]


class TestUnknownPlaceholders:
    """書き間違いは ``{`` ``}`` が残るだけで、どこが悪いか分からない."""

    def test_知らない欄を名指しする(self) -> None:
        assert unknown_placeholders("DE {名前まえ} K") == ("名前まえ",)

    def test_正しい欄は数えない(self) -> None:
        assert unknown_placeholders("{相手コール} DE {自局コール} {RST}") == ()

    def test_マーカーは数えない(self) -> None:
        """``{HORE}`` ``{SK}`` は差し込みではなく、そのまま送る符号."""
        assert unknown_placeholders("{HORE}コンニチハ{RATA} {SK}") == ()

    def test_重複は一度だけ(self) -> None:
        assert unknown_placeholders("{ぼく} {ぼく}") == ("ぼく",)

    def test_出現順を保つ(self) -> None:
        assert unknown_placeholders("{い} {ろ} {は}") == ("い", "ろ", "は")


class TestList:
    def test_型が並ぶ(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        assert d.list_widget.count() == 3
        assert d.list_widget.item(0).text().startswith("CQ")

    def test_一覧にモードも出す(self, qapp, tmp_path, profile) -> None:
        """**どの型がどのモードかは一覧を見て分かる必要がある** (絞り込みに効く)."""
        d = _dialog(tmp_path, profile, _three())
        assert "欧文" in d.list_widget.item(0).text()
        assert "和文" in d.list_widget.item(1).text()
        assert "どちらでも" in d.list_widget.item(2).text()

    def test_選ぶと欄が埋まる(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(1)
        assert d.name_edit.text() == "応答"
        assert d.mode_combo.currentText() == "和文"
        assert d.text_edit.toPlainText() == "{HORE}コンニチハ{RATA}"

    def test_型が無ければ欄は触れない(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, [])
        assert d.list_widget.count() == 0
        assert d.name_edit.isEnabled() is False
        assert d.text_edit.isEnabled() is False


class TestEditing:
    def test_選び直しても編集が残る(self, qapp, tmp_path, profile) -> None:
        """**一番壊れやすいところ。** 書いた内容が選び直しで消えてはいけない."""
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        d.text_edit.setPlainText("CQ TEST DE {自局コール} K")

        d.list_widget.setCurrentRow(2)
        d.list_widget.setCurrentRow(0)

        assert d.text_edit.toPlainText() == "CQ TEST DE {自局コール} K"

    def test_名前を変えると一覧も変わる(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        d.name_edit.setText("CQ (欧文)")
        assert d.list_widget.item(0).text().startswith("CQ (欧文)")

    def test_モードを変えると一覧も変わる(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        values = [v for v, _ in MODE_CHOICES]
        d.mode_combo.setCurrentIndex(values.index("japanese"))
        assert "和文" in d.list_widget.item(0).text()
        assert d.templates()[0].mode == "japanese"

    def test_名前が空なら仮の名前を付ける(self, qapp, tmp_path, profile) -> None:
        """名前の無い型は一覧で選べなくなる."""
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        d.name_edit.setText("   ")
        assert d.templates()[0].name == "新しい型"


class TestAddCopyDelete:
    def test_追加すると末尾に増えて選ばれる(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.add_btn.click()
        assert d.list_widget.count() == 4
        assert d.list_widget.currentRow() == 3
        assert d.name_edit.text() == "新しい型"

    def test_空からでも追加できる(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, [])
        d.add_btn.click()
        assert d.list_widget.count() == 1
        assert d.name_edit.isEnabled() is True

    def test_複製は中身ごと写す(self, qapp, tmp_path, profile) -> None:
        """**似た型を作るのが一番よくある編集である.**"""
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(1)

        d.copy_btn.click()

        assert d.list_widget.count() == 4
        copied = d.templates()[2]
        assert copied.name == "応答 の複製"
        assert copied.mode == "japanese"
        assert copied.text == "{HORE}コンニチハ{RATA}"

    def test_複製は隣に入る(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        d.copy_btn.click()
        assert [t.name for t in d.templates()] == ["CQ", "CQ の複製", "応答", "締め"]

    def test_削除する(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(1)

        d.delete_btn.click()

        assert [t.name for t in d.templates()] == ["CQ", "締め"]

    def test_最後の一つを消しても落ちない(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, [ReplyTemplate(name="CQ", mode="any", text="K")])
        d.delete_btn.click()
        assert d.templates() == []
        assert d.name_edit.isEnabled() is False

    def test_消した型が書き戻らない(self, qapp, tmp_path, profile) -> None:
        """消す直前の欄の内容が、別の型に書き戻されないこと."""
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(1)
        d.name_edit.setText("消される")

        d.delete_btn.click()

        assert [t.name for t in d.templates()] == ["CQ", "締め"]


class TestReorder:
    def test_上へ(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(2)

        d.up_btn.click()

        assert [t.name for t in d.templates()] == ["CQ", "締め", "応答"]
        assert d.list_widget.currentRow() == 1

    def test_下へ(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)

        d.down_btn.click()

        assert [t.name for t in d.templates()] == ["応答", "CQ", "締め"]
        assert d.list_widget.currentRow() == 1

    def test_端では押せない(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        assert d.up_btn.isEnabled() is False
        d.list_widget.setCurrentRow(2)
        assert d.down_btn.isEnabled() is False

    def test_並べ替えても編集が残る(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(2)
        d.text_edit.setPlainText("TU 73 GB {SK}")

        d.up_btn.click()

        assert d.templates()[1].text == "TU 73 GB {SK}"


class TestPreview:
    """**型の本文だけを見ても、何が電波に出るかは分からない.**"""

    def test_経歴を差し込んだ結果を出す(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        assert d.preview_view.toPlainText() == "CQ CQ DE JH0ILL K"

    def test_和文の型では和文側が入る(self, qapp, tmp_path, profile) -> None:
        d = _dialog(
            tmp_path, profile,
            [ReplyTemplate(name="設備", mode="japanese", text="{HORE}リグ ハ {リグ}{RATA}")],
        )
        assert "「FT991」" in d.preview_view.toPlainText()

    def test_書き換えると追いつく(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        d.text_edit.setPlainText("DE {自局コール}")
        assert d.preview_view.toPlainText() == "DE JH0ILL"

    def test_空の欄は疑問符で出る(self, qapp, tmp_path) -> None:
        d = _dialog(
            tmp_path, OperatorProfile(),
            [ReplyTemplate(name="CQ", mode="european", text="DE {自局コール} K")],
        )
        assert d.preview_view.toPlainText() == "DE ? K"


class TestWarnings:
    def test_ホレの中の欧文を見つける(self, qapp, tmp_path, profile) -> None:
        """**一番よく起きる書き間違い。交信中に気づくのでは遅い.**"""
        d = _dialog(
            tmp_path, profile,
            [ReplyTemplate(name="応答", mode="japanese", text="{HORE}コンニチハ RST 599{RATA}")],
        )
        assert any("送信できない文字" in w and "応答" in w for w in d.warnings())

    def test_知らない欄を見つける(self, qapp, tmp_path, profile) -> None:
        d = _dialog(
            tmp_path, profile,
            [ReplyTemplate(name="変な型", mode="any", text="DE {名前まえ} K")],
        )
        assert any("知らない欄" in w and "名前まえ" in w for w in d.warnings())

    def test_正しい型では警告しない(self, qapp, tmp_path, profile) -> None:
        assert _dialog(tmp_path, profile, _three()).warnings() == []

    def test_編集中の内容も見る(self, qapp, tmp_path, profile) -> None:
        """**保存する前の欄をそのまま検証する** (書き戻してから調べる)."""
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(1)
        d.text_edit.setPlainText("{HORE}コンニチハ RST 599{RATA}")
        assert any("送信できない文字" in w for w in d.warnings())


class TestSave:
    def test_保存して読み直せる(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        d.name_edit.setText("CQ (欧文)")
        d.text_edit.setPlainText("CQ TEST DE {自局コール} K")

        d.save()

        loaded = load_templates(tmp_path / "templates.json")
        assert loaded[0].name == "CQ (欧文)"
        assert loaded[0].text == "CQ TEST DE {自局コール} K"
        assert loaded[0].mode == "european"

    def test_並びも保存される(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.list_widget.setCurrentRow(0)
        d.down_btn.click()

        d.save()

        loaded = load_templates(tmp_path / "templates.json")
        assert [t.name for t in loaded] == ["応答", "CQ", "締め"]

    def test_警告があっても保存する(self, qapp, tmp_path, profile) -> None:
        """**書きかけを保存できないと育てるのが苦痛になる.**"""
        d = _dialog(
            tmp_path, profile,
            [ReplyTemplate(name="応答", mode="japanese", text="{HORE}コンニチハ RST 599{RATA}")],
        )

        d.save()

        assert load_templates(tmp_path / "templates.json")[0].name == "応答"
        assert "保存しました" in d.status_label.text()
        assert "送信できない文字" in d.status_label.text()

    def test_問題が無ければ報告だけ(self, qapp, tmp_path, profile) -> None:
        d = _dialog(tmp_path, profile, _three())
        d.save()
        assert d.status_label.text() == "保存しました"

    def test_保存できなければ黙らせない(self, qapp, tmp_path, profile, monkeypatch) -> None:
        d = _dialog(tmp_path, profile, _three())

        import src.app.template_dialog as module

        def boom(*args, **kwargs):
            raise OSError("書き込み失敗")

        monkeypatch.setattr(module, "save_templates", boom)
        d.save()

        assert "保存できませんでした" in d.status_label.text()


class TestOpenedFromTxDialog:
    """送信ダイアログとの繋ぎ目.

    **単体で通っても層をまたぐと壊れる。** このリポジトリで繰り返し踏んでいる。
    """

    def _tx(self, tmp_path, profile, templates, mode="auto"):
        from src.app.tx_dialog import TxDialog
        from src.infer.settings import AppSettings

        path = tmp_path / "templates.json"
        save_templates(templates, path)
        return TxDialog(
            AppSettings(tx_endpoint="127.0.0.1:45679"),
            profile=profile,
            templates_path=path,
            mode=mode,
        )

    def test_型の編集ボタンがある(self, qapp, tmp_path, profile) -> None:
        assert self._tx(tmp_path, profile, _three()).edit_templates_btn.isEnabled()

    def test_閉じたら一覧を作り直す(self, qapp, tmp_path, profile, monkeypatch) -> None:
        """**書いてすぐ使えないと確かめようがない.**"""
        d = self._tx(tmp_path, profile, _three())
        assert d.template_combo.count() == 3

        save_templates(
            [*_three(), ReplyTemplate(name="追加した型", mode="any", text="K")],
            tmp_path / "templates.json",
        )
        monkeypatch.setattr(
            "src.app.template_dialog.TemplateDialog.exec", lambda self: 0
        )
        d.open_template_dialog()

        assert d.template_combo.count() == 4
        assert d.template_combo.itemText(3) == "追加した型"

    def test_モードの絞り込みも効き直す(self, qapp, tmp_path, profile, monkeypatch) -> None:
        """**作り直しで絞り込みを落とさないこと** (`auto` 以外で効く)."""
        d = self._tx(tmp_path, profile, _three(), mode="japanese")
        assert d.template_combo.count() == 2      # 和文 + どちらでも

        save_templates(
            [*_three(), ReplyTemplate(name="欧文の型", mode="european", text="K")],
            tmp_path / "templates.json",
        )
        monkeypatch.setattr(
            "src.app.template_dialog.TemplateDialog.exec", lambda self: 0
        )
        d.open_template_dialog()

        assert d.template_combo.count() == 2      # 増えた欧文の型は出ない

    def test_編集した型がそのまま使える(self, qapp, tmp_path, profile, monkeypatch) -> None:
        """繋ぎ目の本番: 型を書く → 一覧 → 差し込み → 送信文."""
        d = self._tx(tmp_path, profile, _three(), mode="european")

        save_templates(
            [ReplyTemplate(name="CQ", mode="european", text="CQ TEST DE {自局コール} K")],
            tmp_path / "templates.json",
        )
        monkeypatch.setattr(
            "src.app.template_dialog.TemplateDialog.exec", lambda self: 0
        )
        d.open_template_dialog()
        d.apply_template()

        assert d.japanese_edit.toPlainText() == "CQ TEST DE JH0ILL K"

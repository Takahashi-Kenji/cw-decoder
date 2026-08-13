"""返信の型のテスト."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tx.encoder import build_sequence
from src.tx.profile import BilingualField, OperatorProfile
from src.tx.reading import to_sendable_kana
from src.tx.templates import (
    PLACEHOLDERS,
    ReplyTemplate,
    fill,
    load_templates,
    profile_values,
    save_templates,
    templates_for_mode,
    unsendable_in_template,
)


class TestFill:
    def test_欄を差し込む(self) -> None:
        assert fill("{相手コール} DE {自局コール}", {"相手コール": "JA1ABC", "自局コール": "JH0ILL"}) == (
            "JA1ABC DE JH0ILL"
        )

    def test_知らない欄はそのまま残す(self) -> None:
        """**これが効くのは {HORE} と {RATA} である。**

        型の中に書いたマーカーが差し込みで壊れてはいけない。
        """
        assert fill("{HORE}コンニチハ{RATA}", {"相手コール": "JA1ABC"}) == "{HORE}コンニチハ{RATA}"

    def test_マーカーと欄が混ざっていても壊れない(self) -> None:
        text = "{相手コール} DE {自局コール} {HORE}コンニチハ{RATA} K"
        got = fill(text, {"相手コール": "JA1ABC", "自局コール": "JH0ILL"})
        assert got == "JA1ABC DE JH0ILL {HORE}コンニチハ{RATA} K"

    def test_同じ欄が二度出てもよい(self) -> None:
        assert fill("{相手コール} {相手コール}", {"相手コール": "JA1ABC"}) == "JA1ABC JA1ABC"

    def test_閉じていない欄は触らない(self) -> None:
        """``{相手コール`` のように閉じ括弧が無いものは置換の対象にならない.

        壊れやすい点: ``[^}]*`` のような正規表現だと、この閉じていない ``{`` が
        後方の正しい欄まで飲み込んで、正しい欄が置換もされず missing にも
        現れないという「漏れが漏れとして気づけない」事故になる。ここでは
        後方の ``{自局コール}`` はちゃんと埋まらなければならない。
        """
        text = "{相手コール DE {自局コール}"
        assert fill(text, {"相手コール": "JA1ABC", "自局コール": "JH0ILL"}) == "{相手コール DE JH0ILL"

    def test_二重の中括弧は内側だけを欄として扱う(self) -> None:
        """``{{相手コール}}`` は外側を文字として残し、内側だけ差し込む."""
        assert fill("{{相手コール}}", {"相手コール": "JA1ABC"}) == "{JA1ABC}"

    def test_空の中括弧はそのまま残る(self) -> None:
        assert fill("{}", {"相手コール": "JA1ABC"}) == "{}"

    def test_1回のfillでは差し込んだ値の中の欄記法は展開されない(self) -> None:
        """置換後の文字列を再スキャンしない (re.sub の 1 パスの性質を確認する)."""
        got = fill("{相手コール}", {"相手コール": "{自局コール}", "自局コール": "JH0ILL"})
        assert got == "{自局コール}"

    def test_fillを2回続けて呼ぶとたまたま欄記法に見える値も展開される(self) -> None:
        """**注意点**: 値の中にたまたま ``{欄名}`` の形が含まれていて、その値を
        別の fill にもう一度通すと、意図せず展開される。経歴の値 (コールサイン等)
        は通常 ``{}`` を含まないので実害は無いが、呼び出し側は fill を 1 回の
        差し込みで完結させ、結果を再度 fill に通さない設計にすること。
        """
        once = fill("{相手コール}", {"相手コール": "{自局コール}"})
        assert once == "{自局コール}"
        twice = fill(once, {"自局コール": "JH0ILL"})
        assert twice == "JH0ILL"


class TestUnknownMarker:
    """**埋まらなかった欄は ``?`` になる** (2026-08-12)。

    以前は空の値を差し込まず ``missing_placeholders`` が名指しし、
    「埋まっていない欄があります」で送信を止めていた。**その仕組みは廃止した。**
    運用者は送信文に出た ``?`` を見て、必要なら直してから確認・送信する
    (設計書 §2.2)。``[確認]`` の関門そのものは変わらない。
    """

    def test_値が無ければ疑問符になる(self) -> None:
        assert fill("{相手コール} DE JH0ILL", {}) == "? DE JH0ILL"

    def test_空文字も疑問符になる(self) -> None:
        assert fill("{相手コール} DE JH0ILL", {"相手コール": ""}) == "? DE JH0ILL"

    def test_マーカーは触らない(self) -> None:
        """``{HORE}`` は ``PLACEHOLDERS`` に無いので欄ではない."""
        assert fill("{HORE}コンニチハ{RATA}", {}) == "{HORE}コンニチハ{RATA}"

    def test_知らない語も触らない(self) -> None:
        assert fill("{SK}", {}) == "{SK}"

    def test_疑問符は両方の符号表にある(self) -> None:
        """だから和文でも欧文でも送れる (この仕組みの前提)."""
        from src.tx.encoder import find_unsendable

        assert find_unsendable("? DE JH0ILL") == ()
        assert find_unsendable("{HORE}ナマエ ハ ?{RATA}") == ()

    def test_全部埋まっていれば疑問符は出ない(self) -> None:
        filled = fill("{相手コール} DE {自局コール}",
                      {"相手コール": "JA1ABC", "自局コール": "JH0ILL"})
        assert filled == "JA1ABC DE JH0ILL"


class TestProfileValues:
    """**差し込む値は型のモードで変わる。**

    欧文の型に読み (カタカナ) を差し込むと、そのカタカナのせいでモードが和文に
    倒れ、**文中の欧文がまるごと送れなくなる** (実測: ``NAME タロウ`` を入れた
    欧文の型が ``JAABCDEJQCWGMURRST…`` と全滅した)。和文の型のときだけ読みを使う。
    """

    def test_和文の型では読みを使う(self) -> None:
        profile = OperatorProfile(
            callsign="JH0ILL",
            name=BilingualField(european="TARO", japanese="タロウ"),
        )
        values = profile_values(profile, "japanese")
        assert values["名前"] == "タロウ"

    def test_欧文の型では表示形を使う(self) -> None:
        profile = OperatorProfile(name=BilingualField(european="TARO", japanese="タロウ"))
        assert profile_values(profile, "european")["名前"] == "TARO"

    def test_どちらでもの型では表示形を使う(self) -> None:
        """``any`` の型は欧文の略語が主なので表示形に寄せる."""
        profile = OperatorProfile(name=BilingualField(european="TARO", japanese="タロウ"))
        assert profile_values(profile, "any")["名前"] == "TARO"

    def test_もう一方の側で代用しない(self) -> None:
        """**和文側が空でも欧文側の値は使わない** (設計書 §2.2).

        代用すると、書き忘れと「わざと同じにした」の区別がつかなくなる。
        空なら ``?`` を打ち、運用者が送信文を見て判断する。
        """
        profile = OperatorProfile(rig=BilingualField(european="FT991"))
        assert profile_values(profile, "japanese")["リグ"] == "?"
        assert profile_values(profile, "european")["リグ"] == "FT991"

    def test_空の欄も入れる(self) -> None:
        """**欄による例外を作らない。** 値は ``?`` になる."""
        values = profile_values(OperatorProfile(), "japanese")
        assert values["リグ"] == "?"
        assert values["名前"] == "?"
        assert values["自局コール"] == "?"

    def test_コールサインはどちらのモードでも同じ(self) -> None:
        """和文の交信でもコールサインは欧文で送る."""
        profile = OperatorProfile(callsign="JH0ILL")
        assert profile_values(profile, "japanese")["自局コール"] == "JH0ILL"
        assert profile_values(profile, "european")["自局コール"] == "JH0ILL"


class TestReplyTemplate:
    def test_三つの欄を持つ(self) -> None:
        t = ReplyTemplate(name="応答", mode="japanese", text="{相手コール} K")
        assert (t.name, t.mode, t.text) == ("応答", "japanese", "{相手コール} K")

    def test_差し込める欄の一覧がある(self) -> None:
        assert "相手コール" in PLACEHOLDERS
        assert "自局コール" in PLACEHOLDERS
        assert "RST" in PLACEHOLDERS
        assert "HORE" not in PLACEHOLDERS       # マーカーは欄ではない


class TestStore:
    def test_書いて読み戻せる(self, tmp_path) -> None:
        path = tmp_path / "templates.json"
        original = [
            ReplyTemplate(name="CQ", mode="european", text="CQ CQ DE {自局コール} K"),
            ReplyTemplate(name="応答", mode="japanese", text="{相手コール} DE {自局コール} K"),
        ]
        save_templates(original, path)
        assert load_templates(path) == original

    def test_ファイルが無ければ空(self, tmp_path) -> None:
        assert load_templates(tmp_path / "ない.json") == []

    def test_壊れたファイルは空として扱う(self, tmp_path) -> None:
        path = tmp_path / "templates.json"
        path.write_text("{壊れている", encoding="utf-8")
        assert load_templates(path) == []

    def test_壊れたファイルを上書きしない(self, tmp_path) -> None:
        """**読んだだけで消してはいけない。** 手で直せる余地を残す."""
        path = tmp_path / "templates.json"
        path.write_text("{壊れている", encoding="utf-8")
        load_templates(path)
        assert path.read_text(encoding="utf-8") == "{壊れている"

    def test_知らない欄があっても落ちない(self, tmp_path) -> None:
        path = tmp_path / "templates.json"
        path.write_text(
            json.dumps({"templates": [{"name": "A", "text": "K", "未知": 1}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        assert load_templates(path) == [ReplyTemplate(name="A", mode="any", text="K")]

    def test_日本語をそのまま書く(self, tmp_path) -> None:
        """ログと同じで、人が開いて読めることに価値がある."""
        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="応答", text="コンニチハ")], path)
        assert "コンニチハ" in path.read_text(encoding="utf-8")

    def test_templatesが配列でなければ空(self, tmp_path) -> None:
        """``{"templates": "文字列"}`` のような壊れ方でも落ちずに空を返す."""
        path = tmp_path / "templates.json"
        path.write_text(json.dumps({"templates": "文字列"}), encoding="utf-8")
        assert load_templates(path) == []

    def test_型の要素が辞書でなければ無視する(self, tmp_path) -> None:
        """``{"templates": [1, 2]}`` のような要素は型として扱えないので無視する."""
        path = tmp_path / "templates.json"
        path.write_text(json.dumps({"templates": [1, 2]}), encoding="utf-8")
        assert load_templates(path) == []

    def test_欄の値がnullでも落ちない(self, tmp_path) -> None:
        """``{"name": null}`` のように壊れた値でも ``str(None)`` の ``"None"``
        を書き込んではいけない。既定値 (空文字/``"any"``) に倒す。
        """
        path = tmp_path / "templates.json"
        path.write_text(
            json.dumps({"templates": [{"name": None, "mode": None, "text": None}]}),
            encoding="utf-8",
        )
        assert load_templates(path) == [ReplyTemplate(name="", mode="any", text="")]

    def test_大量の型でも読み書きできる(self, tmp_path) -> None:
        """巨大なファイルでも落ちず、内容が正しく往復すること."""
        path = tmp_path / "templates.json"
        many = [ReplyTemplate(name=f"型{i}", text=f"K{i}") for i in range(2000)]
        save_templates(many, path)
        assert load_templates(path) == many

    def test_保存が失敗しても既存のファイルは壊れない(self, tmp_path, monkeypatch) -> None:
        """**書き込みの途中で失敗しても、既存の正しいファイルを壊してはいけない。**

        一時ファイルに書いてから置き換える実装であることを、``json.dump`` を
        壊して確認する。
        """
        path = tmp_path / "templates.json"
        save_templates([ReplyTemplate(name="元", text="K")], path)
        original = path.read_text(encoding="utf-8")

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("書き込み中に落ちた")

        monkeypatch.setattr(json, "dump", _boom)
        with pytest.raises(RuntimeError):
            save_templates([ReplyTemplate(name="新", text="K")], path)

        assert path.read_text(encoding="utf-8") == original
        # 一時ファイルも残さない (ゴミが溜まらないように)
        assert list(tmp_path.glob("*.tmp")) == []


class TestValidation:
    def test_送れる型は空を返す(self) -> None:
        t = ReplyTemplate(
            name="応答",
            mode="japanese",
            text="{相手コール} DE {自局コール} RST {RST} {HORE}コンニチハ{RATA} K",
        )
        assert unsendable_in_template(t) == ""

    def test_小書きカナは送れる扱いになる(self) -> None:
        """**実際の経路では小書きが自動で倒される** (reading.SMALL_KANA_MAP)。

        生の型だけを見ると ``キョウテン`` の ``ョ`` が符号表に無いので
        「送れない」と誤判定する。検証は実際に通る道と同じ道を通ること。
        """
        t = ReplyTemplate(name="小書き", mode="japanese", text="{HORE}キョウテン{RATA}")
        assert unsendable_in_template(t) == ""

    def test_ホレの中の欧文を見つける(self) -> None:
        """**型を書くときに一番間違えやすいところ** (設計書 §4.3).

        RST は欧文で送るものなので、ホレの中に書くと送れない。
        """
        t = ReplyTemplate(name="悪い例", mode="japanese", text="{HORE}コンニチハ RST 599{RATA}")
        assert "R" in unsendable_in_template(t)

    def test_符号表に無い文字を見つける(self) -> None:
        """``+`` は AR プロサインとして符号表にあるため使えない。

        ``#`` のような表に本当に無い文字で確認する (brief 原文の ``+++`` は
        符号表に実在したため、この検証には使えなかった)。
        """
        t = ReplyTemplate(name="悪い例", mode="european", text="CQ DE JH0ILL ###")
        assert "#" in unsendable_in_template(t)

    def test_欄は仮の値で埋めてから調べる(self) -> None:
        """**欄が空のせいで「送れない」と言ってはいけない。**

        埋まっていないことは :func:`missing_placeholders` の仕事である。
        """
        t = ReplyTemplate(name="CQ", mode="european", text="CQ DE {自局コール} K")
        assert unsendable_in_template(t) == ""

    def test_本文が空でも落ちない(self) -> None:
        """``to_sendable_kana`` は空文字を早期リターンするが、その経路が
        壊れていないことを確認する (pykakasi を呼ばずに落ちない)。
        """
        t = ReplyTemplate(name="空", text="")
        assert unsendable_in_template(t) == ""

    def test_経歴の読みがホレの外へ出るのを見つける(self) -> None:
        """**仮の値だけでは原理的に見つからない失敗** (2026-08-11 レビュー I4).

        和文の型で ``{リグ}`` をホレの**外**に置くと、``profile_values`` が
        読み (カタカナ) を返すのでカタカナが欧文の区間に出て、その区間の
        欧文がまるごと送れなくなる。仮の値は数字なのでどちらの表にもあり、
        経歴を渡さない検証ではこの型が「送れる」と判定されてしまう。
        """
        t = ReplyTemplate(
            name="設備", mode="japanese", text="DE {自局コール} RIG {リグ} {HORE}デス{RATA} K"
        )
        profile = OperatorProfile(
            callsign="JH0ILL",
            rig=BilingualField(european="FT-991", japanese="エフティー キュウキュウイチ"),
        )
        assert unsendable_in_template(t, profile) != ""
        # 経歴を渡さない (仮の値だけの) 検証では見つからない = 渡すことに意味がある
        assert unsendable_in_template(t) == ""

    def test_読みをホレの中に置いた型は送れる(self) -> None:
        """上と同じ経歴でも、読みがホレの**中**にあれば送れる (直し方の確認)."""
        t = ReplyTemplate(
            name="設備", mode="japanese", text="DE {自局コール} {HORE}リグ ハ {リグ} デス{RATA} K"
        )
        profile = OperatorProfile(
            callsign="JH0ILL",
            rig=BilingualField(european="FT-991", japanese="エフティー キュウキュウイチ"),
        )
        assert unsendable_in_template(t, profile) == ""

    def test_欧文側と和文側が同じ文字列でも二重括弧にならない(self) -> None:
        """**2026-08-12 に構造で直した欠陥の回帰テスト。**

        以前は ``to_sendable_kana`` が「表示形 → 読み」の辞書
        (``field_readings``) で、欄を差し込んだ**後**の文をもう一度スキャン
        していた。表示形とブラケットの中身が同じ文字列 (``50W`` → ``「50W」``)
        だと、差し込み済みの ``「50W」`` の中の ``50W`` がまた表示形として
        マッチし、``「「50W」」`` と二重に括弧が付いて送れなかった。

        経歴を独立した 2 値にして ``field_readings`` を消したので、
        **差し込んだ後の文をスキャンしなくなった。** 起こりようがない。
        """
        t = ReplyTemplate(
            name="設備", mode="japanese", text="{HORE}シュツリョク ハ {出力} デス{RATA} K"
        )
        profile = OperatorProfile(power=BilingualField(european="50W", japanese="「50W」"))
        assert unsendable_in_template(t, profile) == ""

    def test_欧文側に空白があっても和文の型には影響しない(self) -> None:
        """和文の型は和文側しか見ない (もう一方を混ぜない)."""
        t = ReplyTemplate(
            name="設備", mode="japanese", text="{HORE}シュツリョク ハ {出力} デス{RATA} K"
        )
        profile = OperatorProfile(power=BilingualField(european="50 W", japanese="「50W」"))
        assert unsendable_in_template(t, profile) == ""


class TestModeFilter:
    @pytest.fixture
    def 型たち(self) -> list[ReplyTemplate]:
        return [
            ReplyTemplate(name="欧文", mode="european", text="K"),
            ReplyTemplate(name="和文", mode="japanese", text="K"),
            ReplyTemplate(name="どちらでも", mode="any", text="K"),
        ]

    def test_欧文モードでは和文の型を出さない(self, 型たち) -> None:
        names = [t.name for t in templates_for_mode(型たち, "european")]
        assert names == ["欧文", "どちらでも"]

    def test_和文モードでは欧文の型を出さない(self, 型たち) -> None:
        names = [t.name for t in templates_for_mode(型たち, "japanese")]
        assert names == ["和文", "どちらでも"]

    def test_自動モードでは両方の型が出る(self, 型たち) -> None:
        """**``auto`` は主力の運用モードである。**

        ``auto`` は「相手に合わせて欧文と和文を切り替える」モードなので、
        どちらの型も使う。以前はここが未知のモード扱いで、``any`` 以外の型が
        すべて消えていた (例文 10 個中 9 個が消えた)。しかも理由が出なかった。
        """
        names = [t.name for t in templates_for_mode(型たち, "auto")]
        assert names == ["欧文", "和文", "どちらでも"]

    def test_知らないモードでもanyの型だけは出る(self, 型たち) -> None:
        """未知のモード文字列でも落ちず、``any`` の型だけが残る (fail-closed).

        ``auto`` は ``DisplayMode`` の正規の値なので**ここには当たらない**
        (上のテスト)。当たるのは型ファイルの手編集で壊れた値などである。
        """
        names = [t.name for t in templates_for_mode(型たち, "klingon")]  # type: ignore[arg-type]
        assert names == ["どちらでも"]


class TestExampleFile:
    """添える例文が**実際に送れる**こと.

    運用者が最初に触るのがこれなので、送れない型が混ざっていてはいけない。
    """

    @pytest.fixture
    def 例文(self) -> list[ReplyTemplate]:
        path = Path(__file__).resolve().parent.parent / "docs" / "reply_templates_example.json"
        return load_templates(path)

    @pytest.fixture
    def 読みを入れた経歴(self) -> OperatorProfile:
        """**和文を運用する人なら普通に入れる**経歴 (2026-08-11 レビュー I4).

        コールサインにだけ読みを入れないのは、``JH0ILL`` がそのまま送れる
        文字列であり、読みを入れると ``to_sendable_kana`` の読み辞書が
        ホレの外の ``JH0ILL`` までカタカナに置き換えてしまうためである
        (この置き換えがホレ境界を無視する件は別作業として引き継ぐ。
        ``docs/USAGE.md`` §13 に運用上の注意として書いてある)。
        """
        return OperatorProfile(
            callsign="JH0ILL",
            name=BilingualField(european="TARO", japanese="タロウ"),
            qth=BilingualField(european="KANAGAWA", japanese="カナガワケン ヨコハマシ"),
            rig=BilingualField(european="FT-991", japanese="エフティー キュウキュウイチ"),
            antenna=BilingualField(european="DIPOLE", japanese="ダイポール"),
            power=BilingualField(european="50W", japanese="ゴジュウ ワット"),
        )

    @pytest.fixture
    def 欧文区間を使った経歴(self) -> OperatorProfile:
        """**`「…」` (欧文区間) が使えるようになった運用者の経歴** (2026-08-12).

        以前は和文の型でリグ名等を送るために発音のカタカナ (``エフティー
        キュウキュウイチ``) を読み欄に書く必要があったが、``「…」`` により
        実際の値をそのまま書けるようになった (``FT-991`` → ``「FT991」``)。
        長音・ハイフンは使わない書き方にしてある (``docs/USAGE.md`` §13)。

        **表示形はブラケットの中身と違う文字列にしてある** (``FT-991`` の
        ハイフン、``50 W`` の半角スペース)。同じ文字列にすると、経歴の
        読み辞書が差し込み後の文をもう一度スキャンして二重に括弧が付く
        落とし穴があるため (``TestValidation`` 参照)。
        """
        return OperatorProfile(
            callsign="JH0ILL",
            name=BilingualField(european="TARO", japanese="タロウ"),
            qth=BilingualField(european="KANAGAWA", japanese="カナガワケン ヨコハマシ"),
            rig=BilingualField(european="FT-991", japanese="「FT991」"),
            antenna=BilingualField(european="DIPOLE", japanese="「DP」"),
            power=BilingualField(european="50 W", japanese="「50W」"),
        )

    def test_十個ある(self, 例文) -> None:
        assert len(例文) == 10

    def test_全部送れる(self, 例文) -> None:
        for t in 例文:
            assert unsendable_in_template(t) == "", f"{t.name} が送れない"

    def test_読みを入れた経歴でも全部送れる(self, 例文, 読みを入れた経歴) -> None:
        """**経歴に読みを入れた運用者でも 10 個すべてが送れること** (I4).

        以前は「設備の紹介」が ``{リグ}{アンテナ}{出力}`` をホレの外に置いて
        いたため、読みを入れた運用者だけがこの型を送れなかった
        (実測: ``送信できない文字があります: JAABCDEJQCWRIGANTPWR``)。
        """
        for t in 例文:
            assert unsendable_in_template(t, 読みを入れた経歴) == "", f"{t.name} が送れない"

    def test_設備の紹介がホレの中で完結する(self, 例文) -> None:
        """``「…」`` があるので、欧文をホレの外に出す必要がなくなった.

        以前は ``RIG {リグ} ANT {アンテナ}`` をホレの外に置いていた
        (和文モードの中では R I G が送れないため)。``「…」`` で中に戻せる。
        """
        設備 = next(t for t in 例文 if t.name == "設備の紹介")
        ホレの前 = 設備.text.split("{HORE}")[0]
        assert "RIG" not in ホレの前
        assert "ANT" not in ホレの前
        assert "PWR" not in ホレの前

    def test_欧文区間を使った経歴でも全部送れる(self, 例文, 欧文区間を使った経歴) -> None:
        """**実際の経歴の値 (`「FT991」` `「DP」` `「50W」` `タロウ` `ヨコハマシ`)
        を入れて 10 個すべてが送れること** (brief の Step 5 相当)。

        brief の Step 5 のコマンドは ``fill()`` に直接値を渡すだけで、
        実際のアプリの経路 (``profile_values`` → ``fill`` → ``to_sendable_kana``
        に**同じ経歴を渡す**経路) を通らない。その経路だけで踏む落とし穴が
        あったため (``TestValidation`` の二重括弧のテスト参照)、ここでは
        ``unsendable_in_template(t, profile)`` で実際の経路を通して確かめる。
        """
        for t in 例文:
            assert unsendable_in_template(t, 欧文区間を使った経歴) == "", f"{t.name} が送れない"

    def test_設備の紹介は語間を保ったまま変換される(self, 例文, 欧文区間を使った経歴) -> None:
        """型 → 差し込み → 変換という実際の経路でも、``」`` の直後の語間が
        消えないこと (brief の「語間が保たれているか」の確認)。

        encoder 側の単体テストで ``「…」`` の語間保持は確認済みだが
        (``tests/test_tx_encoder.py`` の ``TestEuropeanSpanWordGaps``)、実際の
        経路 (型 → ``fill`` → ``to_sendable_kana`` → ``build_sequence``) を
        通しても保たれることを、所要秒数で確かめる。符号列や ``WORD_BREAK``
        トークンの個数を見ても分からない (``encode('A B')`` は ``WORD_BREAK``
        を含まない。所要秒数は要素の長さとして語間を表現する)。
        """
        設備 = next(t for t in 例文 if t.name == "設備の紹介")
        values = dict(profile_values(欧文区間を使った経歴, 設備.mode))
        values["相手コール"] = "JA1ABC"
        filled = fill(設備.text, values)
        converted = to_sendable_kana(filled, 欧文区間を使った経歴).text
        assert unsendable_in_template(設備, 欧文区間を使った経歴) == ""
        正しい所要 = build_sequence(converted, 20.0).total_seconds

        # 「」 の直後にあるはずの語間 (半角スペース) をわざと 1 つ消してみて、
        # 所要秒数が短くなることを確認する。消す対象の空白が実在しなければ
        # (= 語間が既に失われていれば) replace は何もせず、この assert 自体が
        # 落ちて回帰に気づける。
        assert "」 " in converted
        削った = converted.replace("」 ", "」", 1)
        assert build_sequence(削った, 20.0).total_seconds < 正しい所要

    def test_名前が重複していない(self, 例文) -> None:
        names = [t.name for t in 例文]
        assert len(names) == len(set(names))

    def test_モードが正しい値(self, 例文) -> None:
        for t in 例文:
            assert t.mode in {"european", "japanese", "any"}, f"{t.name} の mode が不正"

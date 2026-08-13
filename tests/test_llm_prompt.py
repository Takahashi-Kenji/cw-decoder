"""LLM プロンプト構築のテスト."""
from src.llm.prompt import build_kana_to_european


def test_kana_to_european_derives_from_morse_tables():
    table = build_kana_to_european()
    # ・- は欧文 A / 和文 イ (同符号) → イ→A
    assert table["イ"] == "A"
    # ・ は欧文 E / 和文 ヘ → ヘ→E
    assert table["ヘ"] == "E"
    # -・ は欧文 N / 和文 タ → タ→N
    assert table["タ"] == "N"


def test_kana_to_european_is_consistent_with_source_tables():
    from src.tokens.morse_tokens import EUROPEAN_TABLE, JAPANESE_TABLE
    table = build_kana_to_european()
    for code, kana in JAPANESE_TABLE.items():
        if code in EUROPEAN_TABLE and kana not in ("゛", "゜"):
            # 各エントリは元テーブルと矛盾しない
            assert table.get(kana) == EUROPEAN_TABLE[code]


from src.llm.prompt import build_messages


def test_build_messages_returns_system_and_user():
    msgs = build_messages("CQ CQ DE JH0ILL", mode="european")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert "CQ CQ DE JH0ILL" in msgs[1]["content"]


def test_system_prompt_mentions_marker_and_correction():
    msgs = build_messages("ABC", mode="european")
    system = msgs[0]["content"]
    assert "⟦" in system and "⟧" in system   # マーカー説明あり
    assert "訂正" in system                    # 誤り訂正の指示あり


def test_japanese_mode_includes_kana_european_table():
    msgs = build_messages("イロハ", mode="japanese")
    system = msgs[0]["content"]
    assert "欧文化" in system
    assert "イ" in system and "A" in system    # 対応表が埋め込まれている


def test_european_mode_excludes_european_conversion():
    msgs = build_messages("ABC", mode="european")
    system = msgs[0]["content"]
    assert "欧文化" not in system              # 欧文モードでは欧文化指示なし


def test_auto_mode_includes_kana_european_table():
    msgs = build_messages("イロハ", mode="auto")
    system = msgs[0]["content"]
    assert "欧文化" in system
    assert "イ" in system and "A" in system


def test_european_mode_keeps_european_no_japanese_translation():
    """欧文モードは欧文のまま整形し、日本語へ翻訳しない指示を含む."""
    msgs = build_messages("CQ DE JH0ILL", mode="european")
    system = msgs[0]["content"]
    assert "翻訳しない" in system          # 日本語翻訳を禁止
    assert "日本語への清書" not in system   # 日本語清書の指示は出さない


def test_japanese_mode_requests_japanese_cleanup():
    """和文モードは日本語清書を指示し、翻訳禁止文言は含まない."""
    msgs = build_messages("イロハ", mode="japanese")
    system = msgs[0]["content"]
    assert "日本語への清書" in system
    assert "翻訳しない" not in system


def test_error_correction_present_in_all_modes():
    """誤り訂正の指示はどのモードでも含まれる."""
    for mode in ("european", "japanese", "auto"):
        system = build_messages("ABC", mode=mode)[0]["content"]
        assert "訂正" in system
        assert "⟦" in system and "⟧" in system


def test_european_forbids_fabrication():
    """欧文は捏造禁止・確実さ優先.

    コールサインや RST を作られると実害が出るため、和文と違って
    読みやすさを優先させてはいけない。
    """
    system = build_messages("ABC", mode="european")[0]["content"]
    assert "捏造" in system
    assert "「確信ありげな誤り」より「不確実だと分かる出力」を優先" in system


def test_japanese_prioritises_readability_over_certainty():
    """和文は読みやすさ優先.

    電文がカタカナなのは仕様上どうにもならないので、漢字かな交じりへ直すのが
    清書の主目的である。「不確実だと分かる出力を優先」はこれと矛盾するため
    和文には入れない。
    """
    for mode in ("japanese", "auto"):
        system = build_messages("ABC", mode=mode)[0]["content"]
        assert "漢字かな交じり" in system
        assert "読みやすさを優先" in system
        assert "「確信ありげな誤り」より" not in system


def test_japanese_still_requires_marking_guesses():
    """読みやすさ優先でも、推測箇所は必ずマーカーで囲ませる."""
    system = build_messages("ABC", mode="japanese")[0]["content"]
    assert "推測した箇所は必ず" in system


def test_lead_text_is_added_as_reference_only():
    """増分清書: 直前のやり取りは文脈用で、出力させない."""
    messages = build_messages("NEW", mode="japanese", lead_text="OLD")
    system, user = messages[0]["content"], messages[1]["content"]
    assert "参考部分は出力しないこと" in system
    assert "OLD" in user
    assert "NEW" in user
    assert user.index("OLD") < user.index("NEW")


def test_no_lead_text_sends_raw_text_only():
    """従来どおりの呼び方では余計な枠を付けない."""
    messages = build_messages("ABC", mode="japanese")
    assert messages[1]["content"] == "ABC"
    assert "参考部分は出力しないこと" not in messages[0]["content"]


def test_blank_lead_text_is_treated_as_absent():
    messages = build_messages("ABC", mode="japanese", lead_text="   ")
    assert messages[1]["content"] == "ABC"


def test_no_meta_commentary_rule_present_in_all_modes():
    """謝罪・助言などのメタコメント禁止の指示はどのモードでも含まれる."""
    for mode in ("european", "japanese", "auto"):
        system = build_messages("ABC", mode=mode)[0]["content"]
        assert "メタコメント" in system
        assert "謝罪" in system


def test_european_glossary_lists_abbreviations():
    """欧文モードは有効略語のヒントを含み、壊さない指示がある."""
    system = build_messages("CQ DE", mode="european")[0]["content"]
    assert "TNX" in system and "73" in system
    assert "そのまま残す" in system


def test_japanese_glossary_lists_closing_phrases():
    """和文/auto モードは定型表現 (締め言葉) のヒントを含む."""
    for mode in ("japanese", "auto"):
        system = build_messages("オヤスミ", mode=mode)[0]["content"]
        assert "お休みなさい" in system
        assert "よろしくお願いします" in system


def test_ollama_disables_thinking_and_caps_output():
    """清書は文字の変換であって推論ではない。考えさせても待ち時間が延びるだけ."""
    from src.llm.providers.ollama import OllamaProvider

    captured: dict = {}

    def _fake_post(url, json, headers, *, timeout):
        captured.update(json)
        return {"message": {"content": "ok"}}

    import src.llm.providers.ollama as mod
    original = mod.post_json
    mod.post_json = _fake_post
    try:
        OllamaProvider(model="m").transform("ABC", "japanese", timeout=5.0)
    finally:
        mod.post_json = original

    assert captured["think"] is False
    assert captured["options"]["num_predict"] == 512


def test_prosigns_are_explained_in_japanese_modes():
    """[ホレ] を「晴れ」と読み替えられないよう、運用記号だと教える.

    実測で gemma3:4b / gemma4:e4b / gemma3:12b がそろって誤読した。
    """
    for mode in ("japanese", "auto"):
        system = build_messages("ABC", mode=mode)[0]["content"]
        assert "運用記号" in system
        assert "[ホレ]" in system
        assert "[ラタ]" in system


class TestCompactPrompt:
    """短いプロンプト (小さいローカルモデル向け).

    重いプロンプトだと 4B 前後のモデルは例文をそのまま返したり捏造したりする
    (2026-08-08 の実測。docs/llm_refine_result.md)。
    """

    def test_is_much_shorter(self) -> None:
        compact = build_messages("ABC", mode="japanese", compact=True)[0]["content"]
        full = build_messages("ABC", mode="japanese")[0]["content"]
        assert len(compact) < len(full) / 3

    def test_japanese_keeps_the_kanji_goal(self) -> None:
        system = build_messages("ABC", mode="japanese", compact=True)[0]["content"]
        assert "漢字かな交じり" in system

    def test_japanese_omits_the_worked_example(self) -> None:
        """例文を入れると小さいモデルがそれをそのまま返す (実測)."""
        system = build_messages("ABC", mode="japanese", compact=True)[0]["content"]
        assert "こんにちは。天気は晴れです" not in system

    def test_prosigns_are_one_bullet(self) -> None:
        """段落で書くと効かず、箇条書き 1 行だと効いた (実測)."""
        system = build_messages("ABC", mode="japanese", compact=True)[0]["content"]
        assert "[ホレ]" in system
        assert "運用記号" in system

    def test_markers_are_still_requested(self) -> None:
        """赤表示のために推測箇所は囲ませる."""
        system = build_messages("ABC", mode="japanese", compact=True)[0]["content"]
        assert "⟦" in system and "⟧" in system

    def test_kana_table_is_dropped(self) -> None:
        """カナ→欧文対応表 (48 行) は小さいモデルには重すぎる."""
        system = build_messages("ABC", mode="japanese", compact=True)[0]["content"]
        assert "カナ→欧文対応表" not in system

    def test_lead_text_still_works(self) -> None:
        messages = build_messages("NEW", mode="japanese", lead_text="OLD", compact=True)
        assert "参考部分は出力しないこと" in messages[0]["content"]
        assert "OLD" in messages[1]["content"]

    def test_compact_is_off_by_default_in_builder(self) -> None:
        """既定の呼び出しは従来どおり重い版 (設定側で切り替える)."""
        assert "カナ→欧文対応表" in build_messages("ABC", mode="japanese")[0]["content"]


def test_compact_setting_defaults_to_true() -> None:
    """既定のプロバイダはローカル Ollama なので、既定は短い版."""
    from src.infer.settings import AppSettings
    assert AppSettings().llm_compact_prompt is True


class TestEuropeanUsesFullPrompt:
    """短い版は和文だけ。欧文は重い版が良い (実測)."""

    def test_compact_is_ignored_for_european(self) -> None:
        """欧文で compact を指定しても重い版が返る.

        held-out 欧文 10 件で短い版は +4.60pt 悪化、重い版は -0.42pt だった。
        欧文は「触らない規律」が要るので、捏造禁止を明記した重い版が効く。
        """
        compact = build_messages("ABC", mode="european", compact=True)[0]["content"]
        full = build_messages("ABC", mode="european")[0]["content"]
        assert compact == full
        assert "捏造" in compact

    def test_compact_still_applies_to_japanese(self) -> None:
        compact = build_messages("ABC", mode="japanese", compact=True)[0]["content"]
        full = build_messages("ABC", mode="japanese")[0]["content"]
        assert compact != full
        assert len(compact) < len(full) / 3


class TestCutNumbers:
    """短縮数字 (運用者の要望)。CW は数字を文字で送る慣習がある."""

    def test_rule_is_present_in_european(self) -> None:
        system = build_messages("5NN", mode="european")[0]["content"]
        assert "5NN" in system and "599" in system

    def test_existing_digits_must_not_be_changed(self) -> None:
        """歯止めが要る。実測で正しい 559 を 599 に書き換えた."""
        system = build_messages("5NN", mode="european")[0]["content"]
        assert "既にある数字は絶対に変えない" in system

    def test_words_must_not_be_dropped(self) -> None:
        """実測で RST を丸ごと落とした."""
        system = build_messages("5NN", mode="european")[0]["content"]
        assert "語を消さない" in system

    def test_table_covers_the_common_letters(self) -> None:
        from src.llm.prompt import CUT_NUMBERS
        assert CUT_NUMBERS["N"] == "9"
        assert CUT_NUMBERS["T"] == "0"
        assert CUT_NUMBERS["A"] == "1"

    def test_lexicon_is_shared_not_copied(self) -> None:
        """語彙は EUROPEAN_LEXICON が唯一の真正ソース."""
        from src.infer.word_correct import EUROPEAN_LEXICON
        system = build_messages("ABC", mode="european")[0]["content"]
        for word in ("QRZ", "QTH", "TNX"):
            assert word in EUROPEAN_LEXICON
            assert word in system

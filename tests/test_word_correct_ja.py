"""和文の辞書補正のテスト.

**目的は「正確な文」ではなく「正確な意味」** (運用者、2026-08-14)。
したがって受け入れ基準は次のようになる。

* 内容語 (テンキ / クモリ / キオン …) が戻ること
* 助詞の取り違えは意味を壊さないので許容する
* **正しく取れていた内容語を壊さないこと** — こちらは厳格

和文が欧文と違う点:

* 語の切れ目が当てにならない。打鍵者ごとに文節の切り方が違い、WORD_BREAK も
  正しく出ないため、**曖昧一致つきの分割**が要る (欧文は厳密一致で足りる)
* 助詞が 1 文字。曖昧一致を許すとどのカナも別のカナに化けるので**完全一致のみ**
* 濁点・半濁点は独立符号。合成カナの符号は「基本カナ + 濁点」の連結になる
"""
from __future__ import annotations

import pytest

from src.tokens.converter import FALLBACK_CHAR
from src.infer.word_correct import (
    JAPANESE_LEXICON,
    JAPANESE_PARTICLES,
    candidates_for,
    correct_text,
    with_candidates,
    japanese_code_of,
    japanese_target_words,
    nearest_word,
    substitution_cost,
    word_distance,
)


class TestJapaneseSubstitutionCost:
    """置換費用は和文でも**符号の近さ**で決まること."""

    def test_same_kana_is_free(self) -> None:
        assert substitution_cost("テ", "テ", script="japanese") == 0.0

    def test_dakuten_is_closer_than_unrelated_kana(self) -> None:
        """ガ と カ は「濁点 1 個」の差。無関係なカナより近い.

        これは和文特有の強みで、濁点の付き外れという頻出誤りが
        そのまま符号距離に乗る。
        """
        assert (
            substitution_cost("ガ", "カ", script="japanese")
            < substitution_cost("ガ", "ム", script="japanese")
        )

    def test_question_mark_is_cheap(self) -> None:
        """`?` は和文でも「読めなかった」印として安く化ける."""
        assert substitution_cost(FALLBACK_CHAR, "テ", script="japanese") == pytest.approx(0.3)

    def test_european_script_is_unaffected(self) -> None:
        """既定は欧文のまま。和文カナは欧文では未知文字 (1.0)."""
        assert substitution_cost("ア", "B") == 1.0

    def test_one_element_apart_is_cheaper(self) -> None:
        """ニ(-・-・) と ハ(-・・・) は 1 要素差。ム(-) より近い."""
        assert (
            substitution_cost("ニ", "ハ", script="japanese")
            < substitution_cost("ニ", "ム", script="japanese")
        )


class TestJapaneseLexicon:
    """語彙はカテゴリ付きで持ち、カテゴリごとに方針が違うこと."""

    def test_lexicon_is_categorised(self) -> None:
        assert isinstance(JAPANESE_LEXICON, dict)
        assert all(isinstance(v, tuple) for v in JAPANESE_LEXICON.values())

    def test_content_words_are_targets(self) -> None:
        targets = japanese_target_words()
        assert "テンキ" in targets
        assert "クモリ" in targets

    def test_short_words_are_not_targets(self) -> None:
        """**2 文字の語は寄せ先にしない.**

        符号列が短いので、少し違うだけの 3〜4 文字語を片端から引き寄せる。
        2026-08-14 に語彙を 4 倍に広げた直後、``ムンキニ`` が ``サン ニ``
        (``サン`` = 敬称) に化けた。完全一致で切り出す分には短い語も使ってよい。
        """
        assert all(len(w) >= 3 for w in japanese_target_words())

    def test_particles_are_not_targets(self) -> None:
        """助詞を寄せ先にすると 1 文字カナが何にでも化ける."""
        targets = japanese_target_words()
        for particle in JAPANESE_PARTICLES:
            assert particle not in targets

    def test_place_names_are_not_targets(self) -> None:
        """地名・人名は開いた集合。寄せ先にすると正しい固有名詞を壊す."""
        targets = japanese_target_words()
        for word in JAPANESE_LEXICON.get("地名・人名", ()):
            assert word not in targets

    def test_every_word_is_encodable(self) -> None:
        """**語彙の全語が符号に変換できること.**

        和文モールスに**小文字 (ッ ャ ュ ョ) は存在しない**。符号表は大文字だけ
        なので「いってきます」は ``イツテキマス`` と書く。うっかり小文字で書くと
        符号に変換できず、その語は永久に一致しない (静かに効かないだけなので
        気付けない)。2026-08-14 に運用者から挙がった語で実際に踏みかけた。
        """
        for category, words in JAPANESE_LEXICON.items():
            for word in words:
                assert japanese_code_of(word) is not None, f"{category}: {word}"

    def test_no_duplicate_across_categories(self) -> None:
        """同じ語を 2 つのカテゴリに置かない.

        カテゴリごとに寄せ先の可否が違うので、重複すると方針が二重定義になる。
        """
        seen: dict[str, str] = {}
        for category, words in JAPANESE_LEXICON.items():
            for word in words:
                assert word not in seen, f"{word} が {seen.get(word)} と {category} に重複"
                seen[word] = category

    def test_no_word_contains_digits(self) -> None:
        """数字を含む語は触らない規則なので、語彙にも入れない."""
        for words in JAPANESE_LEXICON.values():
            for word in words:
                assert not any(ch.isdigit() for ch in word), word


class TestNearestWordJapanese:
    def test_exact_word_is_returned(self) -> None:
        assert nearest_word("テンキ", script="japanese") == "テンキ"

    def test_one_element_error_is_pulled_back(self) -> None:
        """1 要素だけ違うカナは元の語に戻ること."""
        assert nearest_word("クモソ", script="japanese") == "クモリ"

    def test_far_word_is_left_alone(self) -> None:
        """遠すぎる語は寄せない (捏造しない)."""
        assert nearest_word("ヌヌヌヌ", script="japanese") is None


class TestJapaneseSegmentation:
    """曖昧一致つきの分割 — 和文の中心."""

    def test_particle_is_split_off(self) -> None:
        """テンキハ → テンキ ハ。助詞は 2 文字以上の語彙語に隣接するときだけ切る."""
        assert correct_text("テンキハ").text == "テンキ ハ"

    def test_particles_alone_are_not_split(self) -> None:
        """助詞だけの並びは切らない (どうにでも切れてしまうため)."""
        assert correct_text("ハノガ").text == "ハノガ"

    def test_split_character_is_rejoined(self) -> None:
        """**1 文字が要素の途中で切れて 2 文字になった誤りが戻ること.**

        2026-08-14 の実受信で運用者が見抜いた構造::

            ロ(・-・-) + ム(-) = ・-・-- = テ

        符号列で並べると距離ゼロになる。文字単位の距離ではこうはいかない
        (「1 文字消して 1 文字置換」で高くつく)。**和文の誤りの主役はこれ。**

        助詞は完全一致のみなので ニ のまま残るが、内容語 テンキ が戻れば意味は通る。
        """
        assert correct_text("ロムンキニ").text == "テンキ ニ"

    def test_split_character_is_rejoined_in_ending(self) -> None:
        """運用者のもう 1 つの例: ``ロムマス`` → ``テマス`` (``ロ+ム=テ``)."""
        assert correct_text("ロムマス").text == "テマス"

    def test_wildcard_absorbs_unreadable_character(self) -> None:
        """``?`` は任意個の要素を費用 1 で吸収する.

        ``?ロムマス`` と ``シテマス`` は距離 1 になる (``?`` が ``シ`` の 5 要素を
        吸収し、``ロ+ム`` が ``テ`` に一致する)。1 文字ぶんを挿入として数えると
        高くつきすぎて、``?`` を含む語がどこにも寄らなくなる。
        """
        assert word_distance(f"{FALLBACK_CHAR}ロムマス", "シテマス", "japanese") == 1.0
        # ``?`` が無ければ ``テマス`` の方が近い (こちらは距離 0)
        assert word_distance("ロムマス", "テマス", "japanese") == 0.0

    def test_laughter_is_recognised(self) -> None:
        """CW の笑い (欧文の ``HI HI``) は和文表で ``ヌヘヘ`` / ``ホヘヘ`` になる.

        語彙に入れておかないと辞書が別の語に引き寄せる (運用者、2026-08-14)。
        """
        assert correct_text("ヌヘヘ").text == "ヌヘヘ"
        assert correct_text("ホヘヘ").text == "ホヘヘ"

    def test_correct_text_is_not_broken(self) -> None:
        """既に正しい和文は 1 文字も変えないこと (最重要の歯止め)."""
        for text in ("テンキ ハ ハレ", "アンテナ ハ ダイポール", "コチラ ノ シンゴウ"):
            assert correct_text(text).text == text

    def test_place_name_is_not_pulled_to_lexicon(self) -> None:
        """固有名詞を語彙語に引き寄せて壊さないこと."""
        assert "イチノセキ" in correct_text("イチノセキ").text

    def test_two_character_word_is_not_fuzzy_matched(self) -> None:
        """**2 文字の曖昧一致は許さない.**

        ``ヨ(--) + イ(・-)`` は符号列が ``--・-`` で ``ネ`` と**完全に一致する**。
        語彙に無い 2 文字語は、語彙にある 2 文字語と見分けがつかない。
        実測 (2026-08-14) で ``ヨイ テンキ`` が ``ネ テンキ`` に化けた。
        """
        assert correct_text("ホンジツ ハ ヨイ テンキ デス").text == "ホンジツ ハ ヨイ テンキ デス"

    def test_real_question_mark_is_kept(self) -> None:
        """末尾の ``?`` を外すと語彙語になるなら、その ``?`` は本物なので残す.

        欧文と同じ歯止め (欧文ではこれが無くて 0.42pt 悪化した)。
        実測 (2026-08-14) で ``ワカリマセン?`` の ``?`` を落としていた。
        """
        assert correct_text("ワカリマセン?").text == "ワカリマセン?"
        assert correct_text("ドウゾ?").text == "ドウゾ?"

    def test_short_words_do_not_carve_correct_word(self) -> None:
        """**短い語の完全一致だけで正しい語を砕かないこと.**

        語彙に ``イマ`` ``シタ`` ``カ`` が揃うと ``カイマシタ`` (買いました) が
        ``カ イマ シタ`` に割れる。2026-08-14 に語彙を広げた直後に実際に起きた。
        3 文字以上の語を芯に持たない分割は採らない、という条件で防ぐ。
        **語彙が増えるほどこの条件は効いてくる。**
        """
        text = "スーパー デ カーテン ヲ カイマシタ。"
        assert correct_text(text).text == text

    def test_long_garbled_span_is_repaired(self) -> None:
        """長い塊も、語彙で説明しきれるなら直す.

        実受信の ``サクヤハソチラノホウハンオアメデ`` (余分な ``ン`` が 1 つ
        混ざっている)。運用者の読みでは ``サクヤハ ソチラノ ホウハ オオアメデ``
        で、``ン`` は ``オ`` の読み違いだった。**費用を語数より優先する**ように
        してから直るようになった (それ以前は ``サクヤ ハレ ドウゾ ハ …`` と
        砕くか、諦めるかのどちらかだった)。
        """
        fixed = correct_text("サクヤハソチラノホウハンオアメデ").text
        assert fixed == "サクヤ ハ ソチラ ノ ホウ ハ オオアメ デ"

    def test_unexplainable_span_is_left_alone(self) -> None:
        """説明しきれない塊は**そのまま残す** (無理に砕かない)."""
        text = f"ダ{FALLBACK_CHAR}タタヘマカホヘヘワネ"
        assert correct_text(text).text == text


class TestPunctuationInsideWords:
    """**語の途中の句読点は、句読点ではなく読み違えである.**

    ``デ`` と ``。`` は符号が 1 要素しか違わない (運用者、2026-08-14)::

        デ = ・-・--・・
        。 = ・-・-・・      ← 長音 1 本の差

    長音が短く打たれると ``デ`` が ``。`` に化ける。実受信では長音のばらつきが
    σ 1.236 dot (学習分布の上限 0.25 の約 5 倍) あり、頻繁に起きる。

    以前は ``。`` ``、`` を無条件に区切り記号として保護していたため、
    **文中に出た ``。`` を直せなかった。**
    """

    def test_paragraph_mark_at_the_end_is_kept(self) -> None:
        """文末の ``。`` は本物なので消さないこと (いちばん大事な歯止め)."""
        assert correct_text("ヒンヤリシテマス。").text == "ヒンヤリ シテマス。"

    def test_paragraph_mark_inside_a_word_is_corrected(self) -> None:
        """文中の ``。`` は ``デ`` の読み違えとして直ること."""
        assert correct_text("コチラ。ス").text == "コチラ デス"

    def test_punctuation_keeps_no_space_before_it(self) -> None:
        """句読点の前に空白を入れないこと (``シテマス 。`` にしない)."""
        assert " 。" not in correct_text("ヒンヤリシテマス。").text
        assert " 、" not in correct_text("テンキハクモリ、アメ").text

    def test_real_punctuation_between_words_is_kept(self) -> None:
        """語と語の間の本物の句読点は残すこと."""
        text = "テンキ ハ ハレ、アツイ デス"
        assert correct_text(text).text == text

    def test_trailing_question_mark_survives_fuzzy_matching(self) -> None:
        """**末尾の ``?`` を曖昧一致に飲ませないこと.**

        ``?`` は任意個の要素を吸収するワイルドカードで、語彙語はどれも ``?`` を
        持たない。寄せると必ず ``?`` が消え、「読めなかった」という情報が黙って
        失われる。実測で ``イチド?`` が ``イチド`` に化けた (2026-08-14)。
        """
        text = "ドウゾ?、モウ イチド?、ワカリマセン?"
        assert correct_text(text).text == text

    def test_exact_split_beats_fuzzy_merge(self) -> None:
        """**費用 0 の正しい分割が、曖昧一致を含む短い分割に負けないこと.**

        語数そのものを費用に足していたところ、``ジーピー`` + ``、`` + ``タカサ``
        (3 語・費用 0) より ``ジーピー`` + ``、タカサ``→``イナズマ`` (2 語) が
        勝った。**比べるのは (編集量, 語数) の順**で、語数は同点のときだけ見る。
        """
        text = "アンテナ ハ ジーピー、タカサ ハ 10 メートル"
        assert correct_text(text).text == text


class TestForcedSubstitutions:
    """日常で使われないカナを無条件に置き換える (運用者の指示、2026-08-14)."""

    def test_we_becomes_ima(self) -> None:
        """``ヱ`` は ``イマ`` の末尾の長音が落ちた姿.

            ヱ   = ・--・・
            イマ = ・- + -・・- = ・--・・-    ← 長音 1 本の差

        ``ヱ`` は 1 文字なので曖昧一致の対象にならず、辞書では直せない。
        日常の和文にまず出てこないので、出たら必ず置き換えてよい。
        """
        assert correct_text("ヱ").text == "イマ"
        assert "イマ" in correct_text("コチラハヱ ハレ").text
        assert "ヱ" not in correct_text("コチラハヱ ハレ").text


class TestSpacesAreNotTrusted:
    """**送られてくる語間スペースは当てにならない** (運用者、2026-08-14).

    過去の分析で**誤りの約 30% が語間スペースの過剰挿入** (挿入誤りの 62%) と
    分かっている。空白で区切った塊ごとに独立して処理すると、語の途中に余計な
    スペースが入った語は**原理的に一致しない**。隣り合う塊を繋いで試す。

    繋ぐのは「繋ぐと説明できて、個別では説明できなかった」ときだけ。
    無条件に繋ぐと、正しく分かれている語まで作り直してしまう。
    """

    def test_spurious_space_inside_a_word_is_repaired(self) -> None:
        """``テン キハ`` は ``テンキ ハ`` に直ること (余計なスペースを跨ぐ)."""
        assert correct_text("テン キハ").text == "テンキ ハ"

    def test_spurious_space_across_three_chunks(self) -> None:
        """3 つに割れていても繋ぐ."""
        assert correct_text("ヒン ヤリ シテマス").text == "ヒンヤリ シテマス"

    def test_correctly_spaced_text_is_not_rejoined(self) -> None:
        """**正しく分かれている語は繋がない.** 個別に説明できているため."""
        for text in ("テンキ ハ ハレ", "アンテナ ハ ダイポール", "コチラ ノ シンゴウ"):
            assert correct_text(text).text == text

    def test_merge_does_not_break_proper_nouns(self) -> None:
        """繋いでも説明しきれなければ、元のスペースのまま残すこと."""
        text = "イチノセキ ノ ヤマノウチ"
        assert correct_text(text).text == text

    def test_single_character_fragment_is_absorbed(self) -> None:
        """1 文字の断片も繋ぐ (``シ`` は単独では語にならない).

        1〜2 文字の塊は芯となる 3 文字語を含めないので「直せなかった語」として
        報告もされない。隣と繋がないと永久に直らない。
        """
        assert correct_text("ヒンヤリ シ テマス").text == "ヒンヤリ シテマス"

    def test_merge_never_changes_characters(self) -> None:
        """**繋いだ結果は文字を変えない。切り直すだけ.**

        繋いだ文字列は長いぶん編集量の予算も増えるので、個別なら通らない
        曖昧一致が通ってしまう。実測 (2026-08-14) で ``イカガ デス`` が
        ``イチド デス``、``ミナサン オゲンキ`` が ``ヘンシン オゲンキ`` に化けた。
        文字の直しは塊ごとの処理に任せる。
        """
        for text in ("コチラ ノ シンゴウ ハ イカガ デス カ?", "ミナサン オゲンキ デ、サヨウナラ"):
            assert correct_text(text).text == text

    def test_merge_must_span_a_boundary(self) -> None:
        """元の境界を 1 つも跨がないなら、実際には何も繋いでいないので採らない.

        ``コレデ オワリ`` は繋ぐと ``コレ デ オワリ`` と説明できてしまうが、
        これは隣から芯 (``オワリ``) を借りて元の塊を切り直しているだけ。
        同じ理屈で ``カイマシタ ソチラ`` が ``カ イマ シタ ソチラ`` になりうる。
        """
        for text in ("コレデ オワリ マス", "カイマシタ ソチラ"):
            assert correct_text(text).text == text

    def test_existing_word_is_not_rejoined(self) -> None:
        """それ自体が語彙にある短い語は繋がない.

        ``モウ`` は正しい語なので、``モウ イチド`` を ``モウイチド`` に
        繋ぎ直す理由は無い (意味は同じでも元のテキストを変えることになる)。
        """
        text = "バシヨ ヲ モウ イチド オネガイ シマス"
        assert correct_text(text).text == text


class TestPerformance:
    """**確定テキスト全体を hop (0.5 秒) ごとに GUI スレッドで処理する。**

    2026-08-14 に実機で凍結した。83 文字で 1.8 秒、1,328 文字で 29 秒かかって
    おり、しかも 2 回目も同じ時間だった (キャッシュが全く効いていなかった)。
    確定テキストは末尾に伸びるだけなので、**同じ文字列の再処理はほぼ無料**で
    なければならない。この性質が壊れると必ず GUI が固まる。
    """

    def test_reprocessing_the_same_text_is_cheap(self) -> None:
        import time

        text = ("サササノコチラノ、ムンキニクモリ クモリキンンハ22ド 22ドデ "
                "ヒンヤリモロムマス サクヤハソチラノホウハンオアメデ タイヘン "
                "ダ?タタヘマカホヘヘワネ セ1ムク ") * 8
        correct_text(text)                      # 1 回目 (計算する)
        start = time.perf_counter()
        for _ in range(5):
            correct_text(text)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 5
        # 実測 0.2 ms 程度。50 ms は「キャッシュが効いていない」を捕まえる線
        assert elapsed_ms < 50.0, f"再処理に {elapsed_ms:.1f} ms かかっている"

    def test_candidates_are_not_computed_on_the_hot_path(self) -> None:
        """候補は重い (1 語 60 ms) ので ``correct_text`` では埋めないこと."""
        result = correct_text(f"ダ{FALLBACK_CHAR}タタヘマカホヘヘワネ")
        assert result.unresolved
        assert all(entry.candidates == () for entry in result.unresolved)


class TestJapaneseCanBeDisabled:
    """運用者の要望 (2026-08-14): 和文の語彙補正は切れること."""

    def test_disabled_leaves_text_untouched(self) -> None:
        assert correct_text("テンキハ", japanese_enabled=False).text == "テンキハ"

    def test_disabling_japanese_keeps_european_working(self) -> None:
        """和文を切っても欧文の補正は効くこと (別々に切り替わる)."""
        assert correct_text("CQDE", japanese_enabled=False).text == "CQ DE"


class TestScriptDispatch:
    """語ごとに文字種で辞書を選ぶ (モードを配線しなくてよい)."""

    def test_european_word_in_japanese_text(self) -> None:
        """和文の中の欧文区間もそれぞれ正しい辞書で直ること."""
        result = correct_text("テンキハ CQDE")
        assert "テンキ ハ" in result.text
        assert "CQ DE" in result.text

    def test_digits_are_protected_in_japanese(self) -> None:
        """数字を含む語は和文でも触らない (RST・周波数)."""
        assert correct_text("22ド").text == "22ド"


class TestUnresolvedCandidates:
    """LLM へ渡すための候補 (B 案の土台)."""

    def test_unresolved_word_carries_candidates(self) -> None:
        """寄せきれなかった語には、符号距離が近い候補が付くこと.

        ``ムンキニ`` は ``ロムンキニ`` と違って先頭の要素が失われており、
        辞書だけでは決めきれない (``テンキ`` まで 4 要素あって閾値を超える)。
        **こういう語こそ文脈を読める LLM に渡す**。
        """
        result = correct_text("ムンキニ")
        words = [u.word for u in result.unresolved]
        assert "ムンキニ" in words
        # **候補は既定では空** (1 語 60 ms かかるので hop ごとには計算しない)。
        # LLM を呼ぶときに with_candidates で埋める。
        filled = with_candidates(result.unresolved)
        entry = next(u for u in filled if u.word == "ムンキニ")
        assert len(entry.candidates) >= 1
        # 候補は距離の昇順
        distances = [c.distance for c in entry.candidates]
        assert distances == sorted(distances)

    def test_candidates_come_from_inside_the_chunk(self) -> None:
        """**候補は塊の「どこかに現れる語」で探す.**

        直せなかった塊は文節がまるごと繋がっていることが多い。1 語と丸ごと
        比べると ``サクヤハソチラノホウハンオアメデ`` の候補が
        ``エスダブリユーアール`` になり、LLM を誤導する。
        **無関係な候補は無いよりも悪い。**
        """
        candidates = [
            c.word for c in candidates_for("サクヤハソチラノホウハンオアメデ", limit=6)
        ]
        assert "サクヤ" in candidates
        assert "ソチラ" in candidates

    def test_dot_run_really_contains_the_laughter(self) -> None:
        """点の連なりの中には ``ヌヘヘ`` が**本当に**入っている.

        ``ヌ`` = ・・・・、``ヘ`` = ・ なので ``ヌヘヘ`` は点 6 つ。``ヌヌ`` (点 8 つ)
        の中に含まれる。``ロ+ム=テ`` と同じ現象で、偶然の一致ではなく
        **その音が実際にそう読める**ということ。候補に出るのが正しい。

        (この事実に気付かず「候補が出ないこと」を期待するテストを書いて落ちた。
        符号列で考えると、文字の見た目からくる直感は当てにならない。)
        """
        assert "ヌヘヘ" in [c.word for c in candidates_for("ヌヌヌヌヌヌヌヌ")]

    def test_candidates_are_limited_even_for_garbage(self) -> None:
        """当てはまりの悪い塊では候補を絞ること (無関係な候補は誤導になる)."""
        assert len(candidates_for("ムムムムムム", limit=6)) <= 2

    def test_resolved_word_is_not_listed(self) -> None:
        """直せた語は候補に出さない (LLM に渡すのは迷ったものだけ).

        **既に語彙にある語も「直せなかった」ではない。** 分割が不要なだけなので
        候補に出さない (出すと LLM に無用な候補を渡すことになる)。
        """
        result = correct_text("テンキ ハ ハレ")
        assert result.unresolved == ()

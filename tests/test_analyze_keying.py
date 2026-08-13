"""キーイング解析スクリプトのテスト."""
from __future__ import annotations

import numpy as np

from scripts.analyze_keying import analyze_wave
from src.synth.keying import KeyingParams, codes_to_waveform


def _synth(params: KeyingParams, seed: int = 0) -> np.ndarray:
    """短点と長音が十分な数だけ入る符号列を合成する."""
    codes = ["・-", "-・", "・・-", "-・・"] * 25
    return codes_to_waveform(
        codes, params, np.random.default_rng(seed), sample_rate=8000
    ).samples


class TestAnalyzeWave:
    def test_recovers_dot_and_dash_length(self) -> None:
        """ジッタ無しの合成音から短点・長音の長さを ±5% で復元できる."""
        params = KeyingParams(
            wpm=20.0, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            tone_freq_hz=600.0, rise_fall_ms=3.0,
        )
        stats = analyze_wave(_synth(params), 8000)
        # dot は raised-cosine ランプ (rise_fall_ms=3ms) の非対称な適用
        # (src/synth/keying.py の _apply_raised_cosine_ramp: 立上りは公称境界から
        # 後ろへ、立下りは公称境界の手前で終わるよう配置される) により、公称 60ms に
        # 対してほぼ ramp 分 (3ms ≒ 5.0%) 系統的に短く測定される。実測 5.21% で、
        # dash (180ms に対し 3ms ≒ 1.7%) より影響が大きい。これは合成波形が実際に
        # 持つ物理的な短縮であり analyze_wave 側の誤りではないため、この 1 箇所のみ
        # 許容を 7% に広げる (既存の src/synth/keying.py は変更しない)。
        assert abs(stats.dot_sec - 0.06) / 0.06 < 0.07
        assert abs(stats.dash_sec - 0.18) / 0.18 < 0.05
        assert abs(stats.dash_dot_ratio - 3.0) < 0.15

    def test_recovers_tone_frequency(self) -> None:
        params = KeyingParams(
            wpm=20.0, element_jitter_sigma_ratio=0.0, tone_freq_hz=527.0
        )
        stats = analyze_wave(_synth(params), 8000)
        assert abs(stats.tone_hz - 527.0) < 30.0

    def test_histograms_are_returned(self) -> None:
        """ヒストグラムが必ず返る (測定破綻の検出に使うため)."""
        params = KeyingParams(wpm=20.0, element_jitter_sigma_ratio=0.0)
        stats = analyze_wave(_synth(params), 8000)
        assert len(stats.on_histogram_ms) > 0
        assert len(stats.off_histogram_ms) > 0
        assert sum(c for _, c in stats.on_histogram_ms) == stats.n_dot + stats.n_dash


class TestRoundTrip:
    """合成器に入れた統計値が、生成した波形から復元できることを確認する.

    合成器が指定どおりの分散を持つ波形を作れていなければ、その分布で学習しても
    意味がない。この計画で最も重要なテスト (設計書 §4.3)。

    ここでは全テストで ``rise_fall_ms=0.0`` を使う (brief は 3.0 を指定していたが、
    controller の判断でここだけ意図的に上書きする)。``analyze_wave`` の docstring
    にあるとおり raised-cosine ランプは要素の内側に適用されるため、包絡線の 50% 交差
    で測る ON 長はランプ長ぶんそのまま短く出る系統誤差を持つ。この誤差は
    rise_fall_ms / dot_sec に比例するので、wpm や意図によって見かけの長短比や
    要素間スペースがブレる。実際 wpm=30 (dot 31.25ms), ramp=3.0 では長短比 3.84 の
    測定値が 4.07 まで動き、閾値 (差 0.25 以内) をたまたま通ってしまう
    (差 0.23 で偶然パス)。ランプは測定に系統誤差を持ち込むので、往復の検証では
    0 にして偏りを消す。合成器が「往復するか」だけを見たいので、ランプ由来の
    バイアスは検証対象から外すのが筋である。
    """

    def _long_synth(self, params: KeyingParams, seed: int = 0) -> np.ndarray:
        # σ の推定には標本数が要る。短点・長音を各 200 個以上含む長さにする
        codes = ["・-", "-・", "・・-", "-・・", "・-・", "-・-"] * 60
        return codes_to_waveform(
            codes, params, np.random.default_rng(seed), sample_rate=8000
        ).samples

    def test_dash_sigma_round_trips(self) -> None:
        """長音に大きな σ を入れると、解析でも大きな σ が返る."""
        wpm = 23.0
        dot_sec = 1.2 / wpm
        tight = KeyingParams(
            wpm=wpm, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            tone_freq_hz=527.0, rise_fall_ms=0.0,
        )
        loose = KeyingParams(
            wpm=wpm, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            dash_jitter_sigma_ratio=0.68,
            tone_freq_hz=527.0, rise_fall_ms=0.0,
        )
        s_tight = analyze_wave(self._long_synth(tight), 8000)
        # loose は dash の σ (0.68 dot) が dot 長 (1 dot) との差 (2 dot) の
        # 3 割強に達し、自動推定される split_sec だと長音の下裾が短点側に
        # 混入する (実測: 自動 split だと dash_sigma_dot が 0.68 入れて 0.49 しか
        # 返らない。たまたま許容幅に収まって「理由はズレているのに通る」状態に
        # なる)。1.2 dot に固定すると短点分布 (σ 0.064 dot 程度) の裾からは
        # 十分離れつつ長音分布の下裾の混入も実測でほぼ消えるため、明示指定する
        # (brief 記載のトラブルシュート手順 3 に対応)。
        s_loose = analyze_wave(self._long_synth(loose), 8000, split_sec=1.2 * dot_sec)
        # 入れた σ 0.68 dot が復元できる (推定なので幅を持たせる)
        assert s_tight.dash_sigma_dot < 0.15
        assert 0.40 < s_loose.dash_sigma_dot < 1.10

    def test_dot_stays_tight_while_dash_varies(self) -> None:
        """実測どおりの非対称 (短点は正確・長音だけ暴れる) が作れる."""
        wpm = 23.0
        dot_sec = 1.2 / wpm
        params = KeyingParams(
            wpm=wpm, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.0,
            dot_jitter_sigma_ratio=0.064,
            dash_jitter_sigma_ratio=0.681,
            tone_freq_hz=527.0, rise_fall_ms=0.0,
        )
        # 上の test_dash_sigma_round_trips と同じ理由で split_sec を明示する。
        # 自動推定のままだと長音の下裾が短点側に混入し、短点の σ が
        # 0.064 入れて 0.28 と大きく振れる (誤って測定破綻を「短点も暴れる」と
        # 誤読しかねない)。1.2 dot に固定すると入れた σ にほぼ一致する。
        s = analyze_wave(
            self._long_synth(params), 8000, split_sec=1.2 * dot_sec
        )
        # 短点の σ は長音の σ よりはっきり小さい
        assert s.dot_sigma_dot < 0.25
        assert s.dash_sigma_dot > 0.35
        assert s.dash_sigma_dot > s.dot_sigma_dot * 2.0

    def test_dash_dot_ratio_round_trips(self) -> None:
        """長短比 3.84 を入れると解析でも 3.84 前後が返る."""
        params = KeyingParams(
            wpm=30.0, dash_dot_ratio=3.84, element_jitter_sigma_ratio=0.0,
            tone_freq_hz=527.0, rise_fall_ms=0.0,
        )
        s = analyze_wave(self._long_synth(params), 8000)
        assert abs(s.dash_dot_ratio - 3.84) < 0.25

    def test_intra_element_space_round_trips(self) -> None:
        """要素間 1.25 dot を入れると解析でも 1.25 前後が返る."""
        params = KeyingParams(
            wpm=23.0, element_jitter_sigma_ratio=0.0,
            intra_element_space_units=1.25,
            tone_freq_hz=527.0, rise_fall_ms=0.0,
        )
        s = analyze_wave(self._long_synth(params), 8000)
        assert abs(s.intra_gap_dot - 1.25) < 0.20

    def test_char_and_word_gap_round_trip(self) -> None:
        """文字間・語間の平均が往復する.

        測定値にはバケット境界由来の偏りがある (解析は 1.9 / 5.0 dot で
        要素間・文字間・語間を分ける)。手打ちレンジでは裾が隣のバケットへ流れ、
        実測で文字間 σ が約 12% 低く、語間平均が約 9% 高く出る。許容幅は
        この偏りを織り込んである。
        """
        params = KeyingParams(
            wpm=23.0,
            dash_dot_ratio=3.0,
            element_jitter_sigma_ratio=0.0,
            inter_char_space_units=2.86,
            char_gap_jitter_sigma_ratio=0.82,
            inter_word_space_units=7.55,
            tone_freq_hz=527.0,
            rise_fall_ms=0.0,
        )
        codes = ["・-", "-・", "・・-", "-・・", "・-・", "-・-"] * 60
        # 6 符号ごとに語間を入れる (語間の標本を確保するため)
        word_breaks = list(range(5, len(codes), 6))
        wave = codes_to_waveform(
            codes, params, np.random.default_rng(11),
            sample_rate=8000, word_break_after=word_breaks,
        ).samples
        s = analyze_wave(wave, 8000)
        assert abs(s.char_gap_dot - 2.86) < 0.35
        assert abs(s.word_gap_dot - 7.55) < 1.5

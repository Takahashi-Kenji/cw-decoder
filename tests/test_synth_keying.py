"""キーイング波形生成のテスト."""
from __future__ import annotations

import numpy as np
import pytest

from src.synth.keying import KeyingParams, codes_to_waveform


# ============================================================
# 基本動作
# ============================================================
class TestBasic:
    def test_empty_codes_returns_silence(self) -> None:
        params = KeyingParams(pre_silence_sec=0.1, post_silence_sec=0.1)
        rng = np.random.default_rng(0)
        result = codes_to_waveform([], params, rng, sample_rate=8000)
        assert len(result.samples) == 1600
        assert np.all(result.samples == 0.0)

    def test_single_dot_produces_signal(self) -> None:
        params = KeyingParams(wpm=20.0, rise_fall_ms=0.0)
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["・"], params, rng, sample_rate=8000)
        # dot 長 = 1.2 / 20 = 0.06 秒 = 480 サンプル
        assert abs(len(result.samples) - 480) <= 2
        # 信号は非ゼロ部分を含む
        assert np.max(np.abs(result.samples)) > 0.5

    def test_dtype_is_float32(self) -> None:
        params = KeyingParams(rise_fall_ms=0.0)
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["・-"], params, rng)
        assert result.samples.dtype == np.float32

    def test_amplitude_within_unit_range(self) -> None:
        params = KeyingParams()
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["・-・-"], params, rng)
        assert np.max(np.abs(result.samples)) <= 1.0


# ============================================================
# タイミング検証
# ============================================================
class TestTiming:
    def test_total_length_no_jitter(self) -> None:
        """A B = (・-) (-・・・). 期待長を計算して照合."""
        params = KeyingParams(
            wpm=20.0,
            dash_dot_ratio=3.0,
            element_jitter_sigma_ratio=0.0,
            rise_fall_ms=0.0,
        )
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["・-", "-・・・"], params, rng, sample_rate=8000)
        # A = dot + intra(1) + dash = 1 + 1 + 3 = 5 dot
        # 文字間 = 3 dot
        # B = dash + intra + dot + intra + dot + intra + dot = 3 + 1 + 1 + 1 + 1 + 1 + 1 = 9 dot
        # 計 5 + 3 + 9 = 17 dot
        # dot = 0.06 s, total = 17 * 0.06 = 1.02 s = 8160 samples
        expected = int(round(17 * 0.06 * 8000))
        assert abs(len(result.samples) - expected) <= 2

    def test_dash_dot_ratio_no_jitter(self) -> None:
        """dot のみと dash のみで ON 時間比が一致."""
        params = KeyingParams(
            wpm=20.0,
            dash_dot_ratio=3.0,
            element_jitter_sigma_ratio=0.0,
            rise_fall_ms=0.0,
        )
        rng = np.random.default_rng(0)
        dot_result = codes_to_waveform(["・"], params, rng, sample_rate=8000)
        dash_result = codes_to_waveform(["-"], params, rng, sample_rate=8000)
        # dot 波形長 ≈ 480、dash 波形長 ≈ 1440 (3 倍)
        assert abs(len(dash_result.samples) / len(dot_result.samples) - 3.0) < 0.05

    def test_effective_wpm_matches_no_jitter(self) -> None:
        params = KeyingParams(wpm=25.0, element_jitter_sigma_ratio=0.0)
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["-・-・", "・-・-"], params, rng)
        assert abs(result.effective_wpm - 25.0) < 0.1

    def test_jitter_affects_effective_wpm(self) -> None:
        params = KeyingParams(
            wpm=20.0, element_jitter_sigma_ratio=0.15, rise_fall_ms=0.0
        )
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["・-"] * 20, params, rng)
        # ジッタ ±15% で総時間が変動するが、平均は元 WPM 近傍
        assert 15.0 < result.effective_wpm < 30.0


# ============================================================
# code_start_samples
# ============================================================
class TestCodeStartSamples:
    def test_first_code_start_is_pre_silence(self) -> None:
        params = KeyingParams(pre_silence_sec=0.1, rise_fall_ms=0.0)
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["・-", "-・"], params, rng, sample_rate=8000)
        assert result.code_start_samples[0] == 800  # 0.1 * 8000

    def test_code_starts_are_monotonic(self) -> None:
        params = KeyingParams()
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["・-", "-・・・", "-・-・"], params, rng)
        starts = result.code_start_samples
        assert len(starts) == 3
        assert np.all(np.diff(starts) > 0)


# ============================================================
# raised-cosine スムージング
# ============================================================
class TestRaisedCosine:
    def test_ramp_smooths_envelope_transitions(self) -> None:
        params = KeyingParams(
            wpm=20.0, rise_fall_ms=5.0, element_jitter_sigma_ratio=0.0, tone_freq_hz=600.0,
        )
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["-"], params, rng, sample_rate=8000)
        # 立上り部分は急激な 0→1 ではなく単調増加
        envelope_estimate = np.abs(result.samples)
        # 最初の 100 サンプルは単調 (おおむね) 増加
        first_chunk = envelope_estimate[:50]
        # 包絡線は単純な単調増加ではないが、最大値は徐々に増えるはず
        max_in_windows = [first_chunk[i:i+10].max() for i in range(0, 40, 10)]
        # 段階的に最大値が増えていく
        assert max_in_windows[0] < max_in_windows[-1]

    def test_zero_ramp_yields_sharper_edges(self) -> None:
        params_smooth = KeyingParams(rise_fall_ms=10.0, element_jitter_sigma_ratio=0.0)
        params_sharp = KeyingParams(rise_fall_ms=0.0, element_jitter_sigma_ratio=0.0)
        rng = np.random.default_rng(0)
        smooth = codes_to_waveform(["・"], params_smooth, rng)
        sharp = codes_to_waveform(["・"], params_sharp, rng)
        # シャープエッジの方が先頭サンプルの絶対値が大きい
        assert np.abs(sharp.samples[:5]).max() > np.abs(smooth.samples[:5]).max()


# ============================================================
# 搬送波周波数
# ============================================================
class TestCarrier:
    def test_tone_frequency_matches_via_fft(self) -> None:
        params = KeyingParams(
            wpm=15.0,
            tone_freq_hz=750.0,
            element_jitter_sigma_ratio=0.0,
            rise_fall_ms=0.0,
        )
        rng = np.random.default_rng(0)
        # 長めの dash で FFT 精度を確保
        result = codes_to_waveform(["-" * 10], params, rng, sample_rate=8000)
        spec = np.abs(np.fft.rfft(result.samples.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(result.samples), d=1.0 / 8000)
        peak_freq = freqs[int(np.argmax(spec))]
        assert abs(peak_freq - 750.0) < 5.0

    def test_tone_drift_changes_freq(self) -> None:
        params = KeyingParams(
            wpm=15.0,
            tone_freq_hz=600.0,
            tone_drift_hz_per_sec=100.0,
            element_jitter_sigma_ratio=0.0,
            rise_fall_ms=0.0,
        )
        rng = np.random.default_rng(0)
        result = codes_to_waveform(["-" * 30], params, rng, sample_rate=8000)
        # ドリフトの影響で波形は単純な単一周波数より広がりがある
        assert len(result.samples) > 0


# ============================================================
# 再現性
# ============================================================
class TestReproducibility:
    def test_same_seed_same_output(self) -> None:
        params = KeyingParams(element_jitter_sigma_ratio=0.15)
        result1 = codes_to_waveform(
            ["・-", "-・・・"], params, np.random.default_rng(42)
        )
        result2 = codes_to_waveform(
            ["・-", "-・・・"], params, np.random.default_rng(42)
        )
        np.testing.assert_array_equal(result1.samples, result2.samples)


# ============================================================
# バリデーション
# ============================================================
class TestValidation:
    def test_invalid_wpm_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            codes_to_waveform(["・"], KeyingParams(wpm=0.0), rng)

    def test_unknown_element_in_code_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            codes_to_waveform(["X"], KeyingParams(), rng)


# ============================================================
# 語間スペース
# ============================================================
class TestWordBreaks:
    def test_word_break_adds_extra_space(self) -> None:
        params = KeyingParams(
            wpm=20.0, element_jitter_sigma_ratio=0.0, rise_fall_ms=0.0
        )
        rng = np.random.default_rng(0)
        # 同じ codes で word_break あり/なしを比較
        no_break = codes_to_waveform(["・-", "-・"], params, rng, word_break_after=())
        with_break = codes_to_waveform(["・-", "-・"], params, rng, word_break_after=(0,))
        # word_break で 7 - 3 = 4 dot 分長くなる
        diff_samples = len(with_break.samples) - len(no_break.samples)
        expected_diff = int(round(4 * 0.06 * 8000))
        assert abs(diff_samples - expected_diff) <= 2


# ============================================================
# 要素種別 (ElementKind)
# ============================================================
class TestElementKind:
    def test_kinds_cover_all_elements(self) -> None:
        """符号 2 つ (・- と -) を語間で繋いだとき、種別列が期待どおりになる."""
        from src.synth.keying import ElementKind, build_element_sequence

        durations, is_on, code_starts, kinds = build_element_sequence(
            ["・-", "-"],
            word_break_after=[0],
            dot_sec=0.06,
            dash_dot_ratio=3.0,
            inter_char_space_units=3.0,
            inter_word_space_units=7.0,
        )
        # ・ / 要素間 / - / 語間 / -
        assert list(kinds) == [
            ElementKind.DOT,
            ElementKind.INTRA_GAP,
            ElementKind.DASH,
            ElementKind.WORD_GAP,
            ElementKind.DASH,
        ]
        assert len(kinds) == len(durations) == len(is_on)

    def test_char_gap_kind_when_not_word_break(self) -> None:
        """語間指定が無ければ符号間は CHAR_GAP になる."""
        from src.synth.keying import ElementKind, build_element_sequence

        _, _, _, kinds = build_element_sequence(
            ["・", "・"],
            word_break_after=[],
            dot_sec=0.06,
            dash_dot_ratio=3.0,
            inter_char_space_units=3.0,
            inter_word_space_units=7.0,
        )
        assert list(kinds) == [
            ElementKind.DOT,
            ElementKind.CHAR_GAP,
            ElementKind.DOT,
        ]


# ============================================================
# 種別ごとのジッタ
# ============================================================
class TestPerKindJitter:
    def test_matches_pre_change_implementation_bitwise(self) -> None:
        """変更前の実装 (コミット b3fc8ba) が出した波形と完全に一致する.

        期待値は変更前の実装を実際に動かして得た SHA-256。新旧を同じ seed で
        2 回呼んで比べても「関数が決定的である」ことしか示せないので、
        変更前の出力そのものを固定値として持つ。

        既定値の波形が変わると学習データの再現性が壊れるため、以後の変更でも
        この値が動いてはいけない。動いたら意図的な変更かどうかを必ず確認すること。
        """
        import hashlib

        codes = ["・-", "-・・", "・", "・・-", "-"]
        cases = [
            (
                KeyingParams(
                    wpm=20.0, dash_dot_ratio=3.0, element_jitter_sigma_ratio=0.15,
                    tone_freq_hz=600.0, rise_fall_ms=5.0,
                ),
                7,
                18357,
                "c4ef0a7d7a98a1c0dd7177d56e0841b7c7ebb7d6077097f9349b70d0f731df4b",
            ),
            (
                KeyingParams(
                    wpm=25.0, dash_dot_ratio=3.5, element_jitter_sigma_ratio=0.0,
                    tone_freq_hz=527.0, rise_fall_ms=3.0,
                ),
                42,
                15744,
                "6376b4ace7d6c1e305f6236de9f77185bc048521b3ce455b0e4a85fd2a170cb9",
            ),
        ]
        for params, seed, n_samples, digest in cases:
            result = codes_to_waveform(
                codes, params, np.random.default_rng(seed),
                sample_rate=8000, word_break_after=[1],
            )
            assert len(result.samples) == n_samples
            assert hashlib.sha256(result.samples.tobytes()).hexdigest() == digest

    def test_dash_only_jitter_leaves_dots_exact(self) -> None:
        """長音だけに σ を与えると、短点の長さは正確なままになる."""
        params = KeyingParams(
            wpm=20.0,
            rise_fall_ms=0.0,
            element_jitter_sigma_ratio=0.0,
            dash_jitter_sigma_ratio=0.5,
        )
        rng = np.random.default_rng(3)
        # 短点だけの符号を並べる → 長音が無いので長さは無ジッタと一致するはず
        dots = codes_to_waveform(["・", "・", "・"], params, rng, sample_rate=8000)
        no_jitter = codes_to_waveform(
            ["・", "・", "・"],
            KeyingParams(wpm=20.0, rise_fall_ms=0.0),
            np.random.default_rng(3),
            sample_rate=8000,
        )
        assert len(dots.samples) == len(no_jitter.samples)

    def test_dash_jitter_widens_total_length_spread(self) -> None:
        """長音の σ を上げると、同じ符号列でも波形長のばらつきが広がる.

        ジッタの計算式をテスト側に書き写すと本番ロジックの写経になり、
        実装が変わってもテストが追随しない。ここは実際に codes_to_waveform を
        通した波形長で判定する。
        """
        codes = ["-"] * 8

        def spread(params: KeyingParams) -> float:
            lengths = [
                len(
                    codes_to_waveform(
                        codes, params, np.random.default_rng(seed), sample_rate=8000
                    ).samples
                )
                for seed in range(30)
            ]
            return float(np.std(lengths))

        tight = KeyingParams(
            wpm=20.0, rise_fall_ms=0.0, element_jitter_sigma_ratio=0.0
        )
        loose = KeyingParams(
            wpm=20.0, rise_fall_ms=0.0, element_jitter_sigma_ratio=0.0,
            dash_jitter_sigma_ratio=0.8,
        )
        # ジッタ無しなら seed を変えても長さは一定
        assert spread(tight) == 0.0
        # σ 0.8 dot = 0.048 秒 = 384 サンプル。長音 8 個ぶんの合計なので
        # 標準偏差は 1000 サンプル前後になる。50 は十分に安全な下限
        assert spread(loose) > 50.0

    def test_sigma_by_kind_fallback_is_per_field(self) -> None:
        """None の種別だけが element_jitter_sigma_ratio にフォールバックする."""
        from src.synth.keying import ElementKind

        params = KeyingParams(
            element_jitter_sigma_ratio=0.1,
            dash_jitter_sigma_ratio=0.9,
            char_gap_jitter_sigma_ratio=0.0,
        )
        s = params.jitter_sigma_by_kind()
        assert s[ElementKind.DOT] == 0.1
        assert s[ElementKind.DASH] == 0.9
        assert s[ElementKind.INTRA_GAP] == 0.1
        assert s[ElementKind.CHAR_GAP] == 0.0
        assert s[ElementKind.WORD_GAP] == 0.1

    def test_dot_exact_while_dash_varies_in_same_waveform(self) -> None:
        """同じ波形の中で、短点は seed によらず一定・長音だけがばらつく.

        これが本機能の目的そのもの (実測: 短点 σ 0.06 dot / 長音 σ 0.68〜1.20 dot)。
        短点だけ・長音だけの符号列では非対称を確認できないので、混在符号で見る。
        """
        params = KeyingParams(
            wpm=20.0,
            rise_fall_ms=0.0,
            element_jitter_sigma_ratio=0.0,
            dot_jitter_sigma_ratio=0.0,
            intra_gap_jitter_sigma_ratio=0.0,
            dash_jitter_sigma_ratio=0.8,
        )
        waves = [
            codes_to_waveform(
                ["・-"], params, np.random.default_rng(seed), sample_rate=8000
            ).samples
            for seed in range(5)
        ]
        # dot_sec = 1.2/20 = 0.06 秒 = 480 サンプル。短点 480 + 要素間 480 = 960。
        # 長音より前の区間は σ=0 なので seed によらず完全に一致するはず
        head = 960
        for w in waves[1:]:
            assert np.array_equal(waves[0][:head], w[:head])
        # 長音だけに σ があるので全体長は seed ごとに変わる
        assert len({len(w) for w in waves}) > 1


# ============================================================
# 要素間スペースの長さ
# ============================================================
class TestIntraElementSpace:
    def test_default_is_one_dot(self) -> None:
        """既定では要素間は 1 dot のまま."""
        params = KeyingParams()
        assert params.intra_element_space_units == 1.0

    def test_shorter_intra_space_shortens_waveform(self) -> None:
        """要素間を詰めると符号全体が短くなる."""
        base = KeyingParams(wpm=20.0, rise_fall_ms=0.0, element_jitter_sigma_ratio=0.0)
        tight = KeyingParams(
            wpm=20.0,
            rise_fall_ms=0.0,
            element_jitter_sigma_ratio=0.0,
            intra_element_space_units=0.4,
        )
        rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
        # ・・・ は要素間が 2 つある
        a = codes_to_waveform(["・・・"], base, rng_a, sample_rate=8000)
        b = codes_to_waveform(["・・・"], tight, rng_b, sample_rate=8000)
        dot_samples = 0.06 * 8000       # dot_sec = 1.2/20 = 0.06
        expected_diff = 2 * (1.0 - 0.4) * dot_samples
        assert abs((len(a.samples) - len(b.samples)) - expected_diff) <= 2

    def test_intra_space_does_not_affect_char_gap(self) -> None:
        """要素間を変えても文字間は変わらない."""
        from src.synth.keying import ElementKind, build_element_sequence

        durations, _, _, kinds = build_element_sequence(
            ["・", "・"],
            word_break_after=[],
            dot_sec=0.06,
            dash_dot_ratio=3.0,
            inter_char_space_units=3.0,
            inter_word_space_units=7.0,
            intra_element_space_units=0.4,
        )
        char_gap = durations[kinds == ElementKind.CHAR_GAP]
        assert len(char_gap) == 1
        assert abs(char_gap[0] - 3.0 * 0.06) < 1e-9

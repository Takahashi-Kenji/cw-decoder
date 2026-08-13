"""合成器統合 + Dataset のテスト."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.synth.dataset import (
    DefaultConfigSampler,
    MorseSynthDataset,
    default_config_sampler,
    make_fixed_eval_set,
)
from src.synth.synthesizer import SynthConfig, synthesize_from_text
from src.synth.keying import KeyingParams
from src.synth.noise import RealNoisePool


# ============================================================
# synthesize_from_text
# ============================================================
class TestSynthesizeFromText:
    def test_basic_european(self) -> None:
        rng = np.random.default_rng(42)
        config = SynthConfig(mode="european", snr_db=20.0)
        result = synthesize_from_text("ABC", config, rng)
        assert result.text == "ABC"
        assert result.codes == ["・-", "-・・・", "-・-・"]
        assert result.token_ids.dtype == np.int64
        assert len(result.samples) > 0

    def test_basic_japanese(self) -> None:
        rng = np.random.default_rng(42)
        config = SynthConfig(mode="japanese", snr_db=20.0)
        result = synthesize_from_text("イロハ", config, rng)
        assert result.codes == ["・-", "・-・-", "-・・・"]

    def test_dakuten_expansion(self) -> None:
        # ガ は 2 符号 (カ + 濁点)
        rng = np.random.default_rng(0)
        config = SynthConfig(mode="japanese", snr_db=20.0)
        result = synthesize_from_text("ガ", config, rng)
        assert result.codes == ["・-・・", "・・"]
        assert len(result.token_ids) == 2

    def test_empty_text(self) -> None:
        rng = np.random.default_rng(0)
        config = SynthConfig(mode="european", snr_db=20.0)
        result = synthesize_from_text("", config, rng)
        assert len(result.samples) == 0
        assert len(result.token_ids) == 0


# ============================================================
# MorseSynthDataset 再現性
# ============================================================
class TestDatasetReproducibility:
    def test_same_seed_same_first_sample(self) -> None:
        ds1 = MorseSynthDataset(
            mode_mix={"european": 1.0}, seed=42, max_samples=3
        )
        ds2 = MorseSynthDataset(
            mode_mix={"european": 1.0}, seed=42, max_samples=3
        )
        first1 = next(iter(ds1))
        first2 = next(iter(ds2))
        torch.testing.assert_close(first1[0], first2[0])
        torch.testing.assert_close(first1[1], first2[1])

    def test_max_samples_limits_iteration(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 1.0}, seed=0, max_samples=5
        )
        items = list(ds)
        assert len(items) == 5

    def test_no_seed_still_works(self) -> None:
        ds = MorseSynthDataset(mode_mix={"european": 1.0}, max_samples=2)
        items = list(ds)
        assert len(items) == 2


# ============================================================
# MorseSynthDataset 多モード
# ============================================================
class TestDatasetMultiMode:
    def test_mixed_modes_produce_both(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 0.5, "japanese": 0.5},
            seed=0,
            max_samples=20,
        )
        items = list(ds)
        # 各サンプルは waveform tensor + token_ids tensor
        assert all(len(item) == 2 for item in items)
        # 全 20 サンプルがバラエティを持つ (token_ids が同一でない)
        unique_lengths = {item[1].shape[0] for item in items}
        assert len(unique_lengths) > 1

    def test_zero_weight_mode_excluded(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 1.0, "japanese": 0.0},
            seed=0,
            max_samples=3,
        )
        # 検証: クラッシュせず動く
        items = list(ds)
        assert len(items) == 3

    def test_invalid_weights_raise(self) -> None:
        with pytest.raises(ValueError):
            MorseSynthDataset(mode_mix={"european": -1.0})
        with pytest.raises(ValueError):
            MorseSynthDataset(mode_mix={})


# ============================================================
# DataLoader 並列
# ============================================================
class TestDatasetWithDataLoader:
    def test_single_worker(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 1.0}, seed=42, max_samples=4
        )
        loader = DataLoader(ds, batch_size=1, num_workers=0)
        batches = list(loader)
        assert len(batches) == 4

    def test_multiple_workers_run(self) -> None:
        """num_workers=2 で正常動作 (シリアライズ可能性確認)."""
        ds = MorseSynthDataset(
            mode_mix={"european": 1.0}, seed=42, max_samples=4
        )
        loader = DataLoader(ds, batch_size=1, num_workers=2)
        batches = list(loader)
        # 2 worker でも各 worker max_samples 個ずつ出す = 計 8
        # IterableDataset の挙動: 各 worker が独立イテレート
        assert len(batches) == 8


# ============================================================
# 固定評価セット
# ============================================================
class TestFixedEvalSet:
    def test_deterministic_with_same_seed(self) -> None:
        set1 = make_fixed_eval_set(
            snr_grid=[10.0], wpm_grid=[20.0], samples_per_cell=2, seed=99
        )
        set2 = make_fixed_eval_set(
            snr_grid=[10.0], wpm_grid=[20.0], samples_per_cell=2, seed=99
        )
        assert len(set1) == len(set2) == 2
        for r1, r2 in zip(set1, set2, strict=True):
            np.testing.assert_array_equal(r1.samples, r2.samples)
            np.testing.assert_array_equal(r1.token_ids, r2.token_ids)

    def test_grid_size_matches_product(self) -> None:
        results = make_fixed_eval_set(
            snr_grid=[10.0, 0.0],
            wpm_grid=[15.0, 25.0],
            samples_per_cell=3,
            seed=7,
        )
        # 2 SNR × 2 WPM × 3 samples = 12
        assert len(results) == 12

    def test_snr_is_propagated(self) -> None:
        results = make_fixed_eval_set(
            snr_grid=[15.0], wpm_grid=[20.0], samples_per_cell=1, seed=0
        )
        assert results[0].config.snr_db == 15.0


# ============================================================
# default_config_sampler
# ============================================================
class TestConfigSampler:
    def test_sampled_params_within_spec(self) -> None:
        # 既定 (hand_keying=True) では dash_dot_ratio の範囲が 2.5〜5.0 に拡張される
        # (Task 6, 設計書 §3.1)。
        sampler = default_config_sampler("european")
        rng = np.random.default_rng(0)
        for _ in range(50):
            cfg = sampler(rng)
            assert 8.0 <= cfg.keying.wpm <= 50.0
            assert 2.5 <= cfg.keying.dash_dot_ratio <= 5.0
            assert 400.0 <= cfg.keying.tone_freq_hz <= 900.0
            assert -10.0 <= cfg.snr_db <= 20.0
            assert 0.0 <= cfg.qsb_depth <= 0.8
            assert cfg.qrm_stations in (0, 1, 2)

    def test_sampled_params_within_spec_hand_keying_off(self) -> None:
        # hand_keying=False は従来 (Task 6 以前) の分布のまま.
        sampler = default_config_sampler("european", hand_keying=False)
        rng = np.random.default_rng(0)
        for _ in range(50):
            cfg = sampler(rng)
            assert 8.0 <= cfg.keying.wpm <= 50.0
            assert 2.5 <= cfg.keying.dash_dot_ratio <= 4.0
            assert 400.0 <= cfg.keying.tone_freq_hz <= 900.0
            assert -10.0 <= cfg.snr_db <= 20.0
            assert 0.0 <= cfg.qsb_depth <= 0.8
            assert cfg.qrm_stations in (0, 1, 2)


# ============================================================
# 実ノイズ混合データセット
# ============================================================
class TestRealNoiseMixing:
    @staticmethod
    def _dc_noise_pool() -> RealNoisePool:
        # 定数 (DC) ノイズ → 混合されると波形平均が 0 からずれる
        return RealNoisePool([np.full(8000 * 10, 1.0, dtype=np.float32)])

    def test_noise_prob_one_mixes_noise(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 1.0},
            seed=5,
            max_samples=3,
            noise_pool=self._dc_noise_pool(),
            noise_prob=1.0,
            noise_snr_range=(-10.0, -10.0),
        )
        for wave, _tokens in ds:
            assert abs(float(wave.mean())) > 0.05

    def test_noise_prob_zero_is_clean(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 1.0},
            seed=5,
            max_samples=3,
            noise_pool=self._dc_noise_pool(),
            noise_prob=0.0,
            noise_snr_range=(-10.0, -10.0),
        )
        for wave, _tokens in ds:
            assert abs(float(wave.mean())) < 0.05

    def test_reproducible_with_noise(self) -> None:
        def make() -> list[torch.Tensor]:
            ds = MorseSynthDataset(
                mode_mix={"japanese": 1.0},
                seed=11,
                max_samples=3,
                noise_pool=self._dc_noise_pool(),
                noise_prob=0.5,
                noise_snr_range=(-5.0, 15.0),
            )
            return [w for w, _ in ds]

        for w1, w2 in zip(make(), make(), strict=True):
            torch.testing.assert_close(w1, w2)

    def test_default_has_no_noise_pool(self) -> None:
        ds = MorseSynthDataset(mode_mix={"european": 1.0}, seed=0, max_samples=1)
        assert ds.noise_pool is None


class TestToneFreqRange:
    def test_sampler_respects_tone_freq_range(self) -> None:
        sampler = DefaultConfigSampler("european", tone_freq_range=(590.0, 610.0))
        rng = np.random.default_rng(0)
        for _ in range(20):
            config = sampler(rng)
            assert 590.0 <= config.keying.tone_freq_hz <= 610.0
            assert config.filter_center_hz == config.keying.tone_freq_hz

    def test_default_range_unchanged(self) -> None:
        sampler = DefaultConfigSampler("european")
        rng = np.random.default_rng(0)
        freqs = [sampler(rng).keying.tone_freq_hz for _ in range(50)]
        assert min(freqs) < 500.0 and max(freqs) > 800.0

    def test_dataset_tone_freq_range_applies_per_mode(self) -> None:
        # 欧文+和文の両モードでトーン制約が効き、モードも維持されること
        ds = MorseSynthDataset(
            mode_mix={"european": 0.5, "japanese": 0.5},
            seed=1,
            max_samples=1,
            tone_freq_range=(590.0, 610.0),
        )
        rng = np.random.default_rng(0)
        for mode, sampler in ds._samplers.items():
            cfg = sampler(rng)
            assert cfg.mode == mode
            assert 590.0 <= cfg.keying.tone_freq_hz <= 610.0


from src.synth.dataset import (
    RealNoiseEvalSample,
    make_fixed_real_noise_eval_set,
)
from src.synth.noise import RealNoisePool


def _dummy_noise_pool() -> RealNoisePool:
    rng = np.random.default_rng(0)
    # 帯域内 (約 500Hz) の合成ノイズ 8 秒分
    t = np.arange(8000 * 8) / 8000
    wave = (np.sin(2 * np.pi * 500 * t) * rng.normal(1.0, 0.1, t.size)).astype(np.float32)
    return RealNoisePool([wave], sample_rate=8000)


class TestFixedRealNoiseEvalSet:
    def test_deterministic(self) -> None:
        pool = _dummy_noise_pool()
        kw = dict(noise_pool=pool, snr_grid=[10.0, 0.0], wpm_grid=[20.0],
                  samples_per_cell=2, seed=42, mode="european")
        set1 = make_fixed_real_noise_eval_set(**kw)
        set2 = make_fixed_real_noise_eval_set(**kw)
        assert len(set1) == len(set2) == 4  # 2 snr * 1 wpm * 2
        for a, b in zip(set1, set2, strict=True):
            assert np.array_equal(a.samples, b.samples)
            assert a.token_ids.tolist() == b.token_ids.tolist()
            assert a.eff_snr_db == b.eff_snr_db

    def test_records_effective_snr_close_to_target_for_inband_noise(self) -> None:
        # 帯域内ノイズは BPF で減らないので実効 ≒ 目標
        pool = _dummy_noise_pool()
        samples = make_fixed_real_noise_eval_set(
            noise_pool=pool, snr_grid=[5.0], wpm_grid=[20.0],
            samples_per_cell=3, seed=7, mode="european",
        )
        for s in samples:
            assert s.target_snr_db == 5.0
            assert abs(s.eff_snr_db - 5.0) < 4.0
            assert s.samples.dtype == np.float32
            assert s.token_ids.size > 0

    def test_grid_covers_all_cells(self) -> None:
        pool = _dummy_noise_pool()
        samples = make_fixed_real_noise_eval_set(
            noise_pool=pool, snr_grid=[10.0, 0.0], wpm_grid=[17.0, 25.0],
            samples_per_cell=1, seed=1, mode="european",
        )
        pairs = {(s.target_snr_db, s.wpm) for s in samples}
        assert pairs == {(10.0, 17.0), (10.0, 25.0), (0.0, 17.0), (0.0, 25.0)}


from src.synth.noise import effective_snr_db


class TestSynthEffectiveSnr:
    def _cfg(self, effective: bool, snr: float) -> SynthConfig:
        return SynthConfig(
            mode="european",
            keying=KeyingParams(wpm=20.0, tone_freq_hz=600.0),
            snr_db=snr,
            filter_center_hz=600.0,
            filter_bandwidth_hz=400.0,
            snr_is_effective=effective,
        )

    def test_default_flag_is_false(self) -> None:
        cfg = SynthConfig(mode="european")
        assert cfg.snr_is_effective is False

    def test_effective_mode_hits_target_snr(self) -> None:
        rng = np.random.default_rng(3)
        cfg = self._cfg(effective=True, snr=0.0)
        res = synthesize_from_text("CQ DE JA0XYZ K", cfg, rng, sample_rate=8000)
        # BPF後の結果に対し、クリーン合成との差からノイズ実効SNRを推定するのは難しいので、
        # ここでは「実効0dB は nominal 0dB より明確にノイズが多い (BPF後 SNR が低い)」ことで
        # 分岐が効いていることを確認する。
        rng2 = np.random.default_rng(3)
        cfg_nom = self._cfg(effective=False, snr=0.0)
        res_nom = synthesize_from_text("CQ DE JA0XYZ K", cfg_nom, rng2, sample_rate=8000)
        # 実効0dB はノイズが強いので RMS が nominal 0dB より大きい (BPF後の帯域内で)
        assert float(np.sqrt(np.mean(res.samples.astype(np.float64) ** 2))) > \
               float(np.sqrt(np.mean(res_nom.samples.astype(np.float64) ** 2)))

    def test_nominal_mode_unchanged(self) -> None:
        # snr_is_effective=False は従来と同じ経路 (決定的に一致)
        rng1 = np.random.default_rng(9)
        rng2 = np.random.default_rng(9)
        cfg = self._cfg(effective=False, snr=5.0)
        a = synthesize_from_text("TEST", cfg, rng1, sample_rate=8000)
        b = synthesize_from_text("TEST", cfg, rng2, sample_rate=8000)
        assert np.array_equal(a.samples, b.samples)


class TestEffectiveSnrRangeWiring:
    def test_sampler_none_keeps_nominal(self) -> None:
        sampler = DefaultConfigSampler("european")
        cfg = sampler(np.random.default_rng(0))
        assert cfg.snr_is_effective is False
        assert -10.0 <= cfg.snr_db <= 20.0

    def test_sampler_effective_range_sets_flag_and_range(self) -> None:
        sampler = DefaultConfigSampler("european", effective_snr_range=(-8.0, 25.0))
        for seed in range(20):
            cfg = sampler(np.random.default_rng(seed))
            assert cfg.snr_is_effective is True
            assert -8.0 <= cfg.snr_db <= 25.0

    def test_dataset_effective_range_propagates_to_both_modes(self) -> None:
        ds = MorseSynthDataset(
            mode_mix={"european": 0.5, "japanese": 0.5},
            seed=1, max_samples=0,
            effective_snr_range=(-8.0, 25.0),
        )
        for mode in ("european", "japanese"):
            cfg = ds._samplers[mode](np.random.default_rng(2))
            assert cfg.snr_is_effective is True
            assert cfg.mode == mode
            assert -8.0 <= cfg.snr_db <= 25.0

    def test_dataset_default_is_nominal(self) -> None:
        ds = MorseSynthDataset(mode_mix={"european": 1.0}, seed=1, max_samples=0)
        cfg = ds._samplers["european"](np.random.default_rng(2))
        assert cfg.snr_is_effective is False


class TestHandKeyingSampler:
    def test_log_uniform_stays_in_range(self) -> None:
        from src.synth.dataset import log_uniform

        rng = np.random.default_rng(0)
        vals = np.array([log_uniform(rng, 0.05, 1.30) for _ in range(1000)])
        assert vals.min() >= 0.05
        assert vals.max() <= 1.30
        # 対数一様なので中央値は幾何平均 (≈0.255) 付近。一様なら 0.675 付近になる
        assert np.median(vals) < 0.45

    def test_sampler_sets_per_kind_sigma(self) -> None:
        """hand_keying=True で種別ごとの σ が設定される."""
        from src.synth.dataset import default_config_sampler

        # electronic_keyer_prob=0.0: 混合成分 (I-1) を無効化し、手打ち分岐が
        # 確実に引かれるようにする (このテストの意図は手打ち分岐の範囲検証)
        sampler = default_config_sampler(
            mode="european", hand_keying=True, electronic_keyer_prob=0.0
        )
        cfg = sampler(np.random.default_rng(0))
        k = cfg.keying
        assert k.dot_jitter_sigma_ratio is not None
        assert k.dash_jitter_sigma_ratio is not None
        assert 0.02 <= k.dot_jitter_sigma_ratio <= 0.20
        assert 0.05 <= k.dash_jitter_sigma_ratio <= 1.30
        assert 2.5 <= k.dash_dot_ratio <= 5.0
        assert 1.0 <= k.intra_element_space_units <= 1.3
        assert 2.6 <= k.inter_char_space_units <= 3.2
        assert 5.0 <= k.inter_word_space_units <= 16.0

    def test_extreme_tail_off_caps_dash_sigma(self) -> None:
        """extreme_tail=False で長音 σ の上限が下がる."""
        from src.synth.dataset import default_config_sampler

        # electronic_keyer_prob=0.0: 混合成分が None を混ぜると max() が落ちるため無効化する
        sampler = default_config_sampler(
            mode="european", hand_keying=True, extreme_tail=False,
            electronic_keyer_prob=0.0,
        )
        rng = np.random.default_rng(1)
        vals = [sampler(rng).keying.dash_jitter_sigma_ratio for _ in range(300)]
        assert max(vals) <= 0.70

    def test_hand_keying_off_reproduces_old_distribution(self) -> None:
        """hand_keying=False なら従来どおり種別ごとの σ は None のまま."""
        from src.synth.dataset import default_config_sampler

        sampler = default_config_sampler(mode="european", hand_keying=False)
        k = sampler(np.random.default_rng(0)).keying
        assert k.dot_jitter_sigma_ratio is None
        assert k.dash_jitter_sigma_ratio is None
        assert k.intra_element_space_units == 1.0
        assert k.inter_char_space_units == 3.0
        assert k.inter_word_space_units == 7.0
        assert 2.5 <= k.dash_dot_ratio <= 4.0

    def test_electronic_keyer_is_still_reachable(self) -> None:
        """エレキー相当が「同時条件」として現実的な頻度で引ける.

        周辺分布だけを見てはいけない。各パラメータを独立に引くと、σ が小さい
        サンプルも間隔が教科書どおりのサンプルもそれぞれ存在するのに、
        「両方を同時に満たす」サンプルは 1 件も出ない、ということが起きる
        (実際に 20,000 サンプルで 0 件だった)。守るべき制約は同時条件の方。
        """
        from src.synth.dataset import default_config_sampler

        sampler = default_config_sampler(mode="european", hand_keying=True)
        rng = np.random.default_rng(2)
        clean = 0
        for _ in range(1000):
            k = sampler(rng).keying
            is_textbook_spacing = (
                k.intra_element_space_units == 1.0
                and k.inter_char_space_units == 3.0
                and k.inter_word_space_units == 7.0
            )
            is_legacy_sigma = (
                k.dot_jitter_sigma_ratio is None
                and k.dash_jitter_sigma_ratio is None
            )
            if is_textbook_spacing and is_legacy_sigma and 2.5 <= k.dash_dot_ratio <= 4.0:
                clean += 1
        # electronic_keyer_prob = 0.25 なので 1000 件中 250 件前後。
        # 150 件を下回るなら混合成分が壊れている
        assert clean >= 150

    def test_electronic_keyer_prob_zero_disables_mixture(self) -> None:
        """electronic_keyer_prob=0 なら手打ち分布だけになる (A/B 用)."""
        from src.synth.dataset import default_config_sampler

        sampler = default_config_sampler(
            mode="european", hand_keying=True, electronic_keyer_prob=0.0
        )
        rng = np.random.default_rng(5)
        for _ in range(200):
            k = sampler(rng).keying
            assert k.dash_jitter_sigma_ratio is not None

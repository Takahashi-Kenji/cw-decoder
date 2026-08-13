"""ノイズ・伝搬モデルのテスト."""
from __future__ import annotations

import numpy as np
import pytest

from src.synth.noise import (
    RealNoisePool,
    add_awgn_effective,
    add_real_noise,
    add_awgn,
    add_qrm,
    add_qrn,
    apply_cw_filter,
    apply_qsb,
    effective_snr_db,
    signal_power,
)


def _make_tone(freq_hz: float, duration_sec: float, sample_rate: int = 8000) -> np.ndarray:
    n = int(duration_sec * sample_rate)
    t = np.arange(n) / sample_rate
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


# ============================================================
# AWGN
# ============================================================
class TestAWGN:
    @pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0, 20.0])
    def test_measured_snr_close_to_target(self, snr_db: float) -> None:
        rng = np.random.default_rng(42)
        sig = _make_tone(600.0, 1.0)
        sp = signal_power(sig)
        noisy = add_awgn(sig, snr_db, rng)
        noise = noisy - sig
        np_power = signal_power(noise)
        measured_snr = 10 * np.log10(sp / np_power)
        # ±1 dB 以内
        assert abs(measured_snr - snr_db) < 1.0

    def test_reproducible(self) -> None:
        sig = _make_tone(600.0, 0.5)
        noisy1 = add_awgn(sig, 0.0, np.random.default_rng(7))
        noisy2 = add_awgn(sig, 0.0, np.random.default_rng(7))
        np.testing.assert_array_equal(noisy1, noisy2)

    def test_zero_signal(self) -> None:
        rng = np.random.default_rng(0)
        sig = np.zeros(8000, dtype=np.float32)
        out = add_awgn(sig, 10.0, rng)
        # 全ゼロ入力でもクラッシュしないこと
        assert out.shape == sig.shape


# ============================================================
# QSB
# ============================================================
class TestQSB:
    def test_zero_depth_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        sig = _make_tone(600.0, 1.0)
        out = apply_qsb(sig, depth=0.0, period_s=2.0, sample_rate=8000, rng=rng)
        np.testing.assert_array_equal(out, sig)

    def test_amplitude_modulation_periodic(self) -> None:
        rng = np.random.default_rng(0)
        sig = _make_tone(600.0, 4.0)
        # 周期 1 秒 (4 周期含まれる)
        out = apply_qsb(sig, depth=0.8, period_s=1.0, sample_rate=8000, rng=rng)
        # エンベロープ (絶対値の移動平均) が変動していることを確認
        env = np.abs(out)
        window = 100
        smooth = np.convolve(env, np.ones(window) / window, mode="valid")
        assert smooth.max() / smooth.min() > 2.0  # 振幅が 2 倍以上変動

    @pytest.mark.parametrize("bad_depth", [-0.1, 1.5])
    def test_invalid_depth_raises(self, bad_depth: float) -> None:
        rng = np.random.default_rng(0)
        sig = _make_tone(600.0, 0.5)
        with pytest.raises(ValueError):
            apply_qsb(sig, depth=bad_depth, period_s=1.0, sample_rate=8000, rng=rng)


# ============================================================
# QRM
# ============================================================
class TestQRM:
    def test_zero_stations_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        sig = _make_tone(600.0, 1.0)
        out = add_qrm(sig, num_stations=0, sample_rate=8000, rng=rng)
        np.testing.assert_array_equal(out, sig)

    def test_adds_energy_outside_self_freq(self) -> None:
        rng = np.random.default_rng(123)
        sig = _make_tone(600.0, 2.0)
        out = add_qrm(
            sig,
            num_stations=2,
            sample_rate=8000,
            rng=rng,
            self_freq_hz=600.0,
            freq_range_hz=(300.0, 1000.0),
        )
        # スペクトルに 600 Hz 以外のピークが現れる
        spec = np.abs(np.fft.rfft(out.astype(np.float64) - sig.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(out), d=1.0 / 8000)
        # 主成分は 600 Hz 周辺の外
        peak_freq = freqs[int(np.argmax(spec))]
        assert abs(peak_freq - 600.0) > 50.0


# ============================================================
# QRN
# ============================================================
class TestQRN:
    def test_zero_rate_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        sig = _make_tone(600.0, 1.0)
        out = add_qrn(sig, rate_per_sec=0.0, intensity=1.0, sample_rate=8000, rng=rng)
        np.testing.assert_array_equal(out, sig)

    def test_adds_impulses(self) -> None:
        rng = np.random.default_rng(0)
        sig = _make_tone(600.0, 2.0)
        out = add_qrn(sig, rate_per_sec=20.0, intensity=2.0, sample_rate=8000, rng=rng)
        diff = out - sig
        # インパルスは局所的に大振幅
        assert np.max(np.abs(diff)) > np.std(diff) * 5


# ============================================================
# CW フィルタ
# ============================================================
class TestCWFilter:
    def test_passband_signal_preserved(self) -> None:
        sig = _make_tone(600.0, 1.0)
        out = apply_cw_filter(sig, center_hz=600.0, bandwidth_hz=400.0, sample_rate=8000)
        # 通過帯域の信号はほぼ保持される (RMS 比 0.9 以上)
        rms_in = np.sqrt(signal_power(sig))
        rms_out = np.sqrt(signal_power(out))
        assert rms_out / rms_in > 0.85

    def test_out_of_band_signal_attenuated(self) -> None:
        sig = _make_tone(2000.0, 1.0)
        out = apply_cw_filter(sig, center_hz=600.0, bandwidth_hz=400.0, sample_rate=8000)
        rms_in = np.sqrt(signal_power(sig))
        rms_out = np.sqrt(signal_power(out))
        # 帯域外は大きく減衰
        assert rms_out / rms_in < 0.1

    def test_invalid_passband_raises(self) -> None:
        sig = _make_tone(600.0, 0.5)
        with pytest.raises(ValueError):
            apply_cw_filter(sig, center_hz=5000.0, bandwidth_hz=400.0, sample_rate=8000)


# ============================================================
# 実ノイズ混合 (RealNoisePool / add_real_noise)
# ============================================================
class TestAddRealNoise:
    @pytest.mark.parametrize("snr_db", [-5.0, 0.0, 10.0])
    def test_measured_snr_close_to_target(self, snr_db: float) -> None:
        rng = np.random.default_rng(3)
        sig = _make_tone(600.0, 1.0)
        noise = rng.normal(0.0, 0.3, size=sig.shape).astype(np.float32)
        out = add_real_noise(sig, noise, snr_db)
        added = out - sig
        measured = 10 * np.log10(signal_power(sig) / signal_power(added))
        assert abs(measured - snr_db) < 0.5

    def test_length_mismatch_raises(self) -> None:
        sig = _make_tone(600.0, 1.0)
        noise = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError):
            add_real_noise(sig, noise, 10.0)

    def test_silent_noise_returns_signal_unchanged(self) -> None:
        sig = _make_tone(600.0, 0.5)
        noise = np.zeros_like(sig)
        out = add_real_noise(sig, noise, 10.0)
        np.testing.assert_array_equal(out, sig)

    def test_dtype_preserved(self) -> None:
        rng = np.random.default_rng(0)
        sig = _make_tone(600.0, 0.5)
        noise = rng.normal(0.0, 1.0, size=sig.shape).astype(np.float32)
        out = add_real_noise(sig, noise, 5.0)
        assert out.dtype == np.float32

    def test_silent_signal_scales_noise_to_unit_reference(self) -> None:
        # 無音区間 (キーイング前後) でもノイズが乗ること
        rng = np.random.default_rng(1)
        sig = np.zeros(4000, dtype=np.float32)
        noise = rng.normal(0.0, 0.3, size=sig.shape).astype(np.float32)
        out = add_real_noise(sig, noise, 10.0)
        assert signal_power(out) > 0.0


class TestRealNoisePool:
    @staticmethod
    def _write_wav(path, wave: np.ndarray, sr: int = 8000) -> None:
        import soundfile as sf
        sf.write(path, wave, sr)

    def _make_pool_dir(self, tmp_path, rng: np.random.Generator):
        d = tmp_path / "noise"
        d.mkdir()
        self._write_wav(d / "a.wav", rng.normal(0, 0.1, 8000 * 2).astype(np.float32))
        self._write_wav(d / "b.wav", rng.normal(0, 0.1, 8000 * 5).astype(np.float32))
        return d

    def test_from_dir_loads_all_wavs(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        d = self._make_pool_dir(tmp_path, rng)
        pool = RealNoisePool.from_dir(d, sample_rate=8000)
        assert len(pool) == 2

    def test_from_dir_empty_raises(self, tmp_path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError):
            RealNoisePool.from_dir(d)

    def test_sample_segment_length(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        pool = RealNoisePool.from_dir(self._make_pool_dir(tmp_path, rng))
        seg = pool.sample_segment(12345, np.random.default_rng(1))
        assert seg.shape == (12345,)
        assert seg.dtype == np.float32

    def test_sample_segment_reproducible(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        pool = RealNoisePool.from_dir(self._make_pool_dir(tmp_path, rng))
        seg1 = pool.sample_segment(8000, np.random.default_rng(7))
        seg2 = pool.sample_segment(8000, np.random.default_rng(7))
        np.testing.assert_array_equal(seg1, seg2)

    def test_sample_segment_longer_than_files_wraps(self, tmp_path) -> None:
        # 全ファイルより長い要求 → 循環で埋める
        rng = np.random.default_rng(0)
        pool = RealNoisePool.from_dir(self._make_pool_dir(tmp_path, rng))
        n = 8000 * 20
        seg = pool.sample_segment(n, np.random.default_rng(2))
        assert seg.shape == (n,)
        assert signal_power(seg) > 0.0

    def test_from_dir_resamples_to_target_rate(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        d = tmp_path / "noise48"
        d.mkdir()
        # 48kHz で 1 秒 → 8kHz に変換され 8000 サンプル相当
        self._write_wav(d / "n.wav", rng.normal(0, 0.1, 48000).astype(np.float32), sr=48000)
        pool = RealNoisePool.from_dir(d, sample_rate=8000)
        assert abs(pool.total_samples - 8000) < 80

    def test_from_dir_skips_stereo_by_taking_first_channel(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        d = tmp_path / "stereo"
        d.mkdir()
        stereo = rng.normal(0, 0.1, (8000, 2)).astype(np.float32)
        self._write_wav(d / "s.wav", stereo)
        pool = RealNoisePool.from_dir(d)
        seg = pool.sample_segment(4000, np.random.default_rng(0))
        assert seg.ndim == 1


# ============================================================
# 実効 SNR
# ============================================================
class TestEffectiveSnrDb:
    def test_broadband_awgn_has_positive_bpf_gain(self) -> None:
        """帯域外に広がる AWGN は BPF 後に減るので、実効 SNR は公称より高くなる."""
        rng = np.random.default_rng(0)
        sr = 8000
        # 600 Hz トーン (帯域内の信号)
        t = np.arange(sr * 2) / sr
        sig = np.sin(2 * np.pi * 600 * t).astype(np.float32)
        # 公称 0 dB になるよう広帯域ノイズを作る
        nominal_snr = 0.0
        noise_power = signal_power(sig) / (10.0 ** (nominal_snr / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=sig.size).astype(np.float32)

        eff = effective_snr_db(sig, noise, center_hz=600.0, bandwidth_hz=500.0, sample_rate=sr)
        # BPF が帯域外ノイズを捨てるので実効 SNR は公称 0 dB より数 dB 高い
        assert eff > nominal_snr + 3.0

    def test_inband_noise_has_near_zero_gain(self) -> None:
        """既に帯域内のノイズは BPF でほぼ減らない → 実効 ≒ 公称."""
        rng = np.random.default_rng(1)
        sr = 8000
        t = np.arange(sr * 2) / sr
        sig = np.sin(2 * np.pi * 600 * t).astype(np.float32)
        # 帯域内 (590-610Hz) の狭帯域ノイズ
        narrow = np.sin(2 * np.pi * 600 * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        narrow *= rng.normal(1.0, 0.05, size=sig.size).astype(np.float32)
        eff = effective_snr_db(sig, narrow, center_hz=600.0, bandwidth_hz=500.0, sample_rate=sr)
        nominal = 10.0 * np.log10(signal_power(sig) / signal_power(narrow))
        assert abs(eff - nominal) < 3.0

    def test_zero_noise_returns_inf(self) -> None:
        sr = 8000
        sig = np.ones(sr, dtype=np.float32)
        noise = np.zeros(sr, dtype=np.float32)
        eff = effective_snr_db(sig, noise, center_hz=600.0, bandwidth_hz=500.0, sample_rate=sr)
        assert eff == float("inf")


# ============================================================
# 実効 SNR 目標での AWGN 加算
# ============================================================
class TestAddAwgnEffective:
    def _tone(self, sr: int = 8000, dur: float = 2.0, freq: float = 600.0) -> np.ndarray:
        t = np.arange(int(sr * dur)) / sr
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    @pytest.mark.parametrize("target", [-5.0, 0.0, 10.0])
    @pytest.mark.parametrize("bw", [250.0, 500.0])
    def test_hits_target_effective_snr(self, target: float, bw: float) -> None:
        sr = 8000
        sig = self._tone(sr)
        rng = np.random.default_rng(1)
        noisy = add_awgn_effective(sig, target, 600.0, bw, sr, rng)
        # noisy - sig がスケール済みノイズ。BPF後の実効SNRを測る。
        eff = effective_snr_db(sig, noisy - sig, 600.0, bw, sr)
        assert abs(eff - target) < 1.0

    def test_deterministic(self) -> None:
        sig = self._tone()
        a = add_awgn_effective(sig, 0.0, 600.0, 400.0, 8000, np.random.default_rng(7))
        b = add_awgn_effective(sig, 0.0, 600.0, 400.0, 8000, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_silent_signal_returns_copy(self) -> None:
        sig = np.zeros(8000, dtype=np.float32)
        out = add_awgn_effective(sig, 0.0, 600.0, 400.0, 8000, np.random.default_rng(0))
        assert np.array_equal(out, sig)
        assert out is not sig

    def test_preserves_dtype(self) -> None:
        sig = self._tone().astype(np.float32)
        out = add_awgn_effective(sig, 5.0, 600.0, 400.0, 8000, np.random.default_rng(0))
        assert out.dtype == np.float32

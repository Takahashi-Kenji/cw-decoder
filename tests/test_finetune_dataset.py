"""実信号データセットのテスト."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.finetune.dataset import (
    RealSignalDataset,
    RealSignalSample,
    _parse_txt,
    discover_real_samples,
)
from src.finetune.pipeline import (
    MixedRealSynthDataset,
    split_train_validation,
)
from src.tokens.morse_tokens import text_to_codes

from scripts.finetune import resolve_train_eval_samples


def _make_test_pair(
    dir_: Path,
    stem: str,
    text: str,
    mode: str = "european",
    sr: int = 8000,
    duration_s: float = 0.5,
) -> Path:
    """テスト用 WAV + TXT ペアを作成. WAV path を返す."""
    wav_path = dir_ / f"{stem}.wav"
    txt_path = dir_ / f"{stem}.txt"
    samples = np.zeros(int(duration_s * sr), dtype=np.float32)
    sf.write(wav_path, samples, sr, subtype="PCM_16")
    txt_path.write_text(
        f"mode: {mode}\nsample_rate: {sr}\nduration_s: {duration_s}\n---\n{text}\n",
        encoding="utf-8",
    )
    return wav_path


class TestParseTxt:
    def test_with_header_and_body(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_text("mode: european\nsample_rate: 8000\n---\nABC\n", encoding="utf-8")
        header, body = _parse_txt(p)
        assert header == {"mode": "european", "sample_rate": "8000"}
        assert body == "ABC"

    def test_no_separator(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_text("just plain text\n", encoding="utf-8")
        header, body = _parse_txt(p)
        assert header == {}
        assert body == "just plain text"

    def test_multiline_body(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_text("mode: european\n---\nLINE1\nLINE2\n", encoding="utf-8")
        _, body = _parse_txt(p)
        assert "LINE1" in body and "LINE2" in body


class TestDiscover:
    def test_finds_pairs(self, tmp_path: Path) -> None:
        _make_test_pair(tmp_path, "sample_001_european", "ABC", mode="european")
        _make_test_pair(tmp_path, "sample_002_japanese", "イロハ", mode="japanese")
        samples = discover_real_samples(tmp_path)
        assert len(samples) == 2
        modes = {s.mode for s in samples}
        assert modes == {"european", "japanese"}

    def test_mode_filter(self, tmp_path: Path) -> None:
        _make_test_pair(tmp_path, "a_european", "ABC", mode="european")
        _make_test_pair(tmp_path, "b_japanese", "イロハ", mode="japanese")
        eu_only = discover_real_samples(tmp_path, mode_filter="european")
        assert len(eu_only) == 1
        assert eu_only[0].mode == "european"

    def test_require_text_excludes_empty(self, tmp_path: Path) -> None:
        # 空テキストの TXT
        wav = tmp_path / "empty.wav"
        sf.write(wav, np.zeros(4000, dtype=np.float32), 8000, subtype="PCM_16")
        (tmp_path / "empty.txt").write_text("mode: european\n---\n", encoding="utf-8")
        samples = discover_real_samples(tmp_path, require_text=True)
        assert samples == []

    def test_filename_mode_inference(self, tmp_path: Path) -> None:
        # ヘッダ無し、ファイル名から推定
        wav = tmp_path / "20260612_test_european.wav"
        txt = tmp_path / "20260612_test_european.txt"
        sf.write(wav, np.zeros(4000, dtype=np.float32), 8000, subtype="PCM_16")
        txt.write_text("ABC\n", encoding="utf-8")
        samples = discover_real_samples(tmp_path)
        assert len(samples) == 1
        assert samples[0].mode == "european"

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert discover_real_samples(tmp_path / "nowhere") == []


class TestRealSignalDataset:
    def test_basic_iteration(self, tmp_path: Path) -> None:
        _make_test_pair(tmp_path, "a", "ABC", mode="european")
        _make_test_pair(tmp_path, "b", "XYZ", mode="european")
        samples = discover_real_samples(tmp_path)
        ds = RealSignalDataset(samples)
        assert len(ds) == 2
        for i in range(len(ds)):
            wave, ids = ds[i]
            assert isinstance(wave, torch.Tensor)
            assert isinstance(ids, torch.Tensor)
            assert ids.dtype == torch.long
            assert ids.numel() == 3  # 3 letters

    def test_resamples_if_needed(self, tmp_path: Path) -> None:
        # 44.1 kHz WAV を作る
        wav_path = tmp_path / "highrate.wav"
        sf.write(wav_path, np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")
        txt = tmp_path / "highrate.txt"
        txt.write_text("mode: european\nsample_rate: 44100\n---\nA\n", encoding="utf-8")
        samples = discover_real_samples(tmp_path)
        ds = RealSignalDataset(samples)
        wave, ids = ds[0]
        # 8 kHz にリサンプルされて約 8000 サンプル
        assert abs(wave.numel() - 8000) < 50

    def test_empty_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            RealSignalDataset([])

    def test_sample_at_matches_getitem(self, tmp_path: Path) -> None:
        _make_test_pair(tmp_path, "a_european", "ABC", mode="european")
        _make_test_pair(tmp_path, "b_japanese", "イロハ", mode="japanese")
        samples = discover_real_samples(tmp_path)
        ds = RealSignalDataset(samples)
        for i in range(len(ds)):
            _, ids = ds[i]
            meta = ds.sample_at(i)
            # メタデータのテキストから作った符号長と token 数が一致すること
            assert ids.numel() == len(text_to_codes(meta.text, meta.mode))

    def test_sample_at_skips_dropped_samples(self, tmp_path: Path) -> None:
        """空トークンのサンプルが落ちても sample_at がずれないこと (評価の正しさ)."""
        wav_a = _make_test_pair(tmp_path, "a_european", "A", mode="european")
        wav_b = _make_test_pair(tmp_path, "b_european", "XYZ", mode="european")
        # text_to_codes が空符号列になるサンプル (スペースのみ) を先頭に置く
        samples = [
            RealSignalSample(
                wav_path=wav_a, txt_path=wav_a.with_suffix(".txt"),
                mode="european", text=" ",
            ),
            RealSignalSample(
                wav_path=wav_b, txt_path=wav_b.with_suffix(".txt"),
                mode="european", text="XYZ",
            ),
        ]
        ds = RealSignalDataset(samples)
        assert len(ds) == 1               # 空トークンはスキップされる
        assert ds.sample_at(0).text == "XYZ"


class TestSplit:
    def test_split_deterministic(self) -> None:
        fake = [object() for _ in range(10)]
        a1, b1 = split_train_validation(fake, validation_ratio=0.2, seed=42)
        a2, b2 = split_train_validation(fake, validation_ratio=0.2, seed=42)
        # Same seed should produce same split
        assert [id(x) for x in a1] == [id(x) for x in a2]
        assert [id(x) for x in b1] == [id(x) for x in b2]

    def test_split_sizes(self) -> None:
        fake = list(range(10))
        train, evals = split_train_validation(fake, validation_ratio=0.3, seed=0)
        assert len(train) == 7
        assert len(evals) == 3


class TestMixedDataset:
    def test_mixes_real_and_synth(self, tmp_path: Path) -> None:
        _make_test_pair(tmp_path, "a", "ABC", mode="european")
        _make_test_pair(tmp_path, "b", "XYZ", mode="european")
        samples = discover_real_samples(tmp_path)
        real_ds = RealSignalDataset(samples)
        mixed = MixedRealSynthDataset(
            real_dataset=real_ds,
            mode_mix={"european": 1.0, "japanese": 0.0},
            real_ratio=0.5,
            seed=42,
            max_samples=10,
        )
        items = list(mixed)
        assert len(items) == 10

    def test_invalid_ratio_raises(self, tmp_path: Path) -> None:
        _make_test_pair(tmp_path, "a", "ABC", mode="european")
        samples = discover_real_samples(tmp_path)
        real_ds = RealSignalDataset(samples)
        with pytest.raises(ValueError):
            MixedRealSynthDataset(
                real_dataset=real_ds,
                mode_mix={"european": 1.0},
                real_ratio=0.0,
            )


class TestMixedDatasetRealNoise:
    def test_noise_args_forwarded_and_iterable(self, tmp_path: Path) -> None:
        from src.synth.noise import RealNoisePool

        _make_test_pair(tmp_path, "20260701_000000_european", "ABC")
        samples = discover_real_samples(tmp_path)
        real = RealSignalDataset(samples)
        pool = RealNoisePool([np.full(8000 * 5, 1.0, dtype=np.float32)])
        mixed = MixedRealSynthDataset(
            real_dataset=real,
            mode_mix={"european": 1.0},
            real_ratio=0.5,
            seed=3,
            max_samples=4,
            noise_pool=pool,
            noise_prob=1.0,
            noise_snr_range=(-10.0, -10.0),
        )
        assert mixed.synth_dataset.noise_pool is pool
        assert mixed.synth_dataset.noise_prob == 1.0
        out = list(mixed)
        assert len(out) == 4

    def test_config_sampler_forwarded(self, tmp_path: Path) -> None:
        from src.synth.dataset import DefaultConfigSampler

        _make_test_pair(tmp_path, "20260701_000001_european", "ABC")
        real = RealSignalDataset(discover_real_samples(tmp_path))
        sampler = DefaultConfigSampler("european", tone_freq_range=(590.0, 610.0))
        mixed = MixedRealSynthDataset(
            real_dataset=real,
            mode_mix={"european": 1.0},
            seed=0,
            config_sampler=sampler,
        )
        rng = np.random.default_rng(0)
        cfg = mixed.synth_dataset._samplers["european"](rng)
        assert 590.0 <= cfg.keying.tone_freq_hz <= 610.0


class TestResolveTrainEval:
    def test_fixed_eval_dir_keeps_all_data_for_train(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()
        _make_test_pair(train_dir, "t1_european", "CQ", mode="european")
        _make_test_pair(train_dir, "t2_european", "DE", mode="european")
        _make_test_pair(val_dir, "v1_european", "K", mode="european")

        train, val = resolve_train_eval_samples(
            data_dir=train_dir, eval_dir=val_dir,
            mode_filter=None, eval_ratio=0.2, seed=0,
        )
        assert len(train) == 2   # data_dir 全件が train
        assert len(val) == 1     # eval_dir が固定 val
        assert val[0].text == "K"

    def test_no_eval_dir_falls_back_to_random_split(self, tmp_path: Path) -> None:
        for i in range(5):
            _make_test_pair(tmp_path, f"s{i}_european", "CQ", mode="european")
        train, val = resolve_train_eval_samples(
            data_dir=tmp_path, eval_dir=None,
            mode_filter=None, eval_ratio=0.2, seed=0,
        )
        assert len(train) + len(val) == 5
        assert len(val) >= 1


class TestFinetuneCLIWiring:
    """scripts/finetune.py の CLI 引数が MixedRealSynthDataset まで届くことの確認.

    実際の学習 (モデル読み込み・学習ループ) は回さない。
    MixedRealSynthDataset をスタブに差し替え、渡された kwargs だけを検証する。
    """

    def test_hand_keying_args_forwarded_to_mixed_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_test_pair(tmp_path, "a_european", "ABC", mode="european")
        _make_test_pair(tmp_path, "b_european", "XYZ", mode="european")

        captured: dict = {}

        class _StopEarly(Exception):
            """モデル読み込み等の前で確実に打ち切るための番兵例外."""

        def _fake_mixed_dataset(*args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            raise _StopEarly

        import scripts.finetune as finetune_mod

        monkeypatch.setattr(finetune_mod, "MixedRealSynthDataset", _fake_mixed_dataset)

        with pytest.raises(_StopEarly):
            finetune_mod.main(
                [
                    "--data-dir", str(tmp_path),
                    "--resume", "unused.pt",
                    "--ckpt-dir", str(tmp_path / "ckpt"),
                    "--mix-synth",
                    "--no-extreme-tail",
                    "--electronic-keyer-prob", "0.42",
                ]
            )

        assert captured["hand_keying"] is False   # フラグ未指定なので既定の従来分布
        assert captured["extreme_tail"] is False
        assert captured["electronic_keyer_prob"] == pytest.approx(0.42)

    def test_defaults_use_legacy_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for i in range(5):
            _make_test_pair(tmp_path, f"s{i}_european", "CQ", mode="european")

        captured: dict = {}

        class _StopEarly(Exception):
            pass

        def _fake_mixed_dataset(*args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            raise _StopEarly

        import scripts.finetune as finetune_mod

        monkeypatch.setattr(finetune_mod, "MixedRealSynthDataset", _fake_mixed_dataset)

        with pytest.raises(_StopEarly):
            finetune_mod.main(
                [
                    "--data-dir", str(tmp_path),
                    "--resume", "unused.pt",
                    "--ckpt-dir", str(tmp_path / "ckpt"),
                    "--mix-synth",
                ]
            )

        # 既定は従来分布。手打ち分布はフル学習で実手打ちの TER を 2 倍以上
        # 悪化させたのでオプトインにしてある (--hand-keying)
        assert captured["hand_keying"] is False
        assert captured["extreme_tail"] is True
        assert captured["electronic_keyer_prob"] == pytest.approx(0.25)

"""train.py の実効SNR・実ノイズ混合 CLI 検証のテスト."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.train import (
    apply_lr,
    resolve_effective_snr_range,
    resolve_noise_pool,
    validate_noise_params,
)


class TestResolveEffectiveSnrRange:
    def test_both_none_returns_none(self) -> None:
        assert resolve_effective_snr_range(None, None) is None

    def test_both_given_returns_tuple(self) -> None:
        assert resolve_effective_snr_range(-8.0, 25.0) == (-8.0, 25.0)

    def test_only_min_raises(self) -> None:
        with pytest.raises(ValueError, match="両方"):
            resolve_effective_snr_range(-8.0, None)

    def test_only_max_raises(self) -> None:
        with pytest.raises(ValueError, match="両方"):
            resolve_effective_snr_range(None, 25.0)

    def test_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min"):
            resolve_effective_snr_range(25.0, -8.0)
        with pytest.raises(ValueError, match="min"):
            resolve_effective_snr_range(5.0, 5.0)


class TestValidateNoiseParams:
    def test_none_dir_is_noop(self) -> None:
        # noise_dir が None なら他の値が不正でも例外なし
        validate_noise_params(None, 5.0, 10.0, -10.0)

    def test_valid_params_pass(self) -> None:
        validate_noise_params(Path("data/noise"), 0.5, 0.0, 20.0)

    def test_prob_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="noise-prob"):
            validate_noise_params(Path("data/noise"), 1.5, 0.0, 20.0)
        with pytest.raises(ValueError, match="noise-prob"):
            validate_noise_params(Path("data/noise"), -0.1, 0.0, 20.0)

    def test_snr_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="snr"):
            validate_noise_params(Path("data/noise"), 0.5, 20.0, 0.0)
        with pytest.raises(ValueError, match="snr"):
            validate_noise_params(Path("data/noise"), 0.5, 5.0, 5.0)


class TestResolveNoisePool:
    def test_none_returns_none(self) -> None:
        assert resolve_noise_pool(None) is None

    def test_loads_pool_from_dir(self, tmp_path: Path) -> None:
        # 帯域内相当の短いノイズ wav を1本置く
        sf.write(
            tmp_path / "n.wav",
            np.random.default_rng(0).standard_normal(8000).astype(np.float32),
            8000,
            subtype="PCM_16",
        )
        pool = resolve_noise_pool(tmp_path)
        assert pool is not None
        assert len(pool) == 1


class TestApplyLr:
    """resume 後の学習率再適用 (FT で --lr を効かせるため)."""

    @staticmethod
    def _optimizer(lr: float) -> torch.optim.Optimizer:
        params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
        return torch.optim.AdamW(params, lr=lr)

    def test_sets_lr_on_all_groups(self) -> None:
        opt = self._optimizer(3e-4)
        apply_lr(opt, 5e-5)
        assert all(g["lr"] == pytest.approx(5e-5) for g in opt.param_groups)

    def test_overrides_lr_restored_by_load_state_dict(self) -> None:
        # load_state_dict は保存時の param_groups (lr 込み) で上書きするため、
        # --lr 指定が無視される。apply_lr がこれを打ち消すことを確認する。
        saved = self._optimizer(3e-4).state_dict()
        opt = self._optimizer(5e-5)
        opt.load_state_dict(saved)
        assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)  # 上書きされている
        apply_lr(opt, 5e-5)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-5)


class TestHandKeyingFlags:
    """手打ち合成分布の CLI が MorseSynthDataset まで届くか.

    届かないと、旧分布 (models/full/best.pt の学習条件) を再現する手段が無くなり
    フル学習どうしの比較ができなくなる。
    """

    def _run_main_capturing_dataset(
        self, argv: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> dict:
        captured: dict = {}

        class _StopEarly(Exception):
            """DataLoader 構築より後へ進ませないための番兵例外."""

        def _fake_dataset(*args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            raise _StopEarly

        import scripts.train as train_mod

        monkeypatch.setattr(train_mod, "MorseSynthDataset", _fake_dataset)
        with pytest.raises(_StopEarly):
            train_mod.main(argv)
        return captured

    def test_flags_forwarded_to_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._run_main_capturing_dataset(
            [
                "--steps", "1",
                "--ckpt-dir", str(tmp_path / "ckpt"),
                "--device", "cpu",
                "--no-extreme-tail",
                "--electronic-keyer-prob", "0.42",
            ],
            monkeypatch,
        )
        assert captured["hand_keying"] is False   # 既定は従来分布
        assert captured["extreme_tail"] is False
        assert captured["electronic_keyer_prob"] == pytest.approx(0.42)

    def test_defaults_use_legacy_distribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """既定は従来分布。手打ち分布はフル学習で悪化したのでオプトインにしてある."""
        captured = self._run_main_capturing_dataset(
            ["--steps", "1", "--ckpt-dir", str(tmp_path / "ckpt"), "--device", "cpu"],
            monkeypatch,
        )
        assert captured["hand_keying"] is False
        assert captured["extreme_tail"] is True
        assert captured["electronic_keyer_prob"] == pytest.approx(0.25)

    def test_hand_keying_flag_opts_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = self._run_main_capturing_dataset(
            ["--steps", "1", "--ckpt-dir", str(tmp_path / "ckpt"), "--device", "cpu",
             "--hand-keying"],
            monkeypatch,
        )
        assert captured["hand_keying"] is True

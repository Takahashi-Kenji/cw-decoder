"""掃引スクリプトのスモークテスト (推論なし、キャッシュを直接組み立てる)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.eval.harness import decode_wave
from src.infer.word_break_policy import WordBreakPolicy
from src.train.loop import compute_input_lengths
from src.train.model import CWModel, ModelConfig
from src.train.preprocessing import MelExtractor
from src.tokens.morse_tokens import VOCAB_SIZE, WORD_BREAK_TOKEN_ID

_SPEC = importlib.util.spec_from_file_location(
    "sweep_word_break",
    Path(__file__).resolve().parent.parent / "scripts" / "sweep_word_break.py",
)
assert _SPEC is not None and _SPEC.loader is not None
sweep_mod = importlib.util.module_from_spec(_SPEC)
# dataclasses が cls.__module__ を sys.modules から引くため、実行前に登録する
# (登録しないと frozen dataclass の定義時に AttributeError で落ちる)
sys.modules[_SPEC.name] = sweep_mod
_SPEC.loader.exec_module(sweep_mod)


def _model_and_mel() -> tuple[CWModel, MelExtractor]:
    """``tests/test_eval_harness.py`` と同じ手順で seed 固定の未学習モデルを作る."""
    torch.manual_seed(0)
    model = CWModel(ModelConfig(vocab_size=VOCAB_SIZE))
    model.train(False)
    return model, MelExtractor()


def _cached(token_seq: list[int]) -> "sweep_mod.CachedSample":
    """各時刻で 1 トークンを強く出す log_probs を組み立てる."""
    t = len(token_seq)
    logits = torch.full((t, VOCAB_SIZE), -10.0)
    for j, tok in enumerate(token_seq):
        logits[j, tok] = 0.0
    return sweep_mod.CachedSample(
        log_probs=torch.log_softmax(logits, dim=-1),
        ref_tokens=[1, 2],
        ref_text="AB",
        mode="european",
        name="smoke",
    )


class TestSweepSmoke:
    def test_decode_cached_identity_matches_plain_decode(self) -> None:
        c = _cached([1, 0, 2])
        assert sweep_mod.decode_cached(c.log_probs, WordBreakPolicy()) == [1, 2]

    def test_bias_removes_word_break(self) -> None:
        c = _cached([1, 0, WORD_BREAK_TOKEN_ID, 0, 2])
        with_wb = sweep_mod.decode_cached(c.log_probs, WordBreakPolicy())
        assert WORD_BREAK_TOKEN_ID in with_wb
        without = sweep_mod.decode_cached(
            c.log_probs, WordBreakPolicy(logit_bias=-20.0)
        )
        assert WORD_BREAK_TOKEN_ID not in without

    def test_sweep_returns_one_point_per_grid_cell(self) -> None:
        points = sweep_mod.sweep([_cached([1, 0, 2])], [0.0, -1.0], [0.0, 0.5])
        assert len(points) == 4
        assert {(p["beta"], p["tau"]) for p in points} == {
            (0.0, 0.0), (0.0, 0.5), (-1.0, 0.0), (-1.0, 0.5)
        }

    def test_neighbors_are_within_grid(self) -> None:
        points = sweep_mod.sweep([_cached([1, 0, 2])], [0.0, -1.0], [0.0, 0.5])
        corner = next(p for p in points if p["beta"] == 0.0 and p["tau"] == 0.0)
        nb = sweep_mod.neighbors(points, corner)
        assert len(nb) == 2   # 角なので隣は 2 点


class TestDecodeCachedMatchesDecodeWave:
    """decode_cached (掃引の全結果が依存する前提) と本番 decode_wave の等価性を固定する.

    decode_wave は input_lengths を ctc_greedy_decode に渡して打ち切るのに対し、
    decode_cached は事前にスライス済みのキャッシュを前提にする。打ち切り方の
    経路が違うだけで結果は一致する、という前提をここで固定する。将来どちらかの
    打ち切り経路が編集されて結果がずれたら、このテストが検知する。
    """

    def test_matches_on_nonsilent_waveform(self) -> None:
        model, mel = _model_and_mel()
        device = torch.device("cpu")

        rng = np.random.default_rng(0)
        wave = torch.from_numpy(
            (rng.standard_normal(8000) * 0.1).astype(np.float32)
        )

        # 本番経路: src/eval/harness.decode_wave
        wave_ids = decode_wave(model, mel, wave, device)

        # 掃引キャッシュ経路: cache_keyed_samples と同じ手順で
        # log_softmax → compute_input_lengths → 有効長で切り詰め
        with torch.no_grad():
            t = wave.unsqueeze(0).to(device)
            log_probs = torch.log_softmax(model(mel(t)).float(), dim=-1)
            length = int(compute_input_lengths(
                torch.tensor([wave.numel()], device=device),
                mel.config.hop_length,
                log_probs.size(1),
            )[0])
            cached_log_probs = log_probs[0, :length].cpu()
        cached_ids = sweep_mod.decode_cached(cached_log_probs, WordBreakPolicy())

        # 空列同士の一致では等価性の検証にならないため、非空であることも確認する
        assert len(wave_ids) > 0, "デコード結果が空列 — 等価性の検証にならない"
        assert wave_ids == cached_ids


class TestBestPointTieBreak:
    """同点 TER のときの ``best_point`` のタイブレーク (|beta| → tau の順)."""

    def _point(self, beta: float, tau: float, ter: float) -> dict:
        return {"beta": beta, "tau": tau, "ter": ter}

    def test_prefers_smaller_abs_beta_on_tie(self) -> None:
        points = [
            self._point(beta=-2.0, tau=0.5, ter=0.1),
            self._point(beta=1.0, tau=0.9, ter=0.1),
            self._point(beta=-4.0, tau=0.1, ter=0.1),
        ]
        best = sweep_mod.best_point(points)
        assert best["beta"] == 1.0 and best["tau"] == 0.9

    def test_prefers_smaller_tau_when_abs_beta_also_ties(self) -> None:
        points = [
            self._point(beta=1.0, tau=0.9, ter=0.1),
            self._point(beta=-1.0, tau=0.3, ter=0.1),
            self._point(beta=1.0, tau=0.5, ter=0.1),
        ]
        best = sweep_mod.best_point(points)
        assert best["beta"] == -1.0 and best["tau"] == 0.3

    def test_lower_ter_wins_regardless_of_beta_tau(self) -> None:
        points = [
            self._point(beta=0.0, tau=0.0, ter=0.2),
            self._point(beta=-8.0, tau=0.9, ter=0.05),
        ]
        best = sweep_mod.best_point(points)
        assert best["ter"] == 0.05


class TestNeighborsInteriorPoint:
    """3x3 以上の格子の内側の点で、隣がちょうど 4 つ・β/τ 各方向 1 つずつ返ること."""

    def test_interior_point_has_four_axis_aligned_neighbors(self) -> None:
        points = sweep_mod.sweep(
            [_cached([1, 0, 2])], [0.0, -1.0, -2.0], [0.0, 0.5, 1.0]
        )
        center = next(p for p in points if p["beta"] == -1.0 and p["tau"] == 0.5)
        nb = sweep_mod.neighbors(points, center)

        assert len(nb) == 4
        # beta 方向の隣接 (tau は center と同じ)
        beta_neighbors = {p["beta"] for p in nb if p["tau"] == 0.5}
        assert beta_neighbors == {0.0, -2.0}
        # tau 方向の隣接 (beta は center と同じ)
        tau_neighbors = {p["tau"] for p in nb if p["beta"] == -1.0}
        assert tau_neighbors == {0.0, 1.0}


class TestSynthCheckBoundary:
    """``_synth_check`` の合否境界: 悪化がちょうど +1.00pt なら合格、わずかに超えたら不合格."""

    def test_passes_at_exactly_one_point_zero_pt(self) -> None:
        assert sweep_mod._passes_synth_check(1.0) is True

    def test_fails_just_above_one_point_zero_pt(self) -> None:
        assert sweep_mod._passes_synth_check(1.0000001) is False

    def test_synth_check_result_matches_boundary_at_exact_ter(self) -> None:
        """``_synth_check`` 自体でも境界を確認する (degradation_pt がちょうど 1.00pt).

        base_ter に対し point_ter = base_ter + 0.01 となるよう、全トークンが
        一致するキャッシュ (悪化 0) を土台に、baseline を意図的にずらして
        degradation_pt = 1.00pt を作る。
        """
        cached = [_cached([1, 0, 2])]
        base_ter = sweep_mod.evaluate_cached(
            cached, WordBreakPolicy()
        ).analysis.totals["ter"]
        result = sweep_mod._synth_check(
            cached, base_ter - 0.01, beta=0.0, tau=0.0, label="境界テスト"
        )
        assert result["degradation_pt"] == pytest.approx(1.0)
        assert result["passes"] is True

"""FT CLI の補助ロジックのテスト.

**採否の判定を手作業から外す**ためのもの。これまで「synth_val 悪化 3pt 超で不採用」は
FT のあとに別プロセスで eval_model.py を回して人が見比べていた。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.finetune import subsample_samples, synth_val_verdict


def _samples(n: int) -> list[str]:
    return [f"s{i:02d}" for i in range(n)]


class TestSubsampleSamples:
    """学習曲線 (件数 vs TER) を引くための間引き."""

    def test_件数を指定すると減る(self) -> None:
        got = subsample_samples(_samples(10), 4, seed=42)
        assert len(got) == 4

    def test_同じ_seed_なら同じ結果(self) -> None:
        a = subsample_samples(_samples(10), 4, seed=42)
        b = subsample_samples(_samples(10), 4, seed=42)
        assert a == b

    def test_違う_seed_なら別の組(self) -> None:
        a = subsample_samples(_samples(50), 5, seed=1)
        b = subsample_samples(_samples(50), 5, seed=2)
        assert a != b

    def test_元の順序を保つ(self) -> None:
        """順序が変わると DataLoader の並びまで変わり、seed 以外の差が入る."""
        got = subsample_samples(_samples(10), 5, seed=3)
        assert got == sorted(got)

    def test_None_なら全件(self) -> None:
        assert subsample_samples(_samples(7), None, seed=1) == _samples(7)

    def test_件数が母数以上なら全件(self) -> None:
        assert subsample_samples(_samples(3), 10, seed=1) == _samples(3)

    def test_0_以下はエラー(self) -> None:
        with pytest.raises(SystemExit):
            subsample_samples(_samples(3), 0, seed=1)


class TestSynthValVerdict:
    """「synth_val 悪化 3pt 超で不採用」を自動で言う."""

    def test_改善なら合格(self) -> None:
        v = synth_val_verdict(before=0.4516, after=0.4400, max_degrade_pt=3.0)
        assert v.passed and v.delta_pt == pytest.approx(-1.16, abs=1e-2)

    def test_許容内の悪化は合格(self) -> None:
        assert synth_val_verdict(before=0.4516, after=0.4700, max_degrade_pt=3.0).passed

    def test_許容を超える悪化は不合格(self) -> None:
        v = synth_val_verdict(before=0.4516, after=0.4900, max_degrade_pt=3.0)
        assert not v.passed and v.delta_pt == pytest.approx(3.84, abs=1e-2)

    def test_境界ちょうどは合格(self) -> None:
        assert synth_val_verdict(before=0.40, after=0.43, max_degrade_pt=3.0).passed

    def test_説明文に数値が入る(self) -> None:
        """ログを見た人が判断根拠を追えること."""
        text = synth_val_verdict(before=0.45, after=0.49, max_degrade_pt=3.0).message
        assert "45.00" in text and "49.00" in text and "3.0" in text


class TestBestSelectionUnderSaturation:
    """val が飽和したら best 選択を止め last を採る.

    **full_v5 では best.pt (step 98,000) が last.pt より held-out で 12pt 悪かった。**
    L4 val は 0.06% まで飽和し、評価ごとに ±4pt 揺れていたので、best は
    「たまたま低く出た評価」で固定された。飽和した val に判定能力は無い。
    """

    def test_飽和していなければ従来どおり(self) -> None:
        from scripts.finetune import should_update_best
        assert should_update_best(ter=0.10, best_ter=0.12, saturation=0.01) is True
        assert should_update_best(ter=0.13, best_ter=0.12, saturation=0.01) is False

    def test_飽和したら常に更新する(self) -> None:
        """飽和域では「最新 = best」。揺れで古い重みに固定されない."""
        from scripts.finetune import should_update_best
        assert should_update_best(ter=0.008, best_ter=0.005, saturation=0.01) is True

    def test_初回はNoneでも更新する(self) -> None:
        from scripts.finetune import should_update_best
        assert should_update_best(ter=0.5, best_ter=None, saturation=0.01) is True

    def test_飽和閾値ゼロなら従来と完全に同じ(self) -> None:
        from scripts.finetune import should_update_best
        assert should_update_best(ter=0.008, best_ter=0.005, saturation=0.0) is False

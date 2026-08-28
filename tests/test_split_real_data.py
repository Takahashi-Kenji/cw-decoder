"""本文単位の取り分けのテスト.

**件数で割ってはいけない。** L4 の 2 セッションは同じ 700 本文を 5W と 50W で
録ったもので、件数で割ると同じ本文が train と val の両方に入る。
"""

from __future__ import annotations

from pathlib import Path

from scripts.split_real_data import plan_split
from src.finetune.dataset import RealSignalSample


def make(name: str, text: str) -> RealSignalSample:
    return RealSignalSample(
        wav_path=Path(f"{name}.wav"), txt_path=Path(f"{name}.txt"),
        mode="european", text=text,
    )


class TestPlanSplit:
    def test_同じ本文は同じ側へ行く(self) -> None:
        """**ここが目的そのもの。** 5W と 50W の同一本文が分かれてはいけない."""
        samples = [make(f"a{i}", f"TEXT {i % 5}") for i in range(20)]
        train, val, _ = plan_split(samples, val_texts=2, seed=42)
        assert not ({s.text for s in train} & {s.text for s in val})
        assert len({s.text for s in val}) == 2
        assert len(train) + len(val) == 20

    def test_除外本文はどちらにも入らない(self) -> None:
        """held-out と本文が重なるものを学習に入れると物差しが甘くなる."""
        samples = [make(f"a{i}", f"TEXT {i}") for i in range(6)]
        train, val, dropped = plan_split(
            samples, val_texts=2, seed=42, excluded_texts=frozenset({"TEXT 0", "TEXT 1"}))
        assert len(dropped) == 2
        assert "TEXT 0" not in {s.text for s in train} | {s.text for s in val}
        assert "TEXT 1" not in {s.text for s in train} | {s.text for s in val}
        assert len(train) + len(val) == 4

    def test_同じseedなら同じ分割(self) -> None:
        samples = [make(f"a{i}", f"TEXT {i}") for i in range(30)]
        a = plan_split(samples, val_texts=5, seed=7)[1]
        b = plan_split(samples, val_texts=5, seed=7)[1]
        assert [s.wav_path for s in a] == [s.wav_path for s in b]

    def test_seedが違えば違う分割(self) -> None:
        samples = [make(f"a{i}", f"TEXT {i}") for i in range(30)]
        a = {s.text for s in plan_split(samples, val_texts=5, seed=7)[1]}
        b = {s.text for s in plan_split(samples, val_texts=5, seed=8)[1]}
        assert a != b

    def test_本文より多くvalを求めても壊れない(self) -> None:
        samples = [make(f"a{i}", "SAME") for i in range(4)]
        train, val, _ = plan_split(samples, val_texts=99, seed=1)
        assert len(train) == 0
        assert len(val) == 4

    def test_前後の空白は同じ本文とみなす(self) -> None:
        samples = [make("a", "TEXT"), make("b", " TEXT "), make("c", "OTHER")]
        train, val, _ = plan_split(samples, val_texts=1, seed=3)
        joined = {s.text.strip() for s in val}
        assert len(joined) == 1
        # "TEXT" が val なら 2 件とも val 側に居ること
        assert len(val) == (2 if "TEXT" in joined else 1)

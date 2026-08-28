"""助走 (ノイズ床) を足す処理のテスト.

**この処理の目的は「継ぎ目を手がかりにさせないこと」である。**
前回の貼り合わせは固定位置で 1 回だったので継ぎ目が学習可能な特徴になり、
置換が 46 → 98 に増えた (コミット ad2a391)。テストは
「符号を材料にしない」「段差を作らない」「レベルが揃う」の 3 点を押さえる。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.finetune.lead_in import (
    DEFAULT_SOURCE_S,
    noise_floor_source,
    prepend_lead_in,
    resolve_lead_in_range,
    zero_pad_lengths,
)

SR = 8000


def make_recording(
    *,
    pad_s: float = 0.4,
    head_noise_s: float = 0.06,
    code_s: float = 2.0,
    tail_noise_s: float = 1.0,
    noise_amp: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """8/24 の L4 と同じ形の録音を作る (ゼロ → 60ms ノイズ → 符号 → ノイズ → ゼロ)."""
    rng = np.random.default_rng(seed)

    def noise(sec: float) -> np.ndarray:
        return (rng.normal(0.0, noise_amp, int(sec * SR))).astype(np.float32)

    t = np.arange(int(code_s * SR)) / SR
    keyed = (0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32) + noise(code_s)
    return np.concatenate([
        np.zeros(int(pad_s * SR), dtype=np.float32),
        noise(head_noise_s),
        keyed,
        noise(tail_noise_s),
        np.zeros(int(pad_s * SR), dtype=np.float32),
    ])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


class TestZeroPad:
    def test_前後のゼロを数える(self) -> None:
        wave = make_recording()
        assert zero_pad_lengths(wave) == (int(0.4 * SR), int(0.4 * SR))

    def test_全部ゼロなら全部が先頭パッド(self) -> None:
        assert zero_pad_lengths(np.zeros(100, dtype=np.float32)) == (100, 0)


class TestNoiseFloorSource:
    def test_末尾から連続で取る(self) -> None:
        """**連続していることが効く。** 短い窓を繰り返すと周期構造が立ち、
        実測では素のままより悪くなった (TER 15.69% → 17.50%)。
        """
        wave = make_recording(noise_amp=0.01, tail_noise_s=1.0)
        src = noise_floor_source(wave, SR)
        assert len(src) == int(DEFAULT_SOURCE_S * SR)
        assert rms(src) == pytest.approx(0.01, rel=0.3)     # 符号 (0.3) が入らない
        assert src == pytest.approx(wave[-int(0.4 * SR) - len(src): -int(0.4 * SR)])

    def test_符号が入るなら短くして取り直す(self) -> None:
        """**ここを外すと助走の中に音のある文字を置くことになる。**

        末尾のノイズ床が 0.4 秒しか無い録音で 1.0 秒取ろうとすると符号が入る。
        `SOURCE_FALLBACK_S` を順に試して、汚れていない長さまで下げる。
        """
        wave = make_recording(tail_noise_s=0.4, noise_amp=0.01)
        src = noise_floor_source(wave, SR)
        assert len(src) <= int(0.4 * SR)
        assert rms(src) < 0.05

    def test_末尾が短ければ材料も短くなる(self) -> None:
        wave = make_recording(tail_noise_s=0.25, noise_amp=0.01)
        src = noise_floor_source(wave, SR)
        assert len(src) <= int(0.25 * SR)
        assert rms(src) < 0.05

    def test_末尾まで打鍵が続くならエラー(self) -> None:
        """**黙って助走なしに落ちない。** 材料の無い録音は呼び出し側が数える.

        末尾まで打鍵が続いている録音では、どの長さを取っても符号が入る。
        包絡線の広がり (ON/OFF がある) で気づき、**材料を作らずに知らせる。**
        """
        rng = np.random.default_rng(0)
        t = np.arange(int(0.08 * SR)) / SR
        seg = (0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
        gap = rng.normal(0.0, 0.01, int(0.08 * SR)).astype(np.float32)
        span = np.concatenate([x for _ in range(30) for x in (seg, gap)] + [seg])
        wave = np.concatenate([
            np.zeros(int(0.4 * SR), dtype=np.float32), span,
            np.zeros(int(0.4 * SR), dtype=np.float32)])
        with pytest.raises(ValueError, match="とみなせない"):
            noise_floor_source(wave, SR)

    def test_波形が短すぎればエラー(self) -> None:
        with pytest.raises(ValueError, match="長さがない"):
            noise_floor_source(np.zeros(10, dtype=np.float32) + 1e-3, SR)


class TestPrependLeadIn:
    def test_助走のぶんだけ長くなる(self) -> None:
        wave = make_recording()
        out = prepend_lead_in(wave, SR, 1.0)
        assert len(out) == len(wave) + int(1.0 * SR) - int(0.020 * SR)

    def test_ゼロなら何もしない(self) -> None:
        wave = make_recording()
        assert np.array_equal(prepend_lead_in(wave, SR, 0.0), wave)

    def test_先頭のゼロパッドは残る(self) -> None:
        """**ゼロ → ノイズ床の段差は推論経路にも在る。消してはいけない。**"""
        wave = make_recording()
        head, _ = zero_pad_lengths(prepend_lead_in(wave, SR, 1.0))
        assert head == int(0.4 * SR)

    def test_符号を壊さない(self) -> None:
        """クロスフェード区間より後ろは元の波形とビット一致すること."""
        wave = make_recording()
        pad, n_lead, n_fade = int(0.4 * SR), int(1.0 * SR), int(0.020 * SR)
        out = prepend_lead_in(wave, SR, 1.0)
        assert np.array_equal(out[pad + n_lead:], wave[pad + n_fade:])

    def test_継ぎ目に段差が出ない(self) -> None:
        """継ぎ目のサンプル間差分が、ノイズ床の中の差分と同程度であること.

        **ここが前回の失敗点。** 段差はクリックになり、モデルがキーダウンと読む。
        """
        wave = make_recording()
        out = prepend_lead_in(wave, SR, 1.0)
        pad, n_lead = int(0.4 * SR), int(1.0 * SR)
        junction = out[pad + n_lead - int(0.030 * SR): pad + n_lead + int(0.010 * SR)]
        plain = out[pad + int(0.4 * SR): pad + int(0.5 * SR)]     # 継ぎ目のない区間
        assert np.abs(np.diff(junction)).max() <= np.abs(np.diff(plain)).max() * 2.0

    def test_助走のレベルが元のノイズ床と揃う(self) -> None:
        wave = make_recording()
        out = prepend_lead_in(wave, SR, 1.0)
        pad = int(0.4 * SR)
        assert rms(out[pad: pad + int(0.9 * SR)]) == pytest.approx(
            rms(noise_floor_source(wave, SR)), rel=0.25)

    def test_鏡像で並べるので周期が立たない(self) -> None:
        """材料より長い助走を作るとき、材料長の周期で自己相関が跳ねないこと.

        **既定では材料 1.0 秒 > 助走の多くなので繰り返しは起きない。**
        末尾が短い録音でだけ効く道。
        """
        wave = make_recording()
        src = noise_floor_source(wave, SR)[: int(0.2 * SR)]
        out = prepend_lead_in(wave, SR, 1.0, source=src)
        pad = int(0.4 * SR)
        lead = out[pad: pad + int(0.9 * SR)]
        lead = lead - lead.mean()
        ac = np.correlate(lead, lead, mode="full")[len(lead) - 1:]
        assert ac[len(src)] / ac[0] < 0.5      # そのまま繰り返すと 1.0 近くになる

    def test_材料を渡せば波形から取らない(self) -> None:
        """Dataset は読み込み時に一度だけ材料を作って使い回す."""
        wave = make_recording()
        src = np.full(int(0.2 * SR), 0.5, dtype=np.float32)
        out = prepend_lead_in(wave, SR, 0.5, source=src, crossfade_s=0.0)
        pad = int(0.4 * SR)
        assert out[pad: pad + int(0.4 * SR)] == pytest.approx(0.5)

    def test_助走の長さが変われば継ぎ目の位置も変わる(self) -> None:
        """**これが目的そのもの。** 継ぎ目が固定なら手がかりになる."""
        wave = make_recording()
        a = prepend_lead_in(wave, SR, 0.5)
        b = prepend_lead_in(wave, SR, 1.5)
        assert len(b) - len(a) == int(1.0 * SR)


class TestResolveLeadInRange:
    def test_上限0で無効になる(self) -> None:
        assert resolve_lead_in_range(0.0, 0.0) is None

    def test_範囲を返す(self) -> None:
        assert resolve_lead_in_range(0.2, 2.0) == (0.2, 2.0)

    def test_逆順や負はエラー(self) -> None:
        with pytest.raises(ValueError):
            resolve_lead_in_range(2.0, 1.0)
        with pytest.raises(ValueError):
            resolve_lead_in_range(-1.0, 1.0)


def _write_pair(dirpath, stem: str, wave: np.ndarray, text: str = "CQ TEST"):
    import soundfile as sf
    sf.write(dirpath / f"{stem}.wav", wave, SR, subtype="PCM_16")
    (dirpath / f"{stem}.txt").write_text(
        f"mode: european\nsample_rate: {SR}\n---\n{text}\n", encoding="utf-8")
    return dirpath / f"{stem}.wav"


class TestRealSignalDatasetLeadIn:
    """学習データ側だけに効かせる配線.

    **評価経路には効かせない。** 既定を無効にしてあるのはそのためで、
    ここが既定で有効になると物差しが静かに動く。
    """

    def _dataset(self, tmp_path, **kwargs):
        from src.finetune.dataset import RealSignalDataset, discover_real_samples
        _write_pair(tmp_path, "a_european", make_recording())
        return RealSignalDataset(discover_real_samples(tmp_path), **kwargs)

    def test_既定では波形が変わらない(self, tmp_path) -> None:
        """**評価経路はここを通る。** 既定で助走が付いたら物差しが静かに動く."""
        import soundfile as sf
        wave = make_recording()
        _write_pair(tmp_path, "a_european", wave)
        from src.finetune.dataset import RealSignalDataset, discover_real_samples
        ds = RealSignalDataset(discover_real_samples(tmp_path))
        on_disk, _ = sf.read(tmp_path / "a_european.wav", dtype="float32")
        assert ds[0][0].numpy() == pytest.approx(on_disk)

    def test_助走ありなら取り出すたびに長さが変わる(self, tmp_path) -> None:
        ds = self._dataset(tmp_path, lead_in_range=(0.2, 2.0), seed=42)
        lengths = {int(ds[0][0].shape[0]) for _ in range(8)}
        assert len(lengths) > 1

    def test_ラベルは変わらない(self, tmp_path) -> None:
        """助走を足しても正解トークン列は素のときと同じであること."""
        from src.finetune.dataset import RealSignalDataset, discover_real_samples
        _write_pair(tmp_path, "a_european", make_recording())
        samples = discover_real_samples(tmp_path)
        plain = RealSignalDataset(samples)
        aug = RealSignalDataset(samples, lead_in_range=(0.2, 2.0), seed=42)
        assert torch_equal(aug[0][1], plain[0][1])
        assert int(aug[0][0].shape[0]) > int(plain[0][0].shape[0])

    def test_材料が取れなければ素のまま返し理由を残す(self, tmp_path) -> None:
        """**黙って助走なしに落ちない。** 何件が素のままかを呼び出し側が数える."""
        from src.finetune.dataset import RealSignalDataset, discover_real_samples
        _write_pair(tmp_path, "silent_european", np.zeros(int(1.0 * SR), dtype=np.float32))
        samples = discover_real_samples(tmp_path)
        plain = RealSignalDataset(samples)
        ds = RealSignalDataset(samples, lead_in_range=(0.2, 2.0), seed=42)
        assert len(ds.no_lead_in_reasons) == 1
        assert torch_equal(ds[0][0], plain[0][0])      # 素のまま返る

    def test_範囲が不正ならエラー(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="lead_in_range"):
            self._dataset(tmp_path, lead_in_range=(2.0, 1.0))


def torch_equal(a, b) -> bool:
    import torch
    return bool(torch.equal(a, b))


class TestFilterByDuration:
    """長すぎるサンプルを外す (GPU メモリのピークを決めるのは最長サンプル).

    **バッチはその中の最長に合わせて詰められる。** 実測では 45 秒の 1 件で
    バッチ 16 件が 47 秒分に膨らみ、2.8 GB がシステムメモリへ退避して
    スループットが 2.21 → 0.87 sps に落ちた。
    """

    def _dir(self, tmp_path):
        _write_pair(tmp_path, "short_european", make_recording(code_s=1.0))
        _write_pair(tmp_path, "long_european", make_recording(code_s=8.0))
        from src.finetune.dataset import discover_real_samples
        return discover_real_samples(tmp_path)

    def test_上限を超えるものを外す(self, tmp_path) -> None:
        from src.finetune.dataset import filter_by_duration
        keep, drop = filter_by_duration(self._dir(tmp_path), 5.0)
        assert [s.wav_path.stem for s in keep] == ["short_european"]
        assert [s.wav_path.stem for s in drop] == ["long_european"]

    def test_ゼロなら何も外さない(self, tmp_path) -> None:
        """**既定は無効。** 黙ってデータが減るのが一番まずい."""
        from src.finetune.dataset import filter_by_duration
        keep, drop = filter_by_duration(self._dir(tmp_path), 0.0)
        assert len(keep) == 2 and drop == []

    def test_順序を保つ(self, tmp_path) -> None:
        from src.finetune.dataset import filter_by_duration
        samples = self._dir(tmp_path)
        keep, _ = filter_by_duration(samples, 100.0)
        assert [s.wav_path for s in keep] == [s.wav_path for s in samples]


class TestPadBucket:
    """詰め長の量子化 (GPU アロケータの断片化対策).

    **バッチごとに詰め長が変わると形状の種類が増え、キャッシュが積み上がる。**
    実測で 17 MB のモデルが 15 GB を抱え、システムメモリへ退避して 2.5 倍遅くなった。
    """

    def _batch(self, lengths):
        import torch
        return [(torch.zeros(n), torch.tensor([1, 2], dtype=torch.long)) for n in lengths]

    def test_詰め長が倍数に切り上がる(self) -> None:
        from src.train.collate import cw_collate
        w, _, lens, _ = cw_collate(self._batch([100, 20000]), pad_bucket=16000)
        assert w.shape[1] == 32000
        assert lens.tolist() == [100, 20000]      # 実長は変えない

    def test_ちょうど倍数なら増やさない(self) -> None:
        from src.train.collate import cw_collate
        w, _, _, _ = cw_collate(self._batch([16000]), pad_bucket=16000)
        assert w.shape[1] == 16000

    def test_既定はオフ(self) -> None:
        """**評価経路・ゴールデンの挙動を変えない。** 学習側から明示的に渡す."""
        from src.train.collate import cw_collate
        w, _, _, _ = cw_collate(self._batch([100, 12345]))
        assert w.shape[1] == 12345

    def test_波形の中身は変わらない(self) -> None:
        import torch
        from src.train.collate import cw_collate
        x = torch.arange(500, dtype=torch.float32)
        w, _, _, _ = cw_collate([(x, torch.tensor([1], dtype=torch.long))], pad_bucket=16000)
        assert torch.equal(w[0, :500], x)
        assert torch.count_nonzero(w[0, 500:]) == 0


class TestLengthCappedDataset:
    """長すぎるサンプルを学習中に捨てる.

    **必要メモリはバッチの詰め長に比例する** (実測 batch16: 27 秒で 2.29 GB)。
    合成は中央 7.2 秒だが最長 54 秒あり、バッチ 16 件の最長は中央 31 秒になる。
    **実データだけに上限を掛けても学習の 80% は合成なので効かない** (一度踏んだ)。
    """

    class _Base:
        def __init__(self, lengths):
            self.lengths = lengths

        def __iter__(self):
            import torch
            for n in self.lengths:
                yield torch.zeros(n), torch.tensor([1], dtype=torch.long)

    def test_上限を超えるものを捨てる(self) -> None:
        from src.finetune.pipeline import LengthCappedDataset
        ds = LengthCappedDataset(self._Base([100, 5000, 200]), max_samples_len=1000)
        got = [w.numel() for w, _ in ds]
        assert got == [100, 200]
        assert ds.n_dropped == 1

    def test_ゼロなら何も捨てない(self) -> None:
        from src.finetune.pipeline import LengthCappedDataset
        ds = LengthCappedDataset(self._Base([100, 5000]), max_samples_len=0)
        assert [w.numel() for w, _ in ds] == [100, 5000]
        assert ds.n_dropped == 0

    def test_ラベルはそのまま通す(self) -> None:
        import torch
        from src.finetune.pipeline import LengthCappedDataset
        ds = LengthCappedDataset(self._Base([100]), max_samples_len=1000)
        for _, t in ds:
            assert torch.equal(t, torch.tensor([1], dtype=torch.long))

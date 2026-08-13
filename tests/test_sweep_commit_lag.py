"""commit_lag 掃引スクリプトの単体テスト (モデル不要)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.sweep_commit_lag import (
    SweepRow,
    _simulate_both,
    reference_decode,
    simulate_commit,
    sweep,
    token_cer,
)
from src.infer.engine import FrameToken


class FakeEngine:
    """一定間隔でトークンを返すダミーエンジン.

    投入された波形長に比例した数のトークンを、80 サンプル/フレーム換算で
    0.5 秒おきに配置して返す.
    """

    frame_hop_samples = 80

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]:
        n_frames = waveform.size // 80
        out: list[FrameToken] = []
        # 0.5 秒 = 50 フレームおきに 1 トークン (id は位置から決まるので決定的)
        for start in range(0, n_frames - 5, 50):
            out.append(FrameToken(
                token_id=1 + (start // 50) % 10,
                confidence=0.9,
                frame_start=start,
                frame_end=start + 4,
            ))
        return out


def test_token_cer_identical_is_zero() -> None:
    assert token_cer([1, 2, 3], [1, 2, 3]) == 0.0


def test_token_cer_counts_edits() -> None:
    # 1 置換 / 参照長 3
    assert token_cer([1, 2, 3], [1, 9, 3]) == 1 / 3
    # 1 削除 / 参照長 3
    assert token_cer([1, 2, 3], [1, 3]) == 1 / 3


def test_token_cer_empty_reference() -> None:
    assert token_cer([], []) == 0.0
    assert token_cer([], [1]) == 1.0


def test_simulate_commit_is_deterministic() -> None:
    """同じ入力・同じ設定なら結果が一致すること."""
    engine = FakeEngine()
    wave = np.zeros(8000 * 10, dtype=np.float32)
    a = simulate_commit(engine, wave, commit_lag_s=2.5)
    b = simulate_commit(engine, wave, commit_lag_s=2.5)
    assert a == b
    assert len(a) > 0


def test_smaller_lag_commits_at_least_as_many_raw_tokens() -> None:
    """トリム無しの生の simulate_commit は、commit_lag を小さくすると確定数が
    減らない (末尾がより多く確定する)。

    **この性質は「良いこと」ではない。** 音声ファイルには終わりがあるため、
    lag が大きいほど「ファイル末尾 lag 秒分は音声が尽きて確定できない」区間が
    広がるという、まさに末尾切り捨ての非対称性そのものを表している
    (`docs/commit_lag_sweep_result.md` §1〜2 で経緯を記録した実測バグの原因)。
    実運用 (音声が延々と続くライブ受信) にはこの非対称性は存在しない。

    だからこそ掃引の主指標 (`sweep()`) では、参照・仮説の両方から末尾
    `max(lags)` 秒を除外してから比較し、この副作用を測定から除去している。
    ここでは「生のデコーダの性質としてこの非対称性が確かに存在する」ことを
    記録・固定する目的で残す。トリム済みの主指標がこの副作用を含まないことは
    `test_sweep_trims_reference_by_max_lag` で別途検証している。
    """
    engine = FakeEngine()
    wave = np.zeros(8000 * 10, dtype=np.float32)
    short = simulate_commit(engine, wave, commit_lag_s=0.5)
    long = simulate_commit(engine, wave, commit_lag_s=2.5)
    assert len(short) >= len(long)


def test_sweep_returns_one_row_per_lag() -> None:
    engine = FakeEngine()
    waves = [np.zeros(8000 * 8, dtype=np.float32)]
    rows = sweep(engine, waves, lags=[0.5, 1.0, 2.5])
    assert len(rows) == 3
    assert all(isinstance(r, SweepRow) for r in rows)
    assert [r.commit_lag_s for r in rows] == [0.5, 1.0, 2.5]
    assert all(0.0 <= r.match_rate <= 1.0 for r in rows)
    assert all(0.0 <= r.match_rate_with_finalize <= 1.0 for r in rows)


def test_reference_decode_matches_single_shot_decode() -> None:
    """参照は音声全体を一度に decode_chunk した結果と一致するべき (右文脈最大の定義).

    過去に参照計算が simulate_commit(commit_lag_s=0.0, finalize=True) を使っており、
    ループ中の commit_lag_s=0.0 による早期確定に汚染されていたバグの回帰テスト。
    """
    engine = FakeEngine()
    wave = np.zeros(8000 * 10, dtype=np.float32)  # 10s < 既定 window_s=30s
    ref = reference_decode(engine, wave)
    direct = [t.token_id for t in engine.decode_chunk(wave)]
    assert ref == direct


def test_reference_decode_raises_when_wave_exceeds_window() -> None:
    """window_s を超える音声は、リングバッファの切り詰めで先頭が黙って失われるため
    参照計算を拒否しなければならない (誤った参照を黙って返すのを防ぐ)."""
    engine = FakeEngine()
    wave = np.zeros(8000 * 35, dtype=np.float32)  # 35s > 既定 window_s=30s
    with pytest.raises(ValueError):
        reference_decode(engine, wave, window_s=30.0)


def test_trim_tail_s_zero_keeps_all_tokens() -> None:
    engine = FakeEngine()
    wave = np.zeros(8000 * 10, dtype=np.float32)
    full = simulate_commit(engine, wave, commit_lag_s=0.5)
    trimmed = simulate_commit(engine, wave, commit_lag_s=0.5, trim_tail_s=0.0)
    assert trimmed == full


def test_trim_tail_s_can_exclude_everything() -> None:
    engine = FakeEngine()
    wave = np.zeros(8000 * 10, dtype=np.float32)
    trimmed = simulate_commit(engine, wave, commit_lag_s=0.5, trim_tail_s=20.0)
    assert trimmed == []


def test_sweep_trims_reference_by_max_lag() -> None:
    """主指標は全 lag が同じ区間で比較されるよう、参照からも末尾 max(lags) 秒を
    除外する (レビュー指摘: 末尾非対称性の除去)."""
    engine = FakeEngine()
    wave = np.zeros(8000 * 10, dtype=np.float32)
    rows = sweep(engine, [wave], lags=[0.5, 2.5])
    full_ref = reference_decode(engine, wave)
    trimmed_ref = reference_decode(engine, wave, trim_tail_s=2.5)
    assert len(trimmed_ref) < len(full_ref)
    assert rows[0].n_reference_tokens == len(trimmed_ref)
    assert rows[0].n_reference_tokens_full == len(full_ref)


def test_simulate_both_matches_independent_calls() -> None:
    """sweep() の性能最適化 _simulate_both (redecode ループを 1 回だけ実行して
    主指標・副指標の両方を得る) が、simulate_commit を 2 回独立に呼んだ場合と
    同じ結果になることを保証する (90ファイル実測でも一致を確認済みだが、
    将来の変更に対する自動的な担保としてここでも検証する)."""
    engine = FakeEngine()
    wave = np.zeros(8000 * 10, dtype=np.float32)
    got, got_fin = _simulate_both(
        engine, wave, commit_lag_s=1.5, hop_s=1.0, window_s=30.0,
        head_guard_s=1.0, decode_left_context_s=5.0, sample_rate=8000, trim_tail_s=2.5,
    )
    expected = simulate_commit(engine, wave, commit_lag_s=1.5, trim_tail_s=2.5)
    expected_fin = simulate_commit(engine, wave, commit_lag_s=1.5, finalize=True)
    assert got == expected
    assert got_fin == expected_fin

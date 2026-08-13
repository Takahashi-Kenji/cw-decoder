"""音声波形のエンベロープから語間 (長無音) を検出.

CTC モデルのフレーム位置だけでは語間と文字間を確実に分離できないため、
原音声の振幅エンベロープから「長い無音」を直接検出して語間スペースを
判定する.

戦略:

1. 短窓の peak エンベロープを計算 (1 dot 〜 0.5 dot 程度の窓)
2. 動的閾値で silence_mask を作る
3. 各 token 直前 (silence 中) の連続無音長を計測
4. 「dot 長」を推定 (token 間最小ギャップ等から)
5. 無音長 > 5 dot 相当 を語間とみなす
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.infer.engine import FrameToken


def compute_envelope(
    waveform: np.ndarray,
    sample_rate: int = 8000,
    window_ms: float = 8.0,
) -> np.ndarray:
    """波形の平滑エンベロープ (|x| を矩形窓で移動平均).

    CW 信号は搬送波 (例 600Hz) なので |sin| は 0〜peak を高速振動.
    ~8ms 窓で平均すると ON 区間は 2/π × peak ≒ 0.64 × peak、
    OFF 区間はノイズ振幅レベルに落ちる滑らかなエンベロープが得られる.
    """
    win = max(1, int(sample_rate * window_ms / 1000.0))
    abs_w = np.abs(waveform.astype(np.float32))
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(abs_w, kernel, mode="same")


def detect_silence_mask(
    waveform: np.ndarray,
    sample_rate: int = 8000,
    envelope_window_ms: float = 8.0,
    rel_threshold: float = 0.15,
    closing_samples: int = 80,
) -> np.ndarray:
    """サンプル単位の bool マスク. ``True`` = 無音.

    閾値はピークエンベロープの ``rel_threshold`` 倍.
    ノイズが強いと閾値も高くなり、誤検出を抑える.

    ``closing_samples`` 短い非無音 (ノイズスパイク等) で無音区間が断裂するのを防ぐ
    モルフォロジカル closing. デフォルト 80 サンプル ≒ 10 ms.
    """
    env = compute_envelope(waveform, sample_rate, envelope_window_ms)
    if env.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(env.max())
    if peak < 1e-6:
        return np.ones_like(env, dtype=bool)
    mask = env < peak * rel_threshold
    # Morphological closing: dilate then erode
    if closing_samples > 1:
        kernel = np.ones(closing_samples, dtype=np.uint8)
        # Dilation: convolve and threshold
        dilated = np.convolve(mask.astype(np.uint8), kernel, mode="same") > 0
        # Erosion: convolve and require all 1s within kernel
        eroded = np.convolve(dilated.astype(np.uint8), kernel, mode="same") >= closing_samples
        mask = eroded
    return mask


def estimate_dot_samples(
    silence_mask: np.ndarray,
    min_dot_samples: int = 160,        # ≒ WPM 50 (24ms)
    max_dot_samples: int = 1500,       # ≒ WPM 8 (150ms)
) -> int:
    """音声の OFF ランレングスから 1 dot のサンプル数を推定.

    CW の OFF 区間は 1 dot (要素間), 3 dot (文字間), 7 dot (語間) のみ.
    OFF ラン長のヒストグラムで **下側のモード** ≒ 1 dot に相当する.
    ノイズによる細かい breaks を除外するため ``min_dot_samples`` でクリップ.
    """
    extended_off = np.concatenate(([False], silence_mask, [False]))
    diff = np.diff(extended_off.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size < 2:
        return 640                     # WPM 15 相当のフォールバック
    run_lengths = ends - starts
    # min_dot_samples 以上のランのみを採用 (ノイズ除外)
    filtered = run_lengths[run_lengths >= min_dot_samples]
    if filtered.size == 0:
        return min_dot_samples
    # OFF ラン長の中央値は 1 dot に近い (要素間ギャップが最も多い).
    # 中央値そのものを 1 dot と推定する.
    median_run = float(np.median(filtered))
    return int(max(min_dot_samples, min(max_dot_samples, median_run)))


def detect_word_breaks_from_audio(
    waveform: np.ndarray,
    frame_tokens: Sequence[FrameToken],
    sample_rate: int = 8000,
    hop_samples: int = 80,
    word_gap_dots: float = 6.0,
    silence_envelope_window_ms: float = 10.0,
    silence_rel_threshold: float = 0.12,
    dot_samples_min: int = 300,
    dot_samples_max: int = 1200,
) -> list[bool]:
    """各 ``frame_token`` の直前に語間相当の無音があるか判定.

    **既知の制限**: CTC のフレーム位置と実際の音声タイミングは必ずしも一致せず、
    また SNR が低いと無音検出が断裂的になる. 本関数は **保守的な閾値** で
    動作し、確信できる語間のみを ``True`` とする (偽陰性が多めだが偽陽性は
    最小化する設計). より正確な語間検出には、語間専用トークンを語彙に
    追加して再学習することを推奨.

    Args:
        waveform: 元波形 (8 kHz, float32).
        frame_tokens: ``FrameToken`` リスト.
        sample_rate: サンプリングレート.
        hop_samples: メルスペクトログラム hop_length (CTC frame 単位の正体).
        word_gap_dots: 「無音が dot の何倍以上なら語間か」の閾値.
            CW 規約は語間 7 dot、文字間 3 dot. 6 dot で保守的に分離.
        silence_envelope_window_ms: envelope 計算窓.
        silence_rel_threshold: 無音判定の相対閾値 (ピークに対する比).
        dot_samples_min: dot 長推定の下限.
        dot_samples_max: dot 長推定の上限.

    Returns:
        ``frame_tokens`` と同長の bool 列. ``True`` = この token の前に語間.
        先頭要素は常に ``False``.
    """
    n = len(frame_tokens)
    if n == 0:
        return []
    flags: list[bool] = [False] * n

    silence_mask = detect_silence_mask(
        waveform, sample_rate, silence_envelope_window_ms, silence_rel_threshold
    )

    # 音声から直接 dot 長を推定 (token frame gap は CTC で歪むので使わない)
    dot_samples = estimate_dot_samples(
        silence_mask,
        min_dot_samples=dot_samples_min,
        max_dot_samples=dot_samples_max,
    )
    silence_threshold_samples = int(word_gap_dots * dot_samples)

    # トークン間ウィンドウの「無音合計時間」で判定:
    # CW 規約上、語間は OFF が連続 7 dot、文字間は 3 dot. 厳密に連続でなくとも
    # ウィンドウ内 OFF サンプル合計が 7 dot 以上あれば語間と判断できる.
    for i in range(1, n):
        prev = frame_tokens[i - 1]
        cur = frame_tokens[i]
        prev_sample = prev.frame_end * hop_samples
        cur_sample = cur.frame_start * hop_samples
        if cur_sample <= prev_sample:
            continue
        # token 自身の ON 期間 (frame 周辺 ~1 dot) を除外
        margin = max(1, dot_samples)
        start = min(len(silence_mask), prev_sample + margin)
        end = min(len(silence_mask), max(start, cur_sample - margin))
        seg = silence_mask[start:end]
        if seg.size == 0:
            continue
        total_silence = int(seg.sum())
        if total_silence >= silence_threshold_samples:
            flags[i] = True
    return flags


def _max_true_run(mask: np.ndarray) -> int:
    """1D bool 配列内で True が連続する最大長."""
    if mask.size == 0:
        return 0
    # 差分でブロック境界検出
    extended = np.concatenate(([False], mask, [False]))
    diff = np.diff(extended.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return 0
    return int((ends - starts).max())


__all__ = [
    "compute_envelope",
    "detect_silence_mask",
    "detect_word_breaks_from_audio",
    "estimate_dot_samples",
]

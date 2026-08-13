"""CW キーイング波形生成 (numpy ベクトル化).

要件 §3.2.2 のキーイングパラメータを乱数で変動させながら、符号列を
8 kHz のサンプリングレートで PCM 波形に変換する.

タイミング規約:

- dot 長 :math:`= 1.2 / \\mathrm{WPM}` 秒
- dash 長 = dot × ``dash_dot_ratio``
- 要素間スペース = 1 dot
- 文字間スペース = 3 dot
- 語間スペース = 7 dot (``word_break_after`` で指定された位置のみ)

立上り/立下りは raised-cosine 包絡線でスムージング (キークリック抑制).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from src.tokens.morse_tokens import DASH, DOT, WORD_BREAK_CODE


class ElementKind(IntEnum):
    """要素の種別.

    ジッタの σ を種別ごとに変えるために使う。実測では短点 (σ 0.06〜0.11 dot) と
    長音 (σ 0.68〜1.20 dot) でばらつきが一桁違うため、共通の σ では手打ちを
    再現できない (設計書 §2.3(b))。
    """

    DOT = 0          # 短点 (ON)
    DASH = 1         # 長音 (ON)
    INTRA_GAP = 2    # 同一符号内の要素間 (OFF)
    CHAR_GAP = 3     # 文字間 (OFF)
    WORD_GAP = 4     # 語間 (OFF)


@dataclass
class KeyingParams:
    """キーイングパラメータ.

    - ``wpm``: 速度 (Words Per Minute). 10〜40 を推奨.
    - ``dash_dot_ratio``: dash 対 dot の長さ比. 2.5〜4.0.
    - ``element_jitter_sigma_ratio``: 各要素長に加えるガウシアンジッタの σ
      (dot 長に対する比率). 0.05〜0.20.
    - ``tone_freq_hz``: 搬送波周波数. 400〜900 Hz.
    - ``tone_drift_hz_per_sec``: サンプル内ドリフト量. 0 でドリフト無し.
    - ``rise_fall_ms``: raised-cosine ランプの長さ. 3〜10 ms.
    - ``intra_sample_wpm_drift``: サンプル全体での速度線形変化量.
      0 でドリフト無し.
    """

    wpm: float = 20.0
    dash_dot_ratio: float = 3.0
    element_jitter_sigma_ratio: float = 0.0
    # 同一符号内の要素間スペース (dot 単位)。教科書は 1.0。
    # 実測では平均 1.18〜1.25、最小 0.13 まで詰まる (設計書 §2.2)。
    intra_element_space_units: float = 1.0
    inter_char_space_units: float = 3.0
    inter_word_space_units: float = 7.0
    tone_freq_hz: float = 600.0
    tone_drift_hz_per_sec: float = 0.0
    rise_fall_ms: float = 5.0
    intra_sample_wpm_drift: float = 0.0
    pre_silence_sec: float = 0.0
    post_silence_sec: float = 0.0

    # --- 要素種別ごとのジッタ σ (dot 長に対する比率) ---
    # None ならこの種別は element_jitter_sigma_ratio にフォールバックする。
    # 全部 None なら従来と完全に同一の挙動になる (後方互換)。
    #
    # 実測 (設計書 §2.2): 短点 σ 0.064〜0.107 dot に対し長音 σ 0.681〜1.195 dot と
    # 一桁違う。共通の σ ではこの非対称を作れないため種別ごとに分けた。
    dot_jitter_sigma_ratio: float | None = None
    dash_jitter_sigma_ratio: float | None = None
    intra_gap_jitter_sigma_ratio: float | None = None
    char_gap_jitter_sigma_ratio: float | None = None
    word_gap_jitter_sigma_ratio: float | None = None

    def jitter_sigma_by_kind(self) -> np.ndarray:
        """``ElementKind`` の並び順で σ を返す (長さ 5, float64).

        ``None`` の種別は ``element_jitter_sigma_ratio`` にフォールバックする。
        戻り値は ``kinds`` 配列で fancy index して使う。
        """
        base = self.element_jitter_sigma_ratio
        return np.array(
            [
                base if self.dot_jitter_sigma_ratio is None else self.dot_jitter_sigma_ratio,
                base if self.dash_jitter_sigma_ratio is None else self.dash_jitter_sigma_ratio,
                base if self.intra_gap_jitter_sigma_ratio is None else self.intra_gap_jitter_sigma_ratio,
                base if self.char_gap_jitter_sigma_ratio is None else self.char_gap_jitter_sigma_ratio,
                base if self.word_gap_jitter_sigma_ratio is None else self.word_gap_jitter_sigma_ratio,
            ],
            dtype=np.float64,
        )


@dataclass
class WaveformResult:
    """合成波形と関連メタ情報."""

    samples: np.ndarray         # float32, [-1, 1] 正規化済み
    sample_rate: int
    code_start_samples: np.ndarray  # int64, 各 code の開始サンプル位置
    effective_wpm: float        # ジッタ適用後の実効 WPM


def _apply_raised_cosine_ramp(envelope: np.ndarray, ramp_samples: int) -> np.ndarray:
    """エンベロープの 0↔1 遷移に raised-cosine ランプを適用.

    Args:
        envelope: 0/1 のステップエンベロープ (float32).
        ramp_samples: ランプ長 (サンプル).
    """
    if ramp_samples <= 0:
        return envelope
    n = len(envelope)
    out = envelope.copy()
    diff = np.diff(envelope, prepend=0.0)
    rises = np.where(diff > 0.5)[0]
    falls = np.where(diff < -0.5)[0]
    t = np.arange(ramp_samples, dtype=np.float32)
    rc = (0.5 * (1.0 - np.cos(np.pi * t / ramp_samples))).astype(np.float32)
    rc_down = rc[::-1]
    for r in rises:
        end = min(r + ramp_samples, n)
        out[r:end] = rc[: end - r]
    for f in falls:
        start = max(f - ramp_samples, 0)
        seg = f - start
        out[start:f] = rc_down[:seg]
    return out


def build_element_sequence(
    codes: Sequence[str],
    word_break_after: Sequence[int],
    dot_sec: float,
    dash_dot_ratio: float,
    inter_char_space_units: float,
    inter_word_space_units: float,
    intra_element_space_units: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """符号列を要素 (duration, key_on, kind) のシーケンスに展開.

    **合成器と送信の両方から使う。** 合成器は波形を作るため、送信は電鍵線を
    上下させるために、同じ「いつキーを ON/OFF するか」を必要とする。
    片方だけに実装を持つと必ず食い違うので、ここが唯一の真正ソースである
    (アーキテクチャ原則 2)。

    **送信ではジッタをかけないこと。** 合成器は学習データに人間らしい揺れを
    混ぜるが、送信は機械的に正確であるべきである
    (``KeyingParams`` の各 σ を 0 にする)。

    Returns:
        durations: 各要素の長さ (秒).
        is_on: 各要素のキー状態 (True = ON).
        code_start_element_indices: 各 code が ``durations`` の何番目から始まるか.
        kinds: 各要素の ``ElementKind`` (int64).
    """
    word_break_set = set(word_break_after)
    durations: list[float] = []
    is_on: list[bool] = []
    kinds: list[int] = []
    code_starts: list[int] = []
    # 次の inter-code space を語間 (7 dot) にする必要があるか
    pending_word_break = False

    for i, code in enumerate(codes):
        # WORD_BREAK_CODE は実際の符号を持たず、次の inter-code 区間を語間に拡張
        if code == WORD_BREAK_CODE:
            code_starts.append(len(durations))
            pending_word_break = True
            continue
        if not code:
            code_starts.append(len(durations))
            continue
        if durations:  # 前の code との間に空白を入れる
            is_word_break = pending_word_break or ((i - 1) in word_break_set)
            space_units = (
                inter_word_space_units if is_word_break else inter_char_space_units
            )
            durations.append(space_units * dot_sec)
            is_on.append(False)
            kinds.append(
                ElementKind.WORD_GAP if is_word_break else ElementKind.CHAR_GAP
            )
        pending_word_break = False
        code_starts.append(len(durations))
        for j, elem in enumerate(code):
            if j > 0:
                durations.append(intra_element_space_units * dot_sec)
                is_on.append(False)
                kinds.append(ElementKind.INTRA_GAP)
            if elem == DOT:
                durations.append(dot_sec)
                kinds.append(ElementKind.DOT)
            elif elem == DASH:
                durations.append(dot_sec * dash_dot_ratio)
                kinds.append(ElementKind.DASH)
            else:
                raise ValueError(f"Unknown element {elem!r} in code {code!r}")
            is_on.append(True)

    return (
        np.asarray(durations, dtype=np.float64),
        np.asarray(is_on, dtype=bool),
        np.asarray(code_starts, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64),
    )


def codes_to_waveform(
    codes: Sequence[str],
    params: KeyingParams,
    rng: np.random.Generator,
    sample_rate: int = 8000,
    word_break_after: Sequence[int] = (),
) -> WaveformResult:
    """符号列 → 波形.

    Args:
        codes: ``・`` と ``-`` からなる符号文字列のシーケンス.
            空文字列はスキップ.
        params: キーイングパラメータ.
        rng: 乱数生成器.
        sample_rate: サンプリングレート (Hz).
        word_break_after: 該当インデックスの code 後に語間スペースを挿入.

    Returns:
        合成波形と関連メタ情報.
    """
    if params.wpm <= 0:
        raise ValueError(f"wpm must be > 0, got {params.wpm}")
    if params.dash_dot_ratio <= 0:
        raise ValueError(f"dash_dot_ratio must be > 0")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0")

    if not codes or all(not c for c in codes):
        n_silence = int(
            round((params.pre_silence_sec + params.post_silence_sec) * sample_rate)
        )
        return WaveformResult(
            samples=np.zeros(n_silence, dtype=np.float32),
            sample_rate=sample_rate,
            code_start_samples=np.zeros(0, dtype=np.int64),
            effective_wpm=params.wpm,
        )

    dot_sec = 1.2 / params.wpm

    durations, is_on, code_start_elem_idx, kinds = build_element_sequence(
        codes,
        word_break_after,
        dot_sec,
        params.dash_dot_ratio,
        params.inter_char_space_units,
        params.inter_word_space_units,
        params.intra_element_space_units,
    )

    # 要素種別ごとの σ を引き当ててジッタを乗せる。
    # (旧コメントは「ON 要素のみ」と書いていたが、実際には OFF (スペース) にも
    #  かかっていた。実測でもスペースはばらつくのでこの挙動は正しい。)
    #
    # numpy の normal は scale がスカラでも同値の配列でも同じ乱数列を返すため、
    # 種別ごとの σ が全部同じ値なら従来とビット単位で同一の波形になる。
    sigma = params.jitter_sigma_by_kind()[kinds] * dot_sec
    if np.any(sigma > 0.0):
        # 最小長を dot 長の 10% にクリップ
        durations = np.maximum(durations + rng.normal(0.0, sigma), dot_sec * 0.1)

    sample_counts = np.maximum(
        np.round(durations * sample_rate).astype(np.int64), 1
    )

    # 各 duration の開始サンプル位置 (累積)
    duration_start_samples = np.concatenate(
        ([np.int64(0)], np.cumsum(sample_counts))
    )
    body_samples = int(duration_start_samples[-1])

    # エンベロープ (ステップ関数)
    envelope_body = np.repeat(is_on.astype(np.float32), sample_counts)

    # 前後の無音
    pre_samples = int(round(params.pre_silence_sec * sample_rate))
    post_samples = int(round(params.post_silence_sec * sample_rate))
    if pre_samples > 0 or post_samples > 0:
        envelope = np.concatenate(
            [
                np.zeros(pre_samples, dtype=np.float32),
                envelope_body,
                np.zeros(post_samples, dtype=np.float32),
            ]
        )
    else:
        envelope = envelope_body

    # raised-cosine スムージング (本体のみ)
    ramp_samples = max(0, int(round(params.rise_fall_ms * sample_rate / 1000.0)))
    envelope = _apply_raised_cosine_ramp(envelope, ramp_samples)

    # 搬送波 (位相連続)
    total = len(envelope)
    t_axis = np.arange(total, dtype=np.float64) / sample_rate
    # ドリフトする瞬時周波数
    freq = (
        params.tone_freq_hz
        + params.tone_drift_hz_per_sec * t_axis
    )
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    carrier = np.sin(phase).astype(np.float32)

    samples = (envelope * carrier).astype(np.float32)

    # コード開始位置 (前無音を含めた絶対位置)
    code_start_samples = duration_start_samples[code_start_elem_idx] + pre_samples

    # 実効 WPM: ON 要素の総時間から逆算
    on_total_sec = float(durations[is_on].sum())
    n_dot_units = sum(
        c.count(DOT) + c.count(DASH) * params.dash_dot_ratio
        for c in codes if c and c != WORD_BREAK_CODE
    )
    effective_dot_sec = on_total_sec / n_dot_units if n_dot_units > 0 else dot_sec
    effective_wpm = 1.2 / effective_dot_sec

    return WaveformResult(
        samples=samples,
        sample_rate=sample_rate,
        code_start_samples=code_start_samples,
        effective_wpm=effective_wpm,
    )


__all__ = [
    "ElementKind",
    "KeyingParams",
    "WaveformResult",
    "build_element_sequence",
    "codes_to_waveform",
]

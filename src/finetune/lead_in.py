"""実録音の先頭に助走 (ノイズ床) を足す.

**なぜ要るか**

`data/l4/20260824_5w_18m` の 1200 件は、符号の **60 ms 前**から録音が始まっている
(`collect_item` が `drain()` の直後に送信しており、2026-08-25 の `--lead-in-s` 以前の
収録)。実測では 93% の録音で現行モデルが 1 文字目を落とした。

**音は欠けていない。** 末尾のノイズ床を先頭に回した 60 件で脱落が 86 → 34 に戻った
(コミット ad2a391)。消えていたなら貼っても戻らない。欠けているのは助走だけである。

**前回の貼り合わせがなぜ悪化したか**

固定位置で 1 回貼っただけだったので、**継ぎ目そのものが手がかりになった**
(置換 46 → 98、TER 15.69% → 16.97%)。長さを毎回変えれば継ぎ目の位置は
サンプルごとに動き、モデルは継ぎ目も固定オフセットも当てにできない。

**貼る場所**

先頭のゼロパッド (`pad_silence`) の**後ろ**に入れる。実運用の推論経路は
`sliding_window` が 0.3 秒のゼロを前置し、その後に実音声 (必ずノイズ床から始まる)
が続く。つまり「ゼロ → ノイズ床 → 符号」が正しい形で、ゼロとノイズ床の段差は
**消してはいけない**。消すべきなのは貼り合わせで生じる継ぎ目の方である。

**材料の取り方 — 実測で 2 度作り直した**

1. 「打鍵の終わりを検出してその後ろから取る」→ 実データ 60 件中 30 件で失敗。
   5W はコントラストが 5.3 dB しかなく、**末尾の雑音の尖りを打鍵と読む**。
2. 「録音中で最も静かな 0.2 秒の窓を鏡像で並べる」→ **素のままより悪化**
   (TER 15.69% → 17.50%)。理由は 2 つ。**(a)** 最も静かな窓は AGC が利いた
   文字間から取れるので、実際のノイズ床より低いレベルになる。
   **(b)** 短い窓を繰り返すと周期構造が立つ。
3. **末尾のゼロパッド直前から連続で 1.0 秒**取る → TER 13.02%、先頭の脱落
   57/60 → 22/60。これを採る。

長さは掃いて決めた (0.4s 15.47% / 0.8s 13.98% / **1.0s 13.02%** / 1.2s 12.06% /
1.5s 12.70% / 2.0s 14.73%)。2.0 秒では**挿入が 15 → 55 に跳ねる** —
材料に符号が入った証拠なので、収録した `--tail-s` と同じ 1.0 秒を既定にする。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                       # 実行時に import しない (循環参照になる)
    from src.finetune.dataset import RealSignalDataset

# 継ぎ目のクロスフェード長 (秒)。**元波形の先頭ノイズ床より短いこと。**
# 8/24 は符号まで 60 ms しかないので、20 ms なら符号に食い込まない。
DEFAULT_CROSSFADE_S = 0.020

# 材料として末尾から連続で切り出す長さ (秒)。収録時の `--tail-s` と揃える。
#
# **長さは実測で決めた。** 現行モデルで 8/24 の 60 件をデコードした TER:
# 素のまま 15.69% / 0.4 秒 15.47% / 0.8 秒 13.98% / **1.0 秒 13.02%** /
# 1.2 秒 12.06% / 1.5 秒 12.70% / 2.0 秒 14.73%。
# 2.0 秒では**挿入が 15 → 55 に跳ねる** — 材料に符号が入り、助走の中に
# 幻の文字を置いた証拠である。1.0 秒は収録した `--tail-s` そのもので安全側。
DEFAULT_SOURCE_S = 1.0

# 材料が符号を含んでいたときに順に試す短い長さ (秒)。
SOURCE_FALLBACK_S = (1.0, 0.8, 0.6, 0.4, 0.2)

# 材料に符号が入っていないかを、20 ms 包絡線の p90/p10 (広がり) で見る。
# **符号は ON と OFF を持つので広がる。ノイズ床は広がらない。**
# レベル比 (最も静かな区間の何倍か) でも測ってみたが、AGC が利くと
# 「最も静かな区間」が打鍵中の文字間になり、助走のノイズ床より低く出るため
# 誤検出した (8/25 の 120 件中 25 件)。形で見る方が素直だった。
#
# **実測で較正した値である** (末尾X秒の p90/p10):
#   8/24 5W  : 1.0s max 2.93 / 1.5s 3.69 / 2.0s 3.78
#   8/25 50W : 1.0s max 3.48 / 1.5s 6.64 / 2.0s 8.69
#   実録音   : 1.0s max 1.72 / 2.0s 2.89
# 4.0 なら 1.0 秒は全部通り、符号が混ざる長さは強信号で弾ける。
MAX_ENVELOPE_SPREAD = 4.0

# これ未満しか波形が無ければ材料を作れない (秒)。
MIN_SOURCE_S = 0.05


def zero_pad_lengths(wave: np.ndarray) -> tuple[int, int]:
    """先頭と末尾の**厳密なゼロ**の長さ (サンプル数) を返す.

    `pad_silence` が足したゼロは 16 bit PCM でも正確に 0 なので、
    閾値を置かずに数えられる。
    """
    nonzero = np.flatnonzero(wave)
    if nonzero.size == 0:
        return len(wave), 0
    return int(nonzero[0]), int(len(wave) - 1 - nonzero[-1])


def _envelope_spread(x: np.ndarray, sample_rate: int) -> float:
    """20 ms 包絡線の p90/p10. 符号 (ON/OFF がある) なら大きく、ノイズなら小さい."""
    n = max(1, int(0.020 * sample_rate))
    m = len(x) // n
    if m < 5:
        return 1.0
    env = np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1))
    return float(np.percentile(env, 90) / max(float(np.percentile(env, 10)), 1e-12))


def noise_floor_source(
    wave: np.ndarray,
    sample_rate: int,
    *,
    source_s: float = DEFAULT_SOURCE_S,
) -> np.ndarray:
    """末尾のゼロパッドの直前から**連続で**ノイズ床を取り出す (助走の材料).

    **連続していることが効く。** 短い窓を繰り返して伸ばすと周期構造が立ち、
    実測では素のままより悪くなった (0.2 秒の窓を鏡像で並べた版は TER
    15.69% → 17.50%)。末尾 1.0 秒を連続で使うと 13.02% まで下がる。

    **符号を材料にしてはいけない。** 断片が混ざると助走の中に音のある文字を
    置くことになり、消そうとしている先頭幻覚をこちらから作る (末尾 2.0 秒まで
    遡った版は挿入が 15 → 55 に跳ねた)。純度を測り、汚れていれば短くして試す。

    Raises:
        ValueError: 波形が短すぎるか、どの長さでもノイズ床とみなせないとき。
            **黙って助走なしに落ちない** — 材料を持たない録音が何件あるかは
            呼び出し側が数えて印字する。
    """
    head_zeros, tail_zeros = zero_pad_lengths(wave)
    span = np.asarray(wave[head_zeros: len(wave) - tail_zeros], dtype=np.float64)
    if len(span) < int(MIN_SOURCE_S * sample_rate):
        raise ValueError(
            f"ノイズ床を取れる長さがない ({len(span) / sample_rate:.3f} 秒)"
        )

    for want_s in SOURCE_FALLBACK_S:
        if want_s > source_s:
            continue
        n_want = int(want_s * sample_rate)
        if n_want > len(span):
            continue
        src = span[len(span) - n_want:]
        if _envelope_spread(src, sample_rate) <= MAX_ENVELOPE_SPREAD:
            return np.asarray(src, dtype=np.float32)
    raise ValueError(
        f"末尾のどの長さもノイズ床とみなせない (最短 {SOURCE_FALLBACK_S[-1]} 秒でも"
        f"包絡線の広がりが {MAX_ENVELOPE_SPREAD} を超える)"
    )


def _mirror_tile(src: np.ndarray, length: int) -> np.ndarray:
    """`src` を鏡像で繰り返して `length` サンプルにする.

    そのまま並べると `src` の長さの周期が立つ。1 本おきに反転すると
    継ぎ目が連続になり、周期も崩れる。**for ループを使わない。**
    """
    n_copies = int(np.ceil(length / len(src))) + 1
    copies = np.tile(src, (n_copies, 1))
    copies[1::2] = copies[1::2, ::-1]        # 奇数番を反転 (鏡像)
    return copies.reshape(-1)[:length]


def prepend_lead_in(
    wave: np.ndarray,
    sample_rate: int,
    lead_in_s: float,
    *,
    source: np.ndarray | None = None,
    crossfade_s: float = DEFAULT_CROSSFADE_S,
    source_s: float = DEFAULT_SOURCE_S,
) -> np.ndarray:
    """先頭のゼロパッドの後ろに `lead_in_s` 秒のノイズ床を挿む.

    Args:
        wave: 1 次元波形.
        sample_rate: サンプリングレート.
        lead_in_s: 足す助走の長さ (秒). 0 以下なら何もしない.
        source: ノイズ床の材料. 省略時は `noise_floor_source` で取る
            (**毎回取り直すのは無駄なので、Dataset は読み込み時に一度作って渡す**).
        crossfade_s: 継ぎ目の等電力クロスフェード長 (秒).
        source_s: `source` 省略時に切り出す窓の長さ (秒).
    """
    wave = np.asarray(wave, dtype=np.float32)
    n_lead = int(lead_in_s * sample_rate)
    if n_lead <= 0:
        return wave

    head_zeros, _ = zero_pad_lengths(wave)
    head = wave[:head_zeros]
    body = wave[head_zeros:]

    if source is None:
        source = noise_floor_source(wave, sample_rate, source_s=source_s)

    # クロスフェードは元波形の先頭ノイズ床の中で終わらせる。
    # **符号に食い込ませない。** body が短ければその分だけ縮める。
    n_fade = min(int(crossfade_s * sample_rate), len(body) // 2, n_lead)
    lead = _mirror_tile(source, n_lead)

    if n_fade <= 0:
        return np.concatenate([head, lead, body])

    # 等電力クロスフェード (sqrt raised-cosine)。同じノイズ床どうしなので
    # 直線ランプだと重なり区間だけ電力が落ちて「谷」が手がかりになる。
    t = np.linspace(0.0, np.pi, n_fade, endpoint=False, dtype=np.float32)
    fade_out = np.sqrt(0.5 * (1.0 + np.cos(t)))
    fade_in = np.sqrt(1.0 - fade_out**2)
    joined = lead[-n_fade:] * fade_out + body[:n_fade] * fade_in
    return np.concatenate([head, lead[:-n_fade], joined, body[n_fade:]])


def resolve_lead_in_range(lo: float, hi: float) -> tuple[float, float] | None:
    """助走のランダム長の範囲を決める. 上限 0 なら無効 (``None``)."""
    if hi <= 0.0:
        return None
    if lo < 0.0 or lo > hi:
        raise ValueError(f"--real-lead-in-min/max が不正です: ({lo}, {hi})")
    return (lo, hi)


def report_lead_in(
    dataset: "RealSignalDataset", lead_in_range: tuple[float, float] | None
) -> None:
    """助走の材料を作れなかった件数を印字する.

    **黙って助走なしに落とさない。** 一晩走ってから「半分は助走が付いて
    いなかった」と気づくのでは遅い。
    """
    if lead_in_range is None:
        print("[init] 助走のオーグメンテーション: 無効", flush=True)
        return
    n_no = len(dataset.no_lead_in_reasons)
    print(
        f"[init] 助走 {lead_in_range[0]}〜{lead_in_range[1]} 秒をランダムに前置 "
        f"(材料が取れず素のまま {n_no}/{len(dataset)} 件)",
        flush=True,
    )
    for path, reason in dataset.no_lead_in_reasons[:5]:
        print(f"[init]   助走なし: {path.name}: {reason}", flush=True)
    if n_no > 5:
        print(f"[init]   ... 他 {n_no - 5} 件", flush=True)


__all__ = [
    "DEFAULT_CROSSFADE_S",
    "DEFAULT_SOURCE_S",
    "MAX_ENVELOPE_SPREAD",
    "SOURCE_FALLBACK_S",
    "MIN_SOURCE_S",
    "noise_floor_source",
    "report_lead_in",
    "resolve_lead_in_range",
    "prepend_lead_in",
    "zero_pad_lengths",
]

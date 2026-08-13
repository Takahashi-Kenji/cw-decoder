"""commit_lag_s の掃引.

確定 (committed) は不変な設計のため、``commit_lag_s`` を短くしすぎると
誤りが永久に残る。既定の 2.5 秒は余裕を見た値で実測の裏付けが無いため、
「右文脈が最大限ある状態」を参照として、各 commit_lag で確定したトークン列が
どれだけ一致するかを測る。

比較は**文字ではなくトークン ID 列**で行う (変換表の影響を混ぜない)。

## 主指標と副指標 (cross-check) の 2 本立て

音声ファイルには終わりがあるため、``commit_lag`` を大きくするほど「ファイル末尾
``commit_lag`` 秒分は音声が尽きて確定できないまま」になる区間が広がる。これは
実運用 (音声が延々と続くライブ受信) には存在しない有限長特有の現象であり、
これを比較にそのまま含めると commit_lag が大きいほど不利になる非対称なバイアスが
乗ってしまう (レビューで指摘・実測で確認済み)。

- **主指標**: 参照・仮説の両方から、ファイル末尾 ``max(lags)`` 秒以内に終わる
  トークンを除外してから比較する。全 commit_lag が全く同じ音声区間で評価される
  ため、差は「確定時にどれだけ右文脈があったか」だけになる。
- **副指標 (cross-check)**: 実運用でストリーム終了時に ``finalize()`` を呼ぶ挙動
  (`src/app/workers.py` の `stop()`) を模して、仮説側でも末尾で `finalize()` を
  呼び、末尾を除外しないフル参照と比較する。ただし `finalize()` は右文脈ゼロで
  末尾を確定するため、この指標には「右文脈量の効果」以外のノイズ (末尾一括確定の
  精度) が混ざる。主指標ではなく参考値として使う。

使い方:
    python scripts/sweep_commit_lag.py --checkpoint models/full/best_infer.pt \\
        --audio-dir data/real/train --out docs/commit_lag_sweep.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.infer.engine import FrameToken
from src.infer.sliding_window import CommittedToken, SlidingWindowDecoder

DEFAULT_LAGS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5)


class ChunkEngine(Protocol):
    """``InferenceEngine`` 互換の最小インタフェース (テストで差し替えるため)."""

    frame_hop_samples: int

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]: ...


@dataclass(frozen=True)
class SweepRow:
    """1 つの commit_lag に対する集計結果."""

    commit_lag_s: float
    n_files: int
    n_reference_tokens: int            # 主指標: 末尾 max(lags) 秒を除いた参照の総トークン数
    match_rate: float                  # 主指標: 参照と完全一致したファイルの割合
    cer: float                         # 主指標: トークン単位の編集距離 / 参照長
    match_rate_with_finalize: float    # 副指標 (cross-check): 末尾 finalize() 併用版
    cer_with_finalize: float           # 副指標 (cross-check)
    n_reference_tokens_full: int       # 副指標の分母 (末尾を除かないフル参照の総トークン数)


def token_cer(reference: list[int], hypothesis: list[int]) -> float:
    """トークン ID 列の編集距離を参照長で割った値.

    参照が空のときは、仮説も空なら 0.0、そうでなければ 1.0 を返す.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        cur = [i]
        for j, h in enumerate(hypothesis, start=1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(reference)


def _trim_tail(
    committed: list[CommittedToken], wave_size: int, trim_tail_s: float | None, sample_rate: int
) -> list[CommittedToken]:
    """ファイル末尾 ``trim_tail_s`` 秒以内に終わるトークンを除外する.

    ``trim_tail_s=None`` のときは何も除外しない (元のリストをそのまま返す)。
    """
    if trim_tail_s is None:
        return committed
    cutoff = wave_size - int(trim_tail_s * sample_rate)
    return [t for t in committed if t.absolute_sample_end < cutoff]


def simulate_commit(
    engine: ChunkEngine,
    wave: np.ndarray,
    commit_lag_s: float,
    *,
    hop_s: float = 1.0,
    window_s: float = 30.0,
    head_guard_s: float = 1.0,
    decode_left_context_s: float = 5.0,
    sample_rate: int = 8000,
    finalize: bool = False,
    trim_tail_s: float | None = None,
) -> list[int]:
    """波形を hop ごとに投入し、``commit_lag_s`` に従って確定したトークン ID 列を返す.

    通常のストリーミング確定 (hop ごとに ``redecode()``) を行う。
    ``finalize=True`` のときは、ループを終えた後にさらに ``finalize()`` を呼び、
    まだ暫定だった末尾を一括確定する (実運用でストリーム終了時に ``finalize()``
    を呼ぶ挙動を模した cross-check 用)。

    ``trim_tail_s`` を指定すると、ファイル末尾 ``trim_tail_s`` 秒以内に終わる
    トークンを結果から除外する。掃引の主指標は、全 commit_lag を同じ音声区間で
    比較するためにこれを使う (``sweep()`` のモジュール docstring 参照)。
    ``finalize=True`` と ``trim_tail_s`` を同時に指定する使い方は想定していない
    (finalize 側は「末尾を除外しないフル参照」と比較する cross-check 専用のため)。
    """
    decoder = SlidingWindowDecoder(
        engine,                      # type: ignore[arg-type]  # ChunkEngine で足りる
        window_s=window_s,
        hop_s=hop_s,
        commit_lag_s=commit_lag_s,
        head_guard_s=head_guard_s,
        decode_left_context_s=decode_left_context_s,
        sample_rate=sample_rate,
    )
    hop_samples = int(hop_s * sample_rate)
    for start in range(0, wave.size, hop_samples):
        decoder.push(wave[start:start + hop_samples])
        decoder.redecode()
    view = decoder.finalize() if finalize else decoder.redecode()
    committed = _trim_tail(view.committed, wave.size, trim_tail_s, sample_rate)
    return [t.token_id for t in committed]


def _simulate_both(
    engine: ChunkEngine,
    wave: np.ndarray,
    commit_lag_s: float,
    *,
    hop_s: float,
    window_s: float,
    head_guard_s: float,
    decode_left_context_s: float,
    sample_rate: int,
    trim_tail_s: float | None,
) -> tuple[list[int], list[int]]:
    """``sweep()`` 内部専用: 主指標 (trim 版) と副指標 (finalize 版) を、
    hop ごとの ``redecode()`` ループを 1 回だけ実行して両方求める.

    ``simulate_commit`` を ``finalize`` の有無で 2 回呼ぶと、hop ごとの
    ``redecode()`` ループ (掃引で最もコストが大きい部分) を 2 回実行することになる。
    ``redecode()`` は状態を追加するだけで消さないため、ループ後に
    ``redecode()`` (主指標のスナップショット) → ``finalize()`` (副指標のスナップショット)
    の順で呼んでも、それぞれ ``simulate_commit`` を独立に呼んだ場合と同じ結果になる。
    """
    decoder = SlidingWindowDecoder(
        engine,                      # type: ignore[arg-type]  # ChunkEngine で足りる
        window_s=window_s,
        hop_s=hop_s,
        commit_lag_s=commit_lag_s,
        head_guard_s=head_guard_s,
        decode_left_context_s=decode_left_context_s,
        sample_rate=sample_rate,
    )
    hop_samples = int(hop_s * sample_rate)
    for start in range(0, wave.size, hop_samples):
        decoder.push(wave[start:start + hop_samples])
        decoder.redecode()
    streaming = _trim_tail(decoder.redecode().committed, wave.size, trim_tail_s, sample_rate)
    finalized = decoder.finalize().committed
    return [t.token_id for t in streaming], [t.token_id for t in finalized]


def reference_decode(
    engine: ChunkEngine,
    wave: np.ndarray,
    *,
    hop_s: float = 1.0,
    window_s: float = 30.0,
    head_guard_s: float = 1.0,
    decode_left_context_s: float = 5.0,
    sample_rate: int = 8000,
    trim_tail_s: float | None = None,
) -> list[int]:
    """「右文脈が最大限ある状態」の参照トークン列を返す.

    ``commit_lag_s`` に相当する概念を持たない (どんな lag を掃引していても
    参照は 1 種類だけ)。途中で commit_lag により早期確定させると右文脈最大の
    前提が壊れるため、ループ中は ``redecode()`` を呼ばず ``push()`` だけを行い、
    末尾の ``finalize()`` 一回だけでリング全体を一括確定する。これは音声全体を
    一度に ``engine.decode_chunk()`` した結果と等価になる
    (``tests/test_sweep_commit_lag.py`` の回帰テストで担保)。

    過去に ``simulate_commit(..., commit_lag_s=0.0, finalize=True)`` を参照として
    使っていたが、``simulate_commit`` がループ中も ``redecode()`` を呼ぶ実装だった
    ため ``commit_lag_s=0.0`` は最短 lag 相当の早期確定になってしまい、
    「右文脈最大」の前提が壊れていた不具合があった。それを踏まえ、参照計算は
    この専用関数に分離し、``commit_lag_s`` という無関係な引数を受け取らないように
    した。

    音声長が ``window_s`` を超えると ``SlidingWindowDecoder`` のリングバッファが
    先頭を黙って捨ててしまい、参照がファイル後半 ``window_s`` 秒分だけになる
    (誤った参照を黙って返す方が、例外で気づけるより危険なため送出する)。
    """
    decoder = SlidingWindowDecoder(
        engine,                      # type: ignore[arg-type]  # ChunkEngine で足りる
        window_s=window_s,
        hop_s=hop_s,
        commit_lag_s=0.0,            # finalize() が上書きするので値そのものに意味は無い
        head_guard_s=head_guard_s,
        decode_left_context_s=decode_left_context_s,
        sample_rate=sample_rate,
        # **助走を切る。** 参照の定義は「音声全体を一度に decode_chunk した結果と
        # 等価」であり、そこがこの道具の土台になっている (下のテストが担保)。
        # 2026-08-12 に実運用へ助走を入れたが、**過去の掃引と比べられなくなるので
        # 測定の基準は動かさない**。掃引が比べたいのは commit_lag の違いであって、
        # 参照そのものの精度ではない。
        lead_in_s=0.0,
    )
    if wave.size > decoder.window_samples:
        raise ValueError(
            f"音声長 {wave.size} サンプルが window_s={window_s}s "
            f"({decoder.window_samples} サンプル) を超えています。"
            "finalize() 参照はリングバッファの切り詰めにより先頭が黙って失われるため、"
            "この長さの音声には対応していません。"
        )
    hop_samples = int(hop_s * sample_rate)
    for start in range(0, wave.size, hop_samples):
        decoder.push(wave[start:start + hop_samples])
    committed = _trim_tail(decoder.finalize().committed, wave.size, trim_tail_s, sample_rate)
    return [t.token_id for t in committed]


def sweep(
    engine: ChunkEngine,
    waves: list[np.ndarray],
    lags: list[float] | tuple[float, ...] = DEFAULT_LAGS,
    *,
    hop_s: float = 1.0,
) -> list[SweepRow]:
    """各 commit_lag について、参照との一致を集計する (モジュール docstring 参照)."""
    max_lag = float(max(lags))

    # 主指標: 参照・仮説の両方から末尾 max_lag 秒を除外し、同じ音声区間で比較する。
    trimmed_refs = [reference_decode(engine, w, hop_s=hop_s, trim_tail_s=max_lag) for w in waves]
    n_ref_tokens = sum(len(r) for r in trimmed_refs)

    # 副指標 (cross-check): 末尾を除外しないフル参照。
    full_refs = [reference_decode(engine, w, hop_s=hop_s) for w in waves]
    n_ref_tokens_full = sum(len(r) for r in full_refs)

    rows: list[SweepRow] = []
    for lag in lags:
        matches = 0
        total_edits = 0.0
        matches_fin = 0
        total_edits_fin = 0.0
        for wave, ref, ref_full in zip(waves, trimmed_refs, full_refs, strict=True):
            got, got_fin = _simulate_both(
                engine, wave, commit_lag_s=lag, hop_s=hop_s,
                window_s=30.0, head_guard_s=1.0, decode_left_context_s=5.0,
                sample_rate=8000, trim_tail_s=max_lag,
            )
            if got == ref:
                matches += 1
            total_edits += token_cer(ref, got) * max(1, len(ref))

            if got_fin == ref_full:
                matches_fin += 1
            total_edits_fin += token_cer(ref_full, got_fin) * max(1, len(ref_full))

        rows.append(SweepRow(
            commit_lag_s=lag,
            n_files=len(waves),
            n_reference_tokens=n_ref_tokens,
            match_rate=matches / len(waves) if waves else 0.0,
            cer=total_edits / n_ref_tokens if n_ref_tokens else 0.0,
            match_rate_with_finalize=matches_fin / len(waves) if waves else 0.0,
            cer_with_finalize=total_edits_fin / n_ref_tokens_full if n_ref_tokens_full else 0.0,
            n_reference_tokens_full=n_ref_tokens_full,
        ))
    return rows


def _load_waves(audio_dir: Path, limit: int | None) -> list[np.ndarray]:
    """8 kHz mono float32 に揃えて読み込む."""
    import soundfile as sf
    import soxr

    paths = sorted(audio_dir.glob("*.wav"))
    if limit is not None:
        paths = paths[:limit]
    waves: list[np.ndarray] = []
    for path in paths:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data[:, 0]
        if sr != 8000:
            mono = soxr.resample(mono, sr, 8000).astype(np.float32)
        waves.append(np.ascontiguousarray(mono, dtype=np.float32))
    return waves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/full/best_infer.pt"))
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hop-s", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("docs/commit_lag_sweep.json"))
    args = parser.parse_args()

    from src.infer.engine import InferenceEngine

    engine = InferenceEngine.from_checkpoint(args.checkpoint, device="cpu")
    waves = _load_waves(args.audio_dir, args.limit)
    if not waves:
        raise SystemExit(f"音声が見つかりません: {args.audio_dir}")
    print(f"{len(waves)} ファイルで掃引します")

    rows = sweep(engine, waves, hop_s=args.hop_s)

    print(f"{'lag(s)':>8} {'一致率':>8} {'CER':>8} {'一致率(fin)':>11} {'CER(fin)':>9}")
    for row in rows:
        print(
            f"{row.commit_lag_s:8.2f} {row.match_rate:8.3f} {row.cer:8.4f} "
            f"{row.match_rate_with_finalize:11.3f} {row.cer_with_finalize:9.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()

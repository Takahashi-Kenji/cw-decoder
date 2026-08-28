"""実受信機チェーンからラベル付き録音を集める (L4).

構成 (運用者の設備)::

    無線機 A (ダミーロード) ──電波──> 無線機 B (アンテナ)
        ↑ 電鍵 (COM)                      │ 音声
    scripts/cw_key_server.py          scripts/audio_send.py
        ↑ LAN 45679                       │ LAN 45678
        └────── GPU PC (scripts/collect_l4.py) ──────┘

**送ったテキストがそのままラベルになる。** 書き起こしが要らないので、
一晩回すだけでラベル付き実信号が桁で増える。合成では再現できない
受信機チェーンの癖 (AGC のポンピング・フィルタのリンギング・実ノイズ床) が入る。

**入らないもの**: 打鍵はオートキーヤーなので**人間の手打ちの癖は入らない**。
そこは自局の手打ち録音と、実録音からのタイミング移植が担当する。
この分担を混同すると「L4 を大量に入れたのに手打鍵 TER が下がらない」で
誤った結論を出す。

保存しない条件 (§:class:`RejectReason`)
---------------------------------------
**「打鍵計画」と「実際に出た電波」は違いうる。** 2026-08-03 には、原稿にあるのに
オートキーヤーが送出しなかったと**誤って**判断して held-out のラベルを壊し、
物差しが 1 年近く狂っていた。だからここでは打鍵側の実測と突き合わせ、
少しでも食い違ったら**保存しない**。捨てるのは安い (もう 1 回打てばよい)。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from src.finetune.preprocess import (
    DEFAULT_BPF_BANDWIDTH_HZ,
    DEFAULT_BPF_CENTER_HZ,
    DEFAULT_PAD_SEC,
    bandpass,
    pad_silence,
    write_wav_int16,
)
from src.synth.text_generator import TextGenConfig, generate_text
from src.tokens.morse_tokens import Mode, text_to_codes


class RejectReason(Enum):
    """録れたが**保存しない**理由."""

    ABORTED = "打鍵が中止された"
    ELEMENT_MISMATCH = "打った要素数が計画と違う"
    TOO_QUIET = "音が小さすぎる (無線機 B が A を聞けていない)"
    DROPPED_AUDIO = "受信バッファが溢れて波形に穴がある"
    NO_AUDIO = "音が 1 サンプルも届かなかった"


@dataclass(frozen=True)
class PlanItem:
    """打鍵 1 件の計画."""

    name: str
    text: str
    mode: Mode
    wpm: float


@dataclass(frozen=True)
class CollectConfig:
    """収集の設定."""

    # **打鍵の前にノイズ床を録る長さ。**
    # これが 0 だと音が「無音 → いきなり符号」で始まり、実運用に無い形になる。
    # 実測 (1200 件): 0 のとき **93% の録音で 1 文字目が脱落**した
    # (脱落 1470 個のうち 1188 個が先頭)。録れた音を後から貼り合わせても
    # 直らない — 継ぎ目で置換誤りが増えるだけだった (46 → 98)。
    lead_in_s: float = 1.5
    tail_s: float = 1.0          # 打ち終わってからも録り続ける長さ
    gap_s: float = 10.0          # 次の項目までの間 (終段の熱を冷ます)
    # これ未満は「聞こえていない」とみなす。
    # **単位は dBFS。** 受信側 (`NetworkAudioCapture`) が返す波形は ±1.0 の
    # float であり、int16 のスケール (±32768) ではない。生の RMS で閾値を
    # 持つと 32768 倍ずれて**全件が捨てられる**。
    # 実測のノイズ床は -30 dBFS 前後なので、-60 は「ほぼ無音」の線。
    min_level_db: float = -60.0
    sample_rate: int = 8000
    poll_s: float = 0.2


@dataclass
class ItemResult:
    """1 件の結果."""

    item: PlanItem
    wave: np.ndarray | None = None
    send: Any = None                    # SendResult (打鍵側の実測)
    rejected: RejectReason | None = None


class Keyer(Protocol):
    """打鍵側 (:class:`src.tx.net_key.NetKeyClient`) に求める最小の口."""

    def check(self, text: str, wpm: float) -> Any: ...
    def send(self, text: str, wpm: float) -> Any: ...


class Capture(Protocol):
    """音声側 (:class:`src.infer.net_audio.NetworkAudioCapture`) に求める最小の口."""

    def drain(self) -> list[np.ndarray]: ...
    @property
    def dropped_blocks(self) -> int: ...


def build_session_plan(
    count: int,
    modes: Sequence[Mode],
    wpm_list: Sequence[float],
    seed: int,
    session_id: str = "l4",
    text_config: TextGenConfig | None = None,
) -> list[PlanItem]:
    """セッション 1 回分の打鍵計画を作る.

    **速度とモードは乱数で引かず順に巡らせる。** 一晩回して「30 WPM が 3 件しか
    無い」を避けるため。本文だけが乱数 (seed を固定すれば同じ計画になる)。
    """
    rng = np.random.default_rng(seed)
    config = text_config or TextGenConfig()
    plan: list[PlanItem] = []
    for i in range(count):
        mode = modes[i % len(modes)]
        wpm = float(wpm_list[i % len(wpm_list)])
        text = generate_text(rng, mode, config)
        plan.append(PlanItem(
            name=f"{session_id}_{i + 1:04d}_{mode}",
            text=text, mode=mode, wpm=wpm,
        ))
    return plan


def level_verdict(level_db: float) -> str:
    """受信レベル (dBFS) を人の言葉にする.

    **「繋がっている」と「聞こえている」は違う。** 接続は成立するのに
    レベルが -119 dB だった実例がある (取り込むデバイスが違っていた)。
    一晩回す前にここで気づきたい。

    **RMS だけでノイズ床と信号は区別できない。** 受信機は常に何か鳴っている
    ので、言い切らずに「音は来ている」までにとどめる (信号かどうかは
    録れた WAV を聞くか、デコードして確かめる)。
    """
    if level_db < -70.0:
        return "無音 (取り込むデバイスが違うか、無線機から音が出ていない)"
    if level_db < -6.0:
        return "正常な範囲 (音は来ている)"
    return "過大入力 (歪む恐れ。無線機の音量を下げる)"


def dry_run_error(dry_run: bool) -> str | None:
    """打鍵側が dry-run のときの中止理由を返す (問題なければ ``None``).

    **dry-run では電鍵が動かない。** 録れるのはノイズ床だけなのに、
    ラベルは全文が付く。「音の無い学習データ」が静かに溜まるのが最悪で、
    しかも打鍵側は成功を返すので気づけない。
    """
    if not dry_run:
        return None
    return (
        "打鍵側が dry-run で動いています (電鍵が動きません)。"
        "無線機 PC の cw_key_server.py を --dry-run 無しで起動し直してください"
    )


def fingerprint_error(matches: bool, allow: bool) -> str | None:
    """符号表の指紋が食い違っているときの中止理由を返す (問題なければ ``None``).

    **無線機 PC は別のリポジトリの複製を読む。** 表がずれていると、こちらの
    ラベルと実際に出る電波が食い違ったデータが**静かに**溜まる。一晩集めてから
    気づいても全部使えない。実際に無線機 PC のコードが 1 か月古いまま
    残っていたことがある (`。` の定義が違った)。
    """
    if matches or allow:
        return None
    return (
        "両 PC の符号表が食い違っています (指紋不一致)。"
        "無線機 PC のコードを更新してください。"
        "承知の上で進めるなら --allow-fingerprint-mismatch を付けます"
    )


def load_plan_file(
    path: Path,
    session_id: str,
    modes: Sequence[Mode],
    wpm_list: Sequence[float],
) -> list[PlanItem]:
    """打鍵内容をファイルから読む (**打つ内容を外から全部指定する**).

    弱点を狙った原稿 (recall 0% の記号、``デ``/``テ``/``。`` の取り違え等) を
    ぶつけるための口。1 行 1 件、タブ区切りで:

    ==========================  ==========================================
    ``本文``                    モードと速度は ``modes`` / ``wpm_list`` を巡る
    ``モード<TAB>本文``          モードだけ指定
    ``モード<TAB>速度<TAB>本文``  両方指定
    ==========================  ==========================================

    ``#`` で始まる行と空行は飛ばす。
    **打てない文字はその場でエラーにする** (一晩回してから気づくのでは遅い)。
    """
    plan: list[PlanItem] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        i = len(plan)
        mode: Mode = modes[i % len(modes)]
        wpm = float(wpm_list[i % len(wpm_list)])
        if len(parts) >= 2 and parts[0] in ("european", "japanese"):
            mode = parts[0]                          # type: ignore[assignment]
            if len(parts) >= 3:
                wpm = float(parts[1])
                text = "\t".join(parts[2:]).strip()
            else:
                text = parts[1].strip()
        else:
            text = line
        try:
            codes = text_to_codes(text, mode)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"{path}:{lineno} {mode} で打てません: {text!r} ({exc})") from exc
        if not codes:
            raise SystemExit(f"{path}:{lineno} 空の本文です")
        plan.append(PlanItem(
            name=f"{session_id}_{i + 1:04d}_{mode}", text=text, mode=mode, wpm=wpm,
        ))
    return plan


def collect_item(
    item: PlanItem,
    keyer: Keyer,
    capture: Capture,
    config: CollectConfig,
    *,
    sleep: Callable[[float], None],
) -> ItemResult:
    """1 件を打鍵させ、受信音を録る.

    ``keyer.send`` は打ち終わるまで返らない。そのあいだ受信側は自前のバッファに
    溜めているので、**溢れていないか (``dropped_blocks``) を必ず確かめる**。
    穴の空いた音に全文のラベルが付くのが一番まずい。
    """
    result = ItemResult(item=item)

    dropped_before = capture.dropped_blocks
    capture.drain()                     # 前の項目の残りを捨てる
    if config.lead_in_s > 0:
        # **捨てた後に待つ。** 先に待つと前の項目の残りがラベルに混ざる。
        sleep(config.lead_in_s)         # 符号の前のノイズ床を録る

    send = keyer.send(item.text, item.wpm)
    result.send = send
    if config.tail_s > 0:
        sleep(config.tail_s)            # 末尾の減衰と AGC の戻りを拾う

    blocks = capture.drain()
    if getattr(send, "aborted", False):
        result.rejected = RejectReason.ABORTED
        return result
    if capture.dropped_blocks != dropped_before:
        result.rejected = RejectReason.DROPPED_AUDIO
        return result
    if not blocks:
        result.rejected = RejectReason.NO_AUDIO
        return result

    wave = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    expected = keyer.check(item.text, item.wpm)
    if int(getattr(send, "elements_sent", -1)) != int(getattr(expected, "elements", -2)):
        result.rejected = RejectReason.ELEMENT_MISMATCH
        return result
    rms = float(np.sqrt(np.mean(np.square(wave))))
    level_db = 20.0 * np.log10(rms) if rms > 1e-9 else -120.0
    if level_db < config.min_level_db:
        result.rejected = RejectReason.TOO_QUIET
        return result

    result.wave = wave
    return result


def collect_with_retry(
    item: PlanItem,
    keyer: Keyer,
    capture: Capture,
    config: CollectConfig,
    *,
    sleep: Callable[[float], None],
    reconnect: Callable[[], None],
    max_retries: int = 3,
    error_types: tuple[type[BaseException], ...] = (OSError,),
    no_retry_types: tuple[type[BaseException], ...] = (),
    retry_wait_s: float = 5.0,
) -> ItemResult:
    """1 件を録り、途中で切れたら繋ぎ直して録り直す.

    **一晩の収集では、50 件目で切れて朝に 50 件では 8 時間が無駄になる。**
    打鍵側は待機中の繋ぎ直しを許している (送信中は許さない — 何がどこまで
    出たのか分からない状態で電波を出さないため)。ここは項目と項目の間なので
    繋ぎ直してよい。

    **上限を設ける。** 繋がらないものを一晩叩き続けても仕方がない。

    ``no_retry_types`` は繰り返しても結果が変わらない失敗 (打てない文字を
    含む等、打鍵側の決定的な拒否)。すぐ投げ直す。
    """
    for attempt in range(max_retries + 1):
        try:
            return collect_item(item, keyer, capture, config, sleep=sleep)
        except no_retry_types:
            raise
        except error_types:
            if attempt >= max_retries:
                raise
            sleep(retry_wait_s)
            reconnect()
    raise AssertionError("到達しない")            # pragma: no cover


def measure_recording(wave: np.ndarray, sample_rate: int) -> dict[str, str]:
    """録れた音そのものから実測値を出す (TXT ヘッダに残すため).

    **信号をノイズにどれだけ埋めたかは、後から音を見ないと分からない。**
    セッションの条件 (出力・アッテネータ・バンドの騒がしさ) と実際の録れ高は
    一致しない。学習データを難易度で切り分けたいときにこの値が要る。

    ``contrast_db`` はキーイングの深さの目安 (20 ms 窓 RMS の 90%点 − 10%点)。
    **絶対値で良し悪しを言わないこと** — 実測では、良好に使えている既存の
    学習データが 8.5 dB、ノイズに埋めた収録が 10.4 dB だった。
    セッション間の比較にだけ使う。
    """
    from src.infer.wpm import detect_tone

    win = max(1, int(0.02 * sample_rate))
    usable = wave[: len(wave) // win * win]
    if usable.size == 0:
        return {"tone_hz": "0", "level_db": "-120.0", "contrast_db": "0.0"}
    frames = usable.reshape(-1, win)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    overall = float(np.sqrt(np.mean(np.square(usable))))
    return {
        "tone_hz": f"{detect_tone(wave, sample_rate):.0f}",
        "level_db": f"{20.0 * np.log10(max(overall, 1e-9)):.1f}",
        "contrast_db": f"{float(np.percentile(db, 90) - np.percentile(db, 10)):.1f}",
    }


def write_sample(
    out_dir: Path,
    item: PlanItem,
    wave: np.ndarray,
    sample_rate: int,
    meta: dict[str, str],
    *,
    bpf: tuple[float, float] | None = (DEFAULT_BPF_CENTER_HZ, DEFAULT_BPF_BANDWIDTH_HZ),
    pad_sec: float = DEFAULT_PAD_SEC,
) -> Path:
    """``discover_real_samples`` が読める WAV + TXT で保存する.

    前処理は取り込み経路と同じ (:mod:`src.finetune.preprocess`)。
    未処理の録音を学習に入れると TER 97% まで崩壊することが実測されている。
    """
    if not text_to_codes(item.text, item.mode):
        raise ValueError(f"ラベルをトークン化できません: {item.text!r} ({item.mode})")

    x = np.asarray(wave, dtype=np.float64)
    measured = measure_recording(x, sample_rate)   # **BPF の前**に測る (生の録れ高)
    if bpf is not None:
        x = bandpass(x, sample_rate, *bpf)
    x = pad_silence(x, sample_rate, pad_sec)

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{item.name}.wav"
    write_wav_int16(wav_path, x, sample_rate)

    bpf_note = "none" if bpf is None else f"{bpf[0]:.0f}Hz/{bpf[1]:.0f}Hz"
    header = {
        "mode": item.mode,
        "sample_rate": str(sample_rate),
        "wpm": f"{item.wpm:.0f}",
        "source": "l4_loopback",
        "bpf": bpf_note,
        "pad_sec": f"{pad_sec:.2f}",
        **measured,
        **meta,
    }
    lines = [f"{k}: {v}" for k, v in header.items()]
    wav_path.with_suffix(".txt").write_text(
        "\n".join(lines) + "\n---\n" + item.text + "\n", encoding="utf-8",
    )
    return wav_path


__all__ = [
    "Capture",
    "CollectConfig",
    "ItemResult",
    "Keyer",
    "PlanItem",
    "RejectReason",
    "build_session_plan",
    "collect_item",
    "collect_with_retry",
    "dry_run_error",
    "fingerprint_error",
    "level_verdict",
    "load_plan_file",
    "measure_recording",
    "write_sample",
]

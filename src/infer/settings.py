"""アプリ設定の永続化 (JSON).

要件 §3.4.5: モード・閾値・入力デバイス等の設定をファイルに保存し、
次回起動時に復元する.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from src.tokens.morse_tokens import DisplayMode

DEFAULT_CONFIG_PATH = Path.home() / ".cw-decorder" / "settings.json"


# 設定スキーマ版。フィールドを増減したら上げる。
# v6: AGC を削除 (メル抽出の z-score 正規化と冗長で、適用すると悪化する)
# v7: commit_lag_s の既定を 2.5 → 1.75 (実効右文脈 3.0 → 2.25 秒)。
#     held-out 実録音の CER 実測で、速くかつ精度も良かったため
# v8: hop_s 1.0 → 0.5、commit_lag_s 1.75 → 2.0 (実効右文脈は 2.25 秒のまま)。
#     ブラウザ版と確定条件を揃えた。更新の刻みが半分になり反応が細かくなる。
#     v7 は同日中に v8 へ置き換わっており、実際に保存された設定は存在しない
# v9: line_break_gap_s を追加 (無音による改行)
# v10: squelch_threshold_db の既定を -60 → -25 (実質無効だったスキッシュを有効化)
# v11: word_correct_enabled を追加 (辞書による即時補正)
# v12: decode_device を追加 (既定 cpu。GPU はローカル LLM に空ける)
# v13: llm_highlight_guesses を追加 (推測箇所の赤表示を切替可能に)
# v14: llm_compact_prompt を追加 (小さいローカルモデル向けの短いプロンプト)
# v15: tx_endpoint / tx_wpm を追加 (LAN 越し送信)。
#      結線 (COM ポート・DTR/RTS の役割・極性・PTT) は**アプリに持たせない**。
#      無線機に属する情報であり、打鍵は無線機 PC の CLI が行うため
#      (docs/superpowers/specs/2026-08-10-lan-cw-transmit-design.md §8.1)
CURRENT_SETTINGS_VERSION = 15


@dataclass
class AppSettings:
    """アプリ全体の設定."""

    mode: DisplayMode = "european"
    confidence_threshold: float = 0.5
    input_device: int | None = None       # None = システムデフォルト
    sample_rate: int = 8000
    checkpoint_path: str | None = None
    recording_enabled: bool = False
    recording_dir: str = "data/real"
    window_geometry: dict[str, int] = field(default_factory=dict)
    show_spectrogram: bool = True
    # 未確定 (暫定) テキストをグレーで表示するか。
    # 既定は False。確定と暫定が混ざると読みにくいという運用上の指摘による
    # (2026-08-04)。暫定は文脈が増えると読みが変わるため、追いながら読むには邪魔になる。
    show_provisional: bool = False
    # 自動モードでホレ/ラタにだけ適用する確信度閾値。
    # 実録音でホレの確信度が 0.42〜0.46 に集まり、通常の閾値 0.50 で棄却されて
    # 自動切替が働かなかったため分離した (2026-08-04)。合成欧文100件・実打鍵欧文10件で
    # 0.20 まで下げても偽ホレによる誤切替は 0 件。
    prosign_threshold: float = 0.35
    # 自動モードの欧文中に、欧文表に無く和文表にある符号が来たら和文へ切り替える。
    # ホレ 1 個の検出に依存しない冗長な切替経路 (2026-08-04)。和文→欧文はラタで戻る。
    switch_on_japanese_only: bool = True
    # スキッシュ: **BPF 後**のレベルがこの値 (dBFS) を下回る間、デコーダに無音を
    # 流し込む (捨てるのではなく置き換える。捨てると時間軸が止まる)。
    #
    # 2026-08-07 に -60 (実質無効) から -25 に変更した。メル特徴抽出の z-score
    # 正規化で音量情報が捨てられるため、無信号でもノイズが増幅されて 30 秒に
    # 38 個ものトークン (ほとんど [SN]/[SK]) が出ていた。確信度も 0.6〜0.7 と高く
    # 確信度閾値では防げない。運用者の実信号で -25 が有効と確認済み
    # (docs/word_break_bias.md の付録)。
    #
    # **これより弱い信号は丸ごと落ちる。** 弱い局を追うときは下げること。
    squelch_threshold_db: float = -25.0
    # スキッシュ解除後、レベルが閾値を下回ってもこの秒数はデコードを継続
    squelch_hold_sec: float = 1.0
    # AGC は v6 で削除した。メル抽出がサンプル内 z-score 正規化を行うため
    # (preprocessing.py の MelConfig.normalize) 振幅は特徴量段階で正規化済みで、
    # 追加の AGC は冗長。実測でも入力レベルを 1/50 にしても CER は 1.45% で不変、
    # AGC を適用すると 1.45% → 2.91% と悪化した (時間変化するゲインが包絡を歪める)。
    # そもそも _tick から一度も呼ばれておらず、画面のトグルは無効だった。
    # 帯域通過フィルタ (BPF): CW トーン以外のノイズを除去
    bpf_enabled: bool = True
    bpf_center_hz: float = 600.0          # CW トーン中心周波数 (推奨 600 Hz)
    bpf_bandwidth_hz: float = 400.0       # 通過帯域幅 (推奨 250-500 Hz)

    # --- Phase 6: スライディングウィンドウ再デコード (ライブ連続モード) ---
    # 既定は定数を参照する (以前は 5 が直書きされ、定数を上げた際に取り残された)
    settings_version: int = CURRENT_SETTINGS_VERSION
    window_s: float = 30.0                # リングバッファ保持長 (再開・手動全体用)
    # 再デコード周期。ブラウザ版と揃えて 0.5 (v8)。1.0 だと更新が粗く感じる。
    # CPU 実測で 1 回の再デコードは中央値 32 ms・最大 53 ms なので、
    # hop 500 ms に対して 10 倍の余裕がある (GPU でも同等)。
    hop_s: float = 0.5
    # 確定遅延 (末尾はこの秒数未確定)。効くのは単独値ではなく
    # `commit_lag_s + hop_s / 2` (= 実効右文脈)。hop 0.5 なので実効 2.25 秒で、
    # ブラウザ版 (hop 0.5 / lag 2.0) と完全に同じ。
    # 2026-08-07 に 2.5 → 2.0 に下げた。held-out 実録音での CER 実測で、
    # 実効 2.25 秒が実効 3.0 秒より速くかつ精度も良かったため
    # (docs/commit_lag_sweep_result.md §8)。**hop を変えるときは和を保つこと。**
    commit_lag_s: float = 2.0
    head_guard_s: float = 1.0             # デコード区間先頭の不採用区間 (左文脈なし)
    # CPU 負荷削減: 毎回 window 全体ではなく、確定済み末尾の left_context 秒前から
    # のみ再デコードする (確定済みは不変なので再計算不要). 実効デコード長 ≈
    # left_context + commit_lag + hop = 5.0 + 2.0 + 0.5 = 7.5 秒。CPU 実測 RTF 0.07.
    decode_left_context_s: float = 5.0
    # この長さ以上の無音が空いたら確定テキストに改行を入れる (0 以下なら改行しない)。
    # 送信のターンの切れ目で行を分けるため。根拠と限界は src/infer/line_break.py を参照
    # (送信内の無音は最大 4.4 秒あったので、3.0 秒では送信途中にもまれに改行が入る)。
    line_break_gap_s: float = 3.0
    # パス間タイムスタンプジッタ吸収用の小マージン (中点ウォーターマークの安全余裕).
    # 確定の主機構は midpoint > last_commit_end であり、これは脱落防止の主役ではない.
    commit_jitter_margin_s: float = 0.02

    # --- LLM テキスト清書 ---
    # デコードを走らせるデバイス。``"cpu"`` / ``"cuda"`` / ``"auto"``。
    #
    # **既定は cpu。** デコーダは小さく (17 MB) CPU でも hop 0.5 秒に十分間に合う。
    # GPU はローカル LLM (清書) に空けておく方が全体として速い。運用者の判断による。
    # ``"auto"`` にすると CUDA があれば使う (従来の挙動)。
    decode_device: str = "cpu"

    # CPU デコードに使うスレッド数。0 なら torch の既定に任せる。
    #
    # 2026-08-07 の実測 (30 秒窓 / models/full/best_infer.pt)::
    #
    #     cpu  2 スレッド  164 ms      cpu  8 スレッド  149 ms
    #     cpu  4 スレッド  112 ms      cuda            139 ms
    #
    # **4 が最速で、CUDA より速い。** モデルが小さいので GPU はレイヤごとの
    # 起動費用が勝ってしまう。増やしすぎても同期の費用で遅くなる。
    decode_threads: int = 4

    # 辞書による即時補正 (LLM 不要)。確定テキストの語を切り直し・寄せする。
    # held-out 実録音の欧文で CER 19.25% → 17.15% (-2.09pt)。詳細は
    # src/infer/word_correct.py。既定 True。
    word_correct_enabled: bool = True

    llm_enabled: bool = False
    llm_provider: str = "ollama"          # "claude" | "openai" | "ollama"
    llm_model: str = "llama3.1"
    ollama_endpoint: str = "http://localhost:11434"
    llm_auto: bool = False                # 自動清書トグル (増分方式)
    # LLM が推測・補正した箇所を赤くするか。赤が多いと読みにくいという指摘による
    llm_highlight_guesses: bool = True
    # 短いプロンプトを使うか (207 文字 対 1403 文字)。
    #
    # **既定 True。** 2026-08-08 の実測で、ローカルの小さいモデルは重いプロンプトだと
    # 例文をそのまま返したり (gemma-2-2b-jpn-it) 捏造したり (gemma4:e4b が
    # 「チューリップ」) した。短い版ではどれも直った。
    # クラウドの大きいモデルを使うときは False にすると細かい指示が効く。
    llm_compact_prompt: bool = True
    llm_auto_interval_s: float = 20.0     # 自動清書の最短間隔 (秒)
    # ローカル Ollama の大型モデルは初回ロードで数十秒かかるため余裕を持たせる.
    # Claude/OpenAI は速いのでこの値に達しない.
    llm_timeout_s: float = 120.0

    # 打鍵側 (無線機を繋いだ PC) の `host:port`。空なら送信機能を出さない。
    # **この PC には COM ポートが無い。** 打鍵は向こうでやる。
    # 解析は net_audio.parse_endpoint を使い回す (既定ポートだけ 45679 を渡す)。
    tx_endpoint: str = ""
    # 送信速度。運用者が手で決める (自動追従はしない)。
    tx_wpm: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppSettings:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# 旧 version 既定値 → 新既定への置換表.
# 「保存値が旧既定と一致するなら未変更とみなし新既定へ更新する」ための表。
# 自分で値を変えていたユーザーの設定は書き換えない (一致しないので素通りする)。
# エントリの形式: field: (旧既定, 新既定)
# v4→v5: LLM タイムアウト既定を 30→120 秒へ (Ollama 大型モデルのコールドロード対策).
# v6→v8: commit_lag_s 2.5→2.0、hop_s 1.0→0.5 (実効右文脈 3.0→2.25 秒).
_V1_DEFAULT_REPLACEMENTS: dict[str, tuple[Any, Any]] = {
    "llm_timeout_s": (30.0, 120.0),
    # 長年の既定 2.5 / 1.0 のままのユーザーだけ新既定へ。自分で値を変えていた人の
    # 設定は尊重する (この置換表はそのためにある)。
    # v7 の中間値 1.75 は同日中に 2.0 へ置き換わっており、実際に保存された設定は
    # 存在しないので移行元には含めない (含めると 1.75 を自分で選んだ人の設定を
    # 勝手に書き換えることになる)
    "commit_lag_s": (2.5, 2.0),
    "hop_s": (1.0, 0.5),
    # v9→v10: 実質無効だった -60 のままのユーザーだけ新既定へ
    "squelch_threshold_db": (-60.0, -25.0),
}

def migrate_settings_dict(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """旧 version の設定 dict を最新スキーマへ移行.

    Returns:
        (移行後 dict, 変更があったか).
    """
    # 現行デフォルトを土台に、保存値 (有効キーのみ) を上書き → 欠損フィールドを補完
    valid_keys = {f.name for f in fields(AppSettings)}
    merged = AppSettings().to_dict()
    merged.update({k: v for k, v in data.items() if k in valid_keys})
    version = int(data.get("settings_version", 1))
    changed = False
    if version < CURRENT_SETTINGS_VERSION:
        for field_name, (old_default, new_default) in _V1_DEFAULT_REPLACEMENTS.items():
            if field_name in data and data[field_name] == old_default:
                merged[field_name] = new_default
        merged["settings_version"] = CURRENT_SETTINGS_VERSION
        changed = True
    return merged, changed


def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> AppSettings:
    """設定を JSON から読み込み. ファイルが無い/壊れていれば既定値を返す.

    旧 version の設定はマイグレーション表で補完・置換し、結果をログ出力する.
    """
    path = Path(path)
    if not path.exists():
        return AppSettings()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return AppSettings()
    migrated, changed = migrate_settings_dict(data)
    if changed:
        print(
            f"[settings-migrate] {path} を v{data.get('settings_version', 1)} "
            f"→ v{CURRENT_SETTINGS_VERSION} へ移行しました"
        )
    return AppSettings.from_dict(migrated)


def save_settings(
    settings: AppSettings, path: Path | str = DEFAULT_CONFIG_PATH
) -> None:
    """設定を JSON に保存."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, ensure_ascii=False, indent=2)


__all__ = [
    "CURRENT_SETTINGS_VERSION",
    "DEFAULT_CONFIG_PATH",
    "AppSettings",
    "load_settings",
    "migrate_settings_dict",
    "save_settings",
]

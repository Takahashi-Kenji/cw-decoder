"""メルスペクトログラムの設定値 — **torch を読み込まずに参照できる**.

``src/train/preprocessing.py`` から切り出した。切り出した理由は
配布物の大きさである (詳細は ``src/infer/ctc.py`` の冒頭)。

ONNX 推論では**メル変換が ONNX グラフに焼き込まれている**ので、
実行時に torchaudio は要らない。それでも「1 フレームが何サンプルか」は
ストリーミングの位置合わせに必要なので、値だけをここに置く。

**ここが唯一の真正ソースである。** ``preprocessing.py`` はここから import する。
値を二重に持つと、片方だけ直したときにフレーム位置が静かにずれる。

**このモジュールは torch を import してはいけない。**
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MelConfig:
    """メルスペクトログラム設定 (要件 §3.3.1)."""

    sample_rate: int = 8000
    n_mels: int = 64
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 256          # 2^n で win_length (=200) を覆う最小値
    f_min: float = 50.0
    f_max: float = 4000.0     # Nyquist 直前
    top_db: float = 80.0
    normalize: bool = True

    @property
    def win_length(self) -> int:
        return int(self.sample_rate * self.win_ms / 1000.0)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000.0)


__all__ = ["MelConfig"]

# ブラウザ版 CW デコーダ 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 学習済み CW デコーダモデルを ONNX 化し、ブラウザ上の JavaScript でマイク入力からリアルタイムに欧文・和文をデコードできるようにする。

**Architecture:** メルスペクトログラム変換を `Conv1d` として展開し、モデルと一体の単一 ONNX グラフ (`波形 in → log_probs out`) にする。ブラウザ側は AudioWorklet で 8 kHz 音声を集め、Web Worker 内で ONNX Runtime Web (WASM) 推論 → CTC greedy → スライディングウィンドウ確定 → 符号変換表という流れで処理する。符号表は Python から自動生成し、単一ソース管理を維持する。

**Tech Stack:** PyTorch 2.11 / torchaudio, onnx, onnxruntime (Python 側検証用), Vite, TypeScript, vitest, onnxruntime-web

設計書: `docs/superpowers/specs/2026-08-05-browser-cw-decoder-design.md`

## Global Constraints

- ファイルは UTF-8 (BOM なし)、改行 LF
- コメント・ドキュメント・コミットメッセージは日本語
- Python 3.11+、型ヒント必須 (mypy 互換)、パス操作は `pathlib.Path`
- ruff: `line-length = 120`、`target-version = "py311"`
- 符号表記はドット `・` (U+30FB 中黒)、ダッシュ `-` (U+002D ハイフン)
- 符号定義は `src/tokens/morse_tokens.py` を唯一の真正なソースとする。TypeScript 側に符号表を手で書かない
- 乱数は `np.random.Generator` を引数で受け取る (グローバル `np.random` を使わない)
- テストは `pytest` (Python) / `vitest` (TypeScript)
- ブランチは `feature/browser-decoder` (main から分岐済み)
- モデルは `models/full/best_infer.pt` (4,268,265 パラメータ、vocab 73)
  ※ `src/train/model.py` の `ModelConfig.vocab_size` 既定値 72 は古い。真の値は
  `src/tokens/morse_tokens.VOCAB_SIZE` = 73 (blank + 実符号 71 + WORDBREAK)
- メル設定 (`MelConfig` の既定値、変更しないこと):
  `sample_rate=8000`, `n_mels=64`, `n_fft=256`, `win_length=200`, `hop_length=80`,
  `f_min=50.0`, `f_max=4000.0`, `power=2.0`, `center=True`, `top_db=80.0`, `normalize=True`
- スライディングウィンドウ既定値: `window_s=30.0`, `hop_s=1.0`, `commit_lag_s=2.5`,
  `head_guard_s=1.0`, `decode_left_context_s=5.0`, `commit_jitter_margin_s=0.02`

---

### Task 1: ONNX 化可能なメル変換 (`ConvMelExtractor`)

`torchaudio.transforms.MelSpectrogram` と数値的に一致し、かつ `Conv1d` と行列積だけで書かれたメル変換を作る。これが ONNX グラフに焼き込める形になる。

**Files:**
- Create: `src/train/onnx_mel.py`
- Test: `tests/test_onnx_mel.py`
- Modify: `pyproject.toml` (dev 依存に `onnx`, `onnxruntime` を追加)

**Interfaces:**
- Consumes: `src.train.preprocessing.MelConfig`, `MelExtractor`
- Produces:
  - `class ConvMelExtractor(nn.Module)` — `__init__(self, config: MelConfig | None = None)`,
    `forward(self, waveform: Tensor) -> Tensor` (入力 `(B, T_wave)` または `(T_wave,)`、出力 `(B, n_mels, T_frames)`)

- [ ] **Step 1: dev 依存を追加**

`pyproject.toml` の `[project.optional-dependencies]` の `dev` に 2 行足す。

```toml
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "onnx>=1.16",
    "onnxruntime>=1.18",
]
```

インストール:

```bash
.venv/Scripts/python.exe -m pip install "onnx>=1.16" "onnxruntime>=1.18"
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_onnx_mel.py` を新規作成。

```python
"""ConvMelExtractor が MelExtractor と数値的に一致することの検証."""
from __future__ import annotations

import numpy as np
import torch

from src.train.onnx_mel import ConvMelExtractor
from src.train.preprocessing import MelExtractor


def _reference_and_conv(wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref = MelExtractor().eval()
    conv = ConvMelExtractor().eval()
    with torch.no_grad():
        return ref(wave), conv(wave)


def test_matches_reference_on_noise() -> None:
    """ホワイトノイズで最大絶対誤差 1e-4 未満."""
    rng = np.random.default_rng(0)
    wave = torch.from_numpy(rng.standard_normal(8000 * 3).astype(np.float32) * 0.1)
    ref, got = _reference_and_conv(wave)
    assert got.shape == ref.shape
    assert torch.max(torch.abs(got - ref)).item() < 1e-4


def test_matches_reference_on_tone_bursts() -> None:
    """CW に近い 600 Hz のトーンバーストで一致すること."""
    sr = 8000
    t = np.arange(sr * 2) / sr
    envelope = ((np.sin(2 * np.pi * 3.0 * t) > 0).astype(np.float32))
    wave = torch.from_numpy((np.sin(2 * np.pi * 600.0 * t) * envelope).astype(np.float32))
    ref, got = _reference_and_conv(wave)
    assert torch.max(torch.abs(got - ref)).item() < 1e-4


def test_frame_count_matches() -> None:
    """フレーム数が MelExtractor.frame_count と一致すること."""
    conv = ConvMelExtractor().eval()
    for n in (800, 4001, 8000, 24000):
        wave = torch.zeros(1, n)
        with torch.no_grad():
            out = conv(wave)
        assert out.shape[2] == MelExtractor().frame_count(n)


def test_accepts_1d_and_2d_input() -> None:
    """(T,) と (1, T) のどちらでも同じ結果を返すこと."""
    rng = np.random.default_rng(1)
    wave1d = torch.from_numpy(rng.standard_normal(8000).astype(np.float32) * 0.1)
    conv = ConvMelExtractor().eval()
    with torch.no_grad():
        a = conv(wave1d)
        b = conv(wave1d.unsqueeze(0))
    assert torch.equal(a, b)
```

- [ ] **Step 3: テストが失敗することを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_onnx_mel.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.train.onnx_mel'`

- [ ] **Step 4: 実装する**

`src/train/onnx_mel.py` を新規作成。

**注意点** (ここを外すと一致しない):
- `torch.hann_window` の既定は `periodic=True`。torchaudio もこれを使う
- `win_length` (200) は `n_fft` (256) より短いので、窓を**中央寄せでゼロ詰め**する。
  左パディング量は `(256 - 200) // 2 = 28`
- メルフィルタバンクは `torchaudio.functional.melscale_fbanks` から取る。
  `norm=None`, `mel_scale="htk"` が `MelSpectrogram` の既定値
- `AmplitudeToDB(stype="power", top_db=80)` は
  `10 * log10(clamp(x, min=1e-10))` の後に `max(x_db, x_db.max() - 80)`。
  バッチ 1 前提でテンソル全体の最大値を使う

```python
"""ONNX エクスポート可能なメルスペクトログラム変換.

``torchaudio.transforms.MelSpectrogram`` は内部で ``torch.stft`` を呼ぶが、
ONNX の STFT op はランタイム側の対応が不安定なため、ここでは STFT を
**Conv1d (cos/sin カーネル)** として展開する。Conv1d と行列積と初等演算だけで
構成されるため、どの ONNX ランタイムでも確実に動く。

``MelExtractor`` と数値的に一致することを ``tests/test_onnx_mel.py`` で検証する。
一致が壊れると推論精度が静かに劣化するため、このテストは必ず維持すること。
"""
from __future__ import annotations

import math

import torch
import torchaudio
from torch import Tensor, nn

from src.train.preprocessing import MelConfig


class ConvMelExtractor(nn.Module):
    """波形 → 対数メルスペクトログラム (ONNX エクスポート可能).

    入力: ``(B, T_wave)`` または ``(T_wave,)`` (float32)
    出力: ``(B, n_mels, T_frames)`` (float32)
    """

    def __init__(self, config: MelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MelConfig()
        c = self.config

        self.register_buffer("dft_kernel", self._build_dft_kernel(c), persistent=False)
        self.register_buffer("mel_fb", self._build_mel_fb(c), persistent=False)

    @staticmethod
    def _build_dft_kernel(c: MelConfig) -> Tensor:
        """DFT 基底 × Hann 窓 の Conv1d カーネル ``(2 * n_bins, 1, n_fft)``.

        前半 ``n_bins`` チャネルが実部、後半 ``n_bins`` チャネルが虚部.
        """
        n_fft = c.n_fft
        # periodic=True が torchaudio (torch.stft) の既定
        win = torch.hann_window(c.win_length, periodic=True, dtype=torch.float64)
        # win_length < n_fft のとき torch.stft は窓を中央寄せでゼロ詰めする
        pad_left = (n_fft - c.win_length) // 2
        window = torch.zeros(n_fft, dtype=torch.float64)
        window[pad_left:pad_left + c.win_length] = win

        n_bins = n_fft // 2 + 1
        k = torch.arange(n_bins, dtype=torch.float64).unsqueeze(1)   # (n_bins, 1)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(0)    # (1, n_fft)
        angle = 2.0 * math.pi * k * n / n_fft
        real = torch.cos(angle) * window
        imag = -torch.sin(angle) * window
        kernel = torch.cat([real, imag], dim=0).unsqueeze(1)         # (2*n_bins, 1, n_fft)
        return kernel.to(torch.float32)

    @staticmethod
    def _build_mel_fb(c: MelConfig) -> Tensor:
        """メルフィルタバンク ``(n_mels, n_bins)``.

        自前で式を書かず torchaudio の関数から取る (ズレの余地を作らない).
        """
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=c.n_fft // 2 + 1,
            f_min=c.f_min,
            f_max=c.f_max,
            n_mels=c.n_mels,
            sample_rate=c.sample_rate,
            norm=None,
            mel_scale="htk",
        )                                   # (n_bins, n_mels)
        return fb.transpose(0, 1).contiguous()   # (n_mels, n_bins)

    def forward(self, waveform: Tensor) -> Tensor:
        c = self.config
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        x = waveform.unsqueeze(1)                       # (B, 1, T_wave)

        # center=True 相当の reflect padding
        pad = c.n_fft // 2
        x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")

        spec = torch.nn.functional.conv1d(x, self.dft_kernel, stride=c.hop_length)
        n_bins = c.n_fft // 2 + 1
        real = spec[:, :n_bins, :]
        imag = spec[:, n_bins:, :]
        power = real * real + imag * imag               # (B, n_bins, T_frames)

        mel = torch.matmul(self.mel_fb, power)          # (B, n_mels, T_frames)

        # AmplitudeToDB(stype="power", top_db=80)
        spec_db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        spec_db = torch.maximum(spec_db, spec_db.max() - c.top_db)

        if c.normalize:
            mean = spec_db.mean(dim=(-2, -1), keepdim=True)
            std = spec_db.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            spec_db = (spec_db - mean) / std
        return spec_db


__all__ = ["ConvMelExtractor"]
```

- [ ] **Step 5: テストが通ることを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_onnx_mel.py -v
```

Expected: 4 passed

失敗した場合の切り分け順:
1. `test_frame_count_matches` だけ失敗 → padding 量が違う
2. 誤差が 1e-2 程度 → 窓のゼロ詰め位置 (`pad_left`) を疑う
3. 誤差が桁違い → `melscale_fbanks` の `norm` / `mel_scale` を疑う

- [ ] **Step 6: コミット**

```bash
git add pyproject.toml src/train/onnx_mel.py tests/test_onnx_mel.py
git commit -m "feat: ONNXエクスポート可能なメル変換 ConvMelExtractor を追加"
```

---

### Task 2: ONNX エクスポートスクリプト

`波形 in → log_probs out` の単一 ONNX を出力し、Python の `InferenceEngine` とトークン列が完全一致することを検証する。

**Files:**
- Create: `scripts/export_onnx.py`
- Test: `tests/test_export_onnx.py`

**Interfaces:**
- Consumes: `ConvMelExtractor` (Task 1), `src.train.checkpoint.build_model_from_checkpoint`,
  `src.infer.engine.InferenceEngine`, `ctc_greedy_decode_with_frames`
- Produces:
  - `class OnnxDecoder(nn.Module)` — `__init__(self, model: CWModel, mel: ConvMelExtractor)`,
    `forward(self, wave: Tensor) -> Tensor` (`(1, T_wave)` → `(1, T_frames, 73)` の log_probs)
  - `def export_onnx(checkpoint: Path, out_path: Path) -> None`
    — **エクスポート直後に自己検証し、失敗したら出力を削除して例外を投げること**
    (設計書 §5「エクスポート時の検証 (必須)」)。検証は 2 つ:
    (a) `sample_wav/oubun.wav` の実音声でメル出力の最大絶対誤差 < 1e-4、
    (b) 同じ音声で `InferenceEngine.decode_chunk` と ONNX+CTC のトークン ID 列が完全一致。
    このスクリプトは再学習のたびに実行されるため、テスト任せにすると
    メルグラフの退行が素通りする。`sample_wav/` が無い環境では警告を出してスキップする
  - CLI: `python scripts/export_onnx.py --checkpoint models/full/best_infer.pt --out web/public/model/cw.onnx`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_export_onnx.py` を新規作成。

```python
"""ONNX エクスポートの検証: PyTorch 推論と ONNX 推論のトークン列が一致すること."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.export_onnx import OnnxDecoder, export_onnx
from src.infer.engine import InferenceEngine, ctc_greedy_decode_with_frames
from src.train.onnx_mel import ConvMelExtractor

CHECKPOINT = Path("models/full/best_infer.pt")

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="学習済みチェックポイントが無い環境ではスキップ"
)


def _tone_wave(seconds: float = 3.0) -> np.ndarray:
    """CW に近いトーンバースト波形 (決定的)."""
    sr = 8000
    t = np.arange(int(sr * seconds)) / sr
    envelope = (np.sin(2 * np.pi * 4.0 * t) > 0).astype(np.float32)
    return (np.sin(2 * np.pi * 600.0 * t) * envelope).astype(np.float32)


def test_wrapper_matches_pytorch_engine() -> None:
    """OnnxDecoder (PyTorch のまま) の出力が InferenceEngine と一致すること."""
    from src.train.checkpoint import build_model_from_checkpoint

    engine = InferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
    model = build_model_from_checkpoint(CHECKPOINT, map_location=torch.device("cpu"))
    wrapper = OnnxDecoder(model, ConvMelExtractor()).eval()

    wave = _tone_wave()
    expected = [t.token_id for t in engine.decode_chunk(wave)]

    with torch.no_grad():
        log_probs = wrapper(torch.from_numpy(wave).unsqueeze(0))
    got = [t.token_id for t in ctc_greedy_decode_with_frames(log_probs)[0]]

    assert got == expected


def test_exported_onnx_matches_pytorch_engine(tmp_path: Path) -> None:
    """エクスポートした ONNX を onnxruntime で推論し、トークン列が一致すること."""
    import onnxruntime as ort

    out = tmp_path / "cw.onnx"
    export_onnx(CHECKPOINT, out)
    assert out.exists()

    engine = InferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
    wave = _tone_wave()
    expected = [t.token_id for t in engine.decode_chunk(wave)]

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    log_probs = sess.run(None, {"wave": wave[None, :]})[0]
    got = [
        t.token_id
        for t in ctc_greedy_decode_with_frames(torch.from_numpy(log_probs))[0]
    ]

    assert got == expected


def test_exported_onnx_accepts_variable_length(tmp_path: Path) -> None:
    """時間軸が動的軸になっていること (長さ違いで再エクスポート不要)."""
    import onnxruntime as ort

    out = tmp_path / "cw.onnx"
    export_onnx(CHECKPOINT, out)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

    for seconds in (1.0, 5.0, 8.5):
        wave = _tone_wave(seconds)
        log_probs = sess.run(None, {"wave": wave[None, :]})[0]
        assert log_probs.shape[0] == 1
        assert log_probs.shape[2] == 73
        assert log_probs.shape[1] == len(wave) // 80 + 1
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_export_onnx.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_onnx'`

(`scripts/__init__.py` が無い場合は空ファイルを作る。`pyproject.toml` の `pythonpath = ["."]` によりルートからインポートできる)

- [ ] **Step 3: 実装する**

`scripts/export_onnx.py` を新規作成。

**注意点**:
- `model.eval()` を必ず通す (BatchNorm を推論モードに、Dropout を無効に)
- `torch.onnx.export` は torch 2.9 以降 `dynamo=True` が既定になりうる。
  `dynamic_axes` を有効にするため **`dynamo=False` を明示する**
- 末尾に `log_softmax` を入れて JS 側の処理を減らす

```python
"""学習済みチェックポイントを「波形 in → log_probs out」の単一 ONNX に変換する.

メル変換 (ConvMelExtractor) をモデルと同じグラフに焼き込むことで、ブラウザ側で
前処理を再実装する必要がなくなり、数値のズレという失敗モードが消える。

使い方:
    python scripts/export_onnx.py \\
        --checkpoint models/full/best_infer.pt \\
        --out web/public/model/cw.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from src.train.checkpoint import build_model_from_checkpoint
from src.train.model import CWModel
from src.train.onnx_mel import ConvMelExtractor

OPSET_VERSION = 17


class OnnxDecoder(nn.Module):
    """メル変換 + モデル + log_softmax を 1 つにまとめたエクスポート用ラッパ.

    入力: ``wave (1, T_wave)`` — 8 kHz float32
    出力: ``log_probs (1, T_frames, 73)``
    """

    def __init__(self, model: CWModel, mel: ConvMelExtractor) -> None:
        super().__init__()
        self.mel = mel
        self.model = model

    def forward(self, wave: Tensor) -> Tensor:
        mel = self.mel(wave)
        logits = self.model(mel)
        return torch.log_softmax(logits.float(), dim=-1)


def export_onnx(checkpoint: Path, out_path: Path) -> None:
    """``checkpoint`` を読み込み ``out_path`` に ONNX を書き出す."""
    device = torch.device("cpu")
    model = build_model_from_checkpoint(checkpoint, map_location=device)
    wrapper = OnnxDecoder(model, ConvMelExtractor()).to(device)
    # BatchNorm を推論モードにし Dropout を無効化する (これを忘れると精度が壊れる)
    wrapper.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 8000 * 3, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["wave"],
            output_names=["log_probs"],
            dynamic_axes={"wave": {1: "n_samples"}, "log_probs": {1: "n_frames"}},
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            dynamo=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/full/best_infer.pt"))
    parser.add_argument("--out", type=Path, default=Path("web/public/model/cw.onnx"))
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.out)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"書き出し完了: {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストが通ることを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_export_onnx.py -v
```

Expected: 3 passed

`dynamo=False` が未対応の torch バージョンで `TypeError` になった場合は、その引数を外して再実行する (古い torch では TorchScript 経路が既定のため挙動は同じ)。

- [ ] **Step 5: 実際にエクスポートして手元で確認**

```bash
.venv/Scripts/python.exe scripts/export_onnx.py
```

Expected: `web/public/model/cw.onnx` が 17 MB 前後で生成される

- [ ] **Step 6: 生成物を git 管理外にする**

`cw-decorder/.gitignore` に追記 (無ければ作成):

```
web/public/model/*.onnx
web/node_modules/
web/dist/
```

- [ ] **Step 7: コミット**

```bash
git add scripts/export_onnx.py tests/test_export_onnx.py .gitignore
git commit -m "feat: 波形inからlog_probs outの単一ONNXを出力するスクリプトを追加"
```

---

### Task 3: web プロジェクト初期化と ORT Web の実測 (Step 0 の完了地点)

設計書 §12 の Step 0。**ここが本命のリスク**。ORT Web で LSTM が動くか、シングルスレッドで実用速度が出るかを、他を作り込む前に確かめる。

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/index.html`,
  `web/src/bench.ts`, `web/README.md`
- Create: `docs/browser_ort_bench.md` (実測結果の記録)

**Interfaces:**
- Consumes: `web/public/model/cw.onnx` (Task 2 の生成物)
- Produces: なし (計測のみ)。以降のタスクは `onnxruntime-web` の
  `ort.InferenceSession` を前提にする

- [ ] **Step 1: Vite プロジェクトを作る**

`web/package.json`:

```json
{
  "name": "cw-decoder-web",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "onnxruntime-web": "^1.20.0"
  },
  "devDependencies": {
    "typescript": "^5.6.0",
    "vite": "^6.0.0",
    "vitest": "^2.1.0"
  }
}
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable", "WebWorker"],
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noEmit": true,
    "skipLibCheck": true,
    "types": ["vite/client"]
  },
  "include": ["src", "tests"]
}
```

`web/vite.config.ts`:

```ts
import { defineConfig } from 'vite'

export default defineConfig({
  // onnxruntime-web の .wasm を事前バンドルの対象から外す
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: {
    headers: {
      // ORT Web のマルチスレッドを有効にするためのヘッダ。
      // 無くても動く (シングルスレッド) が、あれば速くなる。
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
```

- [ ] **Step 2: 依存をインストール**

```bash
cd web && npm install
```

- [ ] **Step 3: ベンチ用のページとスクリプトを書く**

`web/index.html`:

```html
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>CW デコーダ (ブラウザ版)</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/bench.ts"></script>
  </body>
</html>
```

`web/src/bench.ts`:

```ts
/**
 * ORT Web の実測ベンチ (設計書 §12 Step 0).
 *
 * 確認すること:
 *   (a) ORT Web の WASM で LSTM が動くか
 *   (b) 1 回のデコードが 0.2〜0.45 秒に収まるか (設計書 §3 の見積もり)
 *   (c) スレッド数による差
 */
import * as ort from 'onnxruntime-web'

const SAMPLE_RATE = 8000

/** CW に近いトーンバースト波形を作る (決定的). */
function toneWave(seconds: number): Float32Array {
  const n = Math.floor(SAMPLE_RATE * seconds)
  const wave = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE
    const envelope = Math.sin(2 * Math.PI * 4.0 * t) > 0 ? 1 : 0
    wave[i] = Math.sin(2 * Math.PI * 600.0 * t) * envelope
  }
  return wave
}

async function benchmark(threads: number): Promise<void> {
  ort.env.wasm.numThreads = threads
  ort.env.wasm.simd = true

  const t0 = performance.now()
  const session = await ort.InferenceSession.create('/model/cw.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  log(`threads=${threads} セッション構築: ${(performance.now() - t0).toFixed(0)} ms`)

  for (const seconds of [5.0, 8.5]) {
    const wave = toneWave(seconds)
    const feed = { wave: new ort.Tensor('float32', wave, [1, wave.length]) }

    // ウォームアップ
    await session.run(feed)

    const runs = 5
    const start = performance.now()
    let frames = 0
    for (let i = 0; i < runs; i++) {
      const out = await session.run(feed)
      frames = out.log_probs.dims[1]
    }
    const ms = (performance.now() - start) / runs
    log(
      `threads=${threads} 音声 ${seconds}s → ${ms.toFixed(1)} ms ` +
        `(RTF=${(ms / 1000 / seconds).toFixed(3)}, frames=${frames})`
    )
  }
  await session.release()
}

function log(message: string): void {
  console.log(message)
  const el = document.getElementById('app')
  if (el) el.innerHTML += `<pre>${message}</pre>`
}

async function main(): Promise<void> {
  log(`crossOriginIsolated = ${globalThis.crossOriginIsolated}`)
  log(`hardwareConcurrency = ${navigator.hardwareConcurrency}`)
  await benchmark(1)
  if (globalThis.crossOriginIsolated) {
    await benchmark(4)
  } else {
    log('crossOriginIsolated=false のためマルチスレッドは計測できません')
  }
}

void main()
```

- [ ] **Step 4: ベンチを実行する**

```bash
cd web && npm run dev
```

ブラウザで表示された URL を開き、ページに出た数値と devtools のコンソール出力を読む。

- [ ] **Step 5: 結果を記録する**

`docs/browser_ort_bench.md` を新規作成し、以下を埋める。数値は Step 4 の実測値。

```markdown
# ORT Web 実測結果 (Step 0)

計測日: 2026-08-05
ブラウザ: (実際に使ったブラウザとバージョン)
CPU: (機種)
モデル: models/full/best_infer.pt → web/public/model/cw.onnx

## 比較対象 (設計書 §3)

| 条件 | 8.5 秒の音声 | RTF |
|---|---|---|
| PyTorch CPU 4 スレッド | 25.6 ms | 0.003 |
| PyTorch CPU 1 スレッド | 72.3 ms | 0.009 |

## 実測

| 条件 | 5.0 秒 | 8.5 秒 | RTF (8.5s) |
|---|---|---|---|
| ORT Web wasm 1 スレッド | | | |
| ORT Web wasm 4 スレッド | | | |

セッション構築時間: ms
crossOriginIsolated:

## 判定

- LSTM は動いたか:
- 8.5 秒のデコードが hop 1.0 秒に対して余裕があるか:
- hop 0.5 秒にできるか (デコード時間 < 0.35 秒か):

## 判定に基づく決定

- `hop_s` の採用値:
- `decode_left_context_s` の採用値:
- int8 量子化の要否:
```

**判定が赤だった場合** (8.5 秒のデコードが 0.8 秒を超える場合) は、ここで止めて次の順に手を打つ:
1. `decode_left_context_s` を 5.0 → 3.0 に下げる (デコード区間が 8.5 → 6.5 秒相当に減る)
2. `hop_s` を 1.0 → 1.5 に延ばす
3. それでも足りなければ int8 動的量子化 (`onnxruntime.quantization.quantize_dynamic`) を試し、
   Task 2 のトークン列一致テストで劣化を確認する

- [ ] **Step 6: コミット**

```bash
git add web/package.json web/package-lock.json web/tsconfig.json web/vite.config.ts \
        web/index.html web/src/bench.ts docs/browser_ort_bench.md
git commit -m "feat: webプロジェクトを初期化しORT Webの実測ベンチを追加"
```

---

### Task 4: `commit_lag` の掃引 (Python のみ、Task 3 と並行可能)

設計書 §6.2。確定遅延 2.5 秒に実測の裏付けを与え、下げられるなら下げる。デスクトップ版も同時に速くなる。

**Files:**
- Create: `scripts/sweep_commit_lag.py`
- Test: `tests/test_sweep_commit_lag.py`
- Create: `docs/commit_lag_sweep_result.md` (実行後に結果を記録)

**Interfaces:**
- Consumes: `src.infer.sliding_window.SlidingWindowDecoder`, `src.infer.engine.InferenceEngine`
- Produces:
  - `@dataclass(frozen=True) class SweepRow` — `commit_lag_s: float`, `n_files: int`,
    `n_reference_tokens: int`, `match_rate: float`, `cer: float`
  - `def simulate_commit(engine, wave, commit_lag_s, *, hop_s=1.0, ...) -> list[int]`
    — 音声を `hop_s` ずつ投入して確定したトークン ID 列を返す
  - `def token_cer(reference: list[int], hypothesis: list[int]) -> float`
  - `def sweep(engine, waves, lags, ...) -> list[SweepRow]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_sweep_commit_lag.py` を新規作成。モデル無しで回せるよう、エンジンは差し替え可能にする。

```python
"""commit_lag 掃引スクリプトの単体テスト (モデル不要)."""
from __future__ import annotations

import numpy as np

from scripts.sweep_commit_lag import SweepRow, simulate_commit, sweep, token_cer
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


def test_smaller_lag_commits_at_least_as_many() -> None:
    """commit_lag を小さくすると確定数は減らない (末尾がより多く確定する)."""
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
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_sweep_commit_lag.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.sweep_commit_lag'`

- [ ] **Step 3: 実装する**

`scripts/sweep_commit_lag.py` を新規作成。

```python
"""commit_lag_s の掃引.

確定 (committed) は不変な設計のため、``commit_lag_s`` を短くしすぎると
誤りが永久に残る。既定の 2.5 秒は余裕を見た値で実測の裏付けが無いため、
「右文脈が最大限ある状態 (finalize) の結果」を参照として、各 commit_lag で
確定したトークン列がどれだけ一致するかを測る。

比較は**文字ではなくトークン ID 列**で行う (変換表の影響を混ぜない)。

使い方:
    python scripts/sweep_commit_lag.py --checkpoint models/full/best_infer.pt \\
        --audio-dir data/keying/val --out docs/commit_lag_sweep.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.infer.engine import FrameToken
from src.infer.sliding_window import SlidingWindowDecoder

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
    n_reference_tokens: int
    match_rate: float      # 参照と完全一致したファイルの割合
    cer: float             # トークン単位の編集距離 / 参照長


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
) -> list[int]:
    """波形を hop ごとに投入し、確定したトークン ID 列を返す.

    ``finalize=True`` のときは末尾で ``finalize()`` を呼び、暫定を全部確定させる
    (これが掃引の参照になる).
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
    return [t.token_id for t in view.committed]


def sweep(
    engine: ChunkEngine,
    waves: list[np.ndarray],
    lags: list[float] | tuple[float, ...] = DEFAULT_LAGS,
    *,
    hop_s: float = 1.0,
) -> list[SweepRow]:
    """各 commit_lag について、参照 (finalize 結果) との一致を集計."""
    references = [
        simulate_commit(engine, w, commit_lag_s=0.0, hop_s=hop_s, finalize=True)
        for w in waves
    ]
    n_ref_tokens = sum(len(r) for r in references)

    rows: list[SweepRow] = []
    for lag in lags:
        matches = 0
        total_edits = 0.0
        for wave, ref in zip(waves, references, strict=True):
            got = simulate_commit(engine, wave, commit_lag_s=lag, hop_s=hop_s)
            if got == ref:
                matches += 1
            total_edits += token_cer(ref, got) * max(1, len(ref))
        rows.append(SweepRow(
            commit_lag_s=lag,
            n_files=len(waves),
            n_reference_tokens=n_ref_tokens,
            match_rate=matches / len(waves) if waves else 0.0,
            cer=total_edits / n_ref_tokens if n_ref_tokens else 0.0,
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

    print(f"{'lag(s)':>8} {'一致率':>8} {'CER':>8}")
    for row in rows:
        print(f"{row.commit_lag_s:8.2f} {row.match_rate:8.3f} {row.cer:8.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストが通ることを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_sweep_commit_lag.py -v
```

Expected: 6 passed

- [ ] **Step 5: 実データで掃引を走らせる**

keyed_val の既定パスは `data/real/val` (`scripts/eval_model.py` の `--keyed-dir` と同じ)。
このリポジトリでは未作成のことがあるため、無ければ `data/real/train` を使う。

```bash
ls data/real/
# val があればそちら、無ければ train
.venv/Scripts/python.exe scripts/sweep_commit_lag.py --audio-dir data/real/train --limit 20
```

- [ ] **Step 6: 結果を記録する**

`docs/commit_lag_sweep_result.md` を新規作成。`docs/word_break_threshold_result.md` と同じ形式で、
採用値と**採用しなかった場合はその理由**を書く。下げられないという結論もそのまま記録すること。

```markdown
# commit_lag 掃引 結果

実施日: 2026-08-05
対象: (使った音声ディレクトリとファイル数)
参照: finalize() の確定結果 (右文脈が最大限ある状態)
比較単位: トークン ID 列 (文字に変換する前)

## 結果

| commit_lag (s) | 一致率 | CER |
|---|---|---|
| 0.50 | | |
| 0.75 | | |
| 1.00 | | |
| 1.50 | | |
| 2.00 | | |
| 2.50 | | |

## 判断

採用値:
理由:
```

- [ ] **Step 7: コミット**

```bash
git add scripts/sweep_commit_lag.py tests/test_sweep_commit_lag.py \
        docs/commit_lag_sweep_result.md docs/commit_lag_sweep.json
git commit -m "feat: commit_lagの掃引スクリプトを追加し実測結果を記録"
```

---

### Task 5: 符号表の TypeScript 生成

`morse_tokens.py` を唯一の真正なソースとしたまま、TypeScript 側で符号表を使えるようにする。

**Files:**
- Create: `scripts/export_tokens.py`
- Create: `web/src/generated/tokens.ts` (生成物だが git 管理する)
- Test: `tests/test_export_tokens.py`

**Interfaces:**
- Consumes: `src.tokens.morse_tokens`
- Produces (TypeScript 側):
  - `export const BLANK_TOKEN_ID: number`
  - `export const WORD_BREAK_TOKEN_ID: number`
  - `export const VOCAB_SIZE: number`
  - `export const DAKUTEN_CHAR: string`
  - `export const HANDAKUTEN_CHAR: string`
  - `export const ID_TO_CODE: readonly string[]` — 添字が token id
  - `export const EUROPEAN_TABLE: Readonly<Record<string, string>>`
  - `export const JAPANESE_TABLE: Readonly<Record<string, string>>`
  - `export const DAKUTEN_COMPOSE: Readonly<Record<string, string>>`
  - `export const HANDAKUTEN_COMPOSE: Readonly<Record<string, string>>`
- Produces (Python 側): `def render_tokens_ts() -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_export_tokens.py` を新規作成。**コミット済みファイルとの差分検出**が主目的。

```python
"""符号表の TypeScript 生成の検証."""
from __future__ import annotations

from pathlib import Path

from scripts.export_tokens import OUTPUT_PATH, render_tokens_ts
from src.tokens.morse_tokens import (
    EUROPEAN_TABLE,
    ID_TO_TOKEN,
    JAPANESE_TABLE,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
)


def test_contains_all_ids() -> None:
    """全トークン ID が ID_TO_CODE に現れること."""
    ts = render_tokens_ts()
    for token in ID_TO_TOKEN.values():
        assert token.code in ts, f"符号 {token.code!r} が生成物に含まれていない"


def test_contains_constants() -> None:
    ts = render_tokens_ts()
    assert f"export const VOCAB_SIZE = {VOCAB_SIZE}" in ts
    assert f"export const WORD_BREAK_TOKEN_ID = {WORD_BREAK_TOKEN_ID}" in ts


def test_marks_generated_and_forbids_edit() -> None:
    """手編集を禁じる注意書きが入っていること."""
    ts = render_tokens_ts()
    assert "自動生成" in ts
    assert "scripts/export_tokens.py" in ts


def test_table_entry_counts() -> None:
    """表のエントリ数が Python 側と一致すること."""
    ts = render_tokens_ts()
    european_block = ts.split("EUROPEAN_TABLE")[1].split("}")[0]
    japanese_block = ts.split("JAPANESE_TABLE")[1].split("}")[0]
    assert european_block.count(":") == len(EUROPEAN_TABLE)
    assert japanese_block.count(":") == len(JAPANESE_TABLE)


def test_committed_file_is_up_to_date() -> None:
    """コミット済みの tokens.ts が最新の生成結果と一致すること.

    符号表を変更したのに再生成を忘れると、Python と TypeScript で
    符号が食い違う。それをこのテストで検出する.
    """
    path = Path(OUTPUT_PATH)
    assert path.exists(), f"{path} が無い。scripts/export_tokens.py を実行すること"
    assert path.read_text(encoding="utf-8") == render_tokens_ts(), (
        "tokens.ts が古い。scripts/export_tokens.py を再実行してコミットすること"
    )
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_export_tokens.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_tokens'`

- [ ] **Step 3: 実装する**

`scripts/export_tokens.py` を新規作成。

```python
"""符号表を TypeScript として生成する.

``src/tokens/morse_tokens.py`` を唯一の真正なソースとする原則を守るため、
TypeScript 側に符号表を手で書かず、ここから生成する。和文符号は実装誤記が
起きやすく、二重定義を作れば必ず事故る。

使い方:
    python scripts/export_tokens.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.tokens.morse_tokens import (
    BLANK_TOKEN_ID,
    DAKUTEN_CHAR,
    DAKUTEN_COMPOSE,
    EUROPEAN_TABLE,
    HANDAKUTEN_CHAR,
    HANDAKUTEN_COMPOSE,
    ID_TO_TOKEN,
    JAPANESE_TABLE,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
)

OUTPUT_PATH = "web/src/generated/tokens.ts"

HEADER = """/**
 * 符号表 (自動生成 — 手で編集しないこと)
 *
 * 生成元: src/tokens/morse_tokens.py
 * 生成方法: python scripts/export_tokens.py
 *
 * 符号定義の唯一の真正なソースは Python 側です。ここを直接書き換えても
 * 次の再生成で失われ、Python 側とずれた状態は tests/test_export_tokens.py
 * が検出します。
 *
 * 符号表記: ドット = ・ (U+30FB 中黒), ダッシュ = - (U+002D ハイフン)
 */
"""


def _js_string(value: str) -> str:
    """JSON 文字列として安全に出力 (中黒などの非 ASCII はそのまま残す)."""
    return json.dumps(value, ensure_ascii=False)


def _render_record(name: str, table: dict[str, str], comment: str) -> str:
    lines = [f"/** {comment} */", f"export const {name}: Readonly<Record<string, string>> = {{"]
    for code, char in table.items():
        lines.append(f"  {_js_string(code)}: {_js_string(char)},")
    lines.append("}")
    return "\n".join(lines)


def render_tokens_ts() -> str:
    """生成する TypeScript の全文を返す."""
    max_id = max(ID_TO_TOKEN)
    id_to_code = [ID_TO_TOKEN[i].code for i in range(max_id + 1)]

    parts: list[str] = [HEADER]
    parts.append(f"export const BLANK_TOKEN_ID = {BLANK_TOKEN_ID}")
    parts.append(f"export const WORD_BREAK_TOKEN_ID = {WORD_BREAK_TOKEN_ID}")
    parts.append(f"export const VOCAB_SIZE = {VOCAB_SIZE}")
    parts.append(f"export const DAKUTEN_CHAR = {_js_string(DAKUTEN_CHAR)}")
    parts.append(f"export const HANDAKUTEN_CHAR = {_js_string(HANDAKUTEN_CHAR)}")
    parts.append("")
    parts.append("/** 添字が token id。0 は CTC blank、末尾は語間 WORDBREAK。 */")
    parts.append("export const ID_TO_CODE: readonly string[] = [")
    for code in id_to_code:
        parts.append(f"  {_js_string(code)},")
    parts.append("]")
    parts.append("")
    parts.append(_render_record("EUROPEAN_TABLE", EUROPEAN_TABLE, "欧文符号表 (符号 → 表示文字)"))
    parts.append("")
    parts.append(_render_record("JAPANESE_TABLE", JAPANESE_TABLE, "和文符号表 (符号 → 表示文字)"))
    parts.append("")
    parts.append(_render_record("DAKUTEN_COMPOSE", DAKUTEN_COMPOSE, "プレーンカナ + 濁点 → 合成カナ"))
    parts.append("")
    parts.append(_render_record("HANDAKUTEN_COMPOSE", HANDAKUTEN_COMPOSE, "プレーンカナ + 半濁点 → 合成カナ"))
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    path = Path(OUTPUT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 改行は LF 固定 (newline="" で Python 側の変換を抑止)
    path.write_text(render_tokens_ts(), encoding="utf-8", newline="")
    print(f"書き出し: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 生成してからテストを通す**

```bash
.venv/Scripts/python.exe scripts/export_tokens.py
.venv/Scripts/python.exe -m pytest tests/test_export_tokens.py -v
```

Expected: 5 passed

- [ ] **Step 5: コミット**

```bash
git add scripts/export_tokens.py tests/test_export_tokens.py web/src/generated/tokens.ts
git commit -m "feat: 符号表をmorse_tokens.pyからTypeScriptへ生成する仕組みを追加"
```

---

### Task 6: ゴールデン fixture の出力

ブラウザ側のデコード結果を Python と突き合わせるための固定データを出す。**リサンプルを検証対象から切り離すため、8 kHz mono に変換済みの波形を fixture にする。**

**Files:**
- Create: `scripts/export_golden.py`
- Create: `web/tests/fixtures/*.f32`, `web/tests/fixtures/golden.json` (生成物、git 管理する)
- Test: `tests/test_export_golden.py`

**Interfaces:**
- Consumes: `src.infer.engine.InferenceEngine`, `src.tokens.converter.TokenConverter`
- Produces:
  - `def load_wave_8k(path: Path, max_seconds: float | None) -> np.ndarray`
  - `def build_golden(engine, name, wave) -> dict[str, object]`
  - `golden.json` の 1 エントリ:
    `{"name": str, "waveFile": str, "nSamples": int, "tokenIds": list[int],
      "confidences": list[float], "textEuropean": str, "textJapanese": str}`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_export_golden.py` を新規作成。

```python
"""ゴールデン fixture 出力の検証."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.export_golden import build_golden, load_wave_8k
from src.infer.engine import FrameToken

SAMPLE = Path("sample_wav/oubun.wav")


class FakeEngine:
    """固定のトークン列を返すダミーエンジン."""

    frame_hop_samples = 80

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]:
        # 欧文 "E" (・) と "T" (-) に相当する符号のトークン ID は
        # morse_tokens 側で決まるため、ここでは 1 と 2 を使う (形の検証のみ)
        return [
            FrameToken(token_id=1, confidence=0.95, frame_start=0, frame_end=3),
            FrameToken(token_id=2, confidence=0.90, frame_start=10, frame_end=13),
        ]


@pytest.mark.skipif(not SAMPLE.exists(), reason="サンプル音声が無い環境ではスキップ")
def test_load_wave_8k_resamples_and_monos() -> None:
    wave = load_wave_8k(SAMPLE, max_seconds=5.0)
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert 8000 * 4 < wave.size <= 8000 * 5 + 8000


def test_build_golden_shape() -> None:
    wave = np.zeros(8000, dtype=np.float32)
    entry = build_golden(FakeEngine(), "dummy", wave)
    assert entry["name"] == "dummy"
    assert entry["waveFile"] == "dummy.f32"
    assert entry["nSamples"] == 8000
    assert entry["tokenIds"] == [1, 2]
    assert len(entry["confidences"]) == 2
    assert isinstance(entry["textEuropean"], str)
    assert isinstance(entry["textJapanese"], str)


def test_golden_json_is_valid_if_present() -> None:
    """生成済みの golden.json が壊れていないこと."""
    path = Path("web/tests/fixtures/golden.json")
    if not path.exists():
        pytest.skip("未生成 (scripts/export_golden.py を実行すること)")
    entries = json.loads(path.read_text(encoding="utf-8"))
    assert entries, "エントリが空"
    for entry in entries:
        wave_path = path.parent / str(entry["waveFile"])
        assert wave_path.exists(), f"{wave_path} が無い"
        assert wave_path.stat().st_size == int(entry["nSamples"]) * 4
        assert len(entry["tokenIds"]) == len(entry["confidences"])
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
.venv/Scripts/python.exe -m pytest tests/test_export_golden.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_golden'`

- [ ] **Step 3: 実装する**

`scripts/export_golden.py` を新規作成。

```python
"""ブラウザ側の検証用ゴールデン fixture を出力する.

移植で最大のリスクは「動くが微妙に精度が落ちている」ことなので、Python の
推論結果を固定データとして出し、ブラウザ側で完全一致を確認できるようにする。

波形は **8 kHz mono float32 に変換済み**のものを出す。リサンプルを検証対象から
切り離すことで、不一致が出たときに「リサンプルのせいか、デコードのせいか」で
悩まずに済む。

使い方:
    python scripts/export_golden.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from src.infer.engine import FrameToken
from src.tokens.converter import TokenConverter


class ChunkEngine(Protocol):
    """``InferenceEngine`` 互換の最小インタフェース (テストで差し替えるため)."""

    def decode_chunk(self, waveform: np.ndarray) -> list[FrameToken]: ...


SAMPLE_SOURCES: tuple[tuple[str, str], ...] = (
    ("oubun", "sample_wav/oubun.wav"),
    ("wabun", "sample_wav/wabun.wav"),
)
MAX_SECONDS = 30.0
OUTPUT_DIR = Path("web/tests/fixtures")


def load_wave_8k(path: Path, max_seconds: float | None = None) -> np.ndarray:
    """音声を 8 kHz mono float32 で読み込む.

    ステートフルな soxr を使う (ステートレスなリサンプルはブロック境界に歪みを
    生じるが、ここは一括変換なので境界問題は起きない).
    """
    import soundfile as sf
    import soxr

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data[:, 0]
    if sr != 8000:
        mono = soxr.resample(mono, sr, 8000).astype(np.float32)
    if max_seconds is not None:
        mono = mono[: int(max_seconds * 8000)]
    return np.ascontiguousarray(mono, dtype=np.float32)


def build_golden(engine: ChunkEngine, name: str, wave: np.ndarray) -> dict[str, object]:
    """1 ファイル分のゴールデンエントリを作る."""
    tokens: list[FrameToken] = engine.decode_chunk(wave)
    token_ids = [t.token_id for t in tokens]
    confidences = [round(float(t.confidence), 6) for t in tokens]

    european = TokenConverter("european").convert(token_ids, confidences)
    japanese = TokenConverter("japanese").convert(token_ids, confidences)

    return {
        "name": name,
        "waveFile": f"{name}.f32",
        "nSamples": int(wave.size),
        "tokenIds": token_ids,
        "confidences": confidences,
        "textEuropean": european.text,
        "textJapanese": japanese.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/full/best_infer.pt"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    from src.infer.engine import InferenceEngine

    engine = InferenceEngine.from_checkpoint(args.checkpoint, device="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for name, source in SAMPLE_SOURCES:
        path = Path(source)
        if not path.exists():
            print(f"スキップ (見つからない): {path}")
            continue
        wave = load_wave_8k(path, MAX_SECONDS)
        (args.out_dir / f"{name}.f32").write_bytes(wave.tobytes())
        entry = build_golden(engine, name, wave)
        entries.append(entry)
        print(f"{name}: {len(entry['tokenIds'])} トークン, {wave.size} サンプル")

    out_json = args.out_dir / "golden.json"
    out_json.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )
    print(f"書き出し: {out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 生成してからテストを通す**

```bash
.venv/Scripts/python.exe scripts/export_golden.py
.venv/Scripts/python.exe -m pytest tests/test_export_golden.py -v
```

Expected: 3 passed

- [ ] **Step 5: コミット**

```bash
git add scripts/export_golden.py tests/test_export_golden.py web/tests/fixtures/
git commit -m "feat: ブラウザ検証用のゴールデンfixtureを出力するスクリプトを追加"
```

---

### Task 7: CTC greedy デコード (TypeScript)

**Files:**
- Create: `web/src/decode/ctc.ts`
- Test: `web/tests/ctc.test.ts`
- Modify: `web/package.json` (vitest 設定は不要、既定で `tests/**/*.test.ts` を拾う)

**Interfaces:**
- Consumes: `web/src/generated/tokens.ts` の `BLANK_TOKEN_ID`
- Produces:
  - `export interface FrameToken { tokenId: number; confidence: number; frameStart: number; frameEnd: number }`
  - `export function ctcGreedyDecodeWithFrames(logProbs: Float32Array, nFrames: number, vocabSize: number, blankId?: number): FrameToken[]`

- [ ] **Step 1: 失敗するテストを書く**

`web/tests/ctc.test.ts` を新規作成。

```ts
import { describe, expect, it } from 'vitest'
import { ctcGreedyDecodeWithFrames } from '../src/decode/ctc'

const VOCAB = 4
const BLANK = 0

/** フレームごとの「最大確率のトークン」から log_probs を組み立てる補助. */
function buildLogProbs(frames: Array<[number, number]>): Float32Array {
  const out = new Float32Array(frames.length * VOCAB).fill(Math.log(0.01))
  frames.forEach(([tokenId, prob], t) => {
    out[t * VOCAB + tokenId] = Math.log(prob)
  })
  return out
}

describe('ctcGreedyDecodeWithFrames', () => {
  it('blank だけなら何も出さない', () => {
    const logProbs = buildLogProbs([[BLANK, 0.9], [BLANK, 0.9]])
    expect(ctcGreedyDecodeWithFrames(logProbs, 2, VOCAB)).toEqual([])
  })

  it('連続する同一トークンを 1 つに畳む', () => {
    const logProbs = buildLogProbs([[1, 0.8], [1, 0.9], [1, 0.7]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB)
    expect(out).toHaveLength(1)
    expect(out[0].tokenId).toBe(1)
    expect(out[0].frameStart).toBe(0)
    expect(out[0].frameEnd).toBe(2)
    // ラン内の最大確率を confidence とする
    expect(out[0].confidence).toBeCloseTo(0.9, 5)
  })

  it('blank を挟んだ同一トークンは 2 つに分かれる', () => {
    const logProbs = buildLogProbs([[1, 0.8], [BLANK, 0.9], [1, 0.7]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB)
    expect(out.map((t) => t.tokenId)).toEqual([1, 1])
    expect(out[0].frameEnd).toBe(0)
    expect(out[1].frameStart).toBe(2)
  })

  it('末尾のトークンも取りこぼさない', () => {
    const logProbs = buildLogProbs([[BLANK, 0.9], [2, 0.8]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 2, VOCAB)
    expect(out.map((t) => t.tokenId)).toEqual([2])
    expect(out[0].frameEnd).toBe(1)
  })

  it('異なるトークンが並ぶ場合は境界を正しく取る', () => {
    const logProbs = buildLogProbs([[1, 0.8], [2, 0.9], [3, 0.7]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB)
    expect(out.map((t) => t.tokenId)).toEqual([1, 2, 3])
    expect(out.map((t) => t.frameStart)).toEqual([0, 1, 2])
    expect(out.map((t) => t.frameEnd)).toEqual([0, 1, 2])
  })

  it('フレーム 0 件なら空', () => {
    expect(ctcGreedyDecodeWithFrames(new Float32Array(0), 0, VOCAB)).toEqual([])
  })
})
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd web && npx vitest run tests/ctc.test.ts
```

Expected: FAIL — `Failed to resolve import "../src/decode/ctc"`

- [ ] **Step 3: 実装する**

`web/src/decode/ctc.ts` を新規作成。`src/infer/engine.py` の
`ctc_greedy_decode_with_frames` の移植。

```ts
/**
 * CTC greedy デコード (フレーム位置付き).
 *
 * 移植元: src/infer/engine.py の ctc_greedy_decode_with_frames
 *
 * フレーム位置を残すのは、スライディングウィンドウで重複区間を取り除き、
 * 確定/暫定の境界を絶対時刻で判定するため。
 */
import { BLANK_TOKEN_ID } from '../generated/tokens'

/** 1 つのデコード済みトークン。frameStart〜frameEnd は inclusive。 */
export interface FrameToken {
  tokenId: number
  confidence: number
  frameStart: number
  frameEnd: number
}

/**
 * log_softmax 済みの `(nFrames, vocabSize)` を平坦化した配列からトークン列を作る。
 *
 * @param logProbs 長さ nFrames * vocabSize の平坦配列 (行優先)
 * @param nFrames 時間フレーム数
 * @param vocabSize 語彙サイズ
 * @param blankId CTC blank の token id
 */
export function ctcGreedyDecodeWithFrames(
  logProbs: Float32Array,
  nFrames: number,
  vocabSize: number,
  blankId: number = BLANK_TOKEN_ID,
): FrameToken[] {
  const out: FrameToken[] = []
  let prev = -1
  let runStart = 0
  let runMax = 0

  for (let t = 0; t < nFrames; t++) {
    const base = t * vocabSize
    let best = 0
    let bestLogProb = logProbs[base]
    for (let v = 1; v < vocabSize; v++) {
      const value = logProbs[base + v]
      if (value > bestLogProb) {
        bestLogProb = value
        best = v
      }
    }
    const conf = Math.exp(bestLogProb)

    if (best === prev) {
      if (best !== blankId && conf > runMax) runMax = conf
      continue
    }
    // 切り替わり: 直前のランを確定する
    if (prev !== -1 && prev !== blankId) {
      out.push({ tokenId: prev, confidence: runMax, frameStart: runStart, frameEnd: t - 1 })
    }
    prev = best
    runStart = t
    runMax = conf
  }

  if (prev !== -1 && prev !== blankId) {
    out.push({ tokenId: prev, confidence: runMax, frameStart: runStart, frameEnd: nFrames - 1 })
  }
  return out
}
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cd web && npx vitest run tests/ctc.test.ts
```

Expected: 6 passed

- [ ] **Step 5: コミット**

```bash
git add web/src/decode/ctc.ts web/tests/ctc.test.ts
git commit -m "feat: CTC greedyデコードをTypeScriptへ移植"
```

---

### Task 8: 符号変換器 (TypeScript)

`src/tokens/converter.py` の `convert()` を固定モード (欧文 / 和文) 限定で移植する。auto モードは初版のスコープ外なので、モード切替の分岐は移植しない。

**Files:**
- Create: `web/src/tokens/converter.ts`
- Test: `web/tests/converter.test.ts`

**Interfaces:**
- Consumes: `web/src/generated/tokens.ts`
- Produces:
  - `export type Mode = 'european' | 'japanese'`
  - `export type FallbackKind = 'TABLE_MISS' | 'LOW_CONFIDENCE'`
  - `export interface FallbackEvent { position: number; inputIndex: number; tokenId: number; code: string; kind: FallbackKind; confidence: number }`
  - `export interface ConvertResult { text: string; fallbackLog: FallbackEvent[] }`
  - `export interface ConvertOptions { mode: Mode; confidenceThreshold?: number; keepLeadingSpace?: boolean }`
  - `export function convert(tokenIds: readonly number[], confidences: readonly number[] | null, options: ConvertOptions): ConvertResult`

- [ ] **Step 1: 失敗するテストを書く**

`web/tests/converter.test.ts` を新規作成。ケースは Python 側の `tests/test_converter.py` と揃える。

```ts
import { describe, expect, it } from 'vitest'
import { convert } from '../src/tokens/converter'
import {
  EUROPEAN_TABLE,
  JAPANESE_TABLE,
  WORD_BREAK_TOKEN_ID,
  ID_TO_CODE,
  BLANK_TOKEN_ID,
} from '../src/generated/tokens'

/** 符号から token id を引く (テスト用の逆引き). */
function idOf(code: string): number {
  const id = ID_TO_CODE.indexOf(code)
  if (id < 0) throw new Error(`符号が見つからない: ${code}`)
  return id
}

describe('convert (欧文)', () => {
  it('基本的な符号を文字にする', () => {
    const ids = [idOf('・'), idOf('-'), idOf('・-')]
    const result = convert(ids, null, { mode: 'european' })
    expect(result.text).toBe('ETA')
  })

  it('WORD_BREAK を空白にする', () => {
    const ids = [idOf('・'), WORD_BREAK_TOKEN_ID, idOf('-')]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E T')
  })

  it('連続する WORD_BREAK は 1 つにまとめる', () => {
    const ids = [idOf('・'), WORD_BREAK_TOKEN_ID, WORD_BREAK_TOKEN_ID, idOf('-')]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E T')
  })

  it('先頭の WORD_BREAK は既定で捨てる', () => {
    const ids = [WORD_BREAK_TOKEN_ID, idOf('・')]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E')
  })

  it('keepLeadingSpace で先頭の空白を残す', () => {
    const ids = [WORD_BREAK_TOKEN_ID, idOf('・')]
    const result = convert(ids, null, { mode: 'european', keepLeadingSpace: true })
    expect(result.text).toBe(' E')
  })

  it('blank は無視する', () => {
    const ids = [BLANK_TOKEN_ID, idOf('・'), BLANK_TOKEN_ID]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E')
  })

  it('確信度が閾値未満なら ? にして LOW_CONFIDENCE を記録する', () => {
    const ids = [idOf('・'), idOf('-')]
    const result = convert(ids, [0.9, 0.2], { mode: 'european', confidenceThreshold: 0.5 })
    expect(result.text).toBe('E?')
    expect(result.fallbackLog).toHaveLength(1)
    expect(result.fallbackLog[0].kind).toBe('LOW_CONFIDENCE')
    expect(result.fallbackLog[0].position).toBe(1)
  })

  it('欧文表に無い和文符号は ? にして TABLE_MISS を記録する', () => {
    // コ (----) は和文表のみ
    const ids = [idOf('----')]
    expect(EUROPEAN_TABLE['----']).toBeUndefined()
    const result = convert(ids, null, { mode: 'european' })
    expect(result.text).toBe('?')
    expect(result.fallbackLog[0].kind).toBe('TABLE_MISS')
  })
})

describe('convert (和文)', () => {
  it('カナに変換する', () => {
    const ids = [idOf('・-'), idOf('・-・-')]
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('イロ')
  })

  it('濁点を直前のカナと合成する', () => {
    // カ (・-・・) + 濁点 (・・) → ガ
    const ids = [idOf('・-・・'), idOf('・・')]
    expect(JAPANESE_TABLE['・・']).toBe('゛')
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('ガ')
  })

  it('半濁点を直前のカナと合成する', () => {
    // ハ (-・・・) + 半濁点 (・・--・) → パ
    const ids = [idOf('-・・・'), idOf('・・--・')]
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('パ')
  })

  it('合成できない位置の濁点は ? にする', () => {
    // イ は濁点と合成できない
    const ids = [idOf('・-'), idOf('・・')]
    const result = convert(ids, null, { mode: 'japanese' })
    expect(result.text).toBe('イ?')
    expect(result.fallbackLog[0].kind).toBe('TABLE_MISS')
  })

  it('先頭の濁点は ? にする', () => {
    const result = convert([idOf('・・')], null, { mode: 'japanese' })
    expect(result.text).toBe('?')
  })

  it('濁点を 2 つ続けても 2 個目は合成しない', () => {
    const ids = [idOf('・-・・'), idOf('・・'), idOf('・・')]
    const result = convert(ids, null, { mode: 'japanese' })
    expect(result.text).toBe('ガ?')
  })

  it('空白のあとの濁点は合成しない', () => {
    const ids = [idOf('・-・・'), WORD_BREAK_TOKEN_ID, idOf('・・')]
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('カ ?')
  })
})

describe('convert (共通)', () => {
  it('空入力は空文字', () => {
    expect(convert([], null, { mode: 'european' }).text).toBe('')
  })

  it('confidences の長さが違えば例外', () => {
    expect(() => convert([1, 2], [0.5], { mode: 'european' })).toThrow()
  })

  it('未知の token id は ? にする', () => {
    const result = convert([9999], null, { mode: 'european' })
    expect(result.text).toBe('?')
    expect(result.fallbackLog[0].code).toBe('<UNKNOWN>')
  })
})
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd web && npx vitest run tests/converter.test.ts
```

Expected: FAIL — `Failed to resolve import "../src/tokens/converter"`

- [ ] **Step 3: 実装する**

`web/src/tokens/converter.ts` を新規作成。

```ts
/**
 * 符号トークン列 → 表示テキスト変換器.
 *
 * 移植元: src/tokens/converter.py の TokenConverter.convert
 *
 * 初版は固定モード (欧文 / 和文) のみを扱う。ホレ/ラタによる自動モード切替は
 * デスクトップ版でも実信号で未解決のためスコープ外 (設計書 §1)。
 *
 * 「?」は 2 種類を区別してログに残す:
 *   TABLE_MISS     — 変換表に該当なし (欧文モード中の和文符号など)
 *   LOW_CONFIDENCE — CTC 事後確率が閾値未満
 */
import {
  DAKUTEN_CHAR,
  DAKUTEN_COMPOSE,
  EUROPEAN_TABLE,
  HANDAKUTEN_CHAR,
  HANDAKUTEN_COMPOSE,
  ID_TO_CODE,
  JAPANESE_TABLE,
  BLANK_TOKEN_ID,
  WORD_BREAK_TOKEN_ID,
} from '../generated/tokens'

export const FALLBACK_CHAR = '?'

export type Mode = 'european' | 'japanese'
export type FallbackKind = 'TABLE_MISS' | 'LOW_CONFIDENCE'

/** 「?」を出したイベント (デバッグ用ログ). */
export interface FallbackEvent {
  /** 最終テキストにおける ? の位置 */
  position: number
  /** 入力 tokenIds のインデックス */
  inputIndex: number
  tokenId: number
  /** 符号、または未知 id のとき '<UNKNOWN>' */
  code: string
  kind: FallbackKind
  confidence: number
}

export interface ConvertResult {
  text: string
  fallbackLog: FallbackEvent[]
}

export interface ConvertOptions {
  mode: Mode
  /** この値**未満**の確信度は ? に置換する。既定 0.5。 */
  confidenceThreshold?: number
  /**
   * 先頭の WORD_BREAK を空白として残すか。既定 false。
   *
   * 確定列と暫定列を別々に変換して連結すると、境界が語間に落ちたとき空白が
   * 消える ('GL 73 CQ' → 'GL 73CQ')。暫定列の変換で true を渡すと防げる。
   * **確定列が既に空白で終わっている場合は false を渡すこと** (二重空白になる)。
   */
  keepLeadingSpace?: boolean
}

export function convert(
  tokenIds: readonly number[],
  confidences: readonly number[] | null,
  options: ConvertOptions,
): ConvertResult {
  if (confidences !== null && confidences.length !== tokenIds.length) {
    throw new Error(
      `confidences length ${confidences.length} != tokenIds length ${tokenIds.length}`,
    )
  }
  const threshold = options.confidenceThreshold ?? 0.5
  const keepLeadingSpace = options.keepLeadingSpace ?? false
  const japanese = options.mode === 'japanese'
  const table = japanese ? JAPANESE_TABLE : EUROPEAN_TABLE

  const outChars: string[] = []
  const log: FallbackEvent[] = []
  /** 直前の出力が「濁点/半濁点と合成可能なカナ」だった位置 */
  let composableAt: number | null = null

  const emitFallback = (
    inputIndex: number,
    tokenId: number,
    code: string,
    kind: FallbackKind,
    confidence: number,
  ): void => {
    log.push({ position: outChars.length, inputIndex, tokenId, code, kind, confidence })
    outChars.push(FALLBACK_CHAR)
  }

  for (let i = 0; i < tokenIds.length; i++) {
    const tid = tokenIds[i]
    if (tid === BLANK_TOKEN_ID) continue

    if (tid === WORD_BREAK_TOKEN_ID) {
      if (outChars.length > 0) {
        if (outChars[outChars.length - 1] !== ' ') outChars.push(' ')
      } else if (keepLeadingSpace) {
        outChars.push(' ')
      }
      composableAt = null
      continue
    }

    const conf = confidences !== null ? confidences[i] : 1.0
    const code = tid >= 0 && tid < ID_TO_CODE.length ? ID_TO_CODE[tid] : undefined

    if (code === undefined) {
      emitFallback(i, tid, '<UNKNOWN>', 'TABLE_MISS', conf)
      composableAt = null
      continue
    }
    if (conf < threshold) {
      emitFallback(i, tid, code, 'LOW_CONFIDENCE', conf)
      composableAt = null
      continue
    }

    const display = table[code]
    if (display === undefined) {
      emitFallback(i, tid, code, 'TABLE_MISS', conf)
      composableAt = null
      continue
    }

    if (japanese && (display === DAKUTEN_CHAR || display === HANDAKUTEN_CHAR)) {
      const composeMap = display === DAKUTEN_CHAR ? DAKUTEN_COMPOSE : HANDAKUTEN_COMPOSE
      if (composableAt !== null) {
        const composed = composeMap[outChars[composableAt]]
        if (composed !== undefined) {
          outChars[composableAt] = composed
          composableAt = null
          continue
        }
      }
      // 直前カナが無い、または合成対象外 → 単独の濁点/半濁点は意味を成さない
      emitFallback(i, tid, code, 'TABLE_MISS', conf)
      composableAt = null
      continue
    }

    outChars.push(display)
    composableAt =
      japanese && (display in DAKUTEN_COMPOSE || display in HANDAKUTEN_COMPOSE)
        ? outChars.length - 1
        : null
  }

  return { text: outChars.join(''), fallbackLog: log }
}
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cd web && npx vitest run tests/converter.test.ts
```

Expected: 15 passed

- [ ] **Step 5: コミット**

```bash
git add web/src/tokens/converter.ts web/tests/converter.test.ts
git commit -m "feat: 符号変換器をTypeScriptへ移植 (固定モードのみ)"
```

---

### Task 9: スライディングウィンドウ確定器 (TypeScript)

**Files:**
- Create: `web/src/decode/sliding-window.ts`
- Test: `web/tests/sliding-window.test.ts`

**Interfaces:**
- Consumes: `web/src/decode/ctc.ts` の `FrameToken`
- Produces:
  - `export interface DecodeEngine { frameHopSamples: number; decodeChunk(wave: Float32Array): Promise<FrameToken[]> }`
  - `export interface CommittedToken { tokenId: number; confidence: number; absoluteSampleStart: number; absoluteSampleEnd: number }`
  - `export interface DecodeView { committed: CommittedToken[]; newlyCommitted: CommittedToken[]; provisional: CommittedToken[] }`
  - `export interface SlidingWindowOptions { windowS?: number; hopS?: number; commitLagS?: number; headGuardS?: number; decodeLeftContextS?: number; commitJitterMarginS?: number; sampleRate?: number }`
  - `export class SlidingWindowDecoder` — `constructor(engine: DecodeEngine, options?: SlidingWindowOptions)`,
    `reset(): void`, `push(audio: Float32Array): void`,
    `redecode(): Promise<DecodeView>`, `finalize(): Promise<DecodeView>`

- [ ] **Step 1: 失敗するテストを書く**

`web/tests/sliding-window.test.ts` を新規作成。モデル不要のフェイクエンジンで境界判定を検証する。

```ts
import { describe, expect, it } from 'vitest'
import { SlidingWindowDecoder, type DecodeEngine } from '../src/decode/sliding-window'
import type { FrameToken } from '../src/decode/ctc'

const SAMPLE_RATE = 8000
const HOP = 80 // 1 フレーム = 80 サンプル = 10 ms

/**
 * 投入された区間の長さに応じて 0.5 秒おきにトークンを返すフェイク。
 * decodeChunk に渡された波形の先頭を時刻 0 とみなす。
 */
class FakeEngine implements DecodeEngine {
  frameHopSamples = HOP
  lastChunkLength = 0

  async decodeChunk(wave: Float32Array): Promise<FrameToken[]> {
    this.lastChunkLength = wave.length
    const nFrames = Math.floor(wave.length / HOP)
    const out: FrameToken[] = []
    for (let start = 0; start + 5 < nFrames; start += 50) {
      out.push({
        tokenId: 1 + ((start / 50) % 10),
        confidence: 0.9,
        frameStart: start,
        frameEnd: start + 4,
      })
    }
    return out
  }
}

function silence(seconds: number): Float32Array {
  return new Float32Array(Math.floor(seconds * SAMPLE_RATE))
}

describe('SlidingWindowDecoder', () => {
  it('音声が無ければ空のビューを返す', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    const view = await decoder.redecode()
    expect(view.committed).toEqual([])
    expect(view.provisional).toEqual([])
  })

  it('commit_lag 圏内のトークンは暫定になる', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine(), { commitLagS: 2.5 })
    decoder.push(silence(6))
    const view = await decoder.redecode()
    expect(view.provisional.length).toBeGreaterThan(0)
    // 暫定は必ず「今 - commit_lag」より後ろで終わる
    const limit = 6 * SAMPLE_RATE - 2.5 * SAMPLE_RATE
    for (const token of view.provisional) {
      expect(token.absoluteSampleEnd).toBeGreaterThanOrEqual(limit)
    }
  })

  it('確定済みトークンは後から変化しない', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    decoder.push(silence(6))
    const first = await decoder.redecode()
    const snapshot = JSON.stringify(first.committed)

    decoder.push(silence(2))
    const second = await decoder.redecode()
    expect(JSON.stringify(second.committed.slice(0, first.committed.length))).toBe(snapshot)
  })

  it('同じトークンを二重に確定しない', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    for (let i = 0; i < 8; i++) {
      decoder.push(silence(1))
      await decoder.redecode()
    }
    const view = await decoder.redecode()
    const ends = view.committed.map((t) => t.absoluteSampleEnd)
    expect(new Set(ends).size).toBe(ends.length)
    // 絶対位置は単調増加
    for (let i = 1; i < ends.length; i++) {
      expect(ends[i]).toBeGreaterThan(ends[i - 1])
    }
  })

  it('finalize は残った暫定をすべて確定する', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    decoder.push(silence(6))
    const before = await decoder.redecode()
    const view = await decoder.finalize()
    expect(view.provisional).toEqual([])
    expect(view.committed.length).toBeGreaterThan(before.committed.length)
  })

  it('reset で状態が消える', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    decoder.push(silence(6))
    await decoder.redecode()
    decoder.reset()
    const view = await decoder.redecode()
    expect(view.committed).toEqual([])
  })

  it('リングバッファは window_s を超えない', async () => {
    const engine = new FakeEngine()
    const decoder = new SlidingWindowDecoder(engine, { windowS: 5, decodeLeftContextS: 5 })
    decoder.push(silence(20))
    await decoder.redecode()
    expect(engine.lastChunkLength).toBeLessThanOrEqual(5 * SAMPLE_RATE)
  })
})
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd web && npx vitest run tests/sliding-window.test.ts
```

Expected: FAIL — `Failed to resolve import "../src/decode/sliding-window"`

- [ ] **Step 3: 実装する**

`web/src/decode/sliding-window.ts` を新規作成。`src/infer/sliding_window.py` の移植。

```ts
/**
 * スライディングウィンドウ再デコード + prefix commit.
 *
 * 移植元: src/infer/sliding_window.py
 *
 * ライブ音声をリングバッファに保持し、hop ごとに窓全体を再デコードする。
 * [max(headGuard, lastCommit), now - commitLag) に入るトークンだけを
 * **不変 (immutable)** に確定する。確定済みは後から変化しないため表示がちらつかない。
 */
import type { FrameToken } from './ctc'

/** 波形チャンクを 1 回デコードするエンジン。 */
export interface DecodeEngine {
  /** 1 フレームあたりのサンプル数 (= mel の hop_length)。 */
  frameHopSamples: number
  decodeChunk(wave: Float32Array): Promise<FrameToken[]>
}

/** 確定 (immutable) トークン。絶対サンプル位置付き。 */
export interface CommittedToken {
  tokenId: number
  confidence: number
  absoluteSampleStart: number
  absoluteSampleEnd: number
}

/** 1 回の再デコード結果のスナップショット。 */
export interface DecodeView {
  /** 確定済み全体 */
  committed: CommittedToken[]
  /** 今回新規に確定したもの */
  newlyCommitted: CommittedToken[]
  /** 暫定 (グレー表示) */
  provisional: CommittedToken[]
}

export interface SlidingWindowOptions {
  windowS?: number
  hopS?: number
  commitLagS?: number
  headGuardS?: number
  decodeLeftContextS?: number
  commitJitterMarginS?: number
  sampleRate?: number
}

export class SlidingWindowDecoder {
  private readonly engine: DecodeEngine
  private readonly sampleRate: number
  private readonly windowSamples: number
  private readonly commitLagSamples: number
  private readonly headGuardSamples: number
  private readonly leftContextSamples: number
  private readonly jitterMarginSamples: number

  private ring = new Float32Array(0)
  /** 累積投入サンプル数 (= 現在時刻) */
  private totalConsumed = 0
  private committed: CommittedToken[] = []
  /** 確定済み末尾の絶対サンプル位置 (中点ウォーターマーク基準) */
  private lastCommitEnd: number | null = null

  constructor(engine: DecodeEngine, options: SlidingWindowOptions = {}) {
    const sampleRate = options.sampleRate ?? 8000
    this.engine = engine
    this.sampleRate = sampleRate
    this.windowSamples = Math.floor((options.windowS ?? 30.0) * sampleRate)
    this.commitLagSamples = Math.floor((options.commitLagS ?? 2.5) * sampleRate)
    this.headGuardSamples = Math.floor((options.headGuardS ?? 1.0) * sampleRate)
    this.leftContextSamples = Math.floor((options.decodeLeftContextS ?? 5.0) * sampleRate)
    this.jitterMarginSamples = Math.floor((options.commitJitterMarginS ?? 0.02) * sampleRate)
  }

  reset(): void {
    this.ring = new Float32Array(0)
    this.totalConsumed = 0
    this.committed = []
    this.lastCommitEnd = null
  }

  /** 音声を追加する (デコードはしない)。窓長を超えた古い分は捨てる。 */
  push(audio: Float32Array): void {
    this.totalConsumed += audio.length
    const merged = new Float32Array(this.ring.length + audio.length)
    merged.set(this.ring, 0)
    merged.set(audio, this.ring.length)
    this.ring =
      merged.length > this.windowSamples ? merged.subarray(merged.length - this.windowSamples) : merged
  }

  /** 確定済み末尾 - leftContext 以降を再デコードし、確定/暫定を更新する。 */
  async redecode(): Promise<DecodeView> {
    return this.decode(this.totalConsumed - this.commitLagSamples)
  }

  /**
   * 送信終了時の最終確定。commitLag を無視して残り全部を確定する。
   *
   * ストリーム終端では右文脈がもう増えないため、暫定として保留していた
   * 末尾トークンをここで確定させる。
   */
  async finalize(): Promise<DecodeView> {
    return this.decode(this.totalConsumed + 1)
  }

  private async decode(commitLimitAbs: number): Promise<DecodeView> {
    if (this.ring.length === 0) {
      return { committed: [...this.committed], newlyCommitted: [], provisional: [] }
    }

    const hop = this.engine.frameHopSamples
    const ringStartAbs = this.totalConsumed - this.ring.length

    // デコード区間の動的短縮 (計算量削減)
    let lastEnd = this.lastCommitEnd
    const anchor = lastEnd ?? ringStartAbs
    const decodeStartAbs = Math.max(ringStartAbs, anchor - this.leftContextSamples)
    const sub = this.ring.subarray(decodeStartAbs - ringStartAbs)
    const frameTokens = await this.engine.decodeChunk(sub)

    // head guard: デコード区間先頭の不採用区間。
    // ただし区間がストリーム先頭のときは先頭信号を捨てない。
    const headCutAbs = decodeStartAbs > 0 ? decodeStartAbs + this.headGuardSamples : 0

    const newly: CommittedToken[] = []
    const provisional: CommittedToken[] = []

    for (const tok of frameTokens) {
      const absStart = decodeStartAbs + tok.frameStart * hop
      const absEnd = decodeStartAbs + tok.frameEnd * hop
      const ct: CommittedToken = {
        tokenId: tok.tokenId,
        confidence: tok.confidence,
        absoluteSampleStart: absStart,
        absoluteSampleEnd: absEnd,
      }
      // 右文脈不足 (commit 境界を終了がまたぐ) → 暫定
      if (absEnd >= commitLimitAbs) {
        provisional.push(ct)
        continue
      }
      // 左文脈なし → 不採用
      if (absStart < headCutAbs) continue
      // 中点ウォーターマーク: 確定済み末尾を中点が超えるものだけ新規確定する。
      // 既確定トークンの再出現は確実にスキップされ、文字間ギャップが小さくても
      // 新規トークンは脱落しない。
      const midpoint = Math.floor((absStart + absEnd) / 2)
      if (lastEnd !== null && midpoint <= lastEnd + this.jitterMarginSamples) continue

      this.committed.push(ct)
      newly.push(ct)
      lastEnd = absEnd
      this.lastCommitEnd = absEnd
    }

    return { committed: [...this.committed], newlyCommitted: newly, provisional }
  }
}
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cd web && npx vitest run tests/sliding-window.test.ts
```

Expected: 7 passed

- [ ] **Step 5: コミット**

```bash
git add web/src/decode/sliding-window.ts web/tests/sliding-window.test.ts
git commit -m "feat: スライディングウィンドウ確定器をTypeScriptへ移植"
```

---

### Task 10: ONNX エンジンとゴールデンテスト

Python の推論結果と TypeScript の推論結果が**完全に一致する**ことを自動テストで確認する。ここを通せば「精度が落ちたかもしれない」という不安が一致/不一致の二値になる。

**Files:**
- Create: `web/src/decode/onnx-engine.ts`
- Test: `web/tests/golden.test.ts`
- Create: `web/vitest.config.ts`

**Interfaces:**
- Consumes: `DecodeEngine` (Task 9), `ctcGreedyDecodeWithFrames` (Task 7),
  `convert` (Task 8), `web/tests/fixtures/` (Task 6), `web/public/model/cw.onnx` (Task 2)
- Produces:
  - `export class OnnxDecodeEngine implements DecodeEngine` —
    `static create(modelUrl: string, options?: { numThreads?: number }): Promise<OnnxDecodeEngine>`,
    `frameHopSamples: number`, `decodeChunk(wave: Float32Array): Promise<FrameToken[]>`,
    `release(): Promise<void>`

- [ ] **Step 1: vitest の設定を追加**

`web/vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    // ONNX の読み込みと推論に時間がかかるため長めに取る
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
})
```

- [ ] **Step 2: 失敗するテストを書く**

`web/tests/golden.test.ts` を新規作成。

```ts
/**
 * ゴールデンテスト: Python の推論結果と完全一致することを確認する.
 *
 * fixture は scripts/export_golden.py が出力した 8 kHz mono float32 の波形と
 * 期待トークン列。リサンプルは検証対象外 (Python 側で済ませてある)。
 *
 * 前提: python scripts/export_onnx.py で web/public/model/cw.onnx が生成済み。
 */
import { readFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it, beforeAll, afterAll } from 'vitest'
import { OnnxDecodeEngine } from '../src/decode/onnx-engine'
import { convert } from '../src/tokens/converter'

const MODEL_PATH = resolve(__dirname, '../public/model/cw.onnx')
const FIXTURE_DIR = resolve(__dirname, 'fixtures')

interface GoldenEntry {
  name: string
  waveFile: string
  nSamples: number
  tokenIds: number[]
  confidences: number[]
  textEuropean: string
  textJapanese: string
}

const hasAssets = existsSync(MODEL_PATH) && existsSync(resolve(FIXTURE_DIR, 'golden.json'))

describe.skipIf(!hasAssets)('ゴールデンテスト', () => {
  let engine: OnnxDecodeEngine
  let entries: GoldenEntry[]

  beforeAll(async () => {
    entries = JSON.parse(readFileSync(resolve(FIXTURE_DIR, 'golden.json'), 'utf-8'))
    engine = await OnnxDecodeEngine.create(MODEL_PATH, { numThreads: 1 })
  })

  afterAll(async () => {
    await engine?.release()
  })

  it('fixture が読み込めている', () => {
    expect(entries.length).toBeGreaterThan(0)
  })

  it('トークン ID 列が Python と完全一致する', async () => {
    for (const entry of entries) {
      const raw = readFileSync(resolve(FIXTURE_DIR, entry.waveFile))
      const wave = new Float32Array(raw.buffer, raw.byteOffset, entry.nSamples)
      const tokens = await engine.decodeChunk(wave)
      expect(tokens.map((t) => t.tokenId), `${entry.name} のトークン列`).toEqual(entry.tokenIds)
    }
  })

  it('確信度が Python と十分近い', async () => {
    for (const entry of entries) {
      const raw = readFileSync(resolve(FIXTURE_DIR, entry.waveFile))
      const wave = new Float32Array(raw.buffer, raw.byteOffset, entry.nSamples)
      const tokens = await engine.decodeChunk(wave)
      tokens.forEach((token, i) => {
        expect(token.confidence, `${entry.name}[${i}] の確信度`).toBeCloseTo(
          entry.confidences[i],
          3,
        )
      })
    }
  })

  it('欧文・和文テキストが Python と一致する', async () => {
    for (const entry of entries) {
      const raw = readFileSync(resolve(FIXTURE_DIR, entry.waveFile))
      const wave = new Float32Array(raw.buffer, raw.byteOffset, entry.nSamples)
      const tokens = await engine.decodeChunk(wave)
      const ids = tokens.map((t) => t.tokenId)
      const confs = tokens.map((t) => t.confidence)
      expect(convert(ids, confs, { mode: 'european' }).text).toBe(entry.textEuropean)
      expect(convert(ids, confs, { mode: 'japanese' }).text).toBe(entry.textJapanese)
    }
  })
})
```

- [ ] **Step 3: テストが失敗することを確認**

```bash
cd web && npx vitest run tests/golden.test.ts
```

Expected: FAIL — `Failed to resolve import "../src/decode/onnx-engine"`

- [ ] **Step 4: 実装する**

`web/src/decode/onnx-engine.ts` を新規作成。

```ts
/**
 * ONNX Runtime Web による推論エンジン.
 *
 * モデルは「波形 in → log_probs out」の単一グラフ (scripts/export_onnx.py が生成)。
 * メル変換がグラフに焼き込まれているため、ここで前処理は一切行わない。
 */
import * as ort from 'onnxruntime-web'
import { ctcGreedyDecodeWithFrames, type FrameToken } from './ctc'
import type { DecodeEngine } from './sliding-window'
import { VOCAB_SIZE } from '../generated/tokens'

/** mel の hop_length。フレーム位置を絶対サンプル位置に直すのに使う。 */
const FRAME_HOP_SAMPLES = 80
/** モデル (17 MB) を保存する Cache API のキー。 */
const MODEL_CACHE_NAME = 'cw-decoder-model-v1'

export interface OnnxEngineOptions {
  /** WASM スレッド数。crossOriginIsolated でない環境では 1 になる。 */
  numThreads?: number
}

/**
 * モデルを取得する。2 回目以降は Cache API から読むので再ダウンロードしない。
 *
 * Cache API が無い環境 (vitest の Node 実行など) では URL をそのまま返し、
 * ORT にファイルを読ませる。
 */
async function loadModel(modelUrl: string): Promise<string | ArrayBuffer> {
  if (typeof caches === 'undefined') return modelUrl
  try {
    const cache = await caches.open(MODEL_CACHE_NAME)
    let response = await cache.match(modelUrl)
    if (!response) {
      await cache.add(modelUrl)
      response = await cache.match(modelUrl)
    }
    if (response) return await response.arrayBuffer()
  } catch {
    // キャッシュが使えなくても動作は続ける (毎回ダウンロードになるだけ)
  }
  return modelUrl
}

export class OnnxDecodeEngine implements DecodeEngine {
  readonly frameHopSamples = FRAME_HOP_SAMPLES

  private constructor(private readonly session: ort.InferenceSession) {}

  static async create(
    modelUrl: string,
    options: OnnxEngineOptions = {},
  ): Promise<OnnxDecodeEngine> {
    ort.env.wasm.simd = true
    ort.env.wasm.numThreads = options.numThreads ?? 1
    const source = await loadModel(modelUrl)
    const session = await ort.InferenceSession.create(source as string, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    return new OnnxDecodeEngine(session)
  }

  async decodeChunk(wave: Float32Array): Promise<FrameToken[]> {
    if (wave.length === 0) return []
    // subarray はビューなので、ORT に渡す前に連続領域へコピーする
    const input = new Float32Array(wave)
    const feeds = { wave: new ort.Tensor('float32', input, [1, input.length]) }
    const output = await this.session.run(feeds)
    const logProbs = output.log_probs
    const nFrames = logProbs.dims[1]
    return ctcGreedyDecodeWithFrames(
      logProbs.data as Float32Array,
      nFrames,
      VOCAB_SIZE,
    )
  }

  async release(): Promise<void> {
    await this.session.release()
  }
}
```

- [ ] **Step 5: テストが通ることを確認**

```bash
cd web && npx vitest run tests/golden.test.ts
```

Expected: 4 passed

**トークン列が一致しない場合の切り分け順**:
1. Task 1 の `tests/test_onnx_mel.py` を再実行 → メル変換のズレ
2. Task 2 の `tests/test_export_onnx.py` を再実行 → ONNX 変換のズレ
3. どちらも通るなら CTC (Task 7) を疑う。`ctcGreedyDecodeWithFrames` に
   Python 側と同じ log_probs を直接食わせて比較する

- [ ] **Step 6: コミット**

```bash
git add web/src/decode/onnx-engine.ts web/tests/golden.test.ts web/vitest.config.ts
git commit -m "feat: ONNXエンジンとPython結果との一致を確認するゴールデンテストを追加"
```

---

### Task 11: 音声取得 (AudioWorklet + フォールバックリサンプラ)

**Files:**
- Create: `web/src/audio/fir-decimator.ts`
- Create: `web/src/audio/capture-worklet.ts` (AudioWorkletProcessor)
- Create: `web/src/audio/capture.ts`
- Test: `web/tests/fir-decimator.test.ts`

**Interfaces:**
- Produces:
  - `export class FirDecimator` — `constructor(sourceRate: number, targetRate: number)`,
    `process(input: Float32Array): Float32Array`, `readonly factor: number`,
    `static supports(sourceRate: number, targetRate: number): boolean`
  - `export interface CaptureOptions { onAudio: (block: Float32Array) => void; deviceId?: string }`
  - `export class AudioCapture` — `static start(options: CaptureOptions): Promise<AudioCapture>`,
    `readonly sampleRate: number`, `readonly usedFallbackResampler: boolean`,
    `readonly analyser: AnalyserNode`, `stop(): Promise<void>`

- [ ] **Step 1: 失敗するテストを書く**

`web/tests/fir-decimator.test.ts` を新規作成。

```ts
import { describe, expect, it } from 'vitest'
import { FirDecimator } from '../src/audio/fir-decimator'

/** 指定周波数の正弦波を作る。 */
function sine(freq: number, rate: number, seconds: number, offset = 0): Float32Array {
  const n = Math.floor(rate * seconds)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = Math.sin((2 * Math.PI * freq * (i + offset)) / rate)
  return out
}

function rms(x: Float32Array): number {
  let sum = 0
  for (const v of x) sum += v * v
  return Math.sqrt(sum / Math.max(1, x.length))
}

describe('FirDecimator', () => {
  it('整数比のみ対応する', () => {
    expect(FirDecimator.supports(48000, 8000)).toBe(true)
    expect(FirDecimator.supports(16000, 8000)).toBe(true)
    expect(FirDecimator.supports(44100, 8000)).toBe(false)
  })

  it('出力長が入力長の 1/factor 前後になる', () => {
    const dec = new FirDecimator(48000, 8000)
    expect(dec.factor).toBe(6)
    const out = dec.process(sine(600, 48000, 1.0))
    expect(out.length).toBeGreaterThan(8000 * 0.95)
    expect(out.length).toBeLessThanOrEqual(8000)
  })

  it('通過帯域の信号は振幅をほぼ保つ', () => {
    const dec = new FirDecimator(48000, 8000)
    const out = dec.process(sine(600, 48000, 1.0))
    // 先頭は FIR の立ち上がりなので後半で評価する
    const tail = out.subarray(Math.floor(out.length / 2))
    expect(rms(tail)).toBeGreaterThan(0.6)
  })

  it('ナイキストを超える信号を減衰させる', () => {
    const dec = new FirDecimator(48000, 8000)
    // 8 kHz 側のナイキストは 4 kHz。9 kHz は折り返す前に落ちるべき
    const out = dec.process(sine(9000, 48000, 1.0))
    const tail = out.subarray(Math.floor(out.length / 2))
    expect(rms(tail)).toBeLessThan(0.05)
  })

  it('ブロック分割しても連続処理と一致する (状態を保持する)', () => {
    const full = sine(600, 48000, 0.5)

    const whole = new FirDecimator(48000, 8000).process(full)

    const streamed = new FirDecimator(48000, 8000)
    const chunks: Float32Array[] = []
    const blockSize = 480
    for (let i = 0; i < full.length; i += blockSize) {
      chunks.push(streamed.process(full.subarray(i, i + blockSize)))
    }
    const joined = new Float32Array(chunks.reduce((n, c) => n + c.length, 0))
    let offset = 0
    for (const c of chunks) {
      joined.set(c, offset)
      offset += c.length
    }

    expect(joined.length).toBe(whole.length)
    for (let i = 0; i < whole.length; i++) {
      expect(joined[i]).toBeCloseTo(whole[i], 5)
    }
  })
})
```

**最後のテストが最重要**。`src/infer/audio.py` に「ステートレスなリサンプルはブロック境界に歪みを生じ、CW の dot/dash を破壊する」と記録されている問題を、そのまま検出する。

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd web && npx vitest run tests/fir-decimator.test.ts
```

Expected: FAIL — `Failed to resolve import "../src/audio/fir-decimator"`

- [ ] **Step 3: FIR デシメータを実装する**

`web/src/audio/fir-decimator.ts` を新規作成。

```ts
/**
 * ステートフルな FIR デシメータ (整数比専用).
 *
 * AudioContext を 8 kHz で作れない環境向けのフォールバック。
 *
 * **状態を保持することが必須**。ステートレスなリサンプルはブロック境界に歪みを
 * 生じ、CW の dot/dash を破壊する (src/infer/audio.py の記録を参照)。
 * 直前ブロックの末尾 numTaps-1 サンプルを保持して連続性を保つ。
 */

/** ローパスの遮断周波数 (目標レートのナイキストの 95%)。 */
const CUTOFF_RATIO = 0.95
/** FIR のタップ数 (奇数。線形位相)。 */
const NUM_TAPS = 121

export class FirDecimator {
  readonly factor: number
  private readonly taps: Float32Array
  /**
   * まだ出力に使い切っていない過去サンプル。
   * 次回の畳み込み窓はこの配列の先頭から始まる (だから phase は不要)。
   */
  private history: Float32Array

  constructor(sourceRate: number, targetRate: number) {
    if (!FirDecimator.supports(sourceRate, targetRate)) {
      throw new Error(`整数比でないレート変換には対応していません: ${sourceRate} → ${targetRate}`)
    }
    this.factor = sourceRate / targetRate
    this.taps = FirDecimator.designLowpass(
      (targetRate / 2) * CUTOFF_RATIO / sourceRate,
      NUM_TAPS,
    )
    this.history = new Float32Array(NUM_TAPS - 1)
  }

  static supports(sourceRate: number, targetRate: number): boolean {
    return Number.isInteger(sourceRate / targetRate) && sourceRate >= targetRate
  }

  /**
   * 窓関数法によるローパス FIR。
   *
   * @param normalizedCutoff 遮断周波数 / サンプリング周波数 (0 < f < 0.5)
   */
  private static designLowpass(normalizedCutoff: number, numTaps: number): Float32Array {
    const taps = new Float32Array(numTaps)
    const center = (numTaps - 1) / 2
    let sum = 0
    for (let i = 0; i < numTaps; i++) {
      const n = i - center
      // sinc
      const sinc =
        n === 0 ? 2 * normalizedCutoff : Math.sin(2 * Math.PI * normalizedCutoff * n) / (Math.PI * n)
      // Hamming 窓
      const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (numTaps - 1))
      taps[i] = sinc * window
      sum += taps[i]
    }
    // 直流利得を 1 に正規化
    for (let i = 0; i < numTaps; i++) taps[i] /= sum
    return taps
  }

  /**
   * 入力ブロックを処理し、間引き済みのサンプルを返す。
   *
   * 「窓が収まる位置まで出力し、残りを丸ごと history に持ち越す」だけで
   * 連続性が保たれる。持ち越し量は numTaps + factor 未満に収まるので増え続けない。
   */
  process(input: Float32Array): Float32Array {
    const taps = this.taps
    const numTaps = taps.length

    // 持ち越し + 今回の入力
    const buffer = new Float32Array(this.history.length + input.length)
    buffer.set(this.history, 0)
    buffer.set(input, this.history.length)

    // 出力できるのは、畳み込み窓が buffer に収まる位置まで
    const nOut = Math.max(0, Math.floor((buffer.length - numTaps) / this.factor) + 1)
    const out = new Float32Array(nOut)
    for (let n = 0; n < nOut; n++) {
      const i = n * this.factor
      let acc = 0
      for (let k = 0; k < numTaps; k++) acc += buffer[i + k] * taps[k]
      out[n] = acc
    }

    // 窓が収まらなかった位置以降を丸ごと持ち越す (次回はここが先頭になる)
    this.history = buffer.slice(nOut * this.factor)
    return out
  }
}
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cd web && npx vitest run tests/fir-decimator.test.ts
```

Expected: 5 passed

「ブロック分割しても一致する」テストが落ちる場合は `phase` / `history` の更新式を見直す。
このテストが通らないまま先に進まないこと (実信号でだけ壊れる不具合になる)。

- [ ] **Step 5: AudioWorklet を実装する**

`web/src/audio/capture-worklet.ts` を新規作成。

```ts
/**
 * 音声取得用の AudioWorkletProcessor.
 *
 * 128 サンプル単位で呼ばれるので、BLOCK_SAMPLES ぶん溜めてから
 * メインスレッドへ postMessage する (メッセージ数を抑える)。
 *
 * このファイルは AudioWorkletGlobalScope で動くため、import は使えない。
 * capture.ts から Blob URL 経由で読み込む。
 */
const WORKLET_SOURCE = `
const BLOCK_SAMPLES = 400   // 8 kHz で 0.05 秒

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.buffer = new Float32Array(BLOCK_SAMPLES)
    this.filled = 0
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || input.length === 0) return true
    const channel = input[0]
    if (!channel) return true

    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.filled++] = channel[i]
      if (this.filled === BLOCK_SAMPLES) {
        const out = this.buffer.slice()
        this.port.postMessage(out, [out.buffer])
        this.buffer = new Float32Array(BLOCK_SAMPLES)
        this.filled = 0
      }
    }
    return true
  }
}

registerProcessor('cw-capture', CaptureProcessor)
`

/** Worklet のソースを Blob URL にして返す (別ファイル配信を不要にする)。 */
export function captureWorkletUrl(): string {
  return URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }))
}
```

- [ ] **Step 6: マイク取得を実装する**

`web/src/audio/capture.ts` を新規作成。

```ts
/**
 * マイク / ライン入力からの音声取得.
 *
 * 重要な制約が 2 つある。
 *
 * 1. getUserMedia の echoCancellation / noiseSuppression / autoGainControl は
 *    既定で有効であり、いずれも CW のトーンを破壊する。**必ず全部 false にする**。
 * 2. リサンプルは状態を保持しなければならない。ステートレスな変換はブロック境界に
 *    歪みを生じ dot/dash を壊す (src/infer/audio.py の記録)。
 *    主系は AudioContext を 8 kHz で作りブラウザ内蔵のリサンプラに任せる。
 *    それが効かない環境では FirDecimator にフォールバックする。
 */
import { FirDecimator } from './fir-decimator'
import { captureWorkletUrl } from './capture-worklet'

const TARGET_SAMPLE_RATE = 8000

export interface CaptureOptions {
  /** 8 kHz の音声ブロックが届くたびに呼ばれる。 */
  onAudio: (block: Float32Array) => void
  deviceId?: string
}

export class AudioCapture {
  private constructor(
    private readonly context: AudioContext,
    private readonly stream: MediaStream,
    private readonly node: AudioWorkletNode,
    readonly analyser: AnalyserNode,
    readonly sampleRate: number,
    readonly usedFallbackResampler: boolean,
  ) {}

  static async start(options: CaptureOptions): Promise<AudioCapture> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: options.deviceId,
        // CW を壊すため必ず無効にする
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    })

    // 主系: AudioContext を 8 kHz で構築する
    let context: AudioContext
    try {
      context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
    } catch {
      context = new AudioContext()
    }
    const needsFallback = context.sampleRate !== TARGET_SAMPLE_RATE

    let decimator: FirDecimator | null = null
    if (needsFallback) {
      if (!FirDecimator.supports(context.sampleRate, TARGET_SAMPLE_RATE)) {
        await context.close()
        stream.getTracks().forEach((t) => t.stop())
        throw new Error(
          `このデバイスのサンプルレート ${context.sampleRate} Hz には対応していません ` +
            `(8000 Hz の整数倍が必要です)`,
        )
      }
      decimator = new FirDecimator(context.sampleRate, TARGET_SAMPLE_RATE)
    }

    const url = captureWorkletUrl()
    try {
      await context.audioWorklet.addModule(url)
    } finally {
      URL.revokeObjectURL(url)
    }

    const source = context.createMediaStreamSource(stream)
    const analyser = context.createAnalyser()
    analyser.fftSize = 2048
    analyser.smoothingTimeConstant = 0.6

    const node = new AudioWorkletNode(context, 'cw-capture')
    node.port.onmessage = (event: MessageEvent<Float32Array>) => {
      const block = event.data
      options.onAudio(decimator ? decimator.process(block) : block)
    }

    source.connect(analyser)
    source.connect(node)
    // Worklet は出力を持たないが、一部ブラウザは destination に繋がないと
    // グラフが動かないため 0 ゲインで接続しておく
    const mute = context.createGain()
    mute.gain.value = 0
    node.connect(mute).connect(context.destination)

    await context.resume()

    return new AudioCapture(
      context,
      stream,
      node,
      analyser,
      context.sampleRate,
      needsFallback,
    )
  }

  async stop(): Promise<void> {
    this.node.port.onmessage = null
    this.node.disconnect()
    this.stream.getTracks().forEach((track) => track.stop())
    await this.context.close()
  }
}
```

- [ ] **Step 7: 型チェックを通す**

```bash
cd web && npx tsc --noEmit
```

Expected: エラーなし

(`fir-decimator.ts` の `process` 内に残っている未使用変数などがあれば消す)

- [ ] **Step 8: コミット**

```bash
git add web/src/audio/ web/tests/fir-decimator.test.ts
git commit -m "feat: マイク入力の取得とステートフルなFIRデシメータを追加"
```

---

### Task 12: デコーダ Worker

推論をメインスレッドから追い出す。UI が固まらないようにするため必須。

**Files:**
- Create: `web/src/worker/protocol.ts`
- Create: `web/src/worker/decoder.worker.ts`
- Create: `web/src/worker/decoder-client.ts`

**Interfaces:**
- Consumes: `OnnxDecodeEngine` (Task 10), `SlidingWindowDecoder` (Task 9), `convert` (Task 8)
- Produces:
  - `protocol.ts`:
    - `export type WorkerRequest = { type: 'init'; modelUrl: string; numThreads: number; commitLagS: number; hopS: number; decodeLeftContextS: number } | { type: 'audio'; block: Float32Array } | { type: 'setMode'; mode: Mode } | { type: 'finalize' } | { type: 'reset' }`
    - `export type WorkerResponse = { type: 'ready' } | { type: 'error'; message: string } | { type: 'text'; committedEuropean: string; provisionalEuropean: string; committedJapanese: string; provisionalJapanese: string; decodeMs: number }`
  - `decoder-client.ts`:
    - `export class DecoderClient` — `static create(options: { modelUrl: string; numThreads: number; commitLagS: number; hopS: number; decodeLeftContextS: number; onText: (r: Extract<WorkerResponse, {type:'text'}>) => void; onError: (message: string) => void }): Promise<DecoderClient>`,
      `pushAudio(block: Float32Array): void`, `finalize(): void`, `reset(): void`, `terminate(): void`

- [ ] **Step 1: プロトコルを定義する**

`web/src/worker/protocol.ts`:

```ts
/** メインスレッドと Decoder Worker の間でやり取りするメッセージ定義. */
import type { FallbackEvent, Mode } from '../tokens/converter'

/** 1 つの表示モード (欧文 or 和文) のレンダリング結果. */
export interface RenderedText {
  committed: string
  provisional: string
  /** 確定テキスト中の「?」の由来 (TABLE_MISS / LOW_CONFIDENCE の区別) */
  committedFallbacks: FallbackEvent[]
  provisionalFallbacks: FallbackEvent[]
}

export type WorkerRequest =
  | {
      type: 'init'
      modelUrl: string
      numThreads: number
      commitLagS: number
      hopS: number
      decodeLeftContextS: number
    }
  | { type: 'audio'; block: Float32Array }
  | { type: 'setMode'; mode: Mode }
  | { type: 'finalize' }
  | { type: 'reset' }

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'error'; message: string }
  | {
      type: 'text'
      /** 同じトークン列を両方の表に通した結果 (追加の推論コストは無い) */
      european: RenderedText
      japanese: RenderedText
      decodeMs: number
    }
```

- [ ] **Step 2: Worker 本体を実装する**

`web/src/worker/decoder.worker.ts`:

```ts
/**
 * デコーダ Worker.
 *
 * 音声ブロックを受け取り、hop ごとに再デコードして確定/暫定テキストを返す。
 * 推論はここで行うのでメインスレッドは固まらない。
 *
 * 欧文列と和文列は同じトークン列を 2 つの変換表に通すだけなので、
 * 追加の推論コストは無い (設計書 §9)。
 */
import { OnnxDecodeEngine } from '../decode/onnx-engine'
import { SlidingWindowDecoder, type DecodeView } from '../decode/sliding-window'
import { convert, type Mode } from '../tokens/converter'
import type { RenderedText, WorkerRequest, WorkerResponse } from './protocol'

const SAMPLE_RATE = 8000

let decoder: SlidingWindowDecoder | null = null
let engine: OnnxDecodeEngine | null = null
let hopSamples = SAMPLE_RATE
let samplesSinceRedecode = 0
let busy = false

function post(message: WorkerResponse): void {
  self.postMessage(message)
}

/** 確定列と暫定列を指定モードで変換する (workers.py の _emit_texts 相当)。 */
function renderMode(view: DecodeView, mode: Mode): RenderedText {
  const committedIds = view.committed.map((t) => t.tokenId)
  const committedConfs = view.committed.map((t) => t.confidence)
  const committed = convert(committedIds, committedConfs, { mode })

  // 確定列と暫定列の境界が語間に落ちると空白が消える ('GL 73 CQ' → 'GL 73CQ')。
  // 確定列が既に空白で終わっていない場合だけ暫定列の先頭空白を残す。
  const keepLeadingSpace = committed.text.length > 0 && !committed.text.endsWith(' ')
  const provisional = convert(
    view.provisional.map((t) => t.tokenId),
    view.provisional.map((t) => t.confidence),
    { mode, keepLeadingSpace },
  )
  return {
    committed: committed.text,
    provisional: provisional.text,
    committedFallbacks: committed.fallbackLog,
    provisionalFallbacks: provisional.fallbackLog,
  }
}

function emit(view: DecodeView, decodeMs: number): void {
  post({
    type: 'text',
    european: renderMode(view, 'european'),
    japanese: renderMode(view, 'japanese'),
    decodeMs,
  })
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  try {
    switch (request.type) {
      case 'init': {
        engine = await OnnxDecodeEngine.create(request.modelUrl, {
          numThreads: request.numThreads,
        })
        decoder = new SlidingWindowDecoder(engine, {
          hopS: request.hopS,
          commitLagS: request.commitLagS,
          decodeLeftContextS: request.decodeLeftContextS,
          sampleRate: SAMPLE_RATE,
        })
        hopSamples = Math.floor(request.hopS * SAMPLE_RATE)
        samplesSinceRedecode = 0
        post({ type: 'ready' })
        break
      }
      case 'audio': {
        if (!decoder) return
        decoder.push(request.block)
        samplesSinceRedecode += request.block.length
        if (samplesSinceRedecode < hopSamples) return
        // 前回の推論が終わっていなければ今回は見送る (キューを溜めない)
        if (busy) return
        samplesSinceRedecode = 0
        busy = true
        try {
          const start = performance.now()
          const view = await decoder.redecode()
          emit(view, performance.now() - start)
        } finally {
          busy = false
        }
        break
      }
      case 'finalize': {
        if (!decoder) return
        const start = performance.now()
        const view = await decoder.finalize()
        emit(view, performance.now() - start)
        break
      }
      case 'reset': {
        decoder?.reset()
        samplesSinceRedecode = 0
        break
      }
      case 'setMode':
        // 表示モードはメインスレッド側で選ぶ (両方を毎回返しているため何もしない)
        break
    }
  } catch (error) {
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
```

- [ ] **Step 3: クライアントラッパを実装する**

`web/src/worker/decoder-client.ts`:

```ts
/** Decoder Worker をメインスレッドから扱うためのラッパ. */
import type { WorkerRequest, WorkerResponse } from './protocol'

type TextResponse = Extract<WorkerResponse, { type: 'text' }>

export interface DecoderClientOptions {
  modelUrl: string
  numThreads: number
  commitLagS: number
  hopS: number
  decodeLeftContextS: number
  onText: (response: TextResponse) => void
  onError: (message: string) => void
}

export class DecoderClient {
  private constructor(private readonly worker: Worker) {}

  static create(options: DecoderClientOptions): Promise<DecoderClient> {
    const worker = new Worker(new URL('./decoder.worker.ts', import.meta.url), {
      type: 'module',
    })
    return new Promise((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const response = event.data
        switch (response.type) {
          case 'ready':
            resolve(new DecoderClient(worker))
            break
          case 'text':
            options.onText(response)
            break
          case 'error':
            options.onError(response.message)
            reject(new Error(response.message))
            break
        }
      }
      worker.onerror = (event) => reject(new Error(event.message))
      const init: WorkerRequest = {
        type: 'init',
        modelUrl: options.modelUrl,
        numThreads: options.numThreads,
        commitLagS: options.commitLagS,
        hopS: options.hopS,
        decodeLeftContextS: options.decodeLeftContextS,
      }
      worker.postMessage(init)
    })
  }

  pushAudio(block: Float32Array): void {
    const message: WorkerRequest = { type: 'audio', block }
    // 転送してコピーを避ける
    this.worker.postMessage(message, [block.buffer])
  }

  finalize(): void {
    this.worker.postMessage({ type: 'finalize' } satisfies WorkerRequest)
  }

  reset(): void {
    this.worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
  }

  terminate(): void {
    this.worker.terminate()
  }
}
```

- [ ] **Step 4: 型チェックを通す**

```bash
cd web && npx tsc --noEmit
```

Expected: エラーなし

- [ ] **Step 5: コミット**

```bash
git add web/src/worker/
git commit -m "feat: 推論をメインスレッドから分離するデコーダWorkerを追加"
```

---

### Task 13: UI

**Files:**
- Create: `web/src/ui/monitor.ts`
- Create: `web/src/ui/app.ts`
- Create: `web/src/style.css`
- Modify: `web/index.html` (エントリを `bench.ts` から `src/main.ts` に切替)
- Create: `web/src/main.ts`
- Move: `web/src/bench.ts` → `web/src/bench/bench.ts` (ベンチは残す)、`web/bench.html` を追加

**Interfaces:**
- Consumes: `AudioCapture` (Task 11), `DecoderClient` (Task 12)
- Produces:
  - `export class SignalMonitor` — `constructor(canvas: HTMLCanvasElement, analyser: AnalyserNode)`,
    `start(): void`, `stop(): void`
  - `export function mountApp(root: HTMLElement): void`

- [ ] **Step 1: 信号モニタを実装する**

`web/src/ui/monitor.ts`:

```ts
/**
 * 信号モニタ: スペクトル表示と入力レベルメータ.
 *
 * CW のトーン周波数が見えるので同調確認に使える。実運用では
 * 「音が届いているか」「周波数が合っているか」の切り分けにこれが要る。
 */
export class SignalMonitor {
  private readonly bins: Uint8Array
  private readonly timeDomain: Float32Array
  private rafId: number | null = null

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly analyser: AnalyserNode,
    private readonly onLevel: (dbfs: number) => void,
  ) {
    this.bins = new Uint8Array(analyser.frequencyBinCount)
    this.timeDomain = new Float32Array(analyser.fftSize)
  }

  start(): void {
    if (this.rafId !== null) return
    const draw = (): void => {
      this.render()
      this.rafId = requestAnimationFrame(draw)
    }
    this.rafId = requestAnimationFrame(draw)
  }

  stop(): void {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId)
    this.rafId = null
  }

  private render(): void {
    const ctx = this.canvas.getContext('2d')
    if (!ctx) return

    this.analyser.getByteFrequencyData(this.bins)
    this.analyser.getFloatTimeDomainData(this.timeDomain)

    let sumSquares = 0
    for (const v of this.timeDomain) sumSquares += v * v
    const rms = Math.sqrt(sumSquares / this.timeDomain.length)
    this.onLevel(rms < 1e-6 ? -120 : 20 * Math.log10(rms))

    const { width, height } = this.canvas
    ctx.clearRect(0, 0, width, height)

    // 8 kHz の AudioContext なので bin 全体が 0〜4 kHz に対応する
    const barWidth = width / this.bins.length
    ctx.fillStyle = '#2f9e6f'
    for (let i = 0; i < this.bins.length; i++) {
      const barHeight = (this.bins[i] / 255) * height
      ctx.fillRect(i * barWidth, height - barHeight, Math.max(1, barWidth), barHeight)
    }

    // 1 kHz ごとの目盛り
    ctx.fillStyle = '#888'
    ctx.font = '10px sans-serif'
    for (let khz = 1; khz < 4; khz++) {
      const x = (khz / 4) * width
      ctx.fillRect(x, 0, 1, height)
      ctx.fillText(`${khz}k`, x + 2, 10)
    }
  }
}
```

- [ ] **Step 2: アプリ本体を実装する**

`web/src/ui/app.ts`:

```ts
/**
 * ブラウザ版 CW デコーダの画面.
 *
 * 表示は 2 モード:
 *   欧文        — 欧文表による解釈のみ
 *   欧文 + 和文 — 同じトークン列を両方の表に通して併記する
 *
 * 確定 (committed) は通常色、暫定 (provisional) はグレーで表示する。
 * 暫定を出すことで体感遅延が commit_lag から切り離される (設計書 §6.1)。
 */
import { AudioCapture } from '../audio/capture'
import { DecoderClient } from '../worker/decoder-client'
import { SignalMonitor } from './monitor'
import type { FallbackEvent } from '../tokens/converter'
import type { WorkerResponse } from '../worker/protocol'

type TextResponse = Extract<WorkerResponse, { type: 'text' }>

const MODEL_URL = '/model/cw.onnx'
// Task 3 の実測と Task 4 の掃引結果に応じて調整する
const HOP_S = 1.0
const COMMIT_LAG_S = 2.5
const DECODE_LEFT_CONTEXT_S = 5.0

const TEMPLATE = `
<h1>CW デコーダ</h1>
<div class="controls">
  <button id="toggle">受信開始</button>
  <label><input type="checkbox" id="show-japanese" /> 和文を併記する</label>
  <span id="status">停止中</span>
</div>
<div class="monitor">
  <canvas id="spectrum" width="640" height="120"></canvas>
  <div class="level">入力レベル: <span id="level">-120.0</span> dBFS</div>
</div>
<section class="output" id="out-european">
  <h2>欧文</h2>
  <p class="text"><span class="committed"></span><span class="provisional"></span></p>
</section>
<section class="output" id="out-japanese" hidden>
  <h2>和文</h2>
  <p class="text"><span class="committed"></span><span class="provisional"></span></p>
</section>
<p class="diag" id="diag"></p>
`

/**
 * WASM のスレッド数を決める。
 *
 * crossOriginIsolated (COOP/COEP) でない環境ではマルチスレッドが使えないため 1。
 * 設計書 §3 の実測により、1 スレッドでも hop に対して余裕がある。
 */
function resolveThreadCount(): number {
  return globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
}

/**
 * テキストを span に描画し、「?」にはその由来をツールチップで付ける。
 *
 * 「?」には 2 種類あり (CLAUDE.md の原則)、デバッグ時に区別できないと困る:
 *   TABLE_MISS     — 変換表に該当なし (欧文モード中の和文符号など)
 *   LOW_CONFIDENCE — CTC 事後確率が閾値未満
 */
function renderInto(target: HTMLElement, text: string, fallbacks: FallbackEvent[]): void {
  target.replaceChildren()
  const byPosition = new Map(fallbacks.map((event) => [event.position, event]))
  let plain = ''

  const flush = (): void => {
    if (plain) {
      target.appendChild(document.createTextNode(plain))
      plain = ''
    }
  }

  for (let i = 0; i < text.length; i++) {
    const event = byPosition.get(i)
    if (event === undefined) {
      plain += text[i]
      continue
    }
    flush()
    const span = document.createElement('span')
    span.className = 'fallback'
    span.textContent = text[i]
    span.title =
      event.kind === 'LOW_CONFIDENCE'
        ? `確信度不足 (${event.confidence.toFixed(2)}) 符号 ${event.code}`
        : `変換表に該当なし 符号 ${event.code}`
    target.appendChild(span)
  }
  flush()
}

export function mountApp(root: HTMLElement): void {
  root.innerHTML = TEMPLATE

  const toggle = root.querySelector<HTMLButtonElement>('#toggle')!
  const showJapanese = root.querySelector<HTMLInputElement>('#show-japanese')!
  const status = root.querySelector<HTMLSpanElement>('#status')!
  const level = root.querySelector<HTMLSpanElement>('#level')!
  const diag = root.querySelector<HTMLParagraphElement>('#diag')!
  const canvas = root.querySelector<HTMLCanvasElement>('#spectrum')!
  const japaneseSection = root.querySelector<HTMLElement>('#out-japanese')!

  const european = {
    committed: root.querySelector<HTMLSpanElement>('#out-european .committed')!,
    provisional: root.querySelector<HTMLSpanElement>('#out-european .provisional')!,
  }
  const japanese = {
    committed: root.querySelector<HTMLSpanElement>('#out-japanese .committed')!,
    provisional: root.querySelector<HTMLSpanElement>('#out-japanese .provisional')!,
  }

  let capture: AudioCapture | null = null
  let client: DecoderClient | null = null
  let monitor: SignalMonitor | null = null
  let numThreads = 1

  /** Worker からの結果を画面に反映する (ライブ受信とファイル入力の共通経路)。 */
  const showResult = (response: TextResponse): void => {
    renderInto(european.committed, response.european.committed, response.european.committedFallbacks)
    renderInto(european.provisional, response.european.provisional, response.european.provisionalFallbacks)
    renderInto(japanese.committed, response.japanese.committed, response.japanese.committedFallbacks)
    renderInto(japanese.provisional, response.japanese.provisional, response.japanese.provisionalFallbacks)
    diag.textContent =
      `推論 ${response.decodeMs.toFixed(0)} ms / hop ${HOP_S}s / ` +
      `確定遅延 ${COMMIT_LAG_S}s / スレッド ${numThreads}`
  }

  showJapanese.addEventListener('change', () => {
    japaneseSection.hidden = !showJapanese.checked
  })

  const stop = async (): Promise<void> => {
    client?.finalize()
    monitor?.stop()
    await capture?.stop()
    client?.terminate()
    capture = null
    client = null
    monitor = null
    toggle.textContent = '受信開始'
    status.textContent = '停止中'
  }

  const start = async (): Promise<void> => {
    toggle.disabled = true
    status.textContent = 'モデル読込中…'
    try {
      numThreads = resolveThreadCount()

      client = await DecoderClient.create({
        modelUrl: MODEL_URL,
        numThreads,
        hopS: HOP_S,
        commitLagS: COMMIT_LAG_S,
        decodeLeftContextS: DECODE_LEFT_CONTEXT_S,
        onText: showResult,
        onError: (message) => {
          status.textContent = `エラー: ${message}`
        },
      })

      status.textContent = 'マイク準備中…'
      capture = await AudioCapture.start({
        onAudio: (block) => client?.pushAudio(block),
      })

      monitor = new SignalMonitor(canvas, capture.analyser, (dbfs) => {
        level.textContent = dbfs.toFixed(1)
      })
      monitor.start()

      toggle.textContent = '受信停止'
      status.textContent = capture.usedFallbackResampler
        ? `受信中 (${capture.sampleRate} Hz → 8000 Hz 変換)`
        : '受信中 (8000 Hz)'
    } catch (error) {
      status.textContent = `エラー: ${error instanceof Error ? error.message : String(error)}`
      await stop()
    } finally {
      toggle.disabled = false
    }
  }

  toggle.addEventListener('click', () => {
    void (capture ? stop() : start())
  })
}
```

- [ ] **Step 3: スタイルとエントリを用意する**

`web/src/style.css`:

```css
:root {
  font-family: system-ui, sans-serif;
  line-height: 1.6;
}

body {
  margin: 0;
  padding: 1.5rem;
  background: #14171a;
  color: #e8e8e8;
}

.controls {
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-bottom: 1rem;
}

button {
  padding: 0.5rem 1.2rem;
  font-size: 1rem;
  cursor: pointer;
}

canvas {
  width: 100%;
  max-width: 640px;
  background: #0b0d0f;
  border: 1px solid #333;
}

.level,
.diag {
  color: #9aa0a6;
  font-size: 0.85rem;
}

.output {
  margin-top: 1.5rem;
}

.output h2 {
  font-size: 0.9rem;
  color: #9aa0a6;
  margin-bottom: 0.25rem;
}

.text {
  font-family: ui-monospace, monospace;
  font-size: 1.4rem;
  letter-spacing: 0.05em;
  word-break: break-all;
  min-height: 3rem;
  margin: 0;
}

/* 確定は通常色、暫定はグレー (後から変わる可能性がある) */
.committed {
  color: #e8e8e8;
}

.provisional {
  color: #6b7076;
}

/* 「?」は由来をツールチップで確認できる (TABLE_MISS / LOW_CONFIDENCE) */
.fallback {
  color: #d08a3e;
  cursor: help;
}
```

`web/src/main.ts`:

```ts
import './style.css'
import { mountApp } from './ui/app'

const root = document.getElementById('app')
if (root) mountApp(root)
```

`web/index.html` の `<script>` 行を差し替える:

```html
    <script type="module" src="/src/main.ts"></script>
```

- [ ] **Step 4: ベンチを別ページに移す**

`web/src/bench.ts` を `web/src/bench/bench.ts` に移動し、`web/bench.html` を作る。

```html
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <title>ORT Web ベンチ</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/bench/bench.ts"></script>
  </body>
</html>
```

`web/vite.config.ts` に複数エントリを追加:

```ts
import { defineConfig } from 'vite'
import { resolve } from 'node:path'

export default defineConfig({
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        bench: resolve(__dirname, 'bench.html'),
      },
    },
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
```

- [ ] **Step 5: 型チェックとテストを通す**

```bash
cd web && npx tsc --noEmit && npx vitest run
```

Expected: 型エラーなし、全テスト pass

- [ ] **Step 6: ビルドが通ることを確認**

```bash
cd web && npm run build
```

Expected: `dist/` が生成される

- [ ] **Step 7: コミット**

```bash
git add -A web/
git commit -m "feat: ブラウザ版CWデコーダのUIを追加"
```

(`bench.ts` の移動も `git add -A` で追跡される)

---

### Task 14: WAV ファイル入力

設計書 §1 の「開発者向けの入口」。録音済み wav をその場でデコードでき、デスクトップ版との突き合わせが手作業でできるようになる。

**Files:**
- Create: `web/src/audio/decode-file.ts`
- Test: `web/tests/decode-file.test.ts`
- Modify: `web/src/ui/app.ts` (ファイル入力の追加)

**Interfaces:**
- Produces:
  - `export function planChunks(totalSamples: number, chunkSamples: number, overlapSamples: number): Array<{ start: number; end: number }>`
  - `export async function fileToWave8k(file: File): Promise<Float32Array>`

- [ ] **Step 1: 失敗するテストを書く**

チャンク分割は純粋関数なので Node でテストできる。`fileToWave8k` はブラウザ API に依存するためテスト対象外にする。

`web/tests/decode-file.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { planChunks } from '../src/audio/decode-file'

describe('planChunks', () => {
  it('短い入力は 1 チャンク', () => {
    expect(planChunks(1000, 8000, 800)).toEqual([{ start: 0, end: 1000 }])
  })

  it('全域を覆う', () => {
    const chunks = planChunks(20000, 8000, 800)
    expect(chunks[0].start).toBe(0)
    expect(chunks[chunks.length - 1].end).toBe(20000)
    // 隣接チャンクに隙間が無い
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].start).toBeLessThan(chunks[i - 1].end)
    }
  })

  it('チャンクが overlap ぶん重なる', () => {
    const chunks = planChunks(20000, 8000, 800)
    expect(chunks[1].start).toBe(8000 - 800)
  })

  it('長さ 0 なら空', () => {
    expect(planChunks(0, 8000, 800)).toEqual([])
  })
})
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd web && npx vitest run tests/decode-file.test.ts
```

Expected: FAIL — `Failed to resolve import "../src/audio/decode-file"`

- [ ] **Step 3: 実装する**

`web/src/audio/decode-file.ts`:

```ts
/**
 * WAV ファイルを 8 kHz mono float32 に変換する (開発者向けの入口).
 *
 * リサンプルは OfflineAudioContext に任せる。ブラウザ内蔵のリサンプラは
 * グラフ内で状態を保持するため、ブロック境界の歪みが生じない。
 */
const TARGET_SAMPLE_RATE = 8000

/**
 * 長い音声を推論しやすい長さに分割する計画を返す。
 *
 * BiLSTM は双方向なので、チャンク境界の前後は文脈が欠ける。
 * overlap ぶん重ねて、呼び出し側が重複区間を捨てられるようにする。
 */
export function planChunks(
  totalSamples: number,
  chunkSamples: number,
  overlapSamples: number,
): Array<{ start: number; end: number }> {
  if (totalSamples <= 0) return []
  if (totalSamples <= chunkSamples) return [{ start: 0, end: totalSamples }]

  const stride = chunkSamples - overlapSamples
  const chunks: Array<{ start: number; end: number }> = []
  for (let start = 0; start < totalSamples; start += stride) {
    const end = Math.min(start + chunkSamples, totalSamples)
    chunks.push({ start, end })
    if (end === totalSamples) break
  }
  return chunks
}

/** ファイルを 8 kHz mono float32 の波形にする。 */
export async function fileToWave8k(file: File): Promise<Float32Array> {
  const bytes = await file.arrayBuffer()

  // decodeAudioData はファイル本来のレートでデコードする
  const decodeContext = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await decodeContext.decodeAudioData(bytes)
  } finally {
    await decodeContext.close()
  }

  // 8 kHz の OfflineAudioContext を通してリサンプル + モノラル化する
  const length = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE))
  const offline = new OfflineAudioContext(1, length, TARGET_SAMPLE_RATE)
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start()
  const rendered = await offline.startRendering()

  return rendered.getChannelData(0).slice()
}
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cd web && npx vitest run tests/decode-file.test.ts
```

Expected: 4 passed

- [ ] **Step 5: UI にファイル入力を足す**

`web/src/ui/app.ts` の `TEMPLATE` の `.controls` に 1 行足す:

```html
  <label class="file">WAV を読む<input type="file" id="wav" accept=".wav,audio/*" /></label>
```

`mountApp` の中、`toggle.addEventListener` の直前に追加する:

```ts
  const wavInput = root.querySelector<HTMLInputElement>('#wav')!

  wavInput.addEventListener('change', () => {
    const file = wavInput.files?.[0]
    if (!file) return
    void (async () => {
      if (capture) await stop()
      status.textContent = `${file.name} を読み込み中…`
      try {
        const { fileToWave8k, planChunks } = await import('../audio/decode-file')
        const wave = await fileToWave8k(file)

        numThreads = resolveThreadCount()
        client = await DecoderClient.create({
          modelUrl: MODEL_URL,
          numThreads,
          hopS: HOP_S,
          commitLagS: COMMIT_LAG_S,
          decodeLeftContextS: DECODE_LEFT_CONTEXT_S,
          // ライブ受信と同じ表示経路を通す
          onText: showResult,
          onError: (message) => {
            status.textContent = `エラー: ${message}`
          },
        })

        // ライブ受信と同じ経路を通す (hop ごとに push して再デコードさせる)
        const blockSamples = 400
        for (const chunk of planChunks(wave.length, blockSamples, 0)) {
          client.pushAudio(wave.slice(chunk.start, chunk.end))
          // Worker に処理の隙を与える
          await new Promise((resolve) => setTimeout(resolve, 0))
        }
        client.finalize()
        status.textContent = `${file.name} をデコードしました (${(wave.length / 8000).toFixed(1)} 秒)`
      } catch (error) {
        status.textContent = `エラー: ${error instanceof Error ? error.message : String(error)}`
      }
    })()
  })
```

`planChunks` はここでは単純なブロック分割 (overlap 0) に使う。スライディング
ウィンドウ側が窓と文脈を管理するため、ここで重ねる必要はない。

- [ ] **Step 6: 型チェックと全テストを通す**

```bash
cd web && npx tsc --noEmit && npx vitest run
```

Expected: 型エラーなし、全テスト pass

- [ ] **Step 7: 手で確認する**

```bash
cd web && npm run dev
```

`sample_wav/oubun.wav` を読み込ませ、`web/tests/fixtures/golden.json` の
`textEuropean` と見比べる (完全一致するとは限らない。ゴールデンテストは一括
デコード、こちらはスライディングウィンドウ経由のため確定条件が異なる)。

- [ ] **Step 8: コミット**

```bash
git add web/src/audio/decode-file.ts web/tests/decode-file.test.ts web/src/ui/app.ts
git commit -m "feat: WAVファイル入力を追加 (デスクトップ版との突き合わせ用)"
```

---

### Task 15: 実信号での確認と README

設計書 §12 Step 5。実際の受信音でデスクトップ版と突き合わせる。

**Files:**
- Create: `web/README.md`
- Modify: `.claude/CLAUDE.md` (ブラウザ版の存在と再生成手順を追記)
- Create: `docs/browser_decoder_field_test.md`

- [ ] **Step 1: README を書く**

`web/README.md`:

```markdown
# ブラウザ版 CW デコーダ

学習済みモデルを ONNX 化し、ブラウザ上でマイク / ライン入力から
リアルタイムに CW をデコードします。

設計書: `../docs/superpowers/specs/2026-08-05-browser-cw-decoder-design.md`

## 準備

モデルと符号表と fixture は Python 側から生成します。
`cw.onnx` は 17 MB あるため git 管理外です。**clone 直後は必ず再生成してください。**

```bash
# リポジトリのルート (cw-decorder/) で実行
.venv/Scripts/python.exe scripts/export_onnx.py      # → web/public/model/cw.onnx
.venv/Scripts/python.exe scripts/export_tokens.py    # → web/src/generated/tokens.ts
.venv/Scripts/python.exe scripts/export_golden.py    # → web/tests/fixtures/
```

## 起動

```bash
cd web
npm install
npm run dev
```

## テスト

```bash
cd web
npm test
```

`tests/golden.test.ts` は Python の推論結果との完全一致を確認します。
モデルや fixture が未生成の場合は自動でスキップされます。

## 注意

- 符号表 (`src/generated/tokens.ts`) は自動生成です。手で編集しないでください。
  唯一の真正なソースは `src/tokens/morse_tokens.py` です
- `getUserMedia` の AGC / ノイズ抑制 / エコーキャンセルはすべて無効にしています。
  有効にすると CW のトーンが壊れます
- ベンチページは `/bench.html` です
- 「WAV を読む」で録音済みファイルをデコードできます。デスクトップ版との
  突き合わせに使ってください
- 「?」にマウスを乗せると由来が出ます (変換表に該当なし / 確信度不足)
```

- [ ] **Step 2: 実信号で確認する**

```bash
cd web && npm run dev
```

無線機の音声を PC の入力に繋ぎ、以下を確認する。

1. スペクトル表示に CW のトーンが見えるか
2. 入力レベルが -40〜-10 dBFS 程度に収まっているか
3. 暫定文字 (グレー) が 1〜2 秒で出るか
4. 確定文字 (通常色) が後から固まり、**確定後に書き換わらない**か
5. 同じ音声をデスクトップ版でもデコードし、結果が一致するか
6. `推論 XX ms` の表示が hop (1000 ms) より十分小さいか

- [ ] **Step 3: 結果を記録する**

`docs/browser_decoder_field_test.md` を作成し、上の 6 項目それぞれの結果を書く。
食い違いがあった場合は、デスクトップ版の出力とブラウザ版の出力を並べて残す。

- [ ] **Step 4: CLAUDE.md に追記する**

`.claude/CLAUDE.md` の「注意事項」に追記:

```markdown
- ブラウザ版 (`web/`) がある。符号表は `scripts/export_tokens.py` で
  `morse_tokens.py` から生成する。**`morse_tokens.py` を変更したら必ず再生成すること**
  (`tests/test_export_tokens.py` が差分を検出する)
- モデルを再学習したら `scripts/export_onnx.py` と `scripts/export_golden.py` を
  再実行してブラウザ版の ONNX と fixture を更新する
```

- [ ] **Step 5: 全テストを通す**

```bash
.venv/Scripts/python.exe -m pytest -q
cd web && npx vitest run && npx tsc --noEmit
```

Expected: すべて pass

- [ ] **Step 6: コミット**

```bash
git add web/README.md .claude/CLAUDE.md docs/browser_decoder_field_test.md
git commit -m "docs: ブラウザ版のREADMEと実信号確認の結果を追加"
```

---

## 実装順序の依存関係

```
Task 1 (メル変換)
  └─ Task 2 (ONNX エクスポート)
       ├─ Task 3 (web 初期化 + 実測)   ← Step 0 の判定ポイント
       └─ Task 6 (ゴールデン fixture)
Task 4 (commit_lag 掃引)  ← Task 1〜3 と独立。並行可能
Task 5 (符号表生成)       ← Task 3 の後 (web/ が必要)
Task 3 → Task 7 (CTC) → Task 9 (スライディングウィンドウ)
Task 5 → Task 8 (変換器)
Task 2, 5, 6, 7, 8, 9 → Task 10 (ONNX エンジン + ゴールデンテスト)
Task 3 → Task 11 (音声取得)
Task 8, 9, 10 → Task 12 (Worker)
Task 11, 12 → Task 13 (UI)
Task 13 → Task 14 (WAV ファイル入力)
Task 14 → Task 15 (実信号確認 + README)
```

**Task 3 が赤 (8.5 秒のデコードが 0.8 秒超) だった場合は、そこで一度止めて設計判断をやり直すこと。**
Task 4 の掃引結果と合わせて `HOP_S` / `COMMIT_LAG_S` / `DECODE_LEFT_CONTEXT_S`
(`web/src/ui/app.ts` の定数) を決め、Task 13 で反映する。

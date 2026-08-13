"""音声入力キャプチャ (sounddevice ラッパ).

要件 §3.4.1:

- マイク/ライン入力をキャプチャ
- 入力デバイスをUIで選択可能
- リングバッファに蓄積し推論スレッドと分離

実装方針:
- 取得は device のネイティブサンプルレートで行う (8 kHz 非対応の機器が多い)
- 取得後に scipy で 8 kHz にリサンプル
- 取得ブロックを Thread-safe ``queue.Queue`` に積む
- レベルメータ用の RMS は同時に計算
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except OSError:                  # 環境によってはバックエンド未インストール
    sd = None                    # type: ignore[assignment]
    _SD_AVAILABLE = False

try:
    import soxr
    _SOXR_AVAILABLE = True
except ImportError:
    soxr = None                  # type: ignore[assignment]
    _SOXR_AVAILABLE = False


@dataclass(frozen=True)
class InputDeviceInfo:
    """入力デバイス情報."""

    index: int
    name: str
    channels: int
    default_samplerate: float


def list_input_devices() -> list[InputDeviceInfo]:
    """入力可能なオーディオデバイス一覧を返す."""
    if not _SD_AVAILABLE or sd is None:
        return []
    result: list[InputDeviceInfo] = []
    try:
        devices = sd.query_devices()
    except sd.PortAudioError:
        return []
    for i, d in enumerate(devices):
        max_in = int(d.get("max_input_channels", 0))
        if max_in <= 0:
            continue
        result.append(InputDeviceInfo(
            index=i,
            name=str(d.get("name", f"device {i}")),
            channels=max_in,
            default_samplerate=float(d.get("default_samplerate", 0.0)),
        ))
    return result


def _resample_to_8k(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """単純整数比リサンプル (scipy polyphase)."""
    if source_sr == 8000:
        return audio
    from scipy.signal import resample_poly
    g = np.gcd(int(source_sr), 8000)
    up = 8000 // g
    down = int(source_sr) // g
    return resample_poly(audio, up, down).astype(np.float32)


class AudioCapture:
    """マイク/ライン入力からの音声キャプチャ.

    ``start`` でキャプチャ開始、``read_block`` で 1 ブロックを取り出す.
    ``level_db_rms`` は最新ブロックの RMS dBFS を保持 (UI レベルメータ用).
    """

    def __init__(
        self,
        device_index: int | None = None,
        target_sample_rate: int = 8000,
        block_duration_s: float = 0.05,
        max_queue_blocks: int = 256,
    ) -> None:
        if not _SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice が初期化できませんでした (PortAudio 未インストール?)"
            )
        self.device_index = device_index
        self.target_sample_rate = target_sample_rate
        self.block_duration_s = block_duration_s
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_blocks)
        self._stream: sd.InputStream | None = None  # type: ignore[name-defined]
        self._source_sr: int | None = None
        self._level_rms: float = 0.0
        self._lock = threading.Lock()
        # soxr ステートフルリサンプラ (start 時に作成)
        self._resampler = None

    @property
    def source_sample_rate(self) -> int | None:
        return self._source_sr

    @property
    def level_db_rms(self) -> float:
        """直近ブロックの RMS dBFS (full scale 比). 無音時は -120 dB."""
        with self._lock:
            rms = self._level_rms
        if rms < 1e-6:
            return -120.0
        return 20.0 * float(np.log10(rms))

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            # オーバーフロー等は警告のみ (ログ機能は呼び出し側で)
            pass
        mono = indata[:, 0].astype(np.float32, copy=True)
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        with self._lock:
            self._level_rms = rms
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # 古いブロックを 1 つ捨てて新規を入れる
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(mono)
            except queue.Empty:
                pass

    def start(self) -> None:
        assert sd is not None
        info = sd.query_devices(self.device_index, kind="input")
        source_sr = int(info["default_samplerate"])
        self._source_sr = source_sr
        # ステートフルストリーミングリサンプラ (soxr)
        if source_sr != self.target_sample_rate and _SOXR_AVAILABLE:
            self._resampler = soxr.ResampleStream(
                source_sr, self.target_sample_rate,
                num_channels=1, dtype="float32",
            )
        else:
            self._resampler = None
        block_frames = max(1, int(self.block_duration_s * source_sr))
        self._stream = sd.InputStream(
            samplerate=source_sr,
            blocksize=block_frames,
            channels=1,
            dtype="float32",
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                # abort() は stop() より即時で、コールバック中の race を回避
                self._stream.abort(ignore_errors=True)
            except Exception:                       # noqa: BLE001
                pass
            try:
                self._stream.close(ignore_errors=True)
            except Exception:                       # noqa: BLE001
                pass
            self._stream = None
        self._resampler = None

    def read_block(self, timeout: float = 0.1) -> np.ndarray | None:
        """1 ブロック取得し、8 kHz にリサンプル. ``None`` はタイムアウト.

        ``drain`` を使う場合は呼ばないこと (バッファ二重化).
        互換性のため残しているが、ストリーミング推論には ``drain`` 推奨.
        """
        try:
            block = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if self._source_sr is None or self._source_sr == 8000:
            return block
        if self._resampler is not None:
            out = self._resampler.resample_chunk(block, last=False)
            return np.asarray(out, dtype=np.float32) if out.size else None
        return _resample_to_8k(block, self._source_sr)

    def drain(self) -> list[np.ndarray]:
        """新規キャプチャ分を soxr ステートフルリサンプラで 8 kHz 変換.

        soxr.ResampleStream は内部状態を保持し、ブロック境界での歪みを
        生じない真のストリーミングリサンプリングを提供する.
        scipy.signal.resample_poly はステートレスで境界歪みが発生し、
        CW の dot/dash を破壊するため使えない.
        """
        blocks: list[np.ndarray] = []
        while True:
            try:
                b = self._queue.get_nowait()
            except queue.Empty:
                break
            blocks.append(b)
        if not blocks:
            return []

        new_native = np.concatenate(blocks)

        if self._source_sr is None or self._source_sr == self.target_sample_rate:
            return [new_native]
        if self._resampler is None:
            # soxr が無い環境のフォールバック (精度低下)
            return [_resample_to_8k(new_native, self._source_sr)]

        out = self._resampler.resample_chunk(new_native, last=False)
        if out.size == 0:
            return []
        return [np.asarray(out, dtype=np.float32)]


__all__ = [
    "AudioCapture",
    "InputDeviceInfo",
    "list_input_devices",
]

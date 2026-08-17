"""配布版が、読めないモデルを指した設定で落ちないことを固定する.

**実際に落ちた (2026-08-16)。** 開発機の設定に `models/full/best_infer.pt` が
残っており、PyTorch を同梱しない配布物がそれを読もうとして
「Unhandled exception in script」で起動できなかった。

利用者の環境でも同じことが起きる:

* 以前 PyTorch 版を使っていて、設定に `.pt` のパスが残っている
* モデルを別の場所へ移した / インストールし直した

**起動できないのが最悪である。** 読めなければ同梱モデルへ戻す。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.app import main_window
from src.infer.settings import AppSettings


class _FakeEngine:
    def __init__(self, path: Path) -> None:
        self.model_path = Path(path)

    def decode_chunk(self, waveform):    # pragma: no cover - 使わない
        return []

    @property
    def frame_hop_samples(self) -> int:
        return 80


class TestFallsBackToBundledModel:
    def test_torch_missing_falls_back_to_onnx(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """**本番で落ちた筋。** `.pt` を読めないとき同梱 ONNX へ戻ること."""
        stale = tmp_path / "best_infer.pt"
        stale.write_bytes(b"x")
        bundled = tmp_path / "cw.onnx"
        bundled.write_bytes(b"x")

        tried: list[Path] = []

        def fake_load(path, device="cpu", threads=0):
            tried.append(Path(path))
            if Path(path).suffix == ".pt":
                raise ModuleNotFoundError("No module named 'torch'")
            return _FakeEngine(path)

        monkeypatch.setattr(main_window, "load_engine", fake_load)
        monkeypatch.setattr(main_window, "default_model_path", lambda: bundled)
        monkeypatch.setattr(main_window, "resolve_model_path", lambda saved: stale)

        engine = main_window._build_engine(AppSettings(checkpoint_path=str(stale)))

        assert tried == [stale, bundled], "同梱モデルへの切り替えが起きていない"
        assert isinstance(engine, _FakeEngine)
        assert engine.model_path == bundled

    def test_broken_model_file_also_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """壊れたファイルを指していても同じこと (原因の種類で分けない)."""
        broken = tmp_path / "こわれた.onnx"
        broken.write_bytes(b"not an onnx")
        bundled = tmp_path / "cw.onnx"
        bundled.write_bytes(b"x")

        def fake_load(path, device="cpu", threads=0):
            if Path(path) == broken:
                raise RuntimeError("Protobuf parsing failed")
            return _FakeEngine(path)

        monkeypatch.setattr(main_window, "load_engine", fake_load)
        monkeypatch.setattr(main_window, "default_model_path", lambda: bundled)
        monkeypatch.setattr(main_window, "resolve_model_path", lambda saved: broken)

        engine = main_window._build_engine(AppSettings(checkpoint_path=str(broken)))
        assert engine.model_path == bundled

    def test_no_infinite_retry_when_bundled_is_the_broken_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """同梱モデル自体が読めないときは、未学習モデルで画面を出すこと.

        **同じパスを二度試して諦めない**ようにする (無限に試すと起動しない)。
        """
        bundled = tmp_path / "cw.onnx"
        bundled.write_bytes(b"x")
        calls: list[Path] = []

        def fake_load(path, device="cpu", threads=0):
            calls.append(Path(path))
            raise RuntimeError("読めない")

        untrained_made = []

        class _FakeInferenceEngine:
            @staticmethod
            def untrained(device="cpu"):
                untrained_made.append(True)
                return _FakeEngine(Path("untrained"))

        monkeypatch.setattr(main_window, "load_engine", fake_load)
        monkeypatch.setattr(main_window, "default_model_path", lambda: bundled)
        monkeypatch.setattr(main_window, "resolve_model_path", lambda saved: bundled)
        monkeypatch.setattr(main_window, "resolve_torch_device", lambda pref: "cpu")
        monkeypatch.setitem(
            __import__("sys").modules, "src.infer.engine",
            type("m", (), {"InferenceEngine": _FakeInferenceEngine}),
        )

        engine = main_window._build_engine(AppSettings(checkpoint_path=str(bundled)))

        assert calls == [bundled], "同じモデルを二度読みにいっている"
        assert untrained_made == [True]
        assert engine.model_path == Path("untrained")

"""run_app.py の --net-source 引数のテスト (アプリは起動しない)."""
from __future__ import annotations

import scripts.run_app as run_app


def test_net_source_is_passed_through(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(*, checkpoint_path=None, net_source=None, device=None):
        captured["checkpoint_path"] = checkpoint_path
        captured["net_source"] = net_source
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    assert run_app.main(["--ckpt", "models/x.pt", "--net-source", "192.168.1.20"]) == 0
    assert captured["checkpoint_path"] == "models/x.pt"
    assert captured["net_source"] == "192.168.1.20"


def test_net_source_defaults_to_none(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(*, checkpoint_path=None, net_source=None, device=None):
        captured["net_source"] = net_source
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    assert run_app.main([]) == 0
    assert captured["net_source"] is None


def test_net_source_with_valid_port_is_passed_through(monkeypatch) -> None:
    """host:port 形式でポートが数値のときは従来どおり run_app_main が呼ばれる."""
    captured: dict[str, object] = {}

    def _fake_main(*, checkpoint_path=None, net_source=None, device=None):
        captured["net_source"] = net_source
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    assert run_app.main(["--net-source", "192.168.1.20:45000"]) == 0
    assert captured["net_source"] == "192.168.1.20:45000"


def test_net_source_invalid_port_rejected_before_app_starts(monkeypatch, capsys) -> None:
    """ポート部が数値でないときは CLI の時点で弾き、run_app_main を呼ばない."""
    called = False

    def _fake_main(*, checkpoint_path=None, net_source=None, device=None):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    result = run_app.main(["--net-source", "192.168.1.20:abc"])
    assert result == 1
    assert called is False
    captured = capsys.readouterr()
    assert "--net-source" in captured.err


def test_device_defaults_to_none_so_settings_win(monkeypatch) -> None:
    """--device 無指定なら None を渡し、設定 (既定 cpu) を尊重する."""
    captured: dict[str, object] = {}

    def _fake_main(*, checkpoint_path=None, net_source=None, device=None):
        captured["device"] = device
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    assert run_app.main([]) == 0
    assert captured["device"] is None


def test_device_is_passed_through(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_main(*, checkpoint_path=None, net_source=None, device=None):
        captured["device"] = device
        return 0

    monkeypatch.setattr(run_app, "run_app_main", _fake_main)
    assert run_app.main(["--device", "cuda"]) == 0
    assert captured["device"] == "cuda"

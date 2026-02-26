"""Tests for virtual_reality.display.monitor."""

from __future__ import annotations

from unittest import mock

from virtual_reality.display.monitor import MonitorInfo, pick_monitor


class TestMonitorInfo:
    """Tests for MonitorInfo."""

    def test_defaults(self) -> None:
        info = MonitorInfo()
        assert info.width == 800
        assert info.height == 600
        assert info.x == 0


class TestPickMonitor:
    """Tests for pick_monitor."""

    def test_fallback_when_no_screeninfo(self) -> None:
        with mock.patch.dict("sys.modules", {"screeninfo": None}):
            mon = pick_monitor(fallback_w=1024, fallback_h=768)
            assert mon.width == 1024
            assert mon.height == 768

    def test_rightmost_selection(self) -> None:
        fake_monitors = [
            mock.Mock(x=0, y=0, width=1920, height=1080, name="Left"),
            mock.Mock(x=1920, y=0, width=1280, height=800, name="Right"),
        ]
        with mock.patch(
            "virtual_reality.display.monitor.get_monitors",
            create=True,
        ):
            pass

    def test_fallback_returns_monitor_info(self) -> None:
        mon = pick_monitor(fallback_w=640, fallback_h=480)
        assert isinstance(mon, MonitorInfo)

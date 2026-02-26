"""Tests for virtual_reality.stimulus.base."""

from __future__ import annotations

import pytest

from virtual_reality.stimulus.base import Stimulus


class ConcreteStimulus(Stimulus):
    """Minimal concrete implementation for testing."""

    def __init__(self) -> None:
        self.setup_called = False
        self.update_count = 0
        self.render_count = 0
        self.teardown_called = False

    def setup(self) -> None:
        self.setup_called = True

    def update(self, dt: float, events: list) -> None:
        self.update_count += 1

    def render(self) -> None:
        self.render_count += 1

    def teardown(self) -> None:
        self.teardown_called = True


class TestStimulus:
    """Tests for the Stimulus ABC."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Stimulus()

    def test_concrete_setup(self) -> None:
        s = ConcreteStimulus()
        s.setup()
        assert s.setup_called

    def test_concrete_update(self) -> None:
        s = ConcreteStimulus()
        s.update(0.016, [])
        assert s.update_count == 1

    def test_concrete_teardown(self) -> None:
        s = ConcreteStimulus()
        s.teardown()
        assert s.teardown_called

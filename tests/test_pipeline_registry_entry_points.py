"""Tests for pipeline stage entry point discovery."""

from rtmdk.pipeline.base import PipelineStage, PipelineContext
from rtmdk.pipeline.registry import StageRegistry


class DummyStage(PipelineStage):
    name = "dummy"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        return ctx


class MockEntryPoint:
    def __init__(self, name, cls):
        self.name = name
        self._cls = cls

    def load(self):
        return self._cls


def test_discover_entry_points_registers_stages(monkeypatch):
    """Auto-discovery registers valid entry points."""
    registry = StageRegistry()

    def mock_entry_points(group=None):
        if group == "rtmdk.pipeline.stages":
            return [MockEntryPoint("dummy", DummyStage)]
        return []

    monkeypatch.setattr("rtmdk.pipeline.registry._entry_points", mock_entry_points)
    registry.discover_entry_points()

    assert "dummy" in registry.list_stages()
    stage = registry.create("dummy")
    assert isinstance(stage, DummyStage)


def test_discover_entry_points_skips_invalid(monkeypatch):
    """Auto-discovery skips entry points that don't subclass PipelineStage."""
    registry = StageRegistry()

    class NotAStage:
        pass

    def mock_entry_points(group=None):
        if group == "rtmdk.pipeline.stages":
            return [MockEntryPoint("bad", NotAStage)]
        return []

    monkeypatch.setattr("rtmdk.pipeline.registry._entry_points", mock_entry_points)
    registry.discover_entry_points()

    assert "bad" not in registry.list_stages()


def test_discover_entry_points_does_not_override_existing(monkeypatch):
    """Auto-discovery does not override manually registered stages."""
    registry = StageRegistry()
    registry.register("dummy", DummyStage)

    class OtherStage(PipelineStage):
        name = "other"

        def process(self, ctx):
            return ctx

    def mock_entry_points(group=None):
        if group == "rtmdk.pipeline.stages":
            return [MockEntryPoint("dummy", OtherStage)]
        return []

    monkeypatch.setattr("rtmdk.pipeline.registry._entry_points", mock_entry_points)
    registry.discover_entry_points()

    # Should keep the original registration
    stage = registry.create("dummy")
    assert isinstance(stage, DummyStage)

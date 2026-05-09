"""Plugin registry for custom pipeline stages.

Usage:
    from rtmdk.pipeline.registry import StageRegistry
    registry = StageRegistry()
    registry.register("my_rerank", MyRerankStage)

    stage = registry.create("my_rerank", **kwargs)
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Type

from rtmdk.pipeline.base import PipelineStage


class StageRegistry:
    """Global registry for PipelineStage subclasses."""

    def __init__(self):
        self._stages: Dict[str, Type[PipelineStage]] = {}

    def register(self, name: str, stage_cls: Type[PipelineStage]) -> None:
        """Register a stage class under a given name."""
        if not issubclass(stage_cls, PipelineStage):
            raise TypeError(f"{stage_cls} must subclass PipelineStage")
        if name in self._stages:
            raise ValueError(f"Stage '{name}' is already registered")
        self._stages[name] = stage_cls

    def create(self, name: str, **kwargs: Any) -> PipelineStage:
        """Instantiate a registered stage."""
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not registered. Available: {list(self._stages.keys())}")
        return self._stages[name](**kwargs)

    def list_stages(self) -> Dict[str, Type[PipelineStage]]:
        """Return a copy of the registered stages map."""
        return dict(self._stages)


# Global singleton for convenience
GLOBAL_REGISTRY = StageRegistry()

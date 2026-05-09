"""Plugin registry for custom pipeline stages.

Usage:
    from rtmdk.pipeline.registry import StageRegistry
    registry = StageRegistry()
    registry.register("my_rerank", MyRerankStage)

    stage = registry.create("my_rerank", **kwargs)

Third-party packages can auto-register via setuptools entry points:
    [rtmdk.pipeline.stages]
    my_rerank = my_package.stages:MyRerankStage
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Type
import logging

try:
    from importlib.metadata import entry_points as _entry_points
except ImportError:
    try:
        from importlib_metadata import entry_points as _entry_points  # type: ignore
    except ImportError:
        _entry_points = None  # type: ignore

from rtmdk.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


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

    def discover_entry_points(self, group: str = "rtmdk.pipeline.stages") -> None:
        """Auto-register stages from setuptools entry points.

        Entry point format in pyproject.toml/setup.cfg:
            [rtmdk.pipeline.stages]
            my_stage = my_package.stages:MyStageClass
        """
        if _entry_points is None:
            logger.debug("importlib.metadata not available, skipping entry point discovery")
            return

        try:
            eps = _entry_points(group=group)
        except TypeError:
            # Python < 3.10 compatibility
            all_eps = _entry_points()
            eps = all_eps.get(group, [])

        for ep in eps:
            try:
                stage_cls = ep.load()
                if not issubclass(stage_cls, PipelineStage):
                    logger.warning("Entry point %s does not subclass PipelineStage", ep.name)
                    continue
                if ep.name in self._stages:
                    logger.debug("Entry point stage %s already registered", ep.name)
                    continue
                self._stages[ep.name] = stage_cls
                logger.info("Auto-registered pipeline stage: %s", ep.name)
            except Exception as exc:
                logger.warning("Failed to load entry point %s: %s", ep.name, exc)


# Global singleton for convenience
GLOBAL_REGISTRY = StageRegistry()

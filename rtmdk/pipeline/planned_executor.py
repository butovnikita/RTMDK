"""Planned pipeline executor — skips stages based on query characteristics.

Integrates QueryPlanner with PipelineExecutor to dynamically select stages
after the route is known, reducing latency for simple queries.

Usage:
    from rtmdk.pipeline import PlannedPipelineExecutor, QueryPlanner
    planner = QueryPlanner(fast_route_skip={"rerank", "calibrate"})
    executor = PlannedPipelineExecutor(stages, planner=planner)
    ctx = executor.run("What is 2+2?", top_k=5)
    # For fast route: skips rerank + calibrate → ~40% faster
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from rtmdk.pipeline.base import PipelineContext, PipelineStage
from rtmdk.pipeline.executor import PipelineExecutor
from rtmdk.pipeline.planner import QueryPlanner


class PlannedPipelineExecutor(PipelineExecutor):
    """Pipeline executor with query-planning optimization.

    Executes embed + route first, then asks the planner which remaining
    stages to run.  This allows route-dependent stage skipping.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        planner: Optional[QueryPlanner] = None,
        webhook_manager: Optional[Any] = None,
    ):
        super().__init__(stages, webhook_manager=webhook_manager)
        self.planner = planner or QueryPlanner()
        # Build name → stage mapping for O(1) lookup
        self._stage_map: Dict[str, PipelineStage] = {}
        for s in stages:
            self._stage_map[s.name] = s

    def run(
        self,
        query_text: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> PipelineContext:
        ctx = PipelineContext(
            query_text=query_text,
            top_k=top_k,
            session_id=session_id,
            embedding=embedding,
        )

        # Phase 1: Always run embed (required for everything else)
        embed_stage = self._stage_map.get("embed")
        if embed_stage:
            ctx = embed_stage.run(ctx)
            self._dispatch_stage_event(ctx, embed_stage)

        # Phase 2: Run route if available
        route_stage = self._stage_map.get("route")
        if route_stage:
            ctx = route_stage.run(ctx)
            self._dispatch_stage_event(ctx, route_stage)

        # Phase 3: Plan remaining stages based on route + query
        plan = self.planner.plan(
            query_text=query_text,
            route=ctx.route,
            top_k=top_k,
            available_stages=list(self._stage_map.keys()),
        )

        # Record skipped stages as zero-latency metrics for observability
        for skipped, reason in plan.skip_reasons.items():
            ctx.add_metric(
                name=skipped,
                latency_ms=0.0,
                input_count=0,
                output_count=0,
                error=None,
                degraded=False,
            )
            # Tag the metric with skip reason via a custom attribute on ctx
            if not hasattr(ctx, "_skip_reasons"):
                ctx._skip_reasons = {}  # type: ignore[attr-defined]
            ctx._skip_reasons[skipped] = reason  # type: ignore[attr-defined]

        # Phase 4: Execute planned stages (preserve original order)
        for stage_name in plan.stage_names:
            if stage_name in ("embed", "route"):
                continue  # Already executed
            stage = self._stage_map.get(stage_name)
            if stage is None:
                continue
            ctx = stage.run(ctx)
            self._dispatch_stage_event(ctx, stage)
            if ctx.skip_remaining:
                break

        return ctx

    def get_plan(self, query_text: str, route: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Preview the execution plan for a query without running it."""
        plan = self.planner.plan(
            query_text=query_text,
            route=route,
            top_k=top_k,
            available_stages=list(self._stage_map.keys()),
        )
        return plan.to_dict()

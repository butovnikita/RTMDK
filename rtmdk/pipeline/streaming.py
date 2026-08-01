"""SSE streaming executor for pipeline stages.

Yields real-time events as each stage completes, enabling live
progress tracking in dashboards and debug UIs.
"""

from __future__ import annotations
import json
import time
from typing import Any, Dict, Iterator, List, Optional

from rtmdk.pipeline.base import PipelineContext, PipelineStage


def _sse_event(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


class StreamingPipelineExecutor:
    """Execute a pipeline and yield SSE-formatted events per stage.

    Usage (FastAPI endpoint):
        async def stream():
            executor = StreamingPipelineExecutor(stages)
            for chunk in executor.run("hello", top_k=5):
                yield chunk

        return StreamingResponse(stream(), media_type="text/event-stream")
    """

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run(
        self,
        query_text: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> Iterator[str]:
        """Yield SSE events as pipeline stages execute.

        Events:
            pipeline_started — list of planned stages
            stage_started    — stage name
            stage_completed  — stage name + metrics
            stage_degraded   — stage name + error info
            pipeline_completed — final results + total latency
        """
        ctx = PipelineContext(
            query_text=query_text,
            top_k=top_k,
            session_id=session_id,
            embedding=embedding,
        )
        stage_names = [s.name for s in self.stages if s.enabled]

        yield _sse_event(
            {
                "event": "pipeline_started",
                "query": query_text,
                "top_k": top_k,
                "session_id": session_id,
                "stages": stage_names,
            }
        )

        total_start = time.perf_counter()

        for stage in self.stages:
            if not stage.enabled:
                continue
            if ctx.skip_remaining:
                break

            yield _sse_event(
                {
                    "event": "stage_started",
                    "stage": stage.name,
                }
            )

            t0 = time.perf_counter()
            try:
                ctx = stage.run(ctx)
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                yield _sse_event(
                    {
                        "event": "stage_degraded",
                        "stage": stage.name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": round(latency_ms, 3),
                    }
                )
                continue

            latency_ms = (time.perf_counter() - t0) * 1000
            latest_metric = ctx.metrics[-1] if ctx.metrics else None
            yield _sse_event(
                {
                    "event": "stage_completed",
                    "stage": stage.name,
                    "latency_ms": round(latency_ms, 3),
                    "output_count": latest_metric.output_count if latest_metric else 0,
                    "degraded": latest_metric.degraded if latest_metric else False,
                    "breaker_state": ctx.breaker_states.get(stage.name),
                }
            )

        total_latency_ms = (time.perf_counter() - total_start) * 1000

        yield _sse_event(
            {
                "event": "pipeline_completed",
                "results": [
                    {"node_id": nid, "score": round(float(score), 6), "content": str(content)[:200]}
                    for nid, score, content in ctx.results[:top_k]
                ],
                "route": ctx.route,
                "total_latency_ms": round(total_latency_ms, 3),
                "degraded_stages": ctx.degraded_stages,
                "breaker_states": ctx.breaker_states,
                "metrics": [m.to_dict() for m in ctx.metrics],
            }
        )

    async def run_async(
        self,
        query_text: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ):
        """Async generator wrapper for run()."""
        # run() is a generator; we can't easily run it in executor.
        # Instead, iterate synchronously — the GIL is fine for yielding.
        for chunk in self.run(query_text, top_k, session_id, embedding):
            yield chunk

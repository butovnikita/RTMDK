"""rtmdk/server/graphql_schema.py — GraphQL schema for RTMDK.

Basic GraphQL API for querying memory, nodes, and health.
"""

from typing import List, Optional

import strawberry


@strawberry.type
class Health:
    status: str
    version: str
    memory_nodes: int


@strawberry.type
class MemoryNode:
    id: str
    content: str
    salience: float
    phase: float
    amplitude: float


@strawberry.type
class MemoryResult:
    node_id: str
    score: float
    content: str


@strawberry.type
class StageMetric:
    stage: str
    latency_ms: float
    error: Optional[str]
    degraded: bool


@strawberry.type
class PipelineMetrics:
    stages: List[StageMetric]
    total_latency_ms: float
    breaker_states: Optional[str] = None


@strawberry.type
class PipelineResult:
    query: str
    results: List[MemoryResult]
    route: Optional[str]
    total: int
    metrics: PipelineMetrics


@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> Health:
        import rtmdk.server.app as app_mod
        memory = getattr(app_mod, "memory", None)
        node_count = 0
        if memory and memory.field is not None:
            node_count = len(memory.field.nodes)
        return Health(status="ok", version="8.2.0", memory_nodes=node_count)

    @strawberry.field
    def node(self, id: str) -> Optional[MemoryNode]:
        import rtmdk.server.app as app_mod
        memory = getattr(app_mod, "memory", None)
        if memory is None or memory.field is None:
            return None
        n = memory.field.nodes.get(id)
        if n is None:
            return None
        content = ""
        if isinstance(n.content, dict):
            content = n.content.get("text", str(n.content))
        else:
            content = str(n.content)
        return MemoryNode(
            id=n.id, content=content, salience=float(n.salience),
            phase=float(n.phase), amplitude=float(n.amplitude))

    @strawberry.field
    def query_pipeline(self, query: str, top_k: int = 5, session_id: Optional[str] = None) -> Optional[PipelineResult]:
        import rtmdk.server.app as app_mod
        memory = getattr(app_mod, "memory", None)
        if memory is None:
            return None
        try:
            result = memory.retrieve_nodes_pipeline(query, top_k=top_k, session_id=session_id)
            formatted = []
            for nid, score, node in result["results"]:
                content = ""
                if isinstance(node.content, dict):
                    content = node.content.get("text", str(node.content))
                else:
                    content = str(node.content)
                formatted.append(MemoryResult(node_id=nid, score=round(float(score), 4), content=content))
            metrics_data = result.get("metrics", {})
            stages = [
                StageMetric(
                    stage=s.get("stage", ""),
                    latency_ms=s.get("latency_ms", 0.0),
                    error=s.get("error"),
                    degraded=s.get("degraded", False),
                )
                for s in metrics_data.get("stages", [])
            ]
            metrics = PipelineMetrics(
                stages=stages,
                total_latency_ms=metrics_data.get("total_latency_ms", 0.0),
                breaker_states=str(metrics_data.get("breaker_states", {})) if metrics_data.get("breaker_states") else None,
            )
            return PipelineResult(
                query=query,
                results=formatted,
                route=result.get("route"),
                total=len(formatted),
                metrics=metrics,
            )
        except Exception as exc:
            raise Exception(f"Pipeline query failed: {exc}")

    @strawberry.field
    def nodes(self, limit: int = 10, offset: int = 0) -> List[MemoryNode]:
        import rtmdk.server.app as app_mod
        memory = getattr(app_mod, "memory", None)
        if memory is None or memory.field is None:
            return []
        results = []
        for nid in list(memory.field.nodes.keys())[offset:offset + limit]:
            n = memory.field.nodes[nid]
            content = ""
            if isinstance(n.content, dict):
                content = n.content.get("text", str(n.content))
            else:
                content = str(n.content)
            results.append(MemoryNode(
                id=n.id, content=content, salience=float(n.salience),
                phase=float(n.phase), amplitude=float(n.amplitude)))
        return results


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_node(self, content: str, salience: Optional[float] = None) -> MemoryNode:
        import rtmdk.server.app as app_mod
        memory = getattr(app_mod, "memory", None)
        if memory is None or memory.field is None:
            raise Exception("Memory not initialized")
        import numpy as np
        nid = memory.add_node(
            embedding=np.zeros(memory.field.cfg.latent_dim, dtype=np.float32),
            content={"text": content},
        )
        n = memory.field.nodes.get(nid)
        if salience is not None:
            n.salience = salience
        return MemoryNode(
            id=n.id, content=content, salience=float(n.salience),
            phase=float(n.phase), amplitude=float(n.amplitude))

    @strawberry.mutation
    def delete_node(self, id: str) -> bool:
        import rtmdk.server.app as app_mod
        memory = getattr(app_mod, "memory", None)
        if memory is None or memory.field is None:
            raise Exception("Memory not initialized")
        if id in memory.field.nodes:
            del memory.field.nodes[id]
            if id in memory.field.node_index:
                memory.field.node_index.remove(id)
            return True
        return False


schema = strawberry.Schema(query=Query, mutation=Mutation)

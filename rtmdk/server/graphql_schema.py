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

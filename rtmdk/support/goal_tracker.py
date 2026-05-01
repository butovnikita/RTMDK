"""Goal tracker for RTMDK."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional

from rtmdk.nodes import GoalNode

if TYPE_CHECKING:
    pass


class GoalTracker:
    """Tracks user goals, subgoals, and completion progress."""

    def __init__(self, max_goals: int = 20, goal_decay: float = 0.995,
                 completion_threshold: float = 0.8):
        self.max_goals = max_goals
        self.goal_decay = goal_decay
        self.completion_threshold = completion_threshold
        self.goals: Dict[str, GoalNode] = {}
        self._history: List[Dict] = []

    def add_goal(self, description: str, goal_id: Optional[str] = None,
                 subgoals: Optional[List[str]] = None,
                 priority: float = 1.0) -> str:
        gid = goal_id or f"goal_{len(self.goals)}_{int(time.time())}"
        self.goals[gid] = GoalNode(
            id=gid, description=description,
            subgoals=subgoals or [], priority=priority
        )
        self._history.append({"action": "add", "goal_id": gid, "time": time.time()})
        self._enforce_max_goals()
        return gid

    def update_completion(self, goal_id: str, completion: float,
                          related_nodes: Optional[List[str]] = None):
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.completion = min(1.0, max(0.0, completion))
            goal.last_updated = time.time()
            if related_nodes:
                goal.related_nodes = list(set(goal.related_nodes + related_nodes))
            if goal.completion >= self.completion_threshold:
                goal.status = "completed"
            self._history.append({
                "action": "update", "goal_id": goal_id,
                "completion": goal.completion, "time": time.time()
            })

    def get_active_goals(self) -> List[GoalNode]:
        return [g for g in self.goals.values() if g.status == "active"]

    def get_goal_relevance(self, node_id: str) -> float:
        """How relevant is a node to current active goals?"""
        if not self.goals:
            return 0.0
        relevance = 0.0
        for goal in self.get_active_goals():
            if node_id in goal.related_nodes:
                relevance += goal.priority * (1.0 - goal.completion)
            # Check subgoals
            for sg in goal.subgoals:
                if sg in node_id or node_id in sg:
                    relevance += goal.priority * 0.5
        return min(1.0, relevance)

    def decay_goals(self):
        """Decay inactive goals over time."""
        to_remove = []
        for gid, goal in self.goals.items():
            if goal.status == "active":
                goal.priority *= self.goal_decay
                if goal.priority < 0.01:
                    goal.status = "abandoned"
                    to_remove.append(gid)
        for gid in to_remove:
            del self.goals[gid]

    def _enforce_max_goals(self):
        active = self.get_active_goals()
        if len(active) > self.max_goals:
            sorted_goals = sorted(active, key=lambda g: g.priority)
            for goal in sorted_goals[:len(active) - self.max_goals]:
                goal.status = "abandoned"

    def get_state(self) -> Dict:
        return {
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
            "history": self._history[-100:],
        }

    def load_state(self, state: Dict):
        for gid, gdata in state.get("goals", {}).items():
            self.goals[gid] = GoalNode.from_dict(gdata)
        self._history = state.get("history", [])

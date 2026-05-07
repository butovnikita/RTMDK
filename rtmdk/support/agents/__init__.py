"""Agent orchestration for RTMDK."""
from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set

if TYPE_CHECKING:
    from rtmdk.nodes import AgentPlan, Hypothesis, ToolCall


class AgentPlanner:
    def __init__(self, max_depth: int = 3, max_tool_calls: int = 5,
                 tool_timeout: float = 15.0):
        self.max_depth = max_depth
        self.max_tool_calls = max_tool_calls
        self.tool_timeout = tool_timeout
        self._visited_tools: Set[str] = set()
        self._call_count = 0

    def create_plan(self, goal: str, available_tools: List[str],
                    context: Dict[str, Any]) -> "AgentPlan":
        from rtmdk.nodes import AgentPlan
        subtasks = self._decompose_goal(goal, context)
        tools_needed = self._select_tools(goal, subtasks, available_tools)
        return AgentPlan(
            goal=goal, subtasks=subtasks, tools_needed=tools_needed,
            estimated_steps=len(subtasks),
            confidence=self._estimate_confidence(goal, subtasks, tools_needed),
            reasoning=f"Decomposed goal into {len(subtasks)} subtasks"
        )

    def _decompose_goal(
            self, goal: str, context: Dict) -> List[Dict[str, Any]]:
        subtasks = []
        subtasks.append({"type": "retrieve",
                         "description": f"Find memories related to: {goal}",
                         "priority": 1})
        if context.get("hypothesis_verification", False):
            subtasks.append(
                {"type": "verify", "description": "Verify causal hypotheses", "priority": 2})
        subtasks.append({"type": "synthesize",
                         "description": f"Synthesize response for: {goal}",
                         "priority": 3})
        return subtasks[:self.max_depth]

    def _select_tools(self, goal: str, subtasks: List[Dict],
                      available_tools: List[str]) -> List[str]:
        selected = []
        for task in subtasks:
            task_type = task.get("type", "")
            for tool in available_tools:
                if task_type in tool.lower() and tool not in selected:
                    selected.append(tool)
        return selected[:self.max_tool_calls]

    def _estimate_confidence(self, goal: str, subtasks: List[Dict],
                             tools: List[str]) -> float:
        base = 0.5
        base += min(0.2, len(subtasks) * 0.05)
        base += min(0.2, len(tools) * 0.05)
        base += 0.1 if len(subtasks) <= self.max_depth else -0.1
        return min(1.0, max(0.0, base))

    def reset(self):
        self._visited_tools.clear()
        self._call_count = 0

    def can_call_tool(self, tool_name: str) -> bool:
        if tool_name in self._visited_tools and tool_name != "retrieve":
            return False
        return self._call_count < self.max_tool_calls

    def record_tool_call(self, tool_name: str):
        self._visited_tools.add(tool_name)
        self._call_count += 1


class HypothesisVerifier:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def verify(self, hypothesis: str, causal_engine: Any,
               active_nodes: List[str]) -> "Hypothesis":
        from rtmdk.nodes import Hypothesis
        evidence_nodes = []
        causal_path = []
        confidence = 0.5
        if causal_engine and hasattr(causal_engine, "causal_effects"):
            for (cause, effect), edge in causal_engine.causal_effects.items():
                if cause in active_nodes or effect in active_nodes:
                    evidence_nodes.append(cause)
                    evidence_nodes.append(effect)
                    causal_path.append(
                        f"{cause} -> {effect} (P={edge.strength:.2f})")
                    confidence = max(
                        confidence, edge.strength * edge.confidence)
        verified = confidence >= self.confidence_threshold
        return Hypothesis(
            statement=hypothesis, confidence=confidence,
            evidence_nodes=list(set(evidence_nodes)),
            causal_path=causal_path, verified=verified,
            verification_score=confidence,
        )


class ToolRouter:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._tool_registry: Dict[str, Callable] = {}
        self._call_history: deque = deque(maxlen=100)

    def register_tool(self, name: str, func: Callable):
        self._tool_registry[name] = func

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> "ToolCall":
        from rtmdk.nodes import ToolCall
        t0 = time.time()
        call = ToolCall(tool_name=tool_name, arguments=arguments)
        if tool_name not in self._tool_registry:
            call.error = f"Tool not registered: {tool_name}"
            call.latency_ms = (time.time() - t0) * 1000
            return call
        try:
            func = self._tool_registry[tool_name]
            result = func(**arguments)
            call.result = result
            call.success = True
        except Exception as e:
            call.error = str(e)
        call.latency_ms = (time.time() - t0) * 1000
        self._call_history.append(call)
        return call

    def get_misuse_rate(self) -> float:
        if not self._call_history:
            return 0.0
        failures = sum(1 for c in self._call_history if not c.success)
        return failures / len(self._call_history)

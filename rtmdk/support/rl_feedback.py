"""RL feedback loop for RTMDK."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    pass


class RLFeedbackLoop:
    """Extracts confidence/uncertainty signals from LLM responses
    and uses them as reinforcement for field updates."""

    def __init__(self, learning_rate: float = 0.01, reward_window: int = 10):
        self.lr = learning_rate
        self.reward_window = reward_window
        self._rewards: deque = deque(maxlen=reward_window)
        self._node_rewards: Dict[str, List[float]] = defaultdict(list)

    def extract_reward_from_response(self, response: str,
                                     context_nodes: List[str]) -> float:
        """Extract reward signal from LLM response text."""
        reward = 0.5  # baseline

        # Confidence markers
        confidence_phrases = [
            "certainly",
            "definitely",
            "clearly",
            "obviously",
            "безусловно",
            "очевидно",
            "точно"]
        uncertainty_phrases = ["not sure", "might be", "could be", "perhaps",
                               "не уверен", "возможно", "кажется", "probably"]

        resp_lower = response.lower()
        for phrase in confidence_phrases:
            if phrase in resp_lower:
                reward += 0.1
        for phrase in uncertainty_phrases:
            reward -= 0.1

        # Fallback: punctuation-based uncertainty estimation
        uncertainty_penalty = (response.count(
            "?") + response.count("возможно")) * 0.15
        reward -= min(0.3, uncertainty_penalty)

        # Length-based signal (too short = unhelpful)
        words = response.split()
        if len(words) < 10:
            reward -= 0.2
        elif len(words) > 200:
            reward -= 0.05  # Very long might be unfocused

        reward = max(0.0, min(1.0, reward))
        self._rewards.append(reward)

        # Distribute reward to context nodes
        for nid in context_nodes:
            self._node_rewards[nid].append(reward)

        return reward

    def get_node_reward(self, node_id: str) -> float:
        """Get average reward for a specific node."""
        rewards = self._node_rewards.get(node_id, [])
        return float(np.mean(rewards)) if rewards else 0.5

    def get_average_reward(self) -> float:
        return float(np.mean(self._rewards)) if self._rewards else 0.5

    def apply_field_updates(self, field: Any):
        """Apply RL-based updates to field parameters."""
        if len(self._rewards) < 3:
            return

        self.get_average_reward()
        reward_trend = 0.0
        if len(self._rewards) >= 2:
            recent = list(self._rewards)[-5:]
            reward_trend = recent[-1] - recent[0]

        # Update node RL rewards
        for nid in field.node_index:
            if nid in field.nodes:
                node = field.nodes[nid]
                node_rl = self.get_node_reward(nid)
                node.rl_reward = node_rl
                # Update goal_relevance based on reward
                if reward_trend > 0.1:
                    node.goal_relevance = min(
                        1.0, node.goal_relevance + self.lr)
                elif reward_trend < -0.1:
                    node.goal_relevance = max(
                        0.0, node.goal_relevance - self.lr)

    def get_state(self) -> Dict:
        return {
            "rewards": list(self._rewards),
            "node_rewards": {k: v[-10:] for k, v in self._node_rewards.items()},
        }

    def load_state(self, state: Dict):
        self._rewards = deque(state.get("rewards", []),
                              maxlen=self.reward_window)
        self._node_rewards = defaultdict(list, state.get("node_rewards", {}))

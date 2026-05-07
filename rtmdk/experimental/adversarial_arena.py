"""rtmdk/production/adversarial_arena.py — Memory Arena / Cognitive Immunization."""
import random
from typing import Dict, List, Any


class AdversarialArena:
    """Self-play adversarial testing for RTMDK robustness.

    Generates attack queries to test memory retrieval robustness.
    Types of attacks:
    1. Adversarial phrasing: same meaning, different words
    2. Memory poisoning: inject misleading nodes
    3. Context confusion: ambiguous queries
    """

    def __init__(self, memory):
        self.memory = memory
        self.attack_stats = {"total": 0, "successful": 0, "failed": 0}

    def generate_adversarial_query(self, original_query: str) -> str:
        """Generate adversarial version of a query."""
        attacks = [
            lambda q: q.lower(),  # Case change
            lambda q: q.replace(
                "?",
                " can you tell me?"),
            # Politeness injection
            lambda q: f"Is it true that {q.lower().rstrip('?')}?",  # Framing
        ]
        return random.choice(attacks)(original_query)

    def test_robustness(
            self, test_queries: List[str], top_k: int = 5) -> Dict[str, Any]:
        """Test retrieval robustness on a set of queries."""
        results = []
        for query in test_queries:
            # Original query
            ctx_orig = self.memory.load_memory_variables(
                {"input": query, "session_id": "test"})

            # Adversarial query
            adv_query = self.generate_adversarial_query(query)
            ctx_adv = self.memory.load_memory_variables(
                {"input": adv_query, "session_id": "test"})

            # Check if results are consistent
            consistent = ctx_orig.get(
                "rtmdk_context", "")[
                :100] == ctx_adv.get(
                "rtmdk_context", "")[
                :100]

            self.attack_stats["total"] += 1
            if consistent:
                self.attack_stats["successful"] += 1
            else:
                self.attack_stats["failed"] += 1

            results.append({"query": query,
                            "adv_query": adv_query,
                            "consistent": consistent})

        return {
            "robustness_rate": self.attack_stats["successful"] / max(self.attack_stats["total"], 1),
            "details": results,
        }

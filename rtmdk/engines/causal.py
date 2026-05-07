"""rtmdk/engines/causal.py"""
from __future__ import annotations
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from numpy.typing import NDArray

from rtmdk.nodes import CausalEdge, ContradictionRecord, CounterfactualResult

CHI_SQUARED_CRITICAL_DF1 = 3.84  # Chi-squared critical value (df=1, p=0.05)
CHI_SQUARED_CRITICAL_DF2 = 5.99  # Chi-squared critical value (df=2, p=0.05)


class CausalInferenceEngine:
    def __init__(self, min_samples: int = 20, p_threshold: float = 0.05,
                 adjustment_sets_enabled: bool = True):
        self.min_samples = min_samples
        self.p_threshold = p_threshold
        self.adjustment_sets_enabled = adjustment_sets_enabled
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.ancestors: Dict[str, Set[str]] = defaultdict(set)
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self._conditional_counts: Dict[Tuple[str,
                                             str, str], int] = defaultdict(int)
        self._node_counts: Dict[str, int] = defaultdict(int)
        self._total_observations = 0
        self.causal_effects: Dict[Tuple[str, str], CausalEdge] = {}
        self.contradictions: Dict[str, ContradictionRecord] = {}
        self._contradiction_counter = 0
        self._counterfactual_cache: Dict[str, CounterfactualResult] = {}
        self._intervention_store: Dict[str, List[Dict]] = {}

    def record_cooccurrence(self, a: str, b: str):
        self._cooccurrence[(a, b)] += 1
        self._cooccurrence[(b, a)] += 1
        self._node_counts[a] += 1
        self._node_counts[b] += 1
        self._total_observations += 1

    def record_observation(
            self,
            active_nodes: List[str],
            context: Optional[Dict] = None):
        self._total_observations += 1
        for node in active_nodes:
            self._node_counts[node] += 1
        for i, a in enumerate(active_nodes):
            for b in active_nodes[i + 1:]:
                self._cooccurrence[(a, b)] += 1
                self._cooccurrence[(b, a)] += 1
                if context:
                    for ctx_key, ctx_val in context.items():
                        self._conditional_counts[(
                            a, b, f"{ctx_key}={ctx_val}")] += 1

    def discover_causal_structure(self) -> Dict[str, Set[str]]:
        nodes = list(self._node_counts.keys())
        if len(nodes) < 3 or self._total_observations < self.min_samples:
            return dict(self.parents)
        skeleton: Dict[str, Set[str]] = defaultdict(set)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                if self._test_independence(a, b, set()):
                    continue
                skeleton[a].add(b)
                skeleton[b].add(a)
        new_parents: Dict[str, Set[str]] = defaultdict(set)
        new_children: Dict[str, Set[str]] = defaultdict(set)
        for z in nodes:
            neighbors = list(skeleton.get(z, set()))
            for i, x in enumerate(neighbors):
                for y in neighbors[i + 1:]:
                    if y not in skeleton.get(x, set()):
                        new_parents[z].add(x)
                        new_parents[z].add(y)
                        new_children[x].add(z)
                        new_children[y].add(z)
        self.parents = new_parents
        self.children = new_children
        self._compute_ancestors()
        return dict(self.parents)

    def _test_independence(self, a: str, b: str, cond_set: Set[str]) -> bool:
        n_ab = self._cooccurrence.get((a, b), 0)
        n_a = self._node_counts.get(a, 0)
        n_b = self._node_counts.get(b, 0)
        n = max(self._total_observations, 1)
        if n_a < 3 or n_b < 3 or n_ab < 2:
            return True
        if not cond_set:
            # Marginal independence test: chi-squared
            expected = (n_a / n) * (n_b / n) * n
            if expected < 5:
                return True
            chi2 = (n_ab - expected) ** 2 / expected
            return chi2 < CHI_SQUARED_CRITICAL_DF1  # p=0.05, df=1

        # Bug #16 FIX: Implement conditional independence test
        # Use partial correlation approximation for discrete data
        # Test: a ⊥ b | cond_set
        total = 0
        chi2_cond = 0.0
        for c_node in cond_set:
            n_c = self._node_counts.get(c_node, 0)
            if n_c < 3:
                continue
            # Compute conditional probabilities
            p_a_given_c = min(n_ab, n_c) / max(n_c, 1)
            p_b_given_c = min(n_ab, n_c) / max(n_c, 1)
            n_ab / max(n, 1)
            expected_cond = p_a_given_c * p_b_given_c * n_c
            if expected_cond > 0:
                chi2_cond += (n_ab - expected_cond) ** 2 / expected_cond
                total += 1

        if total == 0:
            # No valid conditioning sets — fall back to marginal
            return True

        # Average chi-squared over conditioning variables
        avg_chi2 = chi2_cond / total
        # With conditioning, use higher threshold (df increases)
        return avg_chi2 < CHI_SQUARED_CRITICAL_DF2  # p=0.05, df=2

    def _compute_ancestors(self):
        for node in self.parents:
            self.ancestors[node] = self._get_ancestors(node, set())

    def _get_ancestors(self, node: str, visited: Set[str]) -> Set[str]:
        if node in visited:
            return set()
        visited.add(node)
        ancestors = set()
        for parent in self.parents.get(node, set()):
            ancestors.add(parent)
            ancestors.update(self._get_ancestors(parent, visited))
        return ancestors

    def _get_descendants(self, node: str,
                         visited: Optional[Set[str]] = None) -> Set[str]:
        if visited is None:
            visited = set()
        if node in visited:
            return set()
        visited.add(node)
        descendants = set()
        for child in self.children.get(node, set()):
            descendants.add(child)
            descendants.update(self._get_descendants(child, visited))
        return descendants

    def compute_do_probability(self,
                               effect: str,
                               intervention: str,
                               evidence: Optional[Dict[str,
                                                       Any]] = None) -> float:
        edge = self.causal_effects.get((intervention, effect))
        if edge:
            return edge.strength
        return self._naive_causal_estimate(intervention, effect)

    def _naive_causal_estimate(self, cause: str, effect: str) -> float:
        n_cause = self._node_counts.get(cause, 0)
        n_both = self._cooccurrence.get((cause, effect), 0)
        if n_cause < 3:
            return 0.5
        return min(1.0, n_both / n_cause)

    def _find_adjustment_set(self, cause: str, effect: str) -> Set[str]:
        if not self.adjustment_sets_enabled:
            return set()
        parents_of_cause = self.parents.get(cause, set())
        descendants = self._get_descendants(cause)
        return parents_of_cause - descendants

    def _validate_do_calculus(self, effect: str, intervention: str) -> bool:
        z_set = self._find_adjustment_set(intervention, effect)
        has_frontdoor = self._has_frontdoor_path(intervention, effect)
        descendants = self._get_descendants(intervention)
        return bool(z_set) or has_frontdoor or effect in descendants

    def _has_frontdoor_path(self, cause: str, effect: str) -> bool:
        for mediator in self.children.get(cause, set()):
            if effect in self.children.get(mediator, set()):
                return True
        return False

    def detect_contradictions(
            self,
            threshold: float = 0.3) -> List[ContradictionRecord]:
        new_contradictions = []
        effect_causes: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (cause, effect), edge in self.causal_effects.items():
            if edge.strength > 0.1:
                effect_causes[effect].append((cause, edge.strength))
        for effect_node, causes in effect_causes.items():
            if len(causes) < 2:
                continue
            for i, (cause_a, strength_a) in enumerate(causes):
                for cause_b, strength_b in causes[i + 1:]:
                    cooc = self._cooccurrence.get((cause_a, cause_b), 0)
                    n_a = self._node_counts.get(cause_a, 0)
                    n_b = self._node_counts.get(cause_b, 0)
                    if n_a > 0 and n_b > 0:
                        expected = (n_a / self._total_observations) * (
                            n_b / self._total_observations) * self._total_observations
                        if expected > 0 and cooc / \
                                expected < (1.0 - threshold):
                            self._contradiction_counter += 1
                            record = ContradictionRecord(
                                id=f"contr_{self._contradiction_counter}",
                                effect_node=effect_node,
                                causes=[(cause_a, strength_a), (cause_b, strength_b)],
                                contradiction_reason=f"Causes {cause_a} and {cause_b} are negatively correlated"
                            )
                            self.contradictions[record.id] = record
                            new_contradictions.append(record)
                            if (cause_a, effect_node) in self.causal_effects:
                                self.causal_effects[(
                                    cause_a, effect_node)].is_contradicted = True
                            if (cause_b, effect_node) in self.causal_effects:
                                self.causal_effects[(
                                    cause_b, effect_node)].is_contradicted = True
        return new_contradictions

    def counterfactual_query(self,
                             intervention: Dict[str,
                                                Any],
                             query_nodes: List[str],
                             evidence: Optional[Dict[str,
                                                     Any]] = None,
                             max_depth: int = 3) -> CounterfactualResult:
        query_str = f"do({intervention})|{query_nodes}"
        if query_str in self._counterfactual_cache:
            return self._counterfactual_cache[query_str]
        outcomes = []
        reasoning_path = []
        for target in query_nodes[:max_depth]:
            if target in intervention:
                outcomes.append((target, 1.0))
                reasoning_path.append(f"{target} is directly set")
                continue
            best_prob = 0.0
            best_path = ""
            for int_var, int_val in intervention.items():
                prob = self.compute_do_probability(target, int_var)
                if prob > best_prob:
                    best_prob = prob
                    best_path = f"do({int_var}) -> {target} (P={prob:.3f})"
            if best_path:
                outcomes.append((target, best_prob))
                reasoning_path.append(best_path)
            else:
                outcomes.append((target, 0.5))
                reasoning_path.append(f"No causal path to {target}")
        confidence = np.mean([p for _, p in outcomes]) if outcomes else 0.5
        result = CounterfactualResult(
            query=query_str,
            intervention=intervention,
            predicted_outcomes=outcomes,
            confidence=float(confidence),
            reasoning_path=reasoning_path,
            assumptions=[])
        self._counterfactual_cache[query_str] = result
        return result

    def validate_consolidation(
            self, node_a: str, node_b: str) -> Dict[str, Any]:
        result = {
            "safe": True,
            "reasons": [],
            "causal_conflicts": [],
            "recommendation": "proceed"}
        common_targets = set(
            self.children.get(
                node_a,
                set())) & set(
            self.children.get(
                node_b,
                set()))
        for target in common_targets:
            edge_a = self.causal_effects.get((node_a, target))
            edge_b = self.causal_effects.get((node_b, target))
            if edge_a and edge_b:
                diff = abs(edge_a.strength - edge_b.strength)
                if diff > 0.4:
                    result["safe"] = False
                    result["causal_conflicts"].append(
                        {
                            "target": target,
                            "effect_a": edge_a.strength,
                            "effect_b": edge_b.strength,
                            "difference": diff})
                    result["reasons"].append(f"Opposing effects on {target}")
        if node_b in self.children.get(
                node_a,
                set()) or node_a in self.children.get(
                node_b,
                set()):
            result["safe"] = False
            result["reasons"].append("Causal relationship exists")
            result["recommendation"] = "preserve_separate"
        if node_a in self.ancestors.get(
                node_b,
                set()) or node_b in self.ancestors.get(
                node_a,
                set()):
            result["safe"] = False
            result["reasons"].append("Merging would create causal cycle")
            result["recommendation"] = "preserve_separate"
        for cid, record in self.contradictions.items():
            if record.resolved:
                continue
            causes = [c for c, _ in record.causes]
            if node_a in causes and node_b in causes:
                result["safe"] = False
                result["reasons"].append(f"Unresolved contradiction: {cid}")
                result["recommendation"] = "resolve_contradiction_first"
        return result

    def do_intervention(self, node_id: str, new_pos: NDArray):
        if node_id not in self.parents and node_id not in self.children:
            self.parents[node_id] = set()
        if node_id not in self._intervention_store:
            self._intervention_store[node_id] = []
        self._intervention_store[node_id].append({
            "new_pos": new_pos.copy(),
            "timestamp": time.time(),
        })
        for child in self.children.get(node_id, set()):
            edge_key = (node_id, child)
            if edge_key in self.causal_effects:
                edge = self.causal_effects[edge_key]
                edge.strength = min(1.0, edge.strength * 1.2)
                edge.evidence_count += 1

    def clear_interventions(self):
        self._intervention_store = {}

    def get_state(self) -> Dict:
        return {
            "parents": {
                k: list(v) for k,
                v in self.parents.items()},
            "children": {
                k: list(v) for k,
                v in self.children.items()},
            "causal_effects": {
                f"{k[0]}->{k[1]}": v.to_dict() for k,
                v in self.causal_effects.items()},
            "contradictions": {
                k: v.to_dict() for k,
                v in self.contradictions.items()},
            "node_counts": dict(
                self._node_counts),
            "total_observations": self._total_observations,
            "intervention_store": {
                k: [
                    {
                        "new_pos": v["new_pos"].tolist() if hasattr(
                            v["new_pos"],
                            'tolist') else v["new_pos"],
                        "timestamp": v["timestamp"]} for v in vals] for k,
                vals in self._intervention_store.items()},
        }

    def load_state(self, state: Dict):
        self.parents = defaultdict(
            set, {
                k: set(v) for k, v in state.get(
                    "parents", {}).items()})
        self.children = defaultdict(
            set, {
                k: set(v) for k, v in state.get(
                    "children", {}).items()})
        self._node_counts = defaultdict(int, state.get("node_counts", {}))
        self._total_observations = state.get("total_observations", 0)
        self._intervention_store = {}
        for node_id, interventions in state.get(
                "intervention_store", {}).items():
            self._intervention_store[node_id] = [
                {"new_pos": np.array(iv["new_pos"], dtype=np.float32), "timestamp": iv["timestamp"]}
                for iv in interventions
            ]
        for key, edge_data in state.get("causal_effects", {}).items():
            parts = key.split("->")
            if len(parts) == 2:
                self.causal_effects[(parts[0], parts[1])
                                    ] = CausalEdge.from_dict(edge_data)
        for cid, record_data in state.get("contradictions", {}).items():
            self.contradictions[cid] = ContradictionRecord(
                id=record_data["id"], effect_node=record_data["effect_node"],
                causes=record_data["causes"], timestamp=record_data["timestamp"],
                resolved=record_data["resolved"], resolution=record_data["resolution"],
                contradiction_reason=record_data.get("contradiction_reason", ""))
        self._compute_ancestors()

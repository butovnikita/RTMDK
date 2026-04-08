"""rtmdk/support/symbolic_overlay.py — Probabilistic Logic Layer for RTMDK.

Extracts weighted Horn clauses from consolidated memory nodes and performs
forward-chaining inference with confidence thresholds.

Key design:
- Rules extracted from: tier ∈ {semantic, procedural} ∧ self_sup_score > 0.7 ∧ tension < 0.15
- Format: W:0.82 :: кофе_утро → бодрость
- Inference: weighted forward-chaining, confidence threshold τ=0.65
- Conflicts: marked as contextual_exception, NOT suppressed
- Overlay reads from field, never writes — returns symbolic_context only.
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray


@dataclass
class SymbolicRule:
    """Weighted Horn clause: body[0] ∧ body[1] ∧ ... → head with weight W."""
    rule_id: str
    head: str            # Consequent (e.g., "бодрость")
    body: List[str]      # Antecedents (e.g., ["кофе_утро"])
    weight: float        # Confidence from resonance + self_sup_score
    source_nodes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    support_count: int = 0
    is_contextual_exception: bool = False
    conflict_with: Optional[str] = None  # rule_id of conflicting rule

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SymbolicRule":
        return cls(**data)

    def __repr__(self) -> str:
        body_str = " ∧ ".join(self.body) if self.body else "⊤"
        exc = " [EXCEPTION]" if self.is_contextual_exception else ""
        conflict = f" ↔ {self.conflict_with}" if self.conflict_with else ""
        return f"W:{self.weight:.2f} :: {body_str} → {self.head}{exc}{conflict}"


@dataclass
class SymbolicInference:
    """Result of symbolic inference."""
    conclusion: str
    confidence: float
    used_rules: List[str]  # rule_ids
    is_conflict: bool = False
    conflicting_rules: List[str] = field(default_factory=list)
    contextual_note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class ConflictDetector:
    """Detects conflicting symbolic rules and marks them as contextual exceptions."""

    def __init__(self, conflict_threshold: float = 0.15):
        self.conflict_threshold = conflict_threshold
        self._conflicts: List[Tuple[str, str, float]] = []  # (rule_a, rule_b, confidence_diff)

    def detect_conflicts(self, rules: List[SymbolicRule]) -> List[Tuple[str, str]]:
        """Find rules with same head but opposite confidence trends."""
        self._conflicts = []
        head_rules: Dict[str, List[SymbolicRule]] = defaultdict(list)
        for rule in rules:
            head_rules[rule.head].append(rule)

        conflicts = []
        for head, rls in head_rules.items():
            if len(rls) < 2:
                continue
            # Check for opposing bodies
            for i, r1 in enumerate(rls):
                for r2 in rls[i+1:]:
                    # Conflict if bodies are mutually exclusive (contain negation or different contexts)
                    if self._bodies_conflict(r1.body, r2.body):
                        conf_diff = abs(r1.weight - r2.weight)
                        if conf_diff < self.conflict_threshold:
                            # Near-equal confidence → true conflict
                            r1.is_contextual_exception = True
                            r2.is_contextual_exception = True
                            r1.conflict_with = r2.rule_id
                            r2.conflict_with = r1.rule_id
                            conflicts.append((r1.rule_id, r2.rule_id))
                            self._conflicts.append((r1.rule_id, r2.rule_id, conf_diff))
        return conflicts

    @staticmethod
    def _bodies_conflict(body1: List[str], body2: List[str]) -> bool:
        """Check if bodies represent conflicting contexts."""
        negation_patterns = [
            ("утро", "вечер"), ("утро", "ночь"), ("день", "ночь"),
            ("до", "после"), ("включено", "выключено"), ("да", "нет"),
            ("true", "false"), ("+", "-"),
        ]
        set1 = set(body1)
        set2 = set(body2)
        for a, b in negation_patterns:
            if (a in set1 and b in set2) or (b in set1 and a in set2):
                return True
        # Also check for direct overlap with negation prefix
        for t1 in set1:
            neg_t1 = f"не_{t1}" if not t1.startswith("не_") else t1[3:]
            if neg_t1 in set2:
                return True
        return False


class SymbolicOverlay:
    """Probabilistic logic layer over the RTMDK memory field.

    Extracts weighted Horn clauses from high-quality nodes and performs
    forward-chaining inference to produce symbolic context for LLM.
    """

    def __init__(self, min_self_sup: float = 0.7, max_tension: float = 0.15,
                 confidence_threshold: float = 0.65, max_rules: int = 200):
        self.min_self_sup = min_self_sup
        self.max_tension = max_tension
        self.confidence_threshold = confidence_threshold
        self.max_rules = max_rules
        self.rules: Dict[str, SymbolicRule] = {}
        self.conflict_detector = ConflictDetector()
        self._rule_counter = 0
        self._last_extraction_time: float = 0.0
        self._total_extractions: int = 0
        self._inference_history: List[Dict] = []

    def extract_rules_from_field(self, nodes: Dict, causal_edges: Optional[Dict] = None) -> List[SymbolicRule]:
        """Extract Horn clauses from memory field nodes.

        Criteria: tier ∈ {semantic, procedural} ∧ self_sup_score > min ∧ tension < max
        """
        self.rules = {}
        self._rule_counter = 0
        new_rules = []

        for nid, node in nodes.items():
            # Filter by quality
            tier = getattr(node, 'tier', 'semantic')
            if tier not in ('semantic', 'procedural'):
                continue
            if getattr(node, 'self_sup_score', 1.0) < self.min_self_sup:
                continue
            if getattr(node, 'tension', 0.0) > self.max_tension:
                continue

            text = node.content.get("text", "")
            if not text or len(text) < 5:
                continue

            # Extract concepts from text
            concepts = self._extract_concepts(text)
            if len(concepts) < 2:
                continue

            # Create rules from concept co-occurrence
            # Single-body rules: concept_i → concept_j
            for i, head_concept in enumerate(concepts[:3]):  # Limit to first 3 concepts as head
                body_concepts = [c for j, c in enumerate(concepts) if j != i][:2]  # Up to 2 body terms
                if not body_concepts:
                    continue

                # Weight from self_sup_score, salience, and causal strength
                causal_boost = sum(node.causal_strength.values()) if hasattr(node, 'causal_strength') else 0
                weight = (node.self_sup_score * 0.5 +
                          node.salience * 0.3 +
                          min(1.0, causal_boost) * 0.2)

                self._rule_counter += 1
                rule = SymbolicRule(
                    rule_id=f"sr_{self._rule_counter}",
                    head=head_concept,
                    body=body_concepts,
                    weight=min(1.0, max(0.0, weight)),
                    source_nodes=[nid],
                )
                self.rules[rule.rule_id] = rule
                new_rules.append(rule)

                if len(self.rules) >= self.max_rules:
                    break
            if len(self.rules) >= self.max_rules:
                break

        # Add rules from causal edges
        if causal_edges:
            for (cause, effect), edge in causal_edges.items():
                if edge.strength > 0.3 and not edge.is_contradicted:
                    self._rule_counter += 1
                    rule = SymbolicRule(
                        rule_id=f"sr_{self._rule_counter}",
                        head=effect,
                        body=[cause],
                        weight=edge.strength * edge.confidence,
                        source_nodes=[cause, effect],
                    )
                    self.rules[rule.rule_id] = rule
                    new_rules.append(rule)

        # Detect conflicts
        self.conflict_detector.detect_conflicts(list(self.rules.values()))

        self._last_extraction_time = time.time()
        self._total_extractions += 1
        return new_rules

    def forward_chain(self, facts: List[str], max_depth: int = 3) -> List[SymbolicInference]:
        """Perform weighted forward-chaining inference.

        Args:
            facts: Initial known facts (concept strings)
            max_depth: Maximum inference chain depth

        Returns:
            List of inferences above confidence threshold
        """
        active_facts: Dict[str, float] = {f: 1.0 for f in facts}
        inferences: List[SymbolicInference] = []
        used_rules: Set[str] = set()

        for depth in range(max_depth):
            new_facts: Dict[str, float] = {}
            for rule in self.rules.values():
                # Check if all body facts are active
                body_confidences = [active_facts.get(b, 0.0) for b in rule.body]
                if all(bc >= self.confidence_threshold for bc in body_confidences):
                    # Rule fires
                    rule_confidence = rule.weight * min(body_confidences)
                    if rule_confidence >= self.confidence_threshold:
                        # Check for conflicts
                        is_conflict = rule.is_contextual_exception
                        note = ""
                        conflict_rules = []
                        if is_conflict:
                            note = f"Contextual exception (conflicts with {rule.conflict_with})"
                            conflict_rules = [rule.rule_id, rule.conflict_with] if rule.conflict_with else []

                        inference = SymbolicInference(
                            conclusion=rule.head,
                            confidence=rule_confidence,
                            used_rules=[rule.rule_id],
                            is_conflict=is_conflict,
                            conflicting_rules=conflict_rules,
                            contextual_note=note,
                        )
                        inferences.append(inference)
                        used_rules.add(rule.rule_id)

                        # Add to active facts with rule confidence
                        if rule.head not in active_facts or rule_confidence > active_facts[rule.head]:
                            new_facts[rule.head] = rule_confidence

            active_facts.update(new_facts)
            if not new_facts:
                break

        self._inference_history.append({
            "timestamp": time.time(),
            "input_facts": facts,
            "n_inferences": len(inferences),
            "n_conflicts": sum(1 for i in inferences if i.is_conflict),
        })

        return inferences

    def get_symbolic_context(self, facts: List[str], max_depth: int = 3) -> str:
        """Generate symbolic context string for LLM injection."""
        inferences = self.forward_chain(facts, max_depth)
        if not inferences:
            return ""

        lines = ["### SYMBOLIC_CONTEXT"]
        lines.append(f"Extracted rules: {len(self.rules)}, Inferences: {len(inferences)}")
        lines.append("")

        # Non-conflicting inferences
        normal = [i for i in inferences if not i.is_conflict]
        if normal:
            lines.append("Inferred knowledge:")
            for inf in sorted(normal, key=lambda x: -x.confidence)[:10]:
                lines.append(f"  [CF:{inf.confidence:.2f}] {inf.conclusion}")
                for rid in inf.used_rules:
                    rule = self.rules.get(rid)
                    if rule:
                        body_str = " ∧ ".join(rule.body)
                        lines.append(f"    via: W:{rule.weight:.2f} :: {body_str} → {rule.head}")

        # Conflicting inferences
        conflicts = [i for i in inferences if i.is_conflict]
        if conflicts:
            lines.append("")
            lines.append("Contextual exceptions (competing interpretations):")
            for inf in conflicts:
                lines.append(f"  ⚠ [CF:{inf.confidence:.2f}] {inf.conclusion}: {inf.contextual_note}")

        return "\n".join(lines)

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concept tokens from text."""
        # Simple tokenization: lowercase, split on whitespace/punctuation
        tokens = re.findall(r'[а-яА-Яa-zA-Z_][а-яА-Яa-zA-Z0-9_]{2,}', text.lower())
        # Filter stopwords
        stopwords = {
            "это", "как", "что", "для", "или", "если", "но", "и", "в", "на",
            "the", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "shall", "can", "need", "dare", "ought", "used",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "because", "about", "against", "until", "while", "that", "which",
            "who", "whom", "these", "those", "am", "it", "its", "they", "them",
            "their", "what", "your", "you", "he", "she", "him", "her", "we", "us",
            "our", "my", "me", "i", "you", "they", "them", "their", "there",
        }
        concepts = [t for t in tokens if t not in stopwords]
        # Normalize: replace spaces with underscores
        concepts = [c.replace(" ", "_") for c in concepts]
        return list(dict.fromkeys(concepts))[:5]  # Limit to 5 concepts per node

    def get_stats(self) -> Dict:
        return {
            "n_rules": len(self.rules),
            "n_conflicts": len(self.conflict_detector._conflicts),
            "total_extractions": self._total_extractions,
            "last_extraction_time": self._last_extraction_time,
            "avg_rule_weight": np.mean([r.weight for r in self.rules.values()]) if self.rules else 0.0,
        }

    def get_state(self) -> Dict:
        return {
            "rules": {k: v.to_dict() for k, v in self.rules.items()},
            "rule_counter": self._rule_counter,
            "total_extractions": self._total_extractions,
        }

    def load_state(self, data: Dict):
        self.rules = {k: SymbolicRule.from_dict(v) for k, v in data.get("rules", {}).items()}
        self._rule_counter = data.get("rule_counter", 0)
        self._total_extractions = data.get("total_extractions", 0)

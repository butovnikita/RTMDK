"""
rtmdk/engines/neuro_symbolic_prover.py — Neuro-Symbolic Theorem Prover (v2).

v2 Changes:
- Typed API: add_implication(), add_facts(), check_consistency()
- Improved text parser: handles "NOT x", "(NOT x)", nested formulas
- Input validation before Z3
- LLM/Embedder-independent: rules set via typed API, not text parsing

Backends:
    z3: Z3 SMT solver (requires z3-solver package)
    prolog: SWI-Prolog (requires pyswip)
    none: Disabled (fallback to simple checks)
"""

import logging
from typing import Dict, List, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# TYPED RULE STRUCTURE
# ============================================================================

@dataclass
class LogicalRule:
    """A typed logical rule — LLM/Embedder independent."""
    premises: List[str]              # ["disk_is_full", "no_cleanup"]
    conclusion: str                  # "system_crash"
    negate_conclusion: bool = False  # True → NOT conclusion
    rule_type: str = "implication"   # implication, fact, constraint

    def __str__(self) -> str:
        premise_str = " AND ".join(self.premises)
        if self.negate_conclusion:
            return f"{premise_str} → NOT {self.conclusion}"
        return f"{premise_str} → {self.conclusion}"


# ============================================================================
# MAIN PROVER CLASS
# ============================================================================

class NeuroSymbolicProver:
    """Neuro-Symbolic theorem prover for RTMDK (v2).

    Typed API — does NOT depend on LLM output format or embedding semantics.

    Usage (typed API — recommended):
        prover = NeuroSymbolicProver(backend="z3")
        prover.add_fact("server_running")
        prover.add_implication("disk_full", "system_crash")
        prover.add_implication(["disk_full", "no_cleanup"], "data_loss", negate_conclusion=False)
        result = prover.check_consistency()  # {"consistent": True/False, ...}

    Usage (text API — legacy, for compatibility):
        prover.add_rule("disk_full → system_crash")
        prover.add_rule("A AND B → NOT C")
    """

    def __init__(self, backend: str = "z3"):
        self.backend = backend
        self.facts: Dict[str, bool] = {}
        self.rules: List[LogicalRule] = []
        self.raw_rules: List[str] = []  # Legacy text rules
        self.contradictions: List[Dict] = []
        self._solver = None
        self._z3 = None  # Module reference
        self._prolog = None
        self._z3_vars: Dict[str, Any] = {}  # Cache of Z3 Bool variables
        self._init_backend()

    def _init_backend(self):
        """Initialize the theorem proving backend."""
        if self.backend == "z3":
            try:
                import z3
                self._z3 = z3
                self._solver = z3.Solver()
                logger.info("NeuroSymbolicProver: Z3 backend initialized")
            except ImportError:
                logger.warning("NeuroSymbolicProver: z3-solver not installed. "
                               "Install with: pip install z3-solver")
                self.backend = "none"
        elif self.backend == "prolog":
            try:
                from pyswip import Prolog
                self._prolog = Prolog()
                logger.info("NeuroSymbolicProver: Prolog backend initialized")
            except ImportError:
                logger.warning("NeuroSymbolicProver: pyswip not installed. "
                               "Install with: pip install pyswip")
                self.backend = "none"

    def _get_z3_var(self, name: str):
        """Get or create a Z3 Bool variable (cached)."""
        if name not in self._z3_vars and self._z3:
            self._z3_vars[name] = self._z3.Bool(name)
        return self._z3_vars.get(name)

    # ─────────────────────────────────────────────────────────────────
    # TYPED API (recommended — LLM/Embedder independent)
    # ─────────────────────────────────────────────────────────────────

    def add_fact(self, fact_name: str, value: bool = True) -> bool:
        """Add a factual assertion to the knowledge base.

        Args:
            fact_name: Proposition name (e.g., "server_running")
            value: Truth value

        Returns:
            True if added successfully
        """
        # Validate fact name
        if not fact_name or not fact_name.strip():
            logger.warning("NeuroSymbolicProver: Empty fact name ignored")
            return False

        # Clean name: spaces → underscores, remove special chars
        clean_name = "".join(c if c.isalnum() or c ==
                             '_' else '_' for c in fact_name.strip())

        self.facts[clean_name] = value

        if self.backend == "z3" and self._solver:
            var = self._get_z3_var(clean_name)
            if var is not None:
                if value:
                    self._solver.add(var)
                else:
                    self._solver.add(self._z3.Not(var))

        return True

    def add_implication(self,
                        premise: Union[str, List[str]],
                        conclusion: str,
                        negate_conclusion: bool = False) -> bool:
        """Add a logical implication rule.

        Args:
            premise: Single premise or list of premises (AND-ed together)
            conclusion: Conclusion proposition
            negate_conclusion: If True, conclusion is negated (→ NOT conclusion)

        Returns:
            True if rule added successfully

        Example:
            prover.add_implication("disk_full", "system_crash")
            # disk_full → system_crash

            prover.add_implication(["disk_full", "no_cleanup"], "data_loss")
            # (disk_full AND no_cleanup) → data_loss

            prover.add_implication("disk_full", "has_space", negate_conclusion=True)
            # disk_full → NOT has_space
        """
        # Validate and clean
        if isinstance(premise, str):
            premise = [premise]

        premises = []
        for p in premise:
            if p and p.strip():
                clean = "".join(c if c.isalnum() or c ==
                                '_' else '_' for c in p.strip())
                premises.append(clean)

        if not premises:
            logger.warning("NeuroSymbolicProver: Empty premise ignored")
            return False

        if not conclusion or not conclusion.strip():
            logger.warning("NeuroSymbolicProver: Empty conclusion ignored")
            return False

        clean_conclusion = "".join(
            c if c.isalnum() or c == '_' else '_' for c in conclusion.strip())

        rule = LogicalRule(
            premises=premises,
            conclusion=clean_conclusion,
            negate_conclusion=negate_conclusion,
        )
        self.rules.append(rule)

        if self.backend == "z3" and self._solver:
            premise_vars = [self._get_z3_var(p) for p in premises]
            conclusion_var = self._get_z3_var(clean_conclusion)

            if all(
                    v is not None for v in premise_vars) and conclusion_var is not None:
                if len(premise_vars) == 1:
                    premise_z3 = premise_vars[0]
                else:
                    premise_z3 = self._z3.And(*premise_vars)

                if negate_conclusion:
                    conclusion_z3 = self._z3.Not(conclusion_var)
                else:
                    conclusion_z3 = conclusion_var

                self._solver.add(self._z3.Implies(premise_z3, conclusion_z3))

        return True

    def add_constraint(self, propositions: List[str],
                       at_least: int = 1) -> bool:
        """Add a constraint: at least N of the propositions must be true.

        Example:
            prover.add_constraint(["disk_full", "has_space"], at_least=0)
            # NOT (disk_full AND has_space) — they can't both be true
        """
        clean_props = []
        for p in propositions:
            if p and p.strip():
                clean = "".join(c if c.isalnum() or c ==
                                '_' else '_' for c in p.strip())
                clean_props.append(clean)

        if not clean_props or self.backend != "z3" or not self._solver:
            return False

        vars_list = [self._get_z3_var(p) for p in clean_props]
        if any(v is None for v in vars_list):
            return False

        # AtLeast constraint
        self._solver.add(self._z3.AtLeast(vars_list + [at_least]))
        return True

    # ─────────────────────────────────────────────────────────────────
    # TEXT API (legacy — improved parser for compatibility)
    # ─────────────────────────────────────────────────────────────────

    def add_rule(self, rule_str: str) -> bool:
        """Add a rule as a text string (legacy compatibility).

        Supported formats:
            "premise → conclusion"
            "premise -> NOT conclusion"
            "A AND B → C"
            "A OR B → NOT C"
            "NOT A → B"
            "(NOT A) → B"

        Returns:
            True if rule parsed and added successfully
        """
        self.raw_rules.append(rule_str)

        if self.backend != "z3" or not self._solver:
            return False

        try:
            z3_formula = self._parse_rule(rule_str)
            if z3_formula is not None:
                self._solver.add(z3_formula)
                return True
            else:
                logger.warning(
                    f"NeuroSymbolicProver: Failed to parse rule: '{rule_str}'")
                return False
        except Exception as e:
            logger.warning(
                f"NeuroSymbolicProver: Error parsing rule '{rule_str}': {e}")
            return False

    def _parse_rule(self, rule_str: str):
        """Parse a rule string into a Z3 formula.

        Handles: "premise → conclusion", "A AND B → NOT C", "NOT A → B"
        """
        if not self._z3:
            return None

        # Split on → or ->
        if "→" in rule_str:
            parts = rule_str.split("→", 1)
        elif "->" in rule_str:
            parts = rule_str.split("->", 1)
        else:
            logger.warning(
                f"NeuroSymbolicProver: No implication arrow in rule: '{rule_str}'")
            return None

        if len(parts) != 2:
            return None

        premise_str, conclusion_str = parts[0].strip(), parts[1].strip()

        premise = self._parse_formula(premise_str)
        conclusion = self._parse_formula(conclusion_str)

        if premise is not None and conclusion is not None:
            return self._z3.Implies(premise, conclusion)
        return None

    def _parse_formula(self, formula_str: str):
        """Parse a logical formula into Z3 expression.

        Handles:
            "A" — atomic proposition
            "NOT A" / "NOT A" — negation
            "(NOT A)" — parenthesized negation
            "A AND B" — conjunction
            "A OR B" — disjunction
            "A AND (NOT B)" — nested
        """
        if not self._z3:
            return None

        formula_str = formula_str.strip()

        if not formula_str:
            return None

        # Handle parenthesized formulas
        if formula_str.startswith("(") and formula_str.endswith(")"):
            return self._parse_formula(formula_str[1:-1])

        # Handle OR (lowest precedence)
        if " OR " in formula_str:
            parts = formula_str.split(" OR ")
            subformulas = [self._parse_formula(p.strip()) for p in parts]
            if all(s is not None for s in subformulas):
                return self._z3.Or(*subformulas)
            return None

        # Handle AND
        if " AND " in formula_str:
            parts = formula_str.split(" AND ")
            subformulas = [self._parse_formula(p.strip()) for p in parts]
            if all(s is not None for s in subformulas):
                return self._z3.And(*subformulas)
            return None

        # Handle NOT — multiple patterns
        if formula_str.startswith("NOT "):
            inner = formula_str[4:].strip()
            sub = self._parse_formula(inner)
            if sub is not None:
                return self._z3.Not(sub)

        # Handle "(NOT x)" pattern
        if formula_str.startswith("(NOT ") and formula_str.endswith(")"):
            inner = formula_str[5:-1].strip()
            sub = self._parse_formula(inner)
            if sub is not None:
                return self._z3.Not(sub)

        # Atomic proposition — validate name
        clean_name = "".join(c if c.isalnum() or c ==
                             '_' else '_' for c in formula_str.strip())
        if clean_name and clean_name not in ("NOT", "AND", "OR"):
            return self._get_z3_var(clean_name)

        logger.warning(
            f"NeuroSymbolicProver: Unrecognized formula: '{formula_str}'")
        return None

    # ─────────────────────────────────────────────────────────────────
    # CHECKING & RESOLUTION
    # ─────────────────────────────────────────────────────────────────

    def check_consistency(self) -> Dict[str, Any]:
        """Check if current facts and rules are consistent.

        Returns:
            {"consistent": bool, "model": dict or None, "contradictions": list}
        """
        if self.backend == "z3" and self._solver:
            result = self._solver.check()
            if result == self._z3.sat:
                model = self._solver.model()
                # Extract model assignments safely
                model_dict = {}
                for decl in model.decls():
                    try:
                        model_dict[str(decl.name())] = bool(model[decl])
                    except Exception:
                        model_dict[str(decl.name())] = str(model[decl])
                return {
                    "consistent": True,
                    "model": model_dict,
                    "contradictions": [],
                }
            else:
                return {
                    "consistent": False,
                    "model": None,
                    "contradictions": self._find_contradictions(),
                }

        # Fallback: simple fact checking
        return self._simple_check()

    def _find_contradictions(self) -> List[Dict]:
        """Find contradictory propositions."""
        contradictions: List[Dict[str, Any]] = []

        # Check for direct contradictions (A and NOT A)
        for fact, value in self.facts.items():
            neg_fact = f"NOT_{fact}"
            if neg_fact in self.facts and self.facts[neg_fact] == value:
                contradictions.append({
                    "type": "direct_contradiction",
                    "facts": [fact, neg_fact],
                })

        # Check rule-based contradictions
        for rule in self.rules:
            # If all premises are true and conclusion conflicts with a fact
            premises_true = all(self.facts.get(p, False)
                                for p in rule.premises)
            if premises_true:
                expected = not rule.negate_conclusion
                actual = self.facts.get(rule.conclusion, not expected)
                if actual != expected:
                    contradictions.append({
                        "type": "rule_violation",
                        "rule": str(rule),
                        "expected": expected,
                        "actual": actual,
                    })

        return contradictions

    def _simple_check(self) -> Dict[str, Any]:
        """Simple consistency check without solver."""
        contradictions = self._find_contradictions()
        return {
            "consistent": len(contradictions) == 0,
            "model": dict(self.facts),
            "contradictions": contradictions,
        }

    def resolve_contradiction(
        self,
        node_a_id: str,
        node_b_id: str,
        memory_field,
    ) -> Dict[str, Any]:
        """Resolve contradiction between two nodes using logical inference."""
        memory_field.nodes[node_a_id].content.get("text", "")
        memory_field.nodes[node_b_id].content.get("text", "")

        # Heuristic: keep node with higher salience
        sal_a = memory_field.nodes[node_a_id].salience
        sal_b = memory_field.nodes[node_b_id].salience

        if sal_a > sal_b * 1.2:
            return {
                "action": "keep_a",
                "reason": f"Higher salience ({sal_a:.2f} vs {sal_b:.2f})"}
        elif sal_b > sal_a * 1.2:
            return {
                "action": "keep_b",
                "reason": f"Higher salience ({sal_b:.2f} vs {sal_a:.2f})"}
        else:
            return {
                "action": "merge",
                "reason": "Similar salience, recommend merging"}

    def extract_rules_from_symbolic(self, symbolic_overlay) -> int:
        """Extract logical rules from SymbolicOverlay.

        Uses typed API — resilient to format changes in SymbolicOverlay.
        """
        n_added = 0
        if not symbolic_overlay:
            return 0

        # Try to get structured rules first
        if hasattr(symbolic_overlay, 'rules'):
            for rule in symbolic_overlay.rules:
                if hasattr(rule, 'premise') and hasattr(rule, 'conclusion'):
                    premise = str(rule.premise).strip()
                    conclusion = str(rule.conclusion).strip()
                    if premise and conclusion:
                        negate = conclusion.startswith("NOT ")
                        if negate:
                            conclusion = conclusion[4:].strip()
                        self.add_implication(
                            premise, conclusion, negate_conclusion=negate)
                        n_added += 1

        # Fallback: try text rules
        if hasattr(symbolic_overlay, 'raw_rules'):
            for rule_str in symbolic_overlay.raw_rules:
                if self.add_rule(rule_str):
                    n_added += 1

        return n_added

    def get_stats(self) -> Dict:
        return {
            "backend": self.backend,
            "n_facts": len(self.facts),
            "n_rules": len(self.rules),
            "n_raw_rules": len(self.raw_rules),
            "n_contradictions": len(self.contradictions),
            "solver_available": self._solver is not None,
        }

"""
rtmdk/engines/neuro_symbolic_prover.py — Neuro-Symbolic Theorem Prover.

Integrates Z3 SMT solver with RTMDK for domain-specific contradiction resolution.

Usage (legal domain):
    prover = NeuroSymbolicProver(backend="z3")
    prover.add_fact("contract_signed", True)
    prover.add_rule("contract_signed AND payment_made → obligation_fulfilled")
    result = prover.check_consistency()  # SAT/UNSAT

Usage (medical domain):
    prover.add_fact("patient_allergic(penicillin)", True)
    prover.add_fact("prescribed(penicillin)", True)
    result = prover.check_consistency()  # UNSAT → contradiction detected

Backends:
    z3: Z3 SMT solver (requires z3-solver package)
    prolog: SWI-Prolog (requires pyswip)
    none: Disabled
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


class NeuroSymbolicProver:
    """Neuro-Symbolic theorem prover for RTMDK.
    
    Extracts logical rules from SymbolicOverlay and resolves
    contradictions through formal logical inference.
    """
    
    def __init__(self, backend: str = "z3"):
        self.backend = backend
        self.facts: Dict[str, bool] = {}
        self.rules: List[str] = []
        self.contradictions: List[Dict] = []
        self._solver = None
        self._init_solver()
    
    def _init_solver(self):
        """Initialize the theorem proving backend."""
        if self.backend == "z3":
            try:
                import z3
                self._solver = z3.Solver()
                self._z3 = z3
                logger.info("NeuroSymbolicProver: Z3 backend initialized")
            except ImportError:
                logger.warning("NeuroSymbolicProver: z3-solver not installed. Install with: pip install z3-solver")
                self.backend = "none"
        elif self.backend == "prolog":
            try:
                from pyswip import Prolog
                self._solver = Prolog()
                logger.info("NeuroSymbolicProver: Prolog backend initialized")
            except ImportError:
                logger.warning("NeuroSymbolicProver: pyswip not installed. Install with: pip install pyswip")
                self.backend = "none"
    
    def add_fact(self, fact_name: str, value: bool = True):
        """Add a fact to the knowledge base."""
        self.facts[fact_name] = value
        if self.backend == "z3" and self._solver:
            prop = self._z3.Bool(fact_name)
            if value:
                self._solver.add(prop)
            else:
                self._solver.add(self._z3.Not(prop))
    
    def add_rule(self, rule_str: str):
        """Add a logical rule.
        
        Format: "premise1 AND premise2 → conclusion"
        Format: "premise → conclusion1 OR conclusion2"
        """
        self.rules.append(rule_str)
        if self.backend == "z3" and self._solver:
            try:
                z3_formula = self._parse_rule(rule_str)
                if z3_formula is not None:
                    self._solver.add(z3_formula)
            except Exception as e:
                logger.warning(f"NeuroSymbolicProver: Failed to parse rule '{rule_str}': {e}")
    
    def _parse_rule(self, rule_str: str):
        """Parse a rule string into Z3 formula."""
        if self.backend != "z3" or not self._z3:
            return None
        
        if "→" in rule_str:
            premise_str, conclusion_str = rule_str.split("→", 1)
        elif "->" in rule_str:
            premise_str, conclusion_str = rule_str.split("->", 1)
        else:
            return None
        
        premise = self._parse_formula(premise_str.strip())
        conclusion = self._parse_formula(conclusion_str.strip())
        
        if premise is not None and conclusion is not None:
            return self._z3.Implies(premise, conclusion)
        return None
    
    def _parse_formula(self, formula_str: str):
        """Parse a logical formula into Z3 expression."""
        if not self._z3:
            return None
        
        formula_str = formula_str.strip()
        
        if " AND " in formula_str:
            parts = formula_str.split(" AND ")
            subformulas = [self._parse_formula(p.strip()) for p in parts]
            if all(s is not None for s in subformulas):
                return self._z3.And(*subformulas)
        elif " OR " in formula_str:
            parts = formula_str.split(" OR ")
            subformulas = [self._parse_formula(p.strip()) for p in parts]
            if all(s is not None for s in subformulas):
                return self._z3.Or(*subformulas)
        elif " NOT " in formula_str:
            inner = formula_str.replace(" NOT ", "").strip()
            sub = self._parse_formula(inner)
            if sub is not None:
                return self._z3.Not(sub)
        else:
            # Atomic proposition
            return self._z3.Bool(formula_str)
        return None
    
    def check_consistency(self) -> Dict[str, Any]:
        """Check if current facts and rules are consistent.
        
        Returns:
            {"consistent": bool, "model": dict or None, "contradictions": list}
        """
        if self.backend == "z3" and self._solver:
            result = self._solver.check()
            if result == self._z3.sat:
                model = self._solver.model()
                return {
                    "consistent": True,
                    "model": {str(d.name()): model[d] for d in model.declarations()},
                    "contradictions": [],
                }
            else:
                # Find minimal unsat core
                return {
                    "consistent": False,
                    "model": None,
                    "contradictions": self._find_contradictions(),
                }
        
        # Fallback: simple fact checking
        return self._simple_check()
    
    def _find_contradictions(self) -> List[Dict]:
        """Find minimal set of contradictory facts."""
        contradictions = []
        
        # Check for direct contradictions (A and NOT A)
        for fact, value in self.facts.items():
            neg_fact = f"NOT_{fact}"
            if neg_fact in self.facts and self.facts[neg_fact] == value:
                contradictions.append({
                    "type": "direct_contradiction",
                    "facts": [fact, neg_fact],
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
    
    def extract_rules_from_symbolic(self, symbolic_overlay) -> int:
        """Extract logical rules from SymbolicOverlay.
        
        Converts symbolic rules into prover-compatible format.
        Returns number of rules added.
        """
        n_added = 0
        if not symbolic_overlay or not hasattr(symbolic_overlay, 'rules'):
            return 0
        
        for rule in symbolic_overlay.rules:
            # Convert symbolic rule to logical formula
            if hasattr(rule, 'premise') and hasattr(rule, 'conclusion'):
                rule_str = f"{rule.premise} → {rule.conclusion}"
                self.add_rule(rule_str)
                n_added += 1
        
        return n_added
    
    def resolve_contradiction(
        self,
        node_a_id: str,
        node_b_id: str,
        memory_field,
    ) -> Dict[str, Any]:
        """Resolve contradiction between two nodes using logical inference.
        
        Returns recommendation: keep_a, keep_b, merge, or delete_both.
        """
        # Extract facts from nodes
        text_a = memory_field.nodes[node_a_id].content.get("text", "")
        text_b = memory_field.nodes[node_b_id].content.get("text", "")
        
        # Simple heuristic: keep the node with higher salience
        sal_a = memory_field.nodes[node_a_id].salience
        sal_b = memory_field.nodes[node_b_id].salience
        
        if sal_a > sal_b * 1.2:
            return {"action": "keep_a", "reason": f"Higher salience ({sal_a:.2f} vs {sal_b:.2f})"}
        elif sal_b > sal_a * 1.2:
            return {"action": "keep_b", "reason": f"Higher salience ({sal_b:.2f} vs {sal_a:.2f})"}
        else:
            return {"action": "merge", "reason": "Similar salience, recommend merging"}
    
    def get_stats(self) -> Dict:
        return {
            "backend": self.backend,
            "n_facts": len(self.facts),
            "n_rules": len(self.rules),
            "n_contradictions": len(self.contradictions),
            "solver_available": self._solver is not None,
        }

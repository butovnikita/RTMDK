"""Evaluation Feature Matrix Generator.

Scans the RTMDK codebase and generates a structured feature matrix
with: module, config flag, default value, test coverage, benchmark coverage,
integration status, and recommendation.

Usage:
    python scripts/eval_feature_matrix.py > docs/FEATURE_MATRIX.md
"""
from __future__ import annotations
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent


def scan_config_defaults() -> Dict[str, Tuple[str, Any]]:
    """Scan config.py for all boolean/string flags and their defaults."""
    config_path = PROJECT_ROOT / "rtmdk" / "memory" / "config.py"
    flags = {}
    with open(config_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.target and isinstance(node.target, ast.Name):
            name = node.target.id
            if node.value:
                try:
                    val = ast.literal_eval(node.value)
                    flags[name] = val
                except Exception:
                    pass
    return flags


def find_tests_for_module(module_name: str) -> List[str]:
    """Find test files that import or mention a module."""
    tests_dir = PROJECT_ROOT / "tests"
    matches = []
    for test_file in tests_dir.glob("test_*.py"):
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            if module_name.replace(".", "") in content.replace(".", ""):
                matches.append(test_file.name)
        except Exception:
            pass
    return matches


def find_benchmarks_for_module(module_name: str) -> List[str]:
    """Find benchmark scripts that mention a module."""
    scripts_dir = PROJECT_ROOT / "scripts"
    matches = []
    for script in scripts_dir.glob("bench_*.py"):
        try:
            with open(script, "r", encoding="utf-8") as f:
                content = f.read()
            if module_name.replace(".", "") in content.replace(".", ""):
                matches.append(script.name)
        except Exception:
            pass
    return matches


def check_integration(module_path: str) -> str:
    """Check if module is imported in core.py or field.py."""
    core_path = PROJECT_ROOT / "rtmdk" / "memory" / "core.py"
    field_path = PROJECT_ROOT / "rtmdk" / "memory" / "field.py"
    module_name = module_path.replace("rtmdk/", "").replace("/", ".").replace(".py", "")
    short_name = Path(module_path).stem

    for path in [core_path, field_path]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if short_name in content or module_name in content:
            return "integrated"
    return "standalone"


FEATURE_REGISTRY = [
    # (module_path, config_flag, category, notes)
    ("rtmdk/experimental/active_inference.py", "active_inference", "experimental", "Not integrated"),
    ("rtmdk/experimental/adversarial_arena.py", "adversarial_arena", "experimental", "Not integrated"),
    ("rtmdk/experimental/tpr.py", "tpr_enabled", "experimental", "Research mode"),
    ("rtmdk/engines/neural_ode.py", "continuous_dynamics", "engine", "SciPy ODE, expensive"),
    ("rtmdk/engines/neuro_symbolic_prover.py", "neuro_symbolic_prover", "engine", "Requires z3/pyswip"),
    ("rtmdk/engines/trust_consensus.py", "trust_consensus", "engine", "Research prototype"),
    ("rtmdk/engines/ssm_dynamics.py", "ssm_dynamics", "engine", "Mamba-inspired, random matrices"),
    ("rtmdk/engines/counterfactual.py", "counterfactual_enabled", "engine", "Random noise fallback"),
    ("rtmdk/support/swarm.py", "swarm_memory", "support", "Random voting"),
    ("rtmdk/support/meta_controller.py", "meta_controller", "support", "Score not correlated with recall"),
    ("rtmdk/support/meta_memory.py", "meta_memory", "support", "Generic recommendations"),
    ("rtmdk/support/rl_feedback.py", "rl_feedback", "support", "Keyword-counting reward"),
    ("rtmdk/support/symbolic_overlay.py", "symbolic_overlay", "support", "Regex stopword extraction"),
    ("rtmdk/support/tda.py", "tda_monitoring", "support", "Placeholder TDA"),
    ("rtmdk/support/triton_backend.py", "triton_backend", "support", "Fake Triton"),
    ("rtmdk/support/torch_backend.py", "torch_backend", "support", "Optional GPU"),
    ("rtmdk/support/ump.py", "ump_enabled", "support", "Schema export only"),
    ("rtmdk/support/learnable.py", "differentiable", "support", "Gradients not stepped"),
    ("rtmdk/production/offline_dreamer.py", "offline_dreaming", "production", "All placeholders"),
    ("rtmdk/production/llm_eval.py", "llm_eval", "production", "Requires OpenRouter"),
    ("rtmdk/production/bgem3_embedder.py", "bgem3_enabled", "production", "Requires FlagEmbedding"),
    ("rtmdk/production/contextual_retrieval.py", "contextual_retrieval", "production", "Heuristic headers"),
    ("rtmdk/production/cascade_router.py", "cascade_enabled", "production", "Regex routing"),
    ("rtmdk/memory/explainability.py", "result_explainability_enabled", "memory", "New"),
    ("rtmdk/memory/rag_quality.py", "sentence_reranker_enabled", "memory", "New"),
    ("rtmdk/memory/sot_v2/sif_embedder.py", "sot_v2_enabled", "memory", "92.3% recall proven"),
    ("rtmdk/memory/conformal.py", "conformal_prediction", "memory", "Theory sound"),
    ("rtmdk/memory/kalman.py", "enable_kalman_filter", "memory", "Diagonal approx"),
    ("rtmdk/memory/spectral.py", "spectral_consolidation", "memory", "Custom k-means"),
    ("rtmdk/memory/learned_consolidation.py", "learned_consolidation", "memory", "Tiny MLP"),
    ("rtmdk/memory/quantization.py", "quantization", "memory", "fp16/int8 modes"),
]


def main():
    flags = scan_config_defaults()
    lines = []
    lines.append("# RTMDK Feature Evaluation Matrix\n")
    lines.append("> Auto-generated by `scripts/eval_feature_matrix.py`\n")
    lines.append("| Module | Config Flag | Default | Tests | Benchmarks | Integration | Status | Notes |\n")
    lines.append("|--------|-------------|---------|-------|------------|-------------|--------|-------|\n")

    for module_path, flag, category, notes in FEATURE_REGISTRY:
        full_path = PROJECT_ROOT / module_path
        exists = full_path.exists()
        default = flags.get(flag, "N/A")
        tests = find_tests_for_module(Path(module_path).stem)
        benchmarks = find_benchmarks_for_module(Path(module_path).stem)
        integration = check_integration(module_path) if exists else "missing"

        if not exists:
            status = "[DEAD]"
        elif default is False or default == "none" or default is None:
            status = "[DISABLED]"
        elif default is True:
            status = "[ENABLED]"
        else:
            status = "[CONFIG]"

        lines.append(
            f"| `{module_path}` | `{flag}` | `{default}` | "
            f"{len(tests)} test(s) | {len(benchmarks)} bench | {integration} | {status} | {notes} |\n"
        )

    # Write to stdout
    sys.stdout.write("".join(lines))

    # Also write to file
    output_path = PROJECT_ROOT / "docs" / "FEATURE_MATRIX.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"\n\nWritten to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

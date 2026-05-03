"""
rtmdk/cli.py — CLI Tool for RTMDK Management.

Usage:
    python -m rtmdk status
    python -m rtmdk query "What do I know about coffee?"
    python -m rtmdk stats
    python -m rtmdk export --output backup.json
    python -m rtmdk recommend --nodes 50000 --ram 256
"""

import sys
import os
import json
import argparse
from pathlib import Path


def cmd_status():
    """Show memory status."""
    print("RTMDK Status")
    print("=" * 60)
    print("To check status, import your memory instance and run:")
    print("  print(memory.field.stats)")


def cmd_query(query: str, module_path: str = ""):
    """Test a query."""
    print(f"Query: {query}")
    print("=" * 60)
    print("To test queries, use:")
    print("  from rtmdk import create_rtmdk")
    print("  memory = create_rtmdk('local', embedder=my_embedder)")
    print(f"  ctx = memory.load_memory_variables({{'input': '{query}'}})")
    print("  print(ctx['rtmdk_context'])")


def cmd_stats():
    """Show detailed statistics."""
    print("RTMDK Statistics")
    print("=" * 60)
    print("Access via: memory.field.stats")


def cmd_export(output: str = "rtmdk_backup.json"):
    """Export memory to JSON."""
    print(f"Export to: {output}")
    print("=" * 60)
    print("Use: memory.save_session('backup')")


def cmd_recommend(nodes: int = 1000, ram: float = 256, latency: float = 100, use_case: str = "general"):
    """Recommend optimal preset."""
    from rtmdk.utils.preset_recommender import recommend_preset
    
    result = recommend_preset(
        expected_nodes=nodes,
        available_ram_mb=ram,
        max_latency_ms=latency,
        use_case=use_case,
    )
    
    print("RTMDK Preset Recommender")
    print("=" * 60)
    print(f"  Preset:        {result['preset']}")
    print(f"  Est. RAM:      {result['estimated_ram_mb']} MB")
    print(f"  Est. Latency:  {result['estimated_latency_ms']} ms")
    if result['overrides']:
        print(f"  Overrides:     {result['overrides']}")
    print()
    print("Usage:")
    print(f"  from rtmdk import create_rtmdk")
    print(f"  memory = create_rtmdk('{result['preset']}', embedder=embedder)")


def cmd_bootstrap(corpus_path: str, output: str, model_name: str = "all-MiniLM-L6-v2"):
    """Generate SBERT bootstrap projection from corpus."""
    from rtmdk.memory.bootstrap_sbert import run_bootstrap
    
    print("RTMDK SOT Bootstrap")
    print("=" * 60)
    
    if not os.path.exists(corpus_path):
        print(f"Error: corpus file not found: {corpus_path}")
        sys.exit(1)
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Support both list of strings and dict with 'records'
    if isinstance(data, dict) and "records" in data:
        texts = [
            r.get("context", "") + " " + r.get("answer", "")
            for r in data["records"]
        ]
    elif isinstance(data, list):
        texts = [str(item) for item in data]
    else:
        print("Error: corpus must be a list of strings or a dict with 'records'")
        sys.exit(1)
    
    print(f"Corpus: {corpus_path}")
    print(f"Texts:  {len(texts)}")
    print(f"Model:  {model_name}")
    print(f"Output: {output}")
    print("-" * 60)
    
    run_bootstrap(texts, output_path=output, model_name=model_name)
    print(f"Bootstrap projection saved to: {output}")
    print(f"Usage:")
    print(f"  RTMDKConfig(sot_enabled=True, sot_bootstrap_projection='{output}')")


def cmd_list_presets():
    """List all available presets."""
    from rtmdk import list_presets
    
    presets = list_presets()
    
    print("RTMDK Available Presets")
    print("=" * 60)
    print(f"  {'Preset':<15} {'Dim':>5} {'K':>3} {'Decay':>6} {'Engrams':>7} {'Dream':>6} {'Causal':>6} {'SSM':>4}")
    print(f"  {'─'*60}")
    
    for name, p in presets.items():
        print(f"  {name:<15} {p['latent_dim']:>5} {p['top_k']:>3} "
              f"{p['decay_rate']:>6} {'[OK]' if p['engrams'] else '[FAIL]':>7} "
              f"{'[OK]' if p['dreaming'] else '[FAIL]':>6} "
              f"{'[OK]' if p['causal'] else '[FAIL]':>6} "
              f"{'[OK]' if p['ssm'] else '[FAIL]':>4}")


def main():
    parser = argparse.ArgumentParser(description="RTMDK CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # status
    subparsers.add_parser("status", help="Show memory status")
    
    # query
    q_parser = subparsers.add_parser("query", help="Test a query")
    q_parser.add_argument("query", type=str, help="Query text")
    
    # stats
    subparsers.add_parser("stats", help="Show statistics")
    
    # export
    e_parser = subparsers.add_parser("export", help="Export memory")
    e_parser.add_argument("--output", "-o", type=str, default="rtmdk_backup.json")
    
    # recommend
    r_parser = subparsers.add_parser("recommend", help="Recommend preset")
    r_parser.add_argument("--nodes", "-n", type=int, default=1000)
    r_parser.add_argument("--ram", "-r", type=float, default=256)
    r_parser.add_argument("--latency", "-l", type=float, default=100)
    r_parser.add_argument("--use-case", "-u", type=str, default="general")
    
    # presets
    subparsers.add_parser("presets", help="List available presets")
    
    # bootstrap
    b_parser = subparsers.add_parser("bootstrap", help="Generate SBERT bootstrap projection")
    b_parser.add_argument("corpus", type=str, help="Path to corpus JSON")
    b_parser.add_argument("--output", "-o", type=str, default="sot_bootstrap.npz")
    b_parser.add_argument("--model", "-m", type=str, default="all-MiniLM-L6-v2")
    
    args = parser.parse_args()
    
    if args.command == "status":
        cmd_status()
    elif args.command == "query":
        cmd_query(args.query)
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "export":
        cmd_export(args.output)
    elif args.command == "recommend":
        cmd_recommend(args.nodes, args.ram, args.latency, args.use_case)
    elif args.command == "presets":
        cmd_list_presets()
    elif args.command == "bootstrap":
        cmd_bootstrap(args.corpus, args.output, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

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
    if result["overrides"]:
        print(f"  Overrides:     {result['overrides']}")
    print()
    print("Usage:")
    print("  from rtmdk import create_rtmdk")
    print(f"  memory = create_rtmdk('{result['preset']}', embedder=embedder)")


def cmd_pipeline_diagnose(
    memory_file: str = "",
    config_preset: str = "local",
):
    """Diagnose pipeline health and run a smoke test query.

    Usage:
        python -m rtmdk pipeline-diagnose
        python -m rtmdk pipeline-diagnose --memory ~/.rtmdk/memory.json
        python -m rtmdk pipeline-diagnose --preset production
    """
    import numpy as np
    from rtmdk.memory.config import RTMDKConfig
    from rtmdk.memory.core import RTMDKMemory

    print("RTMDK Pipeline Diagnostics")
    print("=" * 60)

    preset_fn = getattr(RTMDKConfig, config_preset, None)
    if preset_fn is None:
        print(f"Unknown preset '{config_preset}', falling back to 'local'")
        preset_fn = RTMDKConfig.local
    cfg = preset_fn()
    mem = RTMDKMemory(config=cfg, embedder=lambda x: np.zeros(cfg.latent_dim, dtype=np.float32))

    if memory_file and os.path.exists(os.path.expanduser(memory_file)):
        print(f"Loading memory from: {memory_file}")
        try:
            mem.load(os.path.expanduser(memory_file))
        except Exception as exc:
            print(f"  [WARN] Could not load: {exc}")
    else:
        print("No memory file — using empty field for smoke test")
        mem.field.add_node(
            embedding=np.zeros(cfg.latent_dim, dtype=np.float32),
            content={"text": "smoke test node"},
            node_id="smoke_0",
        )

    # Build pipeline and check stages
    pipeline = mem.build_pipeline()
    print(f"\nPipeline stages ({len(pipeline.stages)}):")
    for stage in pipeline.stages:
        breaker = "none"
        if stage.circuit_breaker is not None:
            breaker = stage.circuit_breaker.state.value
        print(f"  - {stage.name:20s} enabled={stage.enabled} breaker={breaker}")

    # Smoke test
    print("\nRunning smoke test query...")
    try:
        result = mem.retrieve_nodes_pipeline("smoke test", top_k=3)
        print("  Query:     smoke test")
        print(f"  Results:   {result.get('results_count', 0)}")
        print(f"  Route:     {result.get('route', 'N/A')}")
        print(f"  Latency:   {result.get('total_latency_ms', 0):.2f} ms")
        print(f"  Degraded:  {result.get('degraded_stages', [])}")
        print(f"  Breakers:  {result.get('breaker_states', {})}")
        print("\n[OK] Pipeline is operational")
    except Exception as exc:
        print(f"\n[FAIL] Pipeline query failed: {exc}")
        raise SystemExit(1)


def cmd_bootstrap_sbert(corpus_path: str, output: str, model_name: str = "all-MiniLM-L6-v2"):
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
        texts = [r.get("context", "") + " " + r.get("answer", "") for r in data["records"]]
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
    print("Usage:")
    print(f"  RTMDKConfig(sot_enabled=True, sot_bootstrap_projection='{output}')")


def cmd_bootstrap_fasttext(model_path: str, corpus_path: str, output: str):
    """Generate FastText bootstrap state from gensim model and corpus."""
    from rtmdk.memory.bootstrap_fasttext import run_bootstrap
    from rtmdk.memory.self_organizing_field import SOTokenizer
    import json

    print("RTMDK SOT FastText Bootstrap")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(corpus_path):
        print(f"Error: corpus file not found: {corpus_path}")
        sys.exit(1)

    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "records" in data:
        texts = [r.get("context", "") + " " + r.get("answer", "") for r in data["records"]]
    elif isinstance(data, list):
        texts = [str(item) for item in data]
    else:
        print("Error: corpus must be a list of strings or a dict with 'records'")
        sys.exit(1)

    print(f"Model:  {model_path}")
    print(f"Corpus: {corpus_path}")
    print(f"Texts:  {len(texts)}")
    print(f"Output: {output}")
    print("-" * 60)

    tok = SOTokenizer(latent_dim=64, tokenization_mode="word")
    run_bootstrap(tok, texts=texts, model_path=model_path)
    state = tok.get_state()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(state, f)
    print(f"Bootstrap state saved to: {output}")
    print("Usage:")
    print("  RTMDKConfig(sot_enabled=True, sot_tokenization_mode='word')")
    print("  # Then load state manually or integrate into your pipeline")


def cmd_list_presets():
    """List all available presets."""
    from rtmdk import list_presets
    from rtmdk.config import RTMDKConfig

    preset_names = list_presets()
    preset_map = {
        "local": RTMDKConfig.local,
        "production": RTMDKConfig.production,
        "research": RTMDKConfig.research,
        "enterprise": RTMDKConfig.enterprise,
        "agent": RTMDKConfig.agent,
        "legal": RTMDKConfig.legal,
        "medical": RTMDKConfig.medical,
        "streaming": RTMDKConfig.streaming,
        "sillytavern": RTMDKConfig.sillytavern,
    }

    print("RTMDK Available Presets")
    print("=" * 60)
    print(f"  {'Preset':<15} {'Dim':>5} {'K':>3} {'Decay':>6} {'Engrams':>7} {'Dream':>6} {'Causal':>6} {'SSM':>4}")
    print(f"  {'-'*60}")

    for name in preset_names:
        fn = preset_map.get(name)
        if not fn:
            continue
        cfg = fn()
        print(
            f"  {name:<15} {cfg.latent_dim:>5} {cfg.top_k:>3} "
            f"{cfg.decay_rate:>6} {'[OK]' if cfg.enable_engrams else '[FAIL]':>7} "
            f"{'[OK]' if cfg.offline_dreaming else '[FAIL]':>6} "
            f"{'[OK]' if cfg.causal_traversal else '[FAIL]':>6} "
            f"{'[OK]' if cfg.ssm_dynamics else '[FAIL]':>4}"
        )


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

    # bootstrap-fasttext
    bf_parser = subparsers.add_parser("bootstrap-fasttext", help="Generate FastText bootstrap state")
    bf_parser.add_argument("model_path", type=str, help="Path to gensim KeyedVectors model")
    bf_parser.add_argument("corpus", type=str, help="Path to corpus JSON")
    bf_parser.add_argument("--output", "-o", type=str, default="sot_fasttext.json")

    # pipeline-diagnose
    pd_parser = subparsers.add_parser("pipeline-diagnose", help="Diagnose pipeline health and run smoke test")
    pd_parser.add_argument("--memory", "-m", type=str, default="", help="Path to memory file (optional)")
    pd_parser.add_argument(
        "--preset", "-p", type=str, default="local", help="Config preset: local, production, research"
    )

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
        cmd_bootstrap_sbert(args.corpus, args.output, args.model)
    elif args.command == "bootstrap-fasttext":
        cmd_bootstrap_fasttext(args.model_path, args.corpus, args.output)
    elif args.command == "pipeline-diagnose":
        cmd_pipeline_diagnose(args.memory, args.preset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

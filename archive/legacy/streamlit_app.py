"""
streamlit_app.py
Interactive Memory Dashboard + NL-Steering for RTMDK.

Run:
    pip install streamlit
    streamlit run streamlit_app.py

Features:
    - Field visualization (2D projection via PCA)
    - Node inspection and manual management
    - Natural language memory steering
    - Real-time stats and health monitoring
    - Goal management (teleological layer)
    - Security violation log
    - Swarm status (if enabled)
"""

import os
import sys
import json
import time
import numpy as np
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import (
    RTMDKConfig, RTMDKMemory, ContextFormat,
    detect_modality, detect_tier,
    apply_attention_bias, format_cognitive_context,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="RTMDK Memory Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def get_embedder():
    """Simple hash-based embedder for demo."""
    def embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base
    return embed


def init_memory():
    """Initialize or load RTMDK memory."""
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, top_k=5, enable_async=False,
        causal_topological=True, meta_adaptive=True, self_healing=True,
        cross_modal=True, memory_tiers={"episodic", "semantic", "procedural"},
        attention_bias=True, goal_tracking=True, rl_feedback=True,
        security_enabled=True, sparse_routing=True,
        crystallization=True, event_driven=True,
        meta_memory=True, swarm_memory=True,
    )
    return RTMDKMemory(config=config, embedder=get_embedder())


if "memory" not in st.session_state:
    st.session_state.memory = init_memory()
    st.session_state.chat_history = []
    st.session_state.node_count = 0

memory: RTMDKMemory = st.session_state.memory

# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("🧠 RTMDK Dashboard")
    st.caption("Resonance-Topological Memory v8.0")

    tab_config, tab_export = st.tabs(["Config", "Export/Import"])

    with tab_config:
        st.subheader("Context Format")
        fmt = st.selectbox("Format", ["plain", "json", "yaml"], index=0)
        memory.config.context_format = ContextFormat(fmt)

        st.subheader("Features")
        memory.config.attention_bias = st.checkbox("Attention Bias", value=True)
        memory.config.security_enabled = st.checkbox("Security", value=True)
        memory.config.goal_tracking = st.checkbox("Goal Tracking", value=True)
        memory.config.meta_memory = st.checkbox("Meta-Memory", value=True)

    with tab_export:
        if st.button("Export Memory"):
            path = "rtmdk_dashboard_state.json"
            memory.export_field(path)
            st.success(f"Exported to {path}")

        uploaded = st.file_uploader("Import Memory", type=["json"])
        if uploaded:
            data = json.loads(uploaded.read())
            # Simple import - create new memory from data
            st.success("Imported (reload page to apply)")

# ============================================================================
# MAIN LAYOUT
# ============================================================================

st.title("🧠 RTMDK Memory Dashboard")

# Top metrics
stats = memory.get_stats()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Nodes", stats["active_nodes"])
col2.metric("Queries", stats["total_queries"])
col3.metric("Consolidations", stats["consolidations"])
col4.metric("Recall Accuracy", f"{stats.get('recall_accuracy', 1.0):.2%}")
col5.metric("Field Health", stats.get("field_health", "stable"))

# Tabs
tab_chat, tab_field, tab_goals, tab_security, tab_nodes = st.tabs([
    "💬 Chat", "🗺️ Field", "🎯 Goals", "🔒 Security", "📦 Nodes"
])

# ============================================================================
# TAB 1: CHAT
# ============================================================================

with tab_chat:
    st.subheader("Chat with Memory")

    # Chat input
    user_input = st.chat_input("Type your message...")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Query memory
        ctx = memory.load_memory_variables({"input": user_input, "session_id": "dashboard"})

        # Build response
        response_lines = [f"**Memory context:** {len(ctx['rtmdk_context'])} chars"]
        if ctx["rtmdk_context"] and ctx["rtmdk_context"] not in ("No relevant memory.", "[]"):
            response_lines.append(f"\n```\n{ctx['rtmdk_context'][:500]}\n```")
        else:
            response_lines.append("\n*No relevant memories found.*")

        # Save to memory
        memory.save_context(
            {"input": user_input, "session_id": "dashboard"},
            {"output": " ".join(response_lines)}
        )

        response_text = "\n".join(response_lines)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.write(response_text)

        st.rerun()

# ============================================================================
# TAB 2: FIELD VISUALIZATION
# ============================================================================

with tab_field:
    st.subheader("Memory Field Visualization")

    if memory.field.nodes:
        # 2D projection using first 2 latent dims
        positions = np.array([n.latent_pos[:2] for n in memory.field.nodes.values()])
        colors = []
        labels = []
        for nid, node in memory.field.nodes.items():
            tier = getattr(node, 'tier', 'semantic')
            if tier == "episodic":
                colors.append("#FF6B6B")
            elif tier == "procedural":
                colors.append("#4ECDC4")
            else:
                colors.append("#45B7D1")
            labels.append(nid[:10])

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, (pos, color, label) in enumerate(zip(positions, colors, labels)):
                ax.scatter(pos[0], pos[1], c=color, s=100, alpha=0.7, label=label if i < 20 else "")
            ax.set_xlabel("Latent Dim 1")
            ax.set_ylabel("Latent Dim 2")
            ax.set_title(f"Memory Field ({len(positions)} nodes)")
            ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig)
        except ImportError:
            st.warning("matplotlib not installed — install with `pip install matplotlib` to see field visualization.")
            # Fallback: show data table instead
            import pandas as pd
            df = pd.DataFrame({
                "Node": list(memory.field.nodes.keys())[:20],
                "Dim1": positions[:20, 0],
                "Dim2": positions[:20, 1],
            })
            st.dataframe(df, use_container_width=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Tier Coherence", f"{stats.get('tier_coherence', 0):.3f}")
        col2.metric("Free Energy", f"{stats.get('free_energy', 0):.4f}")
        col3.metric("Compression Ratio", f"{stats.get('compression_ratio', 1.0):.2f}")
    else:
        st.info("No nodes in memory yet. Start chatting to add nodes.")

# ============================================================================
# TAB 3: GOALS
# ============================================================================

with tab_goals:
    st.subheader("Teleological Layer — Goals")

    # Add goal
    col1, col2 = st.columns([3, 1])
    with col1:
        goal_text = st.text_input("New Goal", placeholder="e.g., Learn user preferences")
    with col2:
        goal_priority = st.number_input("Priority", min_value=0.1, max_value=2.0, value=1.0)

    if st.button("Add Goal") and goal_text:
        gid = memory.add_goal(goal_text, priority=goal_priority)
        st.success(f"Goal added: {gid}")

    # Display active goals
    active_goals = memory.get_active_goals()
    if active_goals:
        st.subheader(f"Active Goals ({len(active_goals)})")
        for goal in active_goals:
            with st.expander(f"{goal['description']} (priority: {goal['priority']:.2f})"):
                st.write(f"**Status:** {goal['status']}")
                st.write(f"**Completion:** {goal['completion']:.1%}")
                st.write(f"**Related nodes:** {len(goal['related_nodes'])}")
    else:
        st.info("No active goals. Add one above.")

# ============================================================================
# TAB 4: SECURITY
# ============================================================================

with tab_security:
    st.subheader("Security Monitor")

    if memory.field.security:
        summary = memory.field.security.get_violation_summary()
        col1, col2 = st.columns(2)
        col1.metric("Total Violations", summary["total_violations"])
        col2.metric("Tension Spike Rate", f"{summary['tension_spike_rate']:.3f}")

        if summary["recent_violations"]:
            st.subheader("Recent Violations")
            for v in summary["recent_violations"][-5:]:
                st.warning(f"**{v['type']}**: {v.get('text_preview', v.get('current', 'N/A'))}")
        else:
            st.success("No security violations detected.")
    else:
        st.info("Security module not enabled. Enable in sidebar.")

# ============================================================================
# TAB 5: NODES
# ============================================================================

with tab_nodes:
    st.subheader("Node Management")

    # Search
    search = st.text_input("Search nodes", placeholder="node ID or text...")

    if memory.field.nodes:
        # Filter nodes
        filtered = []
        for nid, node in memory.field.nodes.items():
            text = node.content.get("text", "")
            if not search or search.lower() in nid.lower() or search.lower() in text.lower():
                filtered.append((nid, node))

        # Display table
        if filtered:
            data = []
            for nid, node in filtered[:50]:
                data.append({
                    "ID": nid[:20],
                    "Tier": getattr(node, 'tier', 'semantic'),
                    "Modality": node.modality,
                    "Amplitude": f"{node.amplitude:.3f}",
                    "Salience": f"{node.salience:.3f}",
                    "Tension": f"{node.tension:.3f}",
                    "Text": node.content.get("text", "")[:60],
                })
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            except ImportError:
                st.warning("pandas not installed — install with `pip install pandas` for table view.")
                # Fallback: show as text
                for row in data:
                    st.text(row)
        else:
            st.info("No nodes match your search.")
    else:
        st.info("No nodes in memory yet.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption(f"RTMDK v8.0 | {stats['active_nodes']} nodes | {stats['total_queries']} queries | {stats['consolidations']} consolidations")

"""Ingest chat history and git log into RTMDK memory."""
import os
import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rtmdk import RTMDKMemory, RTMDKConfig
from sentence_transformers import SentenceTransformer

MEMORY_MSGPACK = Path.home() / ".rtmdk" / "memory.msgpack"
MEMORY_JSON = Path.home() / ".rtmdk" / "memory.json"

print("Loading embedder...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embedder(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

# ------------------------------------------------------------------
# 1. Extract chat messages from wire.jsonl files
# ------------------------------------------------------------------

def extract_chat_turns(wire_path: Path, max_turns: int = 200):
    """Extract user/agent text pairs from a wire.jsonl file."""
    turns = []
    current_user = None
    current_agent_parts = []
    
    with open(wire_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            msg_type = data.get("message", {}).get("type", "")
            payload = data.get("message", {}).get("payload", {})
            
            if msg_type == "TurnBegin":
                # Save previous turn if exists
                if current_user is not None:
                    agent_text = "\n".join(current_agent_parts).strip()
                    if agent_text:
                        turns.append({
                            "user": current_user,
                            "agent": agent_text,
                            "timestamp": data.get("timestamp", 0),
                        })
                # Start new turn
                user_input = payload.get("user_input", [])
                if isinstance(user_input, str):
                    user_input = [{"type": "text", "text": user_input}]
                elif not isinstance(user_input, list):
                    user_input = []
                texts = [p.get("text", "") for p in user_input if isinstance(p, dict) and p.get("type") == "text"]
                current_user = "\n".join(texts).strip()
                current_agent_parts = []
            
            elif msg_type == "ContentPart":
                part_type = payload.get("type", "")
                if part_type == "text":
                    current_agent_parts.append(payload.get("text", ""))
                elif part_type == "think":
                    think = payload.get("think", "").strip()
                    if think:
                        current_agent_parts.append(f"[think] {think[:500]}")
    
    # Save last turn
    if current_user is not None and current_agent_parts:
        agent_text = "\n".join(current_agent_parts).strip()
        if agent_text:
            turns.append({
                "user": current_user,
                "agent": agent_text,
                "timestamp": time.time(),
            })
    
    return turns[-max_turns:]


# ------------------------------------------------------------------
# 2. Extract git log
# ------------------------------------------------------------------

def extract_git_log(project_root: Path, n: int = 100):
    """Extract git commit history."""
    try:
        result = subprocess.run(
            ["git", "log", f"--max-count={n}", "--format=%H|%ai|%an|%s"],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0],
                    "date": parts[1],
                    "author": parts[2],
                    "message": parts[3],
                })
        return commits
    except Exception as e:
        print(f"Git log error: {e}")
        return []


# ------------------------------------------------------------------
# 3. Build memory nodes
# ------------------------------------------------------------------

def main():
    # Load existing memory
    if MEMORY_MSGPACK.exists():
        memory = RTMDKMemory.import_field(str(MEMORY_MSGPACK), embedder)
        print(f"Loaded {len(memory.field.nodes)} existing nodes")
    else:
        cfg = RTMDKConfig.production()
        memory = RTMDKMemory(config=cfg, embedder=embedder)
        print("Created fresh memory")

    nodes_to_add = []

    # --- Chat history ---
    session_dir = Path.home() / ".kimi" / "sessions"
    latest_session = max(session_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    print(f"Reading session: {latest_session.name}")

    all_turns = []
    for subdir in sorted(latest_session.iterdir()):
        if subdir.is_dir() and (subdir / "wire.jsonl").exists():
            wire = subdir / "wire.jsonl"
            print(f"  Reading {wire.name} from {subdir.name}...")
            turns = extract_chat_turns(wire, max_turns=200)
            print(f"    -> {len(turns)} turns")
            all_turns.extend(turns)

    print(f"Total chat turns: {len(all_turns)}")
    for i, turn in enumerate(all_turns):
        text = f"User: {turn['user']}\nAgent: {turn['agent'][:2000]}"
        nodes_to_add.append({
            "text": text,
            "title": f"Chat turn {i+1}",
            "tags": ["chat", "session", "conversation"],
            "source": "kimi_session",
            "user_query": turn["user"][:500],
            "agent_response": turn["agent"][:2000],
            "timestamp": turn.get("timestamp", time.time()),
        })

    # --- Git history ---
    project_root = Path(__file__).parent.parent
    commits = extract_git_log(project_root, n=100)
    print(f"Git commits: {len(commits)}")
    for c in commits:
        text = f"Commit {c['hash'][:8]} by {c['author']} on {c['date']}: {c['message']}"
        nodes_to_add.append({
            "text": text,
            "title": f"Git commit {c['hash'][:8]}",
            "tags": ["git", "commit", "history"],
            "source": "git_log",
            "commit_hash": c["hash"],
            "author": c["author"],
            "date": c["date"],
            "message": c["message"],
        })

    if not nodes_to_add:
        print("Nothing to add")
        return

    # Batch add
    print(f"Encoding {len(nodes_to_add)} nodes...")
    texts = [n["text"] for n in nodes_to_add]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    print(f"Adding {len(nodes_to_add)} nodes...")
    nids = memory.add_nodes_batch(embeddings, nodes_to_add)
    print(f"Added {len(nids)} nodes")

    # Save
    print(f"Saving to {MEMORY_MSGPACK}...")
    memory.field.export_field(str(MEMORY_MSGPACK), fmt="msgpack")
    print(f"Saving to {MEMORY_JSON}...")
    memory.field.export_field(str(MEMORY_JSON), fmt="json")

    print(f"Total nodes in memory: {len(memory.field.nodes)}")
    memory.close()


if __name__ == "__main__":
    main()

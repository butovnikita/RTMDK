"""
rtmdk/production/session_persistence.py — Auto-save/Load User Sessions.

Persists memory state per user/session for cross-session continuity.
Features:
- Save/load: memory.save_session("user123"), memory.load_session("user123")
- Auto-save on configurable interval
- Session metadata: created, last_accessed, node_count, size_mb
- List sessions, delete sessions
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class SessionPersistence:
    """Manages per-user memory session save/load.

    Usage:
        sp = SessionPersistence(memory, save_dir="~/.rtmdk/sessions")

        # Save current state:
        sp.save_session("user123")

        # Load saved state:
        sp.load_session("user123")

        # List all sessions:
        sessions = sp.list_sessions()

        # Auto-save every 60 seconds:
        sp.start_auto_save(interval=60)
    """

    def __init__(
        self,
        memory,  # RTMDKMemory instance
        save_dir: str = "~/.rtmdk/sessions",
        auto_save_interval: int = 0,  # 0 = disabled
        max_sessions: int = 1000,
    ):
        self.memory = memory
        self.save_dir = Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval
        self.max_sessions = max_sessions

        self._sessions: Dict[str, Dict] = {}  # session_id → metadata
        self._auto_save_timer: float = 0
        self._load_metadata()

    def save_session(
            self,
            session_id: str,
            metadata: Optional[Dict] = None) -> str:
        """Save current memory state to a session file.

        Args:
            session_id: Unique session/user identifier
            metadata: Additional metadata to store

        Returns:
            Path to saved file
        """
        filepath = self.save_dir / f"{session_id}.json"

        # Collect memory state
        nodes_data = {}
        for nid, node in self.memory.field.nodes.items():
            nodes_data[nid] = {
                "content": node.content,
                "latent_pos": node.latent_pos.tolist() if hasattr(
                    node.latent_pos,
                    'tolist') else list(
                    node.latent_pos),
                "phase": node.phase,
                "amplitude": node.amplitude,
                "salience": node.salience,
                "tier": getattr(
                    node,
                    'tier',
                    'semantic'),
                "created_at": node.created_at,
            }

        session_data = {
            "session_id": session_id,
            "saved_at": time.time(),
            "node_count": len(nodes_data),
            "nodes": nodes_data,
            "metadata": metadata or {},
            "stats": dict(self.memory.field.stats),
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        # Update metadata
        size_mb = filepath.stat().st_size / 1024 / 1024
        self._sessions[session_id] = {
            "session_id": session_id,
            "saved_at": session_data["saved_at"],
            "node_count": len(nodes_data),
            "size_mb": round(size_mb, 2),
            "filepath": str(filepath),
            **(metadata or {}),
        }

        self._save_metadata()
        return str(filepath)

    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load memory state from a session file.

        Args:
            session_id: Session identifier to load

        Returns:
            Session metadata dict, or None if not found
        """
        filepath = self.save_dir / f"{session_id}.json"
        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            session_data = json.load(f)

        # Note: Full node restoration requires access to embedder
        # This method restores basic state
        self.memory.field.nodes.clear()
        self.memory.field.node_index.clear()

        for nid, node_data in session_data.get("nodes", {}).items():
            from rtmdk.nodes import MemoryNode
            node = MemoryNode(
                id=nid,
                latent_pos=node_data["latent_pos"],
                phase=node_data["phase"],
                amplitude=node_data["amplitude"],
                salience=node_data["salience"],
                content=node_data["content"],
            )
            node.tier = node_data.get("tier", "semantic")
            node.created_at = node_data.get("created_at", time.time())
            self.memory.field.nodes[nid] = node
            self.memory.field.node_index.append(nid)

        # Restore stats
        if "stats" in session_data:
            self.memory.field.stats.update(session_data["stats"])

        # Update session metadata
        self._sessions[session_id] = {
            "session_id": session_id,
            "loaded_at": time.time(),
            "saved_at": session_data.get("saved_at"),
            "node_count": session_data.get("node_count", 0),
        }

        return self._sessions[session_id]

    def list_sessions(self) -> List[Dict]:
        """List all available sessions."""
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        filepath = self.save_dir / f"{session_id}.json"
        if filepath.exists():
            filepath.unlink()
            self._sessions.pop(session_id, None)
            self._save_metadata()
            return True
        return False

    def start_auto_save(self, interval: Optional[int] = None):
        """Start auto-save timer (call this periodically in your main loop)."""
        if interval is not None:
            self.auto_save_interval = interval
        if self.auto_save_interval <= 0:
            return

        if time.time() - self._auto_save_timer >= self.auto_save_interval:
            self._auto_save_timer = time.time()
            # Auto-save all active sessions
            for session_id in list(self._sessions.keys()):
                self.save_session(session_id)

    def _save_metadata(self):
        """Save session index file."""
        meta_path = self.save_dir / "sessions_index.json"
        with open(meta_path, 'w') as f:
            json.dump(self._sessions, f, indent=2)

    def _load_metadata(self):
        """Load session index file."""
        meta_path = self.save_dir / "sessions_index.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    self._sessions = json.load(f)
            except (json.JSONDecodeError, KeyError):
                self._sessions = {}

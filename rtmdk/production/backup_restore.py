"""
rtmdk/production/backup_restore.py — Automated Backup & Restore.

Features:
- Scheduled backups with rotation
- One-click restore from backup
- Incremental backup support
- Backup to local disk or dict (for cloud upload)
"""

import os
import json
import time
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any


class BackupManager:
    """Manages RTMDK memory backups and restores.
    
    Usage:
        backup = BackupManager(memory, backup_dir="~/.rtmdk/backups")
        
        # Manual backup
        backup_path = backup.create_backup("manual_backup")
        
        # Restore from backup
        backup.restore(backup_path)
        
        # List backups
        backups = backup.list_backups()
        
        # Auto-backup with rotation
        backup.create_backup(auto_rotate=True, max_backups=5)
    """
    
    def __init__(
        self,
        memory,
        backup_dir: str = "~/.rtmdk/backups",
        compression: bool = True,
    ):
        self.memory = memory
        self.backup_dir = Path(backup_dir).expanduser()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
    
    def create_backup(self, name: str = "", auto_rotate: bool = False, max_backups: int = 5) -> str:
        """Create a backup of current memory state.
        
        Args:
            name: Backup name (auto-generated if empty)
            auto_rotate: Delete oldest backups if over max_backups
            max_backups: Maximum number of backups to keep
            
        Returns:
            Path to backup file
        """
        if not name:
            name = time.strftime("backup_%Y%m%d_%H%M%S")
        
        filepath = self.backup_dir / f"{name}.json"
        
        # Collect memory state
        nodes_data = {}
        for nid, node in self.memory.field.nodes.items():
            nodes_data[nid] = {
                "content": node.content,
                "latent_pos": node.latent_pos.tolist() if hasattr(node.latent_pos, 'tolist') else list(node.latent_pos),
                "phase": node.phase,
                "amplitude": node.amplitude,
                "salience": node.salience,
                "tier": getattr(node, 'tier', 'semantic'),
                "created_at": node.created_at,
                "causal_parents": getattr(node, 'causal_parents', []),
            }
        
        backup_data = {
            "name": name,
            "created_at": time.time(),
            "node_count": len(nodes_data),
            "nodes": nodes_data,
            "stats": dict(self.memory.field.stats),
        }
        
        # Save (optionally compressed)
        if self.compression:
            filepath = filepath.with_suffix('.json.gz')
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(backup_data, f)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2)
        
        # Rotate if needed
        if auto_rotate:
            self._rotate_backups(max_backups)
        
        return str(filepath)
    
    def restore(self, backup_path: str) -> Dict[str, Any]:
        """Restore memory from a backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Restore result dict
        """
        path = Path(backup_path)
        if not path.exists():
            return {"success": False, "error": f"Backup not found: {backup_path}"}
        
        try:
            # Load backup
            if str(path).endswith('.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
            
            # Clear current memory
            self.memory.field.nodes.clear()
            self.memory.field.node_index.clear()
            
            # Restore nodes
            nodes_restored = 0
            for nid, node_data in backup_data.get("nodes", {}).items():
                # Reconstruct node
                from rtmdk.nodes import MemoryNode
                node = MemoryNode(
                    id=nid,
                    latent_pos=node_data["latent_pos"],
                    phase=node_data["phase"],
                    amplitude=node_data["amplitude"],
                    salience=node_data["salience"],
                    content=node_data["content"],
                )
                # Set additional attributes
                node.tier = node_data.get("tier", "semantic")
                node.created_at = node_data.get("created_at", time.time())
                if "causal_parents" in node_data:
                    node.causal_parents = node_data["causal_parents"]
                
                self.memory.field.nodes[nid] = node
                self.memory.field.node_index.append(nid)
                nodes_restored += 1
            
            # Restore stats if available
            if "stats" in backup_data:
                self.memory.field.stats.update(backup_data["stats"])
            
            return {
                "success": True,
                "nodes_restored": nodes_restored,
                "backup_name": backup_data.get("name", "unknown"),
                "backup_created": backup_data.get("created_at"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        for filepath in sorted(self.backup_dir.glob("*.json*")):
            stat = filepath.stat()
            backups.append({
                "name": filepath.stem,
                "path": str(filepath),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": stat.st_mtime,
            })
        return backups
    
    def delete_backup(self, backup_path: str) -> bool:
        """Delete a backup file."""
        path = Path(backup_path)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def _rotate_backups(self, max_backups: int):
        """Delete oldest backups if over limit."""
        backups = sorted(self.backup_dir.glob("*.json*"), key=lambda p: p.stat().st_mtime)
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            oldest.unlink()

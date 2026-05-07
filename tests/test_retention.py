"""Tests for RetentionManager."""

import asyncio
import time

import pytest

from rtmdk.production.retention import RetentionManager, RetentionPolicy


class FakeNode:
    def __init__(self, last_accessed=None, created_at=None):
        self.last_accessed = last_accessed
        self.created_at = created_at


class FakeField:
    def __init__(self, nodes):
        self.nodes = nodes
        self.deleted = []

    def delete_nodes(self, node_ids):
        self.deleted.extend(node_ids)
        for nid in node_ids:
            self.nodes.pop(nid, None)


def test_retention_manager_stats():
    mgr = RetentionManager(None)
    stats = mgr.stats()
    assert stats["pruned_total"] == 0
    assert stats["policy"]["enabled"] is True


def test_retention_manager_set_policy():
    mgr = RetentionManager(None)
    policy = RetentionPolicy(max_age_seconds=3600, max_nodes=100)
    mgr.set_policy(policy)
    assert mgr.policy.max_age_seconds == 3600
    assert mgr.policy.max_nodes == 100


def test_age_based_pruning():
    now = time.time()
    nodes = {
        "old": FakeNode(last_accessed=now - 86400 * 2),
        "recent": FakeNode(last_accessed=now - 3600),
    }
    field = FakeField(nodes)
    mgr = RetentionManager(field)
    mgr.set_policy(RetentionPolicy(max_age_seconds=86400))
    mgr._enforce()
    assert "old" in field.deleted
    assert "recent" not in field.deleted
    assert mgr.stats()["pruned_total"] == 1


def test_count_based_pruning():
    now = time.time()
    nodes = {
        "a": FakeNode(last_accessed=now - 100),
        "b": FakeNode(last_accessed=now - 200),
        "c": FakeNode(last_accessed=now - 300),
    }
    field = FakeField(nodes)
    mgr = RetentionManager(field)
    mgr.set_policy(RetentionPolicy(max_nodes=2))
    mgr._enforce()
    assert "c" in field.deleted
    assert "a" not in field.deleted
    assert "b" not in field.deleted
    assert len(field.nodes) == 2


def test_no_pruning_when_disabled():
    now = time.time()
    nodes = {"a": FakeNode(last_accessed=now - 86400 * 10)}
    field = FakeField(nodes)
    mgr = RetentionManager(field)
    mgr.set_policy(RetentionPolicy(max_age_seconds=1, enabled=False))
    mgr._enforce()
    assert len(field.deleted) == 0


@pytest.mark.asyncio
async def test_start_stop():
    mgr = RetentionManager(None, check_interval=0.1)
    mgr.start()
    await asyncio.sleep(0.05)
    mgr.stop()
    assert not mgr._running

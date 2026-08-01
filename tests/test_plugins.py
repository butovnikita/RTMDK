"""Test Phase 5 plugin architecture."""

import numpy as np
import pytest

from rtmdk.memory.plugins import FieldPlugin, MemoryPort


class TestFieldPlugin:
    def test_protocol_can_be_implemented(self):
        class MyPlugin:
            name = "my_plugin"

            def on_node_added(self, node_id, latent_pos, content):
                pass

            def on_query(self, query_latent, results):
                return results

            def on_consolidate(self, updated_nodes):
                pass

            def get_state(self):
                return {}

            def load_state(self, state):
                pass

        p = MyPlugin()
        assert isinstance(p, FieldPlugin)


class TestMemoryPort:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            MemoryPort()

    def test_concrete_subclass(self):
        class DummyPort(MemoryPort):
            def add(self, embedding, content, **kwargs):
                return "n1"

            def query(self, embedding, top_k=10, session_id=None):
                return []

            def delete(self, node_id):
                return True

            def export(self, path, fmt=None):
                pass

            def import_(self, path):
                pass

            def stats(self):
                return {}

        port = DummyPort()
        assert port.add(np.zeros(10), {}) == "n1"
        assert port.stats() == {}

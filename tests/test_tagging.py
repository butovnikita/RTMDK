"""Tests for rtmdk.production.tagging."""

import pytest
from rtmdk.production.tagging import TaggingSystem


class MockMemory:
    pass


class TestTaggingSystem:
    def test_add_and_get_tags(self):
        tags = TaggingSystem(MockMemory())
        tags.add_tag("n1", "important")
        assert tags.get_tags_for_node("n1") == ["important"]

    def test_add_tags(self):
        tags = TaggingSystem(MockMemory())
        tags.add_tags("n1", ["a", "b"])
        assert set(tags.get_tags_for_node("n1")) == {"a", "b"}

    def test_get_nodes_by_tag(self):
        tags = TaggingSystem(MockMemory())
        tags.add_tag("n1", "coffee")
        tags.add_tag("n2", "coffee")
        assert set(tags.get_nodes_by_tag("coffee")) == {"n1", "n2"}

    def test_remove_tag(self):
        tags = TaggingSystem(MockMemory())
        tags.add_tag("n1", "x")
        tags.remove_tag("n1", "x")
        assert tags.get_tags_for_node("n1") == []

    def test_list_tags(self):
        tags = TaggingSystem(MockMemory())
        tags.add_tag("n1", "x")
        tags.add_tag("n2", "x")
        assert tags.list_tags() == {"x": 2}

    def test_export_import(self):
        tags = TaggingSystem(MockMemory())
        tags.add_tags("n1", ["a", "b"])
        exported = tags.export_tags()
        tags2 = TaggingSystem(MockMemory())
        tags2.import_tags(exported)
        assert set(tags2.get_tags_for_node("n1")) == {"a", "b"}

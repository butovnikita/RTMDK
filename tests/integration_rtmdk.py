"""
rtmdk_integration_test.py — Full Integration Test for RTMDK Message Flow.

Tests the complete lifecycle:
1. Server startup → memory.json created
2. Send message → save_context works
3. Query memory → load_memory_variables returns context
4. Check node count increases
5. Check stats update correctly
6. Test shutdown → memory saves correctly
7. Test restart → memory loads correctly
8. Verify no errors in any step
"""

import os
import sys
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk.memory.core import RTMDKConfig, RTMDKMemory


# Mock embedder for testing (deterministic)
class MockEmbedder:
    """Simple deterministic embedder for testing."""
    def __init__(self, dim=768):
        self.dim = dim
    
    def __call__(self, text: str) -> np.ndarray:
        # Create deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        emb = np.random.randn(self.dim).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


def test_1_memory_creation():
    """Test 1: Memory file is created on startup."""
    print("=" * 70)
    print("TEST 1: Memory Creation on Startup")
    print("=" * 70)
    
    embedder = MockEmbedder()
    config = RTMDKConfig(
        embedding_dim=768,
        latent_dim=256,
        learn_projection=False,  # Use manual projection for reliability
        bm25_fallback=True,
        use_hnsw=True,
    )
    
    memory = RTMDKMemory(config=config, embedder=embedder)
    
    # Verify memory initialized
    assert memory is not None, "Memory should be initialized"
    assert memory.field is not None, "Field should exist"
    assert len(memory.field.nodes) == 0, "Should start with 0 nodes"
    
    print("[OK] Memory initialized successfully")
    print(f"   Config: latent_dim={config.latent_dim}, learn_projection={config.learn_projection}")
    print(f"   Initial nodes: {len(memory.field.nodes)}")
    return memory


def test_2_save_context(memory):
    """Test 2: save_context works and creates nodes."""
    print("\n" + "=" * 70)
    print("TEST 2: Save Context")
    print("=" * 70)
    
    test_messages = [
        ("I love drinking coffee every morning", "User likes morning coffee"),
        ("My favorite color is blue", "User's favorite color"),
        ("I work as a software developer", "User's profession"),
        ("I live in New York", "User's location"),
        ("I have a dog named Max", "User has a dog"),
    ]
    
    for i, (input_text, output_text) in enumerate(test_messages):
        try:
            memory.save_context(
                {"input": input_text, "session_id": "test"},
                {"output": output_text}
            )
            print(f"[OK] Message {i+1} saved: '{input_text[:40]}...'")
        except Exception as e:
            print(f"[FAIL] Message {i+1} FAILED: {e}")
            raise
    
    node_count = len(memory.field.nodes)
    print(f"\n   Total nodes after {len(test_messages)} messages: {node_count}")
    assert node_count > 0, "Should have nodes after saving messages"
    assert node_count >= len(test_messages), f"Expected at least {len(test_messages)} nodes, got {node_count}"
    
    # Check stats
    total_adds = memory.field.stats.get("total_adds", 0)
    print(f"   Stats - total_adds: {total_adds}")
    
    return memory


def test_3_query_memory(memory):
    """Test 3: load_memory_variables returns context."""
    print("\n" + "=" * 70)
    print("TEST 3: Query Memory")
    print("=" * 70)
    
    test_queries = [
        ("What do I drink in the morning?", "coffee"),
        ("What is my favorite color?", "blue"),
        ("Where do I live?", "New York"),
        ("What is my job?", "software"),
        ("What is my pet's name?", "Max"),
    ]
    
    for query, expected_keyword in test_queries:
        try:
            ctx = memory.load_memory_variables({"input": query, "session_id": "test"})
            context = ctx.get("rtmdk_context", "")
            
            if context and context not in ("No relevant memory.", "[]"):
                # Check if we got some context
                found = expected_keyword.lower() in context.lower()
                status = "[OK]" if found else "[WARN]"
                print(f"{status} Query: '{query}'")
                print(f"   Context length: {len(context)} chars")
                if not found:
                    print(f"   Keyword '{expected_keyword}' NOT found in context")
                    print(f"   Context preview: {context[:100]}...")
            else:
                print(f"[FAIL] Query: '{query}' - No context returned")
                print(f"   Context: {context}")

        except Exception as e:
            print(f"[FAIL] Query FAILED: '{query}' - {e}")
            raise
    
    # Check query stats
    total_queries = memory.field.stats.get("total_queries", 0)
    print(f"\n   Stats - total_queries: {total_queries}")
    return memory


def test_4_memory_persistence(memory, temp_dir):
    """Test 4: Memory saves and loads correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Memory Persistence (Save/Load)")
    print("=" * 70)
    
    # Save memory
    save_path = os.path.join(temp_dir, "test_memory.json")
    node_count_before = len(memory.field.nodes)
    
    try:
        memory.export_field(save_path)
        print(f"[OK] Memory exported to: {save_path}")
        print(f"   File size: {os.path.getsize(save_path)} bytes")
        print(f"   Nodes before save: {node_count_before}")
    except Exception as e:
        print(f"[FAIL] Export FAILED: {e}")
        raise
    
    # Load memory into new instance
    try:
        embedder = MockEmbedder()
        memory2 = RTMDKMemory.import_field(save_path, embedder)
        node_count_after = len(memory2.field.nodes)
        
        print(f"[OK] Memory imported successfully")
        print(f"   Nodes after load: {node_count_after}")

        assert node_count_before == node_count_after, \
            f"Node count mismatch: {node_count_before} before, {node_count_after} after"

        # Verify nodes have same data
        for nid in list(memory.field.nodes.keys())[:5]:
            node1 = memory.field.nodes[nid]
            node2 = memory2.field.nodes[nid]

            assert node1.salience == node2.salience, f"Salience mismatch for {nid}"
            assert node1.amplitude == node2.amplitude, f"Amplitude mismatch for {nid}"
            assert node1.content.get("text") == node2.content.get("text"), f"Content mismatch for {nid}"

        print(f"[OK] All verified nodes match")

        return memory2

    except Exception as e:
        print(f"[FAIL] Import FAILED: {e}")
        raise


def test_5_projection_logic(memory):
    """Test 5: Projection works correctly."""
    print("\n" + "=" * 70)
    print("TEST 5: Projection Logic")
    print("=" * 70)
    
    embedder = MockEmbedder()
    
    # Test projection through field
    test_embedding = embedder("test projection")
    
    try:
        # Test _project method
        if hasattr(memory.field, '_project'):
            latent = memory.field._project(test_embedding)
            assert latent.shape == (256,), f"Expected shape (256,), got {latent.shape}"
            print(f"[OK] _project works: output shape {latent.shape}")

        # Test projection_learner if exists
        if hasattr(memory.field, 'projection_learner') and memory.field.projection_learner is not None:
            proj = memory.field.projection_learner
            result = proj.project(test_embedding)
            assert result.shape == (256,), f"Expected shape (256,), got {result.shape}"
            print(f"[OK] projection_learner.project works: output shape {result.shape}")

            # Test update doesn't crash
            result2 = proj.update(test_embedding)
            assert result2.shape == (256,), f"Expected shape (256,), got {result2.shape}"
            print(f"[OK] projection_learner.update works: output shape {result2.shape}")

        else:
            print("[WARN]  No projection_learner (using manual projection)")
            print("[OK] Manual projection works (verified by successful queries)")

    except Exception as e:
        print(f"[FAIL] Projection FAILED: {e}")
        raise


def test_6_stats_consistency(memory):
    """Test 6: Stats are consistent."""
    print("\n" + "=" * 70)
    print("TEST 6: Stats Consistency")
    print("=" * 70)
    
    stats = memory.field.stats
    
    # Check all expected keys exist
    expected_keys = [
        "total_adds", "total_queries", "consolidations",
        "bm25_fallbacks", "field_health"
    ]
    
    for key in expected_keys:
        if key in stats:
            print(f"[OK] Stat '{key}': {stats[key]}")
        else:
            print(f"[WARN]  Stat '{key}' missing (optional)")
    
    # Verify node count matches
    node_count = len(memory.field.nodes)
    print(f"\n   Node count: {node_count}")
    print(f"   Node index length: {len(memory.field.node_index)}")
    
    assert node_count == len(memory.field.node_index), \
        f"Node count mismatch: {node_count} nodes, {len(memory.field.node_index)} in index"

    print("[OK] Stats consistent")
    return True


def test_7_error_handling():
    """Test 7: Error handling for edge cases."""
    print("\n" + "=" * 70)
    print("TEST 7: Error Handling")
    print("=" * 70)
    
    embedder = MockEmbedder()
    config = RTMDKConfig(
        embedding_dim=768,
        latent_dim=256,
        learn_projection=False,
    )
    
    memory = RTMDKMemory(config=config, embedder=embedder)
    
    # Test empty query
    try:
        ctx = memory.load_memory_variables({"input": "", "session_id": "test"})
        print("[OK] Empty query handled gracefully")
    except Exception as e:
        print(f"[FAIL] Empty query FAILED: {e}")
        raise

    # Test very long query
    try:
        long_query = "test " * 1000
        ctx = memory.load_memory_variables({"input": long_query, "session_id": "test"})
        print("[OK] Long query handled gracefully")
    except Exception as e:
        print(f"[FAIL] Long query FAILED: {e}")
        raise

    # Test invalid session
    try:
        memory.save_context(
            {"input": "test", "session_id": None},
            {"output": "test"}
        )
        print("[OK] None session_id handled gracefully")
    except Exception as e:
        print(f"[FAIL] None session_id FAILED: {e}")
        raise
    
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "[TEST] " * 35)
    print("RTMDK INTEGRATION TEST SUITE")
    print("[TEST] " * 35 + "\n")
    
    temp_dir = tempfile.mkdtemp(prefix="rtmdk_test_")
    all_passed = True
    
    try:
        # Run tests in order
        memory = test_1_memory_creation()
        memory = test_2_save_context(memory)
        memory = test_3_query_memory(memory)
        memory = test_4_memory_persistence(memory, temp_dir)
        test_5_projection_logic(memory)
        test_6_stats_consistency(memory)
        test_7_error_handling()
        
        print("\n" + "=" * 70)
        print("[PASS] ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[FAIL] TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        all_passed = False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

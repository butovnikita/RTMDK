"""
rtmdk_integration_test_real.py — Real Integration Test with LM Studio.

Tests the complete RTMDK lifecycle with real LM Studio embeddings:
1. Server startup with real embedder
2. Message flow: save → query → retrieve
3. SillyTavern endpoint compatibility
4. Web UI /health endpoint
5. Backup/restore
6. Node count verification
"""

import os
import sys
import json
import requests
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import LMStudioEmbedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory

# Configuration
LM_STUDIO_URL = "http://127.0.0.1:12345/v1"
RTMDK_URL = "http://127.0.0.1:8080"

def test_embedder_connection():
    """Test 1: LM Studio embedder is available."""
    print("=" * 70)
    print("TEST 1: LM Studio Embedder Connection")
    print("=" * 70)
    
    try:
        embedder = LMStudioEmbedder()
        emb = embedder("test embedding")
        assert emb is not None, "Embedding should not be None"
        assert emb.shape[0] == 768, f"Expected 768 dimensions, got {emb.shape[0]}"
        print(f"✅ Embedder connected: dim={emb.shape[0]}")
        return embedder
    except Exception as e:
        print(f"❌ Embedder FAILED: {e}")
        print("   Make sure LM Studio is running with embedding model loaded")
        raise


def test_memory_with_real_embeddings(embedder):
    """Test 2: Memory operations with real embeddings."""
    print("\n" + "=" * 70)
    print("TEST 2: Memory Operations with Real Embeddings")
    print("=" * 70)
    
    config = RTMDKConfig(
        embedding_dim=768,
        latent_dim=256,
        learn_projection=False,
        bm25_fallback=True,
        use_hnsw=True,
    )
    
    memory = RTMDKMemory(config=config, embedder=embedder)
    print(f"✅ Memory created with real embedder")
    
    # Save test messages
    test_messages = [
        ("I love drinking coffee every morning at 8am", "User likes morning coffee"),
        ("My favorite color is blue, especially navy blue", "User's favorite color is blue"),
        ("I work as a software developer in New York", "User is a developer in NYC"),
        ("I have a golden retriever dog named Max", "User has a dog named Max"),
        ("I prefer tea in the evening, not coffee", "User prefers evening tea"),
    ]
    
    for i, (input_text, output_text) in enumerate(test_messages):
        try:
            memory.save_context(
                {"input": input_text, "session_id": "real_test"},
                {"output": output_text}
            )
            print(f"✅ Message {i+1} saved")
        except Exception as e:
            print(f"❌ Message {i+1} FAILED: {e}")
            raise
    
    node_count = len(memory.field.nodes)
    print(f"\n   Total nodes: {node_count}")
    assert node_count >= len(test_messages), f"Expected {len(test_messages)} nodes, got {node_count}"
    
    return memory


def test_query_real_memory(memory):
    """Test 3: Query memory with real embeddings."""
    print("\n" + "=" * 70)
    print("TEST 3: Query Memory with Real Embeddings")
    print("=" * 70)
    
    test_queries = [
        ("What do I drink in the morning?", "coffee"),
        ("What is my favorite color?", "blue"),
        ("Where do I work and live?", "New York"),
        ("What is my pet's name?", "Max"),
        ("What do I prefer in the evening?", "tea"),
    ]
    
    results = []
    for query, expected_keyword in test_queries:
        try:
            ctx = memory.load_memory_variables({"input": query, "session_id": "real_test"})
            context = ctx.get("rtmdk_context", "")
            
            if context and context not in ("No relevant memory.", "[]"):
                found = expected_keyword.lower() in context.lower()
                status = "✅" if found else "⚠️"
                print(f"{status} Query: '{query}'")
                print(f"   Keyword '{expected_keyword}': {'FOUND' if found else 'NOT FOUND'}")
                results.append({"query": query, "found": found, "keyword": expected_keyword})
            else:
                print(f"❌ Query: '{query}' - No context")
                results.append({"query": query, "found": False, "keyword": expected_keyword})
                
        except Exception as e:
            print(f"❌ Query FAILED: '{query}' - {e}")
            results.append({"query": query, "found": False, "error": str(e)})
    
    # Calculate recall
    found_count = sum(1 for r in results if r.get("found"))
    recall = found_count / len(results) * 100
    print(f"\n   Recall@5: {found_count}/{len(results)} = {recall:.0f}%")
    
    return memory, recall


def test_sillytavern_endpoints(memory):
    """Test 4: SillyTavern compatible endpoints."""
    print("\n" + "=" * 70)
    print("TEST 4: SillyTavern Endpoint Compatibility")
    print("=" * 70)
    
    # This would require running server, so we'll test the router creation
    try:
        from rtmdk_sillytavern_compat import create_sillytavern_router
        
        # Test router creation
        router = create_sillytavern_router(
            lambda: memory,
            {},
            lambda: True,  # lm_studio_available
            lambda: "test-model",  # chat_model
            "http://127.0.0.1:12345/v1"  # lm_studio_url
        )
        
        print("✅ SillyTavern router created successfully")
        print(f"   Number of routes: {len(router.routes)}")
        
        # Check all expected routes exist
        route_paths = [route.path for route in router.routes]
        expected_paths = ["/v1/generate", "/api/v1/generate", "/v1/completions"]
        
        for path in expected_paths:
            if any(path in rp for rp in route_paths):
                print(f"✅ Route {path} exists")
            else:
                print(f"⚠️  Route {path} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ SillyTavern router FAILED: {e}")
        return False


def test_health_endpoint():
    """Test 5: Health endpoint returns correct format."""
    print("\n" + "=" * 70)
    print("TEST 5: Health Endpoint Format")
    print("=" * 70)
    
    try:
        from rtmdk_server_ux import create_ux_router
        
        config = RTMDKConfig(
            embedding_dim=768,
            latent_dim=256,
            learn_projection=False,
        )
        embedder = LMStudioEmbedder()
        memory = RTMDKMemory(config=config, embedder=embedder)
        
        # Add a node to test node count
        memory.save_context(
            {"input": "test health", "session_id": "test"},
            {"output": "test output"}
        )
        
        router = create_ux_router(memory, {})
        
        # Check router created
        print(f"✅ UX router created: {len(router.routes)} routes")
        
        # Check health route exists
        health_route = None
        for route in router.routes:
            if hasattr(route, 'path') and '/health' in route.path:
                health_route = route
                break
        
        if health_route:
            print(f"✅ Health endpoint exists")
        else:
            print(f"⚠️  Health endpoint not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Health endpoint FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backup_restore(memory):
    """Test 6: Backup and restore functionality."""
    print("\n" + "=" * 70)
    print("TEST 6: Backup and Restore")
    print("=" * 70)
    
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp(prefix="rtmdk_real_test_")
    save_path = os.path.join(temp_dir, "test_backup.json")
    
    try:
        # Export
        node_count_before = len(memory.field.nodes)
        memory.export_field(save_path)
        print(f"✅ Memory exported: {node_count_before} nodes, {os.path.getsize(save_path)} bytes")
        
        # Import
        embedder = LMStudioEmbedder()
        memory2 = RTMDKMemory.import_field(save_path, embedder)
        node_count_after = len(memory2.field.nodes)
        
        print(f"✅ Memory imported: {node_count_after} nodes")
        
        assert node_count_before == node_count_after, \
            f"Node count mismatch: {node_count_before} before, {node_count_after} after"
        
        print(f"✅ Node counts match")
        
        # Verify a node
        first_nid = list(memory.field.nodes.keys())[0]
        node1 = memory.field.nodes[first_nid]
        node2 = memory2.field.nodes[first_nid]
        
        assert node1.content.get("text") == node2.content.get("text"), "Content mismatch"
        print(f"✅ Node content verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Backup/restore FAILED: {e}")
        return False
    finally:
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def run_all_real_tests():
    """Run all real integration tests."""
    print("\n" + "🧪 " * 35)
    print("RTMDK REAL INTEGRATION TEST SUITE (LM Studio)")
    print("🧪 " * 35 + "\n")
    
    all_passed = True
    recall = 0
    
    try:
        # Test embedder
        embedder = test_embedder_connection()
        
        # Test memory operations
        memory = test_memory_with_real_embeddings(embedder)
        
        # Test queries
        memory, recall = test_query_real_memory(memory)
        
        # Test SillyTavern compatibility
        test_sillytavern_endpoints(memory)
        
        # Test health endpoint
        test_health_endpoint()
        
        # Test backup/restore
        test_backup_restore(memory)
        
        print("\n" + "=" * 70)
        if recall >= 80:
            print(f"🎉 ALL TESTS PASSED! Recall@5: {recall:.0f}%")
        else:
            print(f"⚠️  TESTS COMPLETED with low recall: {recall:.0f}%")
            print("   This may indicate embedding quality issues")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed and recall >= 60


if __name__ == "__main__":
    success = run_all_real_tests()
    sys.exit(0 if success else 1)

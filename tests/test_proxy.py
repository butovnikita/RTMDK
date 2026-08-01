"""
test_proxy.py — Quick test for RTMDK SillyTavern Proxy.

Usage:
    1. Make sure RTMDK server is running (python rtmdk_server.py)
    2. Make sure Proxy is running (python rtmdk_st_proxy.py)
    3. Run this test: python tests/test_proxy.py
"""

import pytest
import requests

PROXY_URL = "http://127.0.0.1:5000"
RTMDK_URL = "http://127.0.0.1:8080"


@pytest.mark.slow
def test_proxy():
    print("=" * 60)
    print("RTMDK SillyTavern Proxy Test")
    print("=" * 60)

    # Test 1: Proxy status
    print("\n1. Testing proxy status...")
    try:
        resp = requests.get(f"{PROXY_URL}/status", timeout=5)
        if resp.ok:
            status = resp.json()
            print(f"   Proxy: {status.get('proxy', 'unknown')}")
            print(f"   RTMDK: {status.get('rtmdk', {}).get('status', 'unknown')}")
            print(f"   Memory nodes: {status.get('rtmdk', {}).get('memory_nodes', 0)}")
        else:
            print(f"   FAILED: HTTP {resp.status_code}")
    except Exception as e:
        print(f"   FAILED: {e}")
        print("   Make sure proxy is running: python rtmdk_st_proxy.py")
        return

    # Test 2: Send message through proxy
    print("\n2. Testing chat completion...")
    data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, my name is TestUser and I like coffee"}],
        "stream": False,
        "char_name": "TestCharacter",
    }

    try:
        resp = requests.post(f"{PROXY_URL}/v1/chat/completions", json=data, timeout=30)
        if resp.ok:
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   Response: {content[:100]}...")
        else:
            print(f"   FAILED: HTTP {resp.status_code}")
            print(f"   Error: {resp.text[:200]}")
    except Exception as e:
        print(f"   FAILED: {e}")

    # Test 3: Check if memory was saved
    print("\n3. Checking memory...")
    try:
        resp = requests.get(f"{RTMDK_URL}/health", timeout=5)
        if resp.ok:
            health = resp.json()
            nodes = health.get("memory_nodes", 0)
            print(f"   Memory nodes: {nodes}")
            if nodes > 0:
                print("   [OK] Memory is being saved!")
            else:
                print("   [WARN] No nodes in memory")
    except Exception as e:
        print(f"   FAILED: {e}")
        print("   Make sure RTMDK server is running: python rtmdk_server.py")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_proxy()

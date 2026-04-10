"""
rtmdk_sillytavern_launcher.py — Unified Launcher for RTMDK + SillyTavern Proxy.

Starts both RTMDK Server and SillyTavern Proxy with a single command.
Manages both processes and provides status updates.

Usage:
    python rtmdk_sillytavern_launcher.py [--rtmdk-port 8080] [--proxy-port 5000]
"""

import os
import sys
import time
import signal
import subprocess
import requests
import threading
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class LauncherConfig:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.rtmdk_server_script = self.script_dir / "rtmdk_server.py"
        self.rtmdk_proxy_script = self.script_dir / "rtmdk_st_proxy.py"
        self.rtmdk_port = 8080
        self.proxy_port = 5000
        self.lm_studio_url = "http://127.0.0.1:12345/v1"

# Global config
config = LauncherConfig()

# Process handles
rtmdk_process = None
proxy_process = None
shutdown_event = threading.Event()

# ============================================================================
# PROCESS MANAGEMENT
# ============================================================================

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_service(url: str, timeout: int = 30) -> bool:
    """Wait for a service to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.ok:
                return True
        except:
            pass
        time.sleep(1)
    return False

def start_rtmdk_server():
    """Start the RTMDK memory server."""
    global rtmdk_process
    
    if is_port_in_use(config.rtmdk_port):
        print(f"  ⚠️  RTMDK Server already running on port {config.rtmdk_port}")
        return True
    
    print(f"  Starting RTMDK Server on port {config.rtmdk_port}...")
    
    try:
        rtmdk_process = subprocess.Popen(
            [sys.executable, str(config.rtmdk_server_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Capture errors
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            text=True
        )
        
        # Wait for startup
        if wait_for_service(f"http://127.0.0.1:{config.rtmdk_port}/health"):
            print(f"  ✅ RTMDK Server started successfully")
            return True
        else:
            print(f"  ❌ RTMDK Server failed to start. Output:")
            # Print recent output for debugging
            try:
                output = rtmdk_process.communicate(timeout=2)[0]
                for line in output.split('\n')[-5:]:
                    if line.strip(): print(f"    {line}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"  ❌ Error starting RTMDK Server: {e}")
        return False

def start_proxy():
    """Start the SillyTavern proxy."""
    global proxy_process
    
    if is_port_in_use(config.proxy_port):
        print(f"  ⚠️  Proxy already running on port {config.proxy_port}")
        return True
    
    print(f"  Starting SillyTavern Proxy on port {config.proxy_port}...")
    
    try:
        proxy_process = subprocess.Popen(
            [sys.executable, str(config.rtmdk_proxy_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Capture errors
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            text=True
        )
        
        # Wait for startup
        if wait_for_service(f"http://127.0.0.1:{config.proxy_port}/status"):
            print(f"  ✅ SillyTavern Proxy started successfully")
            return True
        else:
            print(f"  ❌ SillyTavern Proxy failed to start. Output:")
            # Print recent output for debugging
            try:
                output = proxy_process.communicate(timeout=2)[0]
                for line in output.split('\n')[-5:]:
                    if line.strip(): print(f"    {line}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"  ❌ Error starting Proxy: {e}")
        return False

def shutdown_services():
    """Gracefully shutdown all services."""
    print("\n\n🛑 Shutting down services...")
    
    global rtmdk_process, proxy_process
    
    if proxy_process:
        print("  Stopping Proxy...")
        proxy_process.terminate()
        try:
            proxy_process.wait(timeout=5)
        except:
            proxy_process.kill()
        print("  ✅ Proxy stopped")
    
    if rtmdk_process:
        print("  Stopping RTMDK Server...")
        rtmdk_process.terminate()
        try:
            rtmdk_process.wait(timeout=5)
        except:
            rtmdk_process.kill()
        print("  ✅ RTMDK Server stopped")
    
    print("👋 All services stopped. Goodbye!")

# ============================================================================
# STATUS MONITOR
# ============================================================================

def print_status():
    """Print current status of all services."""
    print("\n" + "=" * 60)
    print("  RTMDK SillyTavern Launcher — Status")
    print("=" * 60)
    
    # Check RTMDK Server
    try:
        resp = requests.get(f"http://127.0.0.1:{config.rtmdk_port}/health", timeout=3)
        if resp.ok:
            health = resp.json()
            print(f"  RTMDK Server:  ✅ Running (port {config.rtmdk_port})")
            print(f"    Memory nodes: {health.get('memory_nodes', 0)}")
            print(f"    LM Studio:    {'✅ Connected' if health.get('lm_studio') else '❌ Disconnected'}")
        else:
            print(f"  RTMDK Server:  ❌ Error (HTTP {resp.status_code})")
    except:
        print(f"  RTMDK Server:  ❌ Not running")
    
    # Check Proxy
    try:
        resp = requests.get(f"http://127.0.0.1:{config.proxy_port}/status", timeout=3)
        if resp.ok:
            status = resp.json()
            print(f"  Proxy:         ✅ Running (port {config.proxy_port})")
            rtmdk_status = status.get('rtmdk', {}).get('status', 'unknown')
            print(f"    RTMDK link:   {'✅ Connected' if rtmdk_status == 'ok' else '❌ Disconnected'}")
        else:
            print(f"  Proxy:         ❌ Error (HTTP {resp.status_code})")
    except:
        print(f"  Proxy:         ❌ Not running")
    
    # LM Studio
    try:
        resp = requests.get(f"{config.lm_studio_url}/models", timeout=3)
        if resp.ok:
            models = resp.json().get('data', [])
            print(f"  LM Studio:     ✅ Connected ({len(models)} models)")
        else:
            print(f"  LM Studio:     ❌ Error")
    except:
        print(f"  LM Studio:     ❌ Not running (start LM Studio first)")
    
    print("=" * 60)

def print_connection_info():
    """Print SillyTavern connection settings."""
    print("\n📱 SillyTavern Connection Settings:")
    print(f"  API Type:    OpenAI")
    print(f"  Base URL:    http://127.0.0.1:{config.proxy_port}/v1")
    print(f"  API Key:     (any value, not checked)")
    print(f"  Model:       (any model name)")

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RTMDK SillyTavern Launcher")
    parser.add_argument("--rtmdk-port", type=int, default=8080, help="RTMDK Server port")
    parser.add_argument("--proxy-port", type=int, default=5000, help="Proxy port")
    parser.add_argument("--status", action="store_true", help="Show status only")
    args = parser.parse_args()
    
    config.rtmdk_port = args.rtmdk_port
    config.proxy_port = args.proxy_port
    
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, lambda s, f: shutdown_services())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_services())
    
    print("\n" + "🚀 " * 30)
    print("  RTMDK SillyTavern Launcher v1.0.0")
    print("🚀 " * 30)
    
    # Status mode
    if args.status:
        print_status()
        return
    
    # Start services
    print("\n📦 Starting services...")
    
    # Check LM Studio first
    try:
        resp = requests.get(f"{config.lm_studio_url}/models", timeout=3)
        if not resp.ok:
            print("  ⚠️  LM Studio not responding. Make sure it's running!")
    except:
        print("  ⚠️  LM Studio not running. Start LM Studio before using SillyTavern!")
    
    # Start RTMDK Server
    if not start_rtmdk_server():
        print("❌ Failed to start RTMDK Server. Exiting.")
        return
    
    # Start Proxy
    if not start_proxy():
        print("❌ Failed to start Proxy. Exiting.")
        shutdown_services()
        return
    
    # Print status
    print_status()
    print_connection_info()
    
    print("\n💡 Press Ctrl+C to stop all services")
    print("=" * 60)
    
    # Keep running and show periodic status
    try:
        status_interval = 60  # seconds
        while not shutdown_event.is_set():
            time.sleep(status_interval)
            print_status()
    except KeyboardInterrupt:
        pass
    
    shutdown_services()

if __name__ == "__main__":
    main()

"""
rtmdk_monitor.py — Real-time RTMDK Server Monitor.

Monitors server state every 30 seconds for 10 minutes:
- Node count
- Memory file size
- LM Studio status
- Health status

Usage:
    python rtmdk_monitor.py [--interval 30] [--duration 600] [--server http://127.0.0.1:8080]
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from pathlib import Path

def monitor_server(server_url="http://127.0.0.1:8080", interval=30, duration=600):
    """Monitor RTMDK server state."""
    
    print("=" * 70)
    print("RTMDK SERVER MONITOR")
    print("=" * 70)
    print(f"Server: {server_url}")
    print(f"Interval: {interval}s")
    print(f"Duration: {duration}s ({duration//60} minutes)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop monitoring early.")
    print()
    
    log = []
    start_time = time.time()
    iteration = 0
    
    try:
        while time.time() - start_time < duration:
            iteration += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            elapsed = int(time.time() - start_time)
            
            print(f"[{elapsed:4d}s] Checking server... ", end="", flush=True)
            
            entry = {
                "timestamp": timestamp,
                "elapsed_seconds": elapsed,
                "iteration": iteration
            }
            
            try:
                # Check health
                resp = requests.get(f"{server_url}/health", timeout=5)
                if resp.ok:
                    health = resp.json()
                    entry["status"] = "OK"
                    entry["memory_nodes"] = health.get("memory_nodes", 0)
                    entry["lm_studio"] = health.get("lm_studio", False)
                    entry["version"] = health.get("version", "?")
                    print(f"OK | Nodes: {entry['memory_nodes']} | LM Studio: {entry['lm_studio']}")
                else:
                    entry["status"] = f"HTTP {resp.status_code}"
                    print(f"HTTP {resp.status_code}")
                    
            except requests.exceptions.ConnectionError:
                entry["status"] = "Connection refused"
                print("CONNECTION REFUSED")
            except requests.exceptions.Timeout:
                entry["status"] = "Timeout"
                print("TIMEOUT")
            except Exception as e:
                entry["status"] = f"Error: {e}"
                print(f"ERROR: {e}")
            
            # Check memory file
            memory_file = os.path.expanduser("~/.rtmdk/memory.json")
            if os.path.exists(memory_file):
                entry["memory_file_size_mb"] = round(os.path.getsize(memory_file) / 1024 / 1024, 2)
                entry["memory_file_mod"] = datetime.fromtimestamp(os.path.getmtime(memory_file)).strftime('%H:%M:%S')
                print(f"           Memory file: {entry['memory_file_size_mb']}MB (modified: {entry['memory_file_mod']})")
            else:
                entry["memory_file_size_mb"] = 0
                entry["memory_file_mod"] = None
                print("           Memory file: NOT FOUND")
            
            log.append(entry)
            
            # Save log
            log_path = "rtmdk_monitor_log.json"
            with open(log_path, 'w') as f:
                json.dump({
                    "server": server_url,
                    "start_time": datetime.now().isoformat(),
                    "interval": interval,
                    "duration": duration,
                    "entries": log
                }, f, indent=2, default=str)
            
            # Wait for next check
            if time.time() - start_time < duration:
                time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user (Ctrl+C)")
    
    # Summary
    print("\n" + "=" * 70)
    print("MONITORING SUMMARY")
    print("=" * 70)
    
    if log:
        print(f"Total checks: {len(log)}")
        print(f"Time span: {log[-1]['elapsed_seconds']}s")
        
        # Node count changes
        node_counts = [e.get("memory_nodes", 0) for e in log if e.get("memory_nodes") is not None]
        if node_counts:
            print(f"Node count: started at {node_counts[0]}, ended at {node_counts[-1]}")
            if node_counts[0] == node_counts[-1]:
                print("⚠️  Node count did NOT change during monitoring!")
            else:
                print(f"✅ Node count changed by {node_counts[-1] - node_counts[0]} nodes")
        
        # LM Studio status
        lm_statuses = [e.get("lm_studio") for e in log if "lm_studio" in e]
        if lm_statuses:
            lm_ok = sum(1 for s in lm_statuses if s)
            print(f"LM Studio: available {lm_ok}/{len(lm_statuses)} times")
        
        # Connection status
        ok_count = sum(1 for e in log if e.get("status") == "OK")
        print(f"Successful checks: {ok_count}/{len(log)}")
        
        # Memory file changes
        file_sizes = [e.get("memory_file_size_mb", 0) for e in log]
        if file_sizes:
            print(f"Memory file: started at {file_sizes[0]}MB, ended at {file_sizes[-1]}MB")
    
    print(f"\nLog saved to: rtmdk_monitor_log.json")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--duration", type=int, default=600, help="Total duration in seconds")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="Server URL")
    args = parser.parse_args()
    
    monitor_server(args.server, args.interval, args.duration)

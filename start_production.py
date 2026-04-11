#!/usr/bin/env python3
"""RTMDK Production Launcher — No SillyTavern modules.

Starts the production-ready RTMDK server with:
- OpenAI-compatible API
- RTMDK Memory
- Dashboard UI
- Security (API Key auth)
- Auto-save

Usage:
    python start_production.py
    python start_production.py --port 9090
"""

import os
import sys
import argparse

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk.server.app import main, SERVER_PORT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTMDK Production Server")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--api-key", type=str, default=None, help="API key for auth")
    parser.add_argument("--no-auth", action="store_true", help="Disable API key auth")
    args = parser.parse_args()

    if args.port != SERVER_PORT:
        os.environ["RTMDK_PORT"] = str(args.port)
    if args.api_key:
        os.environ["RTMDK_API_KEY"] = args.api_key
    if args.no_auth:
        os.environ["RTMDK_ENABLE_API_AUTH"] = "false"

    main()

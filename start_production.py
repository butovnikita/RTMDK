#!/usr/bin/env python3
"""RTMDK Production Launcher — No SillyTavern modules."""

import os
import sys
import argparse

# Set env vars BEFORE importing the app
if "--no-auth" in sys.argv:
    os.environ["RTMDK_ENABLE_API_AUTH"] = "false"
    sys.argv.remove("--no-auth")

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk.server.app import main, SERVER_PORT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTMDK Production Server")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--api-key", type=str, default=None, help="API key for auth")
    args = parser.parse_args()

    if args.port != SERVER_PORT:
        os.environ["RTMDK_PORT"] = str(args.port)
    if args.api_key:
        os.environ["RTMDK_API_KEY"] = args.api_key

    main()

"""RTMDK Package — Entry Point.

Usage:
    python -m rtmdk              # Run server
    python -m rtmdk status       # CLI commands
    python -m rtmdk bootstrap corpus.json
"""

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()  # pick up .env (RTMDK_PORT, RTMDK_API_KEY, ...) if present
except ImportError:
    pass  # python-dotenv is optional; real env vars still work

if len(sys.argv) > 1 and sys.argv[1] in {
    "status",
    "query",
    "stats",
    "export",
    "recommend",
    "presets",
    "bootstrap",
    "bootstrap-fasttext",
    "pipeline-diagnose",
}:
    from rtmdk.cli import main as cli_main

    cli_main()
else:
    from rtmdk.server.app import main as server_main

    server_main()

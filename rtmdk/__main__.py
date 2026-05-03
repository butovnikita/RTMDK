"""RTMDK Package — Entry Point.

Usage:
    python -m rtmdk              # Run server
    python -m rtmdk status       # CLI commands
    python -m rtmdk bootstrap corpus.json
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] in {
    "status", "query", "stats", "export", "recommend", "presets", "bootstrap"
}:
    from rtmdk.cli import main as cli_main
    cli_main()
else:
    from rtmdk.server.app import main as server_main
    server_main()

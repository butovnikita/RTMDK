"""
RTMDK Server — Main Entry Point.

Usage:
    python main.py
    python -m rtmdk.main
"""

import sys
import os

# Add parent directory to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk_server import main

if __name__ == "__main__":
    main()

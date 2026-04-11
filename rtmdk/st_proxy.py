"""
RTMDK SillyTavern Proxy — Entry Point.

Usage:
    python st_proxy.py
    python -m rtmdk.st_proxy
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk_st_proxy import main

if __name__ == "__main__":
    main()

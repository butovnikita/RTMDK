"""rtmdk/production/json_logger.py — Structured JSON logging formatter.

Usage:
    from rtmdk.production.json_logger import setup_json_logging
    setup_json_logging()
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Emit log records as JSON lines."""

    def __init__(self, static_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
            "source": {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            },
        }
        payload.update(self.static_fields)
        # Add exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_json_logging(level: int = logging.INFO, service: str = "rtmdk") -> None:
    """Configure root logger to output JSON to stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter(static_fields={"service": service}))
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)

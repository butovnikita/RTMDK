"""Structured JSON logging configuration for RTMDK.

Usage:
    from rtmdk.utils.json_logger import setup_json_logging
    setup_json_logging(level=logging.INFO)

Produces lines like:
    {"timestamp": "2026-05-07T10:30:00", "level": "INFO", "logger": "rtmdk.memory", "message": "...", "context": {...}}
"""

from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Emit log records as JSON objects."""

    def __init__(self, static_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self.static_fields,
        }
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "context"):
            obj["context"] = record.context
        return json.dumps(obj, ensure_ascii=False, default=str)


def setup_json_logging(
    level: int = logging.INFO,
    static_fields: Optional[Dict[str, Any]] = None,
    stream: Any = sys.stderr,
) -> None:
    """Configure root logger to emit JSON lines."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter(static_fields=static_fields))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

"""RTMDK Server Module — Production OpenAI-compatible API server (No SillyTavern)."""

from rtmdk.server.app import app as server_app, main

__all__ = ["server_app", "main"]

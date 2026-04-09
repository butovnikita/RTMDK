"""
rtmdk/production/circuit_breaker.py — Graceful Degradation.

Prevents cascading failures when LLM API is unavailable.
"""

import time
from typing import Dict, Any, Callable, Optional


class CircuitBreaker:
    """Circuit breaker for LLM API calls.
    
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    
    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        def call_llm():
            with cb:
                return llm_api_call()
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
    
    def __enter__(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
        else:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            self.success_count += 1
        return False
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "threshold": self.failure_threshold,
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass

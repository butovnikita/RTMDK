"""
rtmdk/production/rate_limiter.py — Request Rate Limiting.

Protects RTMDK from abuse with configurable rate limits.
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict


class RateLimiter:
    """Rate limiter for RTMDK queries.
    
    Usage:
        limiter = RateLimiter(max_queries_per_minute=60, max_queries_per_hour=1000)
        
        if limiter.allow_request("user123"):
            # Process request
            pass
        else:
            # Rate limited
            pass
    """
    
    def __init__(
        self,
        max_per_minute: int = 60,
        max_per_hour: int = 1000,
        max_per_day: int = 10000,
    ):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self._requests: Dict[str, List[float]] = defaultdict(list)
    
    def allow_request(self, client_id: str) -> bool:
        """Check if a request is allowed for this client."""
        now = time.time()
        requests = self._requests[client_id]
        
        # Clean old entries
        cutoff_hour = now - 3600
        cutoff_day = now - 86400
        requests = [t for t in requests if t > cutoff_day]
        self._requests[client_id] = requests
        
        # Check limits
        minute_count = sum(1 for t in requests if t > now - 60)
        hour_count = sum(1 for t in requests if t > cutoff_hour)
        day_count = len(requests)
        
        if minute_count >= self.max_per_minute:
            return False
        if hour_count >= self.max_per_hour:
            return False
        if day_count >= self.max_per_day:
            return False
        
        requests.append(now)
        return True
    
    def get_remaining(self, client_id: str) -> Dict[str, int]:
        """Get remaining requests for each window."""
        now = time.time()
        requests = self._requests.get(client_id, [])
        
        minute_count = sum(1 for t in requests if t > now - 60)
        hour_count = sum(1 for t in requests if t > now - 3600)
        day_count = len(requests)
        
        return {
            "per_minute": max(0, self.max_per_minute - minute_count),
            "per_hour": max(0, self.max_per_hour - hour_count),
            "per_day": max(0, self.max_per_day - day_count),
        }

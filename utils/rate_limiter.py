"""
Sliding Window Rate Limiter

Generic per-operation rate limiting using a sliding time window.
Prevents rapid-fire execution of any categorized operation.

Usage:
    limiter = RateLimiter({"api_call": 60, "email_send": 3})
    if limiter.check("api_call"):
        do_api_call()
    else:
        print("Rate limited — try again later")

Contributed by: Kendra
"""

import logging
import time
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default limits (per minute) — override via constructor
DEFAULT_LIMITS: Dict[str, int] = {
    "default": 100,
}

WINDOW_SECONDS = 60.0


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks timestamps per operation category. Prunes entries outside
    the window on each check. Thread-safe for single-threaded async;
    for multi-threaded use, add a lock around check().
    """

    def __init__(self, limits: Optional[Dict[str, int]] = None):
        """
        Args:
            limits: Dict mapping operation names to max calls per minute.
                    Unknown operations default to 100/min.
        """
        self._limits = dict(DEFAULT_LIMITS)
        if limits:
            self._limits.update(limits)
        self._buckets: Dict[str, list] = defaultdict(list)

    def check(self, operation: str) -> bool:
        """
        Check if an operation is allowed under its rate limit.

        Returns True if under the limit (and records the call),
        False if rate limited (call is NOT recorded).
        """
        now = time.time()
        bucket = self._buckets[operation]

        # Prune entries outside the window
        bucket[:] = [t for t in bucket if now - t < WINDOW_SECONDS]

        limit = self._limits.get(operation, self._limits.get("default", 100))
        if len(bucket) >= limit:
            logger.warning(
                f"Rate limited: '{operation}' ({len(bucket)}/{limit} per minute)"
            )
            return False

        bucket.append(now)
        return True

    def set_limit(self, operation: str, per_minute: int) -> None:
        """Override the limit for an operation at runtime."""
        self._limits[operation] = per_minute

    def reset(self, operation: Optional[str] = None) -> None:
        """Clear rate limit history for one or all operations."""
        if operation:
            self._buckets.pop(operation, None)
        else:
            self._buckets.clear()

    def remaining(self, operation: str) -> int:
        """Return how many calls remain in the current window."""
        now = time.time()
        bucket = self._buckets.get(operation, [])
        active = [t for t in bucket if now - t < WINDOW_SECONDS]
        limit = self._limits.get(operation, self._limits.get("default", 100))
        return max(0, limit - len(active))

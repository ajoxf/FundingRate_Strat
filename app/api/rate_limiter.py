"""
Rate Limiter for API Requests

Implements a token bucket algorithm to control request rates and prevent
API throttling. Supports configurable rates and burst allowances.

OKX API Rate Limits (as of 2024):
- Public endpoints: 10 requests/second
- Account endpoints: 5 requests/second
- Trade endpoints: 20 requests/second
- Market data: 20 requests/second
"""

import time
import threading
from collections import defaultdict
from typing import Optional, Dict


class RateLimiter:
    """
    Token bucket rate limiter implementation.

    The token bucket algorithm allows for controlled bursting while maintaining
    a long-term average rate. Tokens are added at a fixed rate, and each request
    consumes one token.

    Usage:
        limiter = RateLimiter(rate=10, burst=20)  # 10/sec, burst of 20

        # Blocking wait
        limiter.wait()  # Waits if necessary to respect rate limit
        make_request()

        # Non-blocking check
        if limiter.acquire(blocking=False):
            make_request()
        else:
            handle_rate_limit()
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst: Optional[int] = None,
        name: str = 'default'
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Requests per second allowed
            burst: Maximum burst size (defaults to rate)
            name: Name for logging/identification
        """
        self.rate = rate
        self.burst = burst if burst is not None else int(rate)
        self.name = name

        # Token bucket state
        self._tokens = float(self.burst)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

        # Statistics
        self._total_requests = 0
        self._total_waits = 0
        self._total_wait_time = 0.0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.burst,
            self._tokens + elapsed * self.rate
        )
        self._last_update = now

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        Acquire a token from the bucket.

        Args:
            blocking: If True, wait for a token. If False, return immediately.
            timeout: Maximum time to wait (seconds). None = wait indefinitely.

        Returns:
            True if token acquired, False if not (only when blocking=False
            or timeout exceeded)
        """
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        with self._lock:
            while True:
                self._refill()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._total_requests += 1
                    return True

                if not blocking:
                    return False

                if deadline is not None and time.monotonic() >= deadline:
                    return False

                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self.rate

                # Don't exceed deadline
                if deadline is not None:
                    wait_time = min(wait_time, deadline - time.monotonic())

                if wait_time > 0:
                    self._total_waits += 1
                    self._total_wait_time += wait_time

                    # Release lock while sleeping
                    self._lock.release()
                    try:
                        time.sleep(wait_time)
                    finally:
                        self._lock.acquire()

    def wait(self, timeout: float = None) -> bool:
        """
        Wait for a token (convenience method).

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if token acquired, False if timeout exceeded
        """
        return self.acquire(blocking=True, timeout=timeout)

    def try_acquire(self) -> bool:
        """
        Try to acquire a token without waiting.

        Returns:
            True if token was available, False otherwise
        """
        return self.acquire(blocking=False)

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            'name': self.name,
            'rate': self.rate,
            'burst': self.burst,
            'total_requests': self._total_requests,
            'total_waits': self._total_waits,
            'total_wait_time': round(self._total_wait_time, 3),
            'avg_wait_time': (
                round(self._total_wait_time / self._total_waits, 3)
                if self._total_waits > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        with self._lock:
            self._tokens = float(self.burst)
            self._last_update = time.monotonic()
            self._total_requests = 0
            self._total_waits = 0
            self._total_wait_time = 0.0


class RateLimiterGroup:
    """
    Manage multiple rate limiters for different endpoint types.

    Usage:
        limiters = RateLimiterGroup({
            'public': 10,
            'trade': 20,
            'account': 5,
        })

        limiters.wait('public')
        make_public_request()

        limiters.wait('trade')
        make_trade_request()
    """

    def __init__(self, rates: Dict[str, float], burst_multiplier: float = 2.0):
        """
        Initialize rate limiter group.

        Args:
            rates: Dict mapping endpoint type to requests per second
            burst_multiplier: Burst size as multiple of rate
        """
        self.limiters: Dict[str, RateLimiter] = {}

        for name, rate in rates.items():
            burst = int(rate * burst_multiplier)
            self.limiters[name] = RateLimiter(rate=rate, burst=burst, name=name)

    def wait(self, endpoint_type: str, timeout: float = None) -> bool:
        """
        Wait for rate limit on specific endpoint type.

        Args:
            endpoint_type: Type of endpoint ('public', 'trade', etc.)
            timeout: Maximum wait time

        Returns:
            True if acquired, False if timeout or unknown endpoint type
        """
        limiter = self.limiters.get(endpoint_type)
        if limiter:
            return limiter.wait(timeout=timeout)
        return True  # Allow if unknown endpoint type

    def try_acquire(self, endpoint_type: str) -> bool:
        """Try to acquire without waiting."""
        limiter = self.limiters.get(endpoint_type)
        if limiter:
            return limiter.try_acquire()
        return True

    @property
    def stats(self) -> Dict[str, Dict]:
        """Get statistics for all rate limiters."""
        return {
            name: limiter.stats
            for name, limiter in self.limiters.items()
        }

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self.limiters.values():
            limiter.reset()


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts based on API responses.

    Automatically backs off when rate limit errors are received
    and gradually increases rate when requests succeed.
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 20.0,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
        name: str = 'adaptive'
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            initial_rate: Starting requests per second
            min_rate: Minimum rate (never go below this)
            max_rate: Maximum rate (never exceed this)
            backoff_factor: Multiply rate by this on error
            recovery_factor: Multiply rate by this on success
            name: Name for identification
        """
        super().__init__(rate=initial_rate, name=name)

        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor

        self._consecutive_successes = 0
        self._consecutive_errors = 0

    def report_success(self) -> None:
        """Report a successful request. May increase rate."""
        with self._lock:
            self._consecutive_successes += 1
            self._consecutive_errors = 0

            # Gradually increase rate after consecutive successes
            if self._consecutive_successes >= 10:
                new_rate = min(self.max_rate, self.rate * self.recovery_factor)
                if new_rate != self.rate:
                    self.rate = new_rate
                    self.burst = int(new_rate * 2)
                self._consecutive_successes = 0

    def report_error(self, is_rate_limit: bool = True) -> None:
        """
        Report a failed request. Backs off if rate limit error.

        Args:
            is_rate_limit: True if error was due to rate limiting
        """
        with self._lock:
            self._consecutive_errors += 1
            self._consecutive_successes = 0

            if is_rate_limit:
                new_rate = max(self.min_rate, self.rate * self.backoff_factor)
                if new_rate != self.rate:
                    self.rate = new_rate
                    self.burst = int(new_rate * 2)

    def reset(self) -> None:
        """Reset to initial state."""
        super().reset()
        with self._lock:
            self.rate = self.initial_rate
            self.burst = int(self.initial_rate * 2)
            self._consecutive_successes = 0
            self._consecutive_errors = 0


# Default rate limiters for OKX API
def create_okx_rate_limiters() -> RateLimiterGroup:
    """
    Create rate limiters configured for OKX API limits.

    Returns:
        RateLimiterGroup with limiters for each endpoint type
    """
    return RateLimiterGroup({
        'public': 10,     # Public endpoints
        'market': 20,     # Market data
        'account': 5,     # Account info
        'trade': 20,      # Trading endpoints
    })

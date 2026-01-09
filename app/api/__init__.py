"""
API Module - OKX API Client and Rate Limiting

This module provides:
- OKXClient: Full-featured OKX API client for funding rate and trading
- RateLimiter: Token bucket rate limiter for API throttling
- RateLimiterGroup: Manage multiple rate limiters
"""

from .okx_client import OKXClient, OKXClientError, create_okx_client
from .rate_limiter import (
    RateLimiter,
    RateLimiterGroup,
    AdaptiveRateLimiter,
    create_okx_rate_limiters
)

__all__ = [
    'OKXClient',
    'OKXClientError',
    'create_okx_client',
    'RateLimiter',
    'RateLimiterGroup',
    'AdaptiveRateLimiter',
    'create_okx_rate_limiters',
]

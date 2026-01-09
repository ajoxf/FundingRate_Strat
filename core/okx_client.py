"""
OKX API Client - Standalone version for core module
"""

import os
import json
import time
import hmac
import base64
import hashlib
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, List

import requests


class OKXClientError(Exception):
    """Exception raised for OKX API errors."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"OKX API Error [{code}]: {message}")


class SimpleRateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()

    def wait(self):
        """Wait until a token is available."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class OKXClient:
    """OKX API Client for funding rate and trading operations."""

    BASE_URL = 'https://www.okx.com'

    def __init__(
        self,
        api_key: str = '',
        api_secret: str = '',
        passphrase: str = '',
        demo_mode: bool = False,
        timeout: int = 10,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.demo_mode = demo_mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = self.BASE_URL
        self.session = requests.Session()
        self.rate_limiter = SimpleRateLimiter(rate=10, burst=20)

    @classmethod
    def from_env(cls, demo_mode: bool = False) -> 'OKXClient':
        """Create client from environment variables."""
        return cls(
            api_key=os.environ.get('OKX_API_KEY', ''),
            api_secret=os.environ.get('OKX_API_SECRET', ''),
            passphrase=os.environ.get('OKX_PASSPHRASE', ''),
            demo_mode=demo_mode
        )

    def _get_timestamp(self) -> str:
        """Get ISO 8601 timestamp for API requests."""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _sign(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate HMAC-SHA256 signature."""
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _get_headers(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        """Generate request headers."""
        timestamp = self._get_timestamp()
        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers.update({
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': self._sign(timestamp, method, path, body),
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase,
            })

        if self.demo_mode:
            headers['x-simulated-trading'] = '1'

        return headers

    def _request(self, method: str, path: str, params: Dict = None, data: Dict = None, retry: int = 0) -> Dict:
        """Make API request with rate limiting and retry logic."""
        self.rate_limiter.wait()

        url = self.base_url + path
        body = json.dumps(data) if data else ''
        headers = self._get_headers(method, path, body)

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            else:
                response = self.session.post(url, data=body, headers=headers, timeout=self.timeout)

            response.raise_for_status()
            result = response.json()

            if result.get('code') != '0':
                error_code = result.get('code', '-1')
                error_msg = result.get('msg', 'Unknown error')

                if error_code in ['50011', '50013'] and retry < self.max_retries:
                    time.sleep(2 ** retry)
                    return self._request(method, path, params, data, retry + 1)

                raise OKXClientError(error_code, error_msg)

            return result

        except requests.exceptions.RequestException as e:
            if retry < self.max_retries:
                time.sleep(2 ** retry)
                return self._request(method, path, params, data, retry + 1)
            raise

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """Get current funding rate for a perpetual swap."""
        path = '/api/v5/public/funding-rate'
        params = {'instId': symbol}

        try:
            result = self._request('GET', path, params=params)
            if result.get('data'):
                data = result['data'][0]
                return {
                    'symbol': data.get('instId'),
                    'fundingRate': data.get('fundingRate'),
                    'nextFundingRate': data.get('nextFundingRate'),
                    'fundingTime': data.get('fundingTime'),
                    'nextFundingTime': data.get('nextFundingTime'),
                }
        except OKXClientError as e:
            print(f"Error fetching funding rate: {e}")
        return None

    def get_funding_rate_history(self, symbol: str, limit: int = 100, before: str = None) -> List[Dict]:
        """Get historical funding rates."""
        path = '/api/v5/public/funding-rate-history'
        params = {'instId': symbol, 'limit': str(min(limit, 100))}
        if before:
            params['before'] = before

        try:
            result = self._request('GET', path, params=params)
            if result.get('data'):
                return [
                    {
                        'symbol': item.get('instId'),
                        'funding_rate': float(item.get('fundingRate', 0)),
                        'realized_rate': float(item.get('realizedRate')) if item.get('realizedRate') else None,
                        'funding_time': int(item.get('fundingTime', 0)),
                    }
                    for item in result['data']
                ]
        except OKXClientError as e:
            print(f"Error fetching funding rate history: {e}")
        return []

    def get_funding_rate_history_all(self, symbol: str, periods: int = 90) -> List[Dict]:
        """Fetch all historical funding rates with pagination."""
        all_rates = []
        before_ts = None
        remaining = periods

        while remaining > 0:
            limit = min(remaining, 100)
            rates = self.get_funding_rate_history(symbol, limit=limit, before=before_ts)

            if not rates:
                break

            all_rates.extend(rates)
            remaining -= len(rates)

            if rates:
                before_ts = str(rates[-1]['funding_time'])

            if len(rates) < limit:
                break

        return all_rates

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for current swap price."""
        path = '/api/v5/market/ticker'
        params = {'instId': symbol}

        try:
            result = self._request('GET', path, params=params)
            if result.get('data'):
                data = result['data'][0]
                return {
                    'symbol': data.get('instId'),
                    'last': data.get('last'),
                    'bidPx': data.get('bidPx'),
                    'askPx': data.get('askPx'),
                    'ts': data.get('ts'),
                }
        except OKXClientError as e:
            print(f"Error fetching ticker: {e}")
        return None

    def get_positions(self, symbol: str = None) -> List[Dict]:
        """Get current positions (requires authentication)."""
        if not self.api_key:
            return []

        path = '/api/v5/account/positions'
        params = {'instId': symbol} if symbol else {}

        try:
            result = self._request('GET', path, params=params)
            if result.get('data'):
                return [
                    {
                        'symbol': pos.get('instId'),
                        'position_side': pos.get('posSide'),
                        'position_size': float(pos.get('pos', 0)),
                        'avg_price': float(pos.get('avgPx', 0)) if pos.get('avgPx') else 0,
                    }
                    for pos in result['data']
                    if float(pos.get('pos', 0)) != 0
                ]
        except OKXClientError as e:
            print(f"Error fetching positions: {e}")
        return []

    def place_order(self, symbol: str, side: str, size: float, order_type: str = 'market', price: float = None) -> Dict:
        """Place an order."""
        if not self.api_key:
            return {'success': False, 'message': 'Authentication required'}

        path = '/api/v5/trade/order'
        data = {
            'instId': symbol,
            'tdMode': 'cross',
            'side': side,
            'ordType': order_type,
            'sz': str(size),
        }

        if order_type == 'limit' and price:
            data['px'] = str(price)

        try:
            result = self._request('POST', path, data=data)
            if result.get('data'):
                order_data = result['data'][0]
                return {
                    'order_id': order_data.get('ordId'),
                    'success': order_data.get('sCode') == '0',
                    'message': order_data.get('sMsg', ''),
                }
        except OKXClientError as e:
            return {'success': False, 'message': str(e)}
        return {'success': False, 'message': 'Unknown error'}

    def close_position(self, symbol: str) -> Dict:
        """Close all positions for a symbol."""
        if not self.api_key:
            return {'success': False, 'message': 'Authentication required'}

        path = '/api/v5/trade/close-position'
        data = {'instId': symbol, 'mgnMode': 'cross', 'posSide': 'net'}

        try:
            self._request('POST', path, data=data)
            return {'success': True, 'message': 'Position closed'}
        except OKXClientError as e:
            return {'success': False, 'message': str(e)}


# Global client instance
_okx_client: Optional[OKXClient] = None


def get_okx_client() -> Optional[OKXClient]:
    """Get or create global OKX client."""
    global _okx_client
    if _okx_client is None:
        from .database import get_settings
        settings = get_settings()
        if settings.get('api_key'):
            _okx_client = OKXClient(
                api_key=settings['api_key'],
                api_secret=settings['api_secret'],
                passphrase=settings['api_passphrase'],
            )
        else:
            _okx_client = OKXClient()
    return _okx_client


def create_okx_client(api_key: str = '', api_secret: str = '', passphrase: str = '') -> OKXClient:
    """Create a new OKX client."""
    return OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)

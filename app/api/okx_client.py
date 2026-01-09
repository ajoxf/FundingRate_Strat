"""
OKX API Client for Funding Rate Trading System

This module provides a comprehensive client for interacting with the OKX API,
specifically optimized for funding rate data and perpetual swap trading.

Endpoints used:
- GET /api/v5/public/funding-rate - Current funding rate
- GET /api/v5/public/funding-rate-history - Historical funding rates
- GET /api/v5/market/ticker - Swap price data
- GET /api/v5/account/positions - Current positions
- POST /api/v5/trade/order - Place orders
- POST /api/v5/trade/close-position - Close positions
"""

import os
import json
import time
import hmac
import base64
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import requests

from .rate_limiter import RateLimiter


class OKXClientError(Exception):
    """Exception raised for OKX API errors."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"OKX API Error [{code}]: {message}")


class OKXClient:
    """
    OKX API Client for funding rate and trading operations.

    Supports both public (unauthenticated) and private (authenticated) endpoints.
    Includes rate limiting to prevent API throttling.

    Usage:
        # Public endpoints (no auth required)
        client = OKXClient()
        funding = client.get_funding_rate('BTC-USDT-SWAP')

        # Private endpoints (auth required)
        client = OKXClient(
            api_key='your-key',
            api_secret='your-secret',
            passphrase='your-passphrase'
        )
        positions = client.get_positions()
    """

    BASE_URL = 'https://www.okx.com'
    DEMO_URL = 'https://www.okx.com'  # OKX uses same URL, different header for demo

    # Rate limits per endpoint type (requests per second)
    RATE_LIMITS = {
        'public': 10,      # Public endpoints
        'account': 5,      # Account endpoints
        'trade': 20,       # Trade endpoints
        'market': 20,      # Market data endpoints
    }

    def __init__(
        self,
        api_key: str = '',
        api_secret: str = '',
        passphrase: str = '',
        demo_mode: bool = False,
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize OKX API client.

        Args:
            api_key: OKX API key (required for private endpoints)
            api_secret: OKX API secret (required for private endpoints)
            passphrase: OKX API passphrase (required for private endpoints)
            demo_mode: Use demo trading environment
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.demo_mode = demo_mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = self.BASE_URL

        # Initialize session for connection pooling
        self.session = requests.Session()

        # Initialize rate limiters for different endpoint types
        self.rate_limiters = {
            endpoint_type: RateLimiter(rate, burst=rate * 2)
            for endpoint_type, rate in self.RATE_LIMITS.items()
        }

    @classmethod
    def from_env(cls, demo_mode: bool = False) -> 'OKXClient':
        """
        Create client from environment variables.

        Expected env vars:
            OKX_API_KEY
            OKX_API_SECRET
            OKX_PASSPHRASE
        """
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
        """
        Generate HMAC-SHA256 signature for authenticated requests.

        Args:
            timestamp: ISO 8601 timestamp
            method: HTTP method (GET/POST)
            path: API endpoint path
            body: Request body for POST requests

        Returns:
            Base64-encoded signature
        """
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _get_headers(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        """
        Generate request headers.

        For authenticated requests, includes:
            - OK-ACCESS-KEY
            - OK-ACCESS-SIGN
            - OK-ACCESS-TIMESTAMP
            - OK-ACCESS-PASSPHRASE
            - x-simulated-trading (for demo mode)
        """
        timestamp = self._get_timestamp()
        headers = {
            'Content-Type': 'application/json',
        }

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

    def _get_rate_limiter(self, path: str) -> RateLimiter:
        """Get appropriate rate limiter based on endpoint path."""
        if '/public/' in path or '/market/' in path:
            return self.rate_limiters['public']
        elif '/account/' in path:
            return self.rate_limiters['account']
        elif '/trade/' in path:
            return self.rate_limiters['trade']
        return self.rate_limiters['market']

    def _request(
        self,
        method: str,
        path: str,
        params: Dict = None,
        data: Dict = None,
        retry: int = 0
    ) -> Dict:
        """
        Make API request with rate limiting and retry logic.

        Args:
            method: HTTP method
            path: API endpoint path
            params: Query parameters for GET requests
            data: JSON body for POST requests
            retry: Current retry attempt

        Returns:
            API response data

        Raises:
            OKXClientError: On API error
            requests.exceptions.RequestException: On network error
        """
        # Apply rate limiting
        rate_limiter = self._get_rate_limiter(path)
        rate_limiter.wait()

        url = self.base_url + path
        body = json.dumps(data) if data else ''
        headers = self._get_headers(method, path, body)

        try:
            if method.upper() == 'GET':
                response = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
            else:
                response = self.session.post(
                    url, data=body, headers=headers, timeout=self.timeout
                )

            response.raise_for_status()
            result = response.json()

            # Check for API-level errors
            if result.get('code') != '0':
                error_code = result.get('code', '-1')
                error_msg = result.get('msg', 'Unknown error')

                # Retry on rate limit errors
                if error_code in ['50011', '50013'] and retry < self.max_retries:
                    time.sleep(2 ** retry)  # Exponential backoff
                    return self._request(method, path, params, data, retry + 1)

                raise OKXClientError(error_code, error_msg)

            return result

        except requests.exceptions.RequestException as e:
            if retry < self.max_retries:
                time.sleep(2 ** retry)
                return self._request(method, path, params, data, retry + 1)
            raise

    # =========================================================================
    # Public Endpoints - No Authentication Required
    # =========================================================================

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Get current funding rate for a perpetual swap.

        GET /api/v5/public/funding-rate

        Args:
            symbol: Instrument ID (e.g., 'BTC-USDT-SWAP')

        Returns:
            Dict with:
                - symbol: Instrument ID
                - funding_rate: Current funding rate (decimal)
                - next_funding_rate: Predicted next funding rate
                - funding_time: Current funding time (ms timestamp)
                - next_funding_time: Next funding time (ms timestamp)
        """
        path = '/api/v5/public/funding-rate'
        params = {'instId': symbol}

        try:
            result = self._request('GET', path, params=params)

            if result.get('data'):
                data = result['data'][0]
                return {
                    'symbol': data.get('instId'),
                    'funding_rate': float(data.get('fundingRate', 0)),
                    'next_funding_rate': (
                        float(data.get('nextFundingRate'))
                        if data.get('nextFundingRate') else None
                    ),
                    'funding_time': int(data.get('fundingTime', 0)),
                    'next_funding_time': int(data.get('nextFundingTime', 0)),
                }
        except OKXClientError as e:
            print(f"Error fetching funding rate: {e}")

        return None

    def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100,
        before: str = None,
        after: str = None
    ) -> List[Dict]:
        """
        Get historical funding rates.

        GET /api/v5/public/funding-rate-history

        OKX returns max 100 records per call. Use pagination params
        for more data.

        Args:
            symbol: Instrument ID (e.g., 'BTC-USDT-SWAP')
            limit: Number of records (max 100)
            before: Pagination - return records before this timestamp
            after: Pagination - return records after this timestamp

        Returns:
            List of dicts with:
                - symbol: Instrument ID
                - funding_rate: Funding rate at that time
                - realized_rate: Actual settled rate
                - funding_time: Funding time (ms timestamp)
        """
        path = '/api/v5/public/funding-rate-history'
        params = {'instId': symbol, 'limit': str(min(limit, 100))}

        if before:
            params['before'] = before
        if after:
            params['after'] = after

        try:
            result = self._request('GET', path, params=params)

            if result.get('data'):
                return [
                    {
                        'symbol': item.get('instId'),
                        'funding_rate': float(item.get('fundingRate', 0)),
                        'realized_rate': (
                            float(item.get('realizedRate'))
                            if item.get('realizedRate') else None
                        ),
                        'funding_time': int(item.get('fundingTime', 0)),
                    }
                    for item in result['data']
                ]
        except OKXClientError as e:
            print(f"Error fetching funding rate history: {e}")

        return []

    def get_funding_rate_history_all(
        self,
        symbol: str,
        periods: int = 90
    ) -> List[Dict]:
        """
        Fetch all historical funding rates up to specified number of periods.

        Handles pagination automatically since OKX limits to 100 per call.

        Args:
            symbol: Instrument ID
            periods: Number of funding periods to fetch (each = 8 hours)

        Returns:
            List of funding rate records, newest first
        """
        all_rates = []
        before_ts = None
        remaining = periods

        while remaining > 0:
            limit = min(remaining, 100)
            rates = self.get_funding_rate_history(
                symbol,
                limit=limit,
                before=before_ts
            )

            if not rates:
                break

            all_rates.extend(rates)
            remaining -= len(rates)

            # Get oldest timestamp for next pagination
            if rates:
                before_ts = str(rates[-1]['funding_time'])

            # Stop if we got fewer than requested (no more data)
            if len(rates) < limit:
                break

        return all_rates

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get ticker data for current swap price.

        GET /api/v5/market/ticker

        Args:
            symbol: Instrument ID (e.g., 'BTC-USDT-SWAP')

        Returns:
            Dict with:
                - symbol: Instrument ID
                - last_price: Last traded price
                - bid_price: Best bid price
                - ask_price: Best ask price
                - high_24h: 24h high
                - low_24h: 24h low
                - volume_24h: 24h volume
                - timestamp: Data timestamp (ms)
        """
        path = '/api/v5/market/ticker'
        params = {'instId': symbol}

        try:
            result = self._request('GET', path, params=params)

            if result.get('data'):
                data = result['data'][0]
                return {
                    'symbol': data.get('instId'),
                    'last_price': float(data.get('last', 0)),
                    'bid_price': float(data.get('bidPx', 0)),
                    'ask_price': float(data.get('askPx', 0)),
                    'high_24h': float(data.get('high24h', 0)),
                    'low_24h': float(data.get('low24h', 0)),
                    'volume_24h': float(data.get('vol24h', 0)),
                    'timestamp': int(data.get('ts', 0)),
                }
        except OKXClientError as e:
            print(f"Error fetching ticker: {e}")

        return None

    def get_instruments(self, inst_type: str = 'SWAP') -> List[Dict]:
        """
        Get available instruments.

        GET /api/v5/public/instruments

        Args:
            inst_type: Instrument type ('SWAP', 'FUTURES', 'SPOT')

        Returns:
            List of instrument details
        """
        path = '/api/v5/public/instruments'
        params = {'instType': inst_type}

        try:
            result = self._request('GET', path, params=params)

            if result.get('data'):
                return [
                    {
                        'symbol': item.get('instId'),
                        'base_ccy': item.get('baseCcy'),
                        'quote_ccy': item.get('quoteCcy'),
                        'settle_ccy': item.get('settleCcy'),
                        'contract_val': float(item.get('ctVal', 0)),
                        'tick_size': float(item.get('tickSz', 0)),
                        'lot_size': float(item.get('lotSz', 0)),
                        'min_size': float(item.get('minSz', 0)),
                    }
                    for item in result['data']
                ]
        except OKXClientError as e:
            print(f"Error fetching instruments: {e}")

        return []

    # =========================================================================
    # Private Endpoints - Authentication Required
    # =========================================================================

    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Get current positions.

        GET /api/v5/account/positions

        Requires authentication.

        Args:
            symbol: Optional filter by instrument ID

        Returns:
            List of position dicts with:
                - symbol: Instrument ID
                - position_side: 'long', 'short', or 'net'
                - position_size: Position quantity
                - avg_price: Average entry price
                - unrealized_pnl: Unrealized P&L
                - leverage: Position leverage
                - liquidation_price: Liquidation price
                - margin: Position margin
        """
        if not self.api_key:
            print("Warning: get_positions requires authentication")
            return []

        path = '/api/v5/account/positions'
        params = {}
        if symbol:
            params['instId'] = symbol

        try:
            result = self._request('GET', path, params=params)

            if result.get('data'):
                return [
                    {
                        'symbol': pos.get('instId'),
                        'position_side': pos.get('posSide'),
                        'position_size': float(pos.get('pos', 0)),
                        'avg_price': float(pos.get('avgPx', 0)) if pos.get('avgPx') else 0,
                        'unrealized_pnl': float(pos.get('upl', 0)) if pos.get('upl') else 0,
                        'leverage': float(pos.get('lever', 1)),
                        'liquidation_price': float(pos.get('liqPx', 0)) if pos.get('liqPx') else 0,
                        'margin': float(pos.get('margin', 0)) if pos.get('margin') else 0,
                    }
                    for pos in result['data']
                    if float(pos.get('pos', 0)) != 0  # Only return non-zero positions
                ]
        except OKXClientError as e:
            print(f"Error fetching positions: {e}")

        return []

    def get_account_balance(self, currency: str = 'USDT') -> Optional[Dict]:
        """
        Get account balance.

        GET /api/v5/account/balance

        Args:
            currency: Currency to check (e.g., 'USDT')

        Returns:
            Dict with balance info
        """
        if not self.api_key:
            print("Warning: get_account_balance requires authentication")
            return None

        path = '/api/v5/account/balance'
        params = {'ccy': currency}

        try:
            result = self._request('GET', path, params=params)

            if result.get('data'):
                data = result['data'][0]
                details = data.get('details', [])
                for detail in details:
                    if detail.get('ccy') == currency:
                        return {
                            'currency': detail.get('ccy'),
                            'available': float(detail.get('availBal', 0)),
                            'frozen': float(detail.get('frozenBal', 0)),
                            'total': float(detail.get('cashBal', 0)),
                            'equity': float(detail.get('eq', 0)),
                        }
        except OKXClientError as e:
            print(f"Error fetching account balance: {e}")

        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = 'market',
        price: float = None,
        reduce_only: bool = False,
        client_order_id: str = None
    ) -> Dict:
        """
        Place an order.

        POST /api/v5/trade/order

        Args:
            symbol: Instrument ID (e.g., 'BTC-USDT-SWAP')
            side: 'buy' or 'sell'
            size: Order size (contracts)
            order_type: 'market' or 'limit'
            price: Limit price (required for limit orders)
            reduce_only: Only reduce existing position
            client_order_id: Custom order ID

        Returns:
            Dict with:
                - order_id: OKX order ID
                - client_order_id: Custom order ID
                - success: True if order placed
                - message: Error message if failed
        """
        if not self.api_key:
            return {'success': False, 'message': 'Authentication required'}

        path = '/api/v5/trade/order'
        data = {
            'instId': symbol,
            'tdMode': 'cross',  # Cross margin mode
            'side': side,
            'ordType': order_type,
            'sz': str(size),
        }

        if order_type == 'limit' and price:
            data['px'] = str(price)

        if reduce_only:
            data['reduceOnly'] = True

        if client_order_id:
            data['clOrdId'] = client_order_id

        try:
            result = self._request('POST', path, data=data)

            if result.get('data'):
                order_data = result['data'][0]
                return {
                    'order_id': order_data.get('ordId'),
                    'client_order_id': order_data.get('clOrdId'),
                    'success': order_data.get('sCode') == '0',
                    'message': order_data.get('sMsg', ''),
                }
        except OKXClientError as e:
            return {'success': False, 'message': str(e)}

        return {'success': False, 'message': 'Unknown error'}

    def close_position(
        self,
        symbol: str,
        position_side: str = 'net',
        margin_mode: str = 'cross'
    ) -> Dict:
        """
        Close all positions for a symbol.

        POST /api/v5/trade/close-position

        Args:
            symbol: Instrument ID
            position_side: 'long', 'short', or 'net'
            margin_mode: 'cross' or 'isolated'

        Returns:
            Dict with success status and message
        """
        if not self.api_key:
            return {'success': False, 'message': 'Authentication required'}

        path = '/api/v5/trade/close-position'
        data = {
            'instId': symbol,
            'mgnMode': margin_mode,
            'posSide': position_side,
        }

        try:
            result = self._request('POST', path, data=data)
            return {'success': True, 'message': 'Position closed'}
        except OKXClientError as e:
            return {'success': False, 'message': str(e)}

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Cancel an order.

        POST /api/v5/trade/cancel-order

        Args:
            symbol: Instrument ID
            order_id: Order ID to cancel

        Returns:
            Dict with success status
        """
        if not self.api_key:
            return {'success': False, 'message': 'Authentication required'}

        path = '/api/v5/trade/cancel-order'
        data = {
            'instId': symbol,
            'ordId': order_id,
        }

        try:
            result = self._request('POST', path, data=data)
            return {'success': True, 'message': 'Order cancelled'}
        except OKXClientError as e:
            return {'success': False, 'message': str(e)}

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """
        Get order details.

        GET /api/v5/trade/order

        Args:
            symbol: Instrument ID
            order_id: Order ID

        Returns:
            Order details dict
        """
        if not self.api_key:
            return None

        path = '/api/v5/trade/order'
        params = {
            'instId': symbol,
            'ordId': order_id,
        }

        try:
            result = self._request('GET', path, params=params)

            if result.get('data'):
                order = result['data'][0]
                return {
                    'order_id': order.get('ordId'),
                    'symbol': order.get('instId'),
                    'side': order.get('side'),
                    'size': float(order.get('sz', 0)),
                    'filled_size': float(order.get('fillSz', 0)),
                    'price': float(order.get('px', 0)) if order.get('px') else None,
                    'avg_price': float(order.get('avgPx', 0)) if order.get('avgPx') else None,
                    'status': order.get('state'),
                    'order_type': order.get('ordType'),
                    'create_time': int(order.get('cTime', 0)),
                }
        except OKXClientError as e:
            print(f"Error fetching order: {e}")

        return None


# Convenience function to create client from environment
def create_okx_client(demo_mode: bool = False) -> OKXClient:
    """Create OKX client from environment variables."""
    return OKXClient.from_env(demo_mode=demo_mode)

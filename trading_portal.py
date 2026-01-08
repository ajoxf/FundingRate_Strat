"""
OKX Funding Rate Mean Reversion Trading System

This system implements a mean reversion strategy based on funding rates
for OKX perpetual swaps. When funding rates deviate significantly from
their rolling mean (measured by Z-score), positions are taken expecting
reversion to the mean.

Strategy Logic:
- Z-Score >= +2.0σ (funding unusually HIGH) → Go SHORT to collect funding
- Z-Score <= -2.0σ (funding unusually LOW) → Go LONG to collect funding
- Exit when Z-Score returns to ±0.2σ (mean reversion complete)
- Stop Loss at ±6.0σ
"""

import os
import json
import time
import hmac
import base64
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Any

import requests
from flask import Flask, render_template, jsonify, request, g
from apscheduler.schedulers.background import BackgroundScheduler

# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

DATABASE = 'funding_rate.db'
OKX_BASE_URL = 'https://www.okx.com'

# Default Settings
DEFAULT_SETTINGS = {
    'symbol': 'BTC-USDT-SWAP',
    'lookback_periods': 90,  # 90 x 8 hours = 720 hours = 30 days
    'entry_std_dev': 2.0,
    'exit_std_dev': 0.2,
    'stop_loss_std_dev': 6.0,
    'position_size': 1.0,  # Contract size
    'paper_mode': True,
    'api_key': '',
    'api_secret': '',
    'api_passphrase': '',
}

# =============================================================================
# Database Functions
# =============================================================================

def get_db():
    """Get database connection for current thread."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


def get_db_connection():
    """Get database connection outside Flask context."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


@app.teardown_appcontext
def close_connection(exception):
    """Close database connection when app context ends."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            lookback_periods INTEGER DEFAULT 90,
            entry_std_dev REAL DEFAULT 2.0,
            exit_std_dev REAL DEFAULT 0.2,
            stop_loss_std_dev REAL DEFAULT 6.0,
            position_size REAL DEFAULT 1.0,
            paper_mode INTEGER DEFAULT 1,
            api_key TEXT DEFAULT '',
            api_secret TEXT DEFAULT '',
            api_passphrase TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Funding rate history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS funding_rate_history (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            funding_rate REAL NOT NULL,
            funding_time TIMESTAMP NOT NULL,
            realized_rate REAL,
            swap_price REAL,
            z_score REAL,
            mean_rate REAL,
            std_dev REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, funding_time)
        )
    ''')

    # Create index on funding_time for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_funding_time
        ON funding_rate_history(symbol, funding_time DESC)
    ''')

    # Trades table for tracking positions and P&L
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,  -- 'long' or 'short'
            entry_price REAL NOT NULL,
            exit_price REAL,
            entry_funding_rate REAL,
            exit_funding_rate REAL,
            entry_z_score REAL,
            exit_z_score REAL,
            position_size REAL NOT NULL,
            price_pnl REAL DEFAULT 0,
            funding_pnl REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            funding_periods_held INTEGER DEFAULT 0,
            status TEXT DEFAULT 'open',  -- 'open', 'closed', 'stopped'
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP,
            exit_reason TEXT,  -- 'mean_reversion', 'stop_loss', 'manual'
            paper_trade INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Funding payments received/paid during open positions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS funding_payments (
            id INTEGER PRIMARY KEY,
            trade_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            funding_rate REAL NOT NULL,
            payment_amount REAL NOT NULL,
            funding_time TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        )
    ''')

    # Price snapshots for tracking swap prices
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            swap_price REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Initialize default settings if not exists
    cursor.execute('SELECT COUNT(*) FROM settings')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO settings (symbol, lookback_periods, entry_std_dev,
                                  exit_std_dev, stop_loss_std_dev, position_size, paper_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (DEFAULT_SETTINGS['symbol'], DEFAULT_SETTINGS['lookback_periods'],
              DEFAULT_SETTINGS['entry_std_dev'], DEFAULT_SETTINGS['exit_std_dev'],
              DEFAULT_SETTINGS['stop_loss_std_dev'], DEFAULT_SETTINGS['position_size'],
              1 if DEFAULT_SETTINGS['paper_mode'] else 0))

    conn.commit()
    conn.close()


def get_settings() -> Dict:
    """Get current settings from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM settings ORDER BY id DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'id': row['id'],
            'symbol': row['symbol'],
            'lookback_periods': row['lookback_periods'],
            'entry_std_dev': row['entry_std_dev'],
            'exit_std_dev': row['exit_std_dev'],
            'stop_loss_std_dev': row['stop_loss_std_dev'],
            'position_size': row['position_size'],
            'paper_mode': bool(row['paper_mode']),
            'api_key': row['api_key'],
            'api_secret': row['api_secret'],
            'api_passphrase': row['api_passphrase'],
        }
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict) -> bool:
    """Save settings to database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE settings SET
                symbol = ?,
                lookback_periods = ?,
                entry_std_dev = ?,
                exit_std_dev = ?,
                stop_loss_std_dev = ?,
                position_size = ?,
                paper_mode = ?,
                api_key = ?,
                api_secret = ?,
                api_passphrase = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = (SELECT MAX(id) FROM settings)
        ''', (
            settings['symbol'],
            settings['lookback_periods'],
            settings['entry_std_dev'],
            settings['exit_std_dev'],
            settings['stop_loss_std_dev'],
            settings['position_size'],
            1 if settings['paper_mode'] else 0,
            settings.get('api_key', ''),
            settings.get('api_secret', ''),
            settings.get('api_passphrase', ''),
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False
    finally:
        conn.close()


# =============================================================================
# OKX API Client
# =============================================================================

class OKXClient:
    """OKX API Client for funding rate and trading operations."""

    def __init__(self, api_key: str = '', api_secret: str = '', passphrase: str = ''):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = OKX_BASE_URL
        self.session = requests.Session()

    def _get_timestamp(self) -> str:
        """Get ISO format timestamp for API requests."""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _sign(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate signature for authenticated requests."""
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _get_headers(self, method: str, path: str, body: str = '') -> Dict:
        """Get headers for authenticated requests."""
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
        return headers

    def _request(self, method: str, path: str, params: Dict = None,
                 data: Dict = None) -> Dict:
        """Make API request."""
        url = self.base_url + path
        body = json.dumps(data) if data else ''
        headers = self._get_headers(method, path, body)

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            else:
                response = self.session.post(url, data=body, headers=headers, timeout=10)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return {'code': '-1', 'msg': str(e), 'data': []}

    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Get current funding rate for a symbol.
        GET /api/v5/public/funding-rate
        """
        path = '/api/v5/public/funding-rate'
        params = {'instId': symbol}
        result = self._request('GET', path, params=params)

        if result.get('code') == '0' and result.get('data'):
            data = result['data'][0]
            return {
                'symbol': data.get('instId'),
                'funding_rate': float(data.get('fundingRate', 0)),
                'next_funding_rate': float(data.get('nextFundingRate', 0)) if data.get('nextFundingRate') else None,
                'funding_time': int(data.get('fundingTime', 0)),
                'next_funding_time': int(data.get('nextFundingTime', 0)),
            }
        return {}

    def get_funding_rate_history(self, symbol: str, limit: int = 100,
                                  before: str = None, after: str = None) -> List[Dict]:
        """
        Get historical funding rates.
        GET /api/v5/public/funding-rate-history
        Max 100 records per call.
        """
        path = '/api/v5/public/funding-rate-history'
        params = {'instId': symbol, 'limit': str(limit)}
        if before:
            params['before'] = before
        if after:
            params['after'] = after

        result = self._request('GET', path, params=params)

        if result.get('code') == '0' and result.get('data'):
            return [
                {
                    'symbol': item.get('instId'),
                    'funding_rate': float(item.get('fundingRate', 0)),
                    'realized_rate': float(item.get('realizedRate', 0)) if item.get('realizedRate') else None,
                    'funding_time': int(item.get('fundingTime', 0)),
                }
                for item in result['data']
            ]
        return []

    def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker data for swap price.
        GET /api/v5/market/ticker
        """
        path = '/api/v5/market/ticker'
        params = {'instId': symbol}
        result = self._request('GET', path, params=params)

        if result.get('code') == '0' and result.get('data'):
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
        return {}

    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Get current positions.
        GET /api/v5/account/positions
        Requires authentication.
        """
        path = '/api/v5/account/positions'
        params = {}
        if symbol:
            params['instId'] = symbol

        result = self._request('GET', path, params=params)

        if result.get('code') == '0' and result.get('data'):
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
            ]
        return []

    def place_order(self, symbol: str, side: str, size: float,
                   order_type: str = 'market', price: float = None) -> Dict:
        """
        Place an order.
        POST /api/v5/trade/order

        Args:
            symbol: Instrument ID (e.g., 'BTC-USDT-SWAP')
            side: 'buy' or 'sell'
            size: Position size
            order_type: 'market' or 'limit'
            price: Required for limit orders
        """
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

        result = self._request('POST', path, data=data)

        if result.get('code') == '0' and result.get('data'):
            order_data = result['data'][0]
            return {
                'order_id': order_data.get('ordId'),
                'client_order_id': order_data.get('clOrdId'),
                'success': order_data.get('sCode') == '0',
                'message': order_data.get('sMsg', ''),
            }
        return {'success': False, 'message': result.get('msg', 'Unknown error')}

    def close_position(self, symbol: str, position_side: str = 'net') -> Dict:
        """
        Close all positions for a symbol.
        POST /api/v5/trade/close-position
        """
        path = '/api/v5/trade/close-position'
        data = {
            'instId': symbol,
            'mgnMode': 'cross',
            'posSide': position_side,
        }

        result = self._request('POST', path, data=data)

        if result.get('code') == '0':
            return {'success': True, 'message': 'Position closed'}
        return {'success': False, 'message': result.get('msg', 'Unknown error')}


# Global OKX client instance
okx_client = OKXClient()


def update_okx_client():
    """Update OKX client with current settings."""
    global okx_client
    settings = get_settings()
    okx_client = OKXClient(
        api_key=settings.get('api_key', ''),
        api_secret=settings.get('api_secret', ''),
        passphrase=settings.get('api_passphrase', '')
    )


# =============================================================================
# Z-Score Calculation
# =============================================================================

def calculate_z_score(funding_rates: List[float]) -> Dict:
    """
    Calculate Z-score from a list of funding rates.

    Returns:
        Dict with z_score, mean, std_dev, current_rate
    """
    if len(funding_rates) < 2:
        return {
            'z_score': 0,
            'mean': funding_rates[0] if funding_rates else 0,
            'std_dev': 0,
            'current_rate': funding_rates[0] if funding_rates else 0,
        }

    current_rate = funding_rates[0]
    historical_rates = funding_rates[1:]  # Exclude current for calculation

    mean = sum(historical_rates) / len(historical_rates)
    variance = sum((r - mean) ** 2 for r in historical_rates) / len(historical_rates)
    std_dev = variance ** 0.5

    if std_dev == 0:
        z_score = 0
    else:
        z_score = (current_rate - mean) / std_dev

    return {
        'z_score': z_score,
        'mean': mean,
        'std_dev': std_dev,
        'current_rate': current_rate,
    }


def get_signal(z_score: float, settings: Dict, current_position: str = None) -> Dict:
    """
    Determine trading signal based on Z-score and current position.

    Args:
        z_score: Current Z-score
        settings: Strategy settings
        current_position: 'long', 'short', or None

    Returns:
        Dict with signal type and details
    """
    entry_threshold = settings['entry_std_dev']
    exit_threshold = settings['exit_std_dev']
    stop_loss_threshold = settings['stop_loss_std_dev']

    signal = {
        'action': 'hold',
        'type': None,
        'reason': None,
        'z_score': z_score,
    }

    # If we have a position, check for exit signals first
    if current_position == 'short':
        # Short position: exit if Z-score returns to mean or hits stop loss
        if abs(z_score) <= exit_threshold:
            signal['action'] = 'close'
            signal['type'] = 'exit'
            signal['reason'] = 'mean_reversion'
        elif z_score <= -stop_loss_threshold:
            # Z-score went very negative (funding went negative) - stop loss
            signal['action'] = 'close'
            signal['type'] = 'stop_loss'
            signal['reason'] = 'stop_loss'

    elif current_position == 'long':
        # Long position: exit if Z-score returns to mean or hits stop loss
        if abs(z_score) <= exit_threshold:
            signal['action'] = 'close'
            signal['type'] = 'exit'
            signal['reason'] = 'mean_reversion'
        elif z_score >= stop_loss_threshold:
            # Z-score went very positive (funding went positive) - stop loss
            signal['action'] = 'close'
            signal['type'] = 'stop_loss'
            signal['reason'] = 'stop_loss'

    else:
        # No position - check for entry signals
        if z_score >= entry_threshold:
            # High funding rate - go SHORT to collect funding
            signal['action'] = 'open'
            signal['type'] = 'short'
            signal['reason'] = f'Z-score {z_score:.2f} >= {entry_threshold}'
        elif z_score <= -entry_threshold:
            # Low funding rate - go LONG to collect funding
            signal['action'] = 'open'
            signal['type'] = 'long'
            signal['reason'] = f'Z-score {z_score:.2f} <= -{entry_threshold}'

    return signal


def calculate_annualized_rate(funding_rate: float) -> float:
    """
    Calculate annualized funding rate.
    Funding is paid every 8 hours = 3 times per day = 1095 times per year.
    """
    return funding_rate * 3 * 365 * 100  # Return as percentage


# =============================================================================
# Data Collection
# =============================================================================

def bootstrap_funding_history(symbol: str, periods: int = 90):
    """
    Fetch historical funding rates on startup.
    OKX allows max 100 records per call, so we may need multiple calls.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    all_rates = []
    before_ts = None

    # Fetch funding rate history (may need multiple calls for 90 periods)
    remaining = periods
    while remaining > 0:
        limit = min(remaining, 100)
        rates = okx_client.get_funding_rate_history(
            symbol,
            limit=limit,
            before=before_ts
        )

        if not rates:
            break

        all_rates.extend(rates)
        remaining -= len(rates)

        # Get the oldest timestamp for next pagination
        if rates:
            before_ts = str(rates[-1]['funding_time'])

        # Safety check to avoid infinite loop
        if len(rates) < limit:
            break

    # Store in database
    for rate in all_rates:
        funding_time = datetime.fromtimestamp(rate['funding_time'] / 1000, tz=timezone.utc)
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO funding_rate_history
                (symbol, funding_rate, funding_time, realized_rate)
                VALUES (?, ?, ?, ?)
            ''', (
                rate['symbol'],
                rate['funding_rate'],
                funding_time.isoformat(),
                rate['realized_rate'],
            ))
        except sqlite3.IntegrityError:
            pass  # Ignore duplicates

    conn.commit()
    conn.close()

    print(f"Bootstrapped {len(all_rates)} historical funding rates for {symbol}")
    return len(all_rates)


def update_funding_data():
    """Update funding rate data - called periodically."""
    settings = get_settings()
    symbol = settings['symbol']

    # Get current funding rate
    funding_data = okx_client.get_funding_rate(symbol)
    if not funding_data:
        print("Failed to fetch funding rate")
        return

    # Get current ticker for swap price
    ticker = okx_client.get_ticker(symbol)
    swap_price = ticker.get('last_price', 0) if ticker else 0

    # Get historical rates for Z-score calculation
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT funding_rate FROM funding_rate_history
        WHERE symbol = ?
        ORDER BY funding_time DESC
        LIMIT ?
    ''', (symbol, settings['lookback_periods']))

    historical_rates = [row['funding_rate'] for row in cursor.fetchall()]

    # Add current rate at the beginning for Z-score calculation
    current_rate = funding_data['funding_rate']
    all_rates = [current_rate] + historical_rates

    # Calculate Z-score
    z_data = calculate_z_score(all_rates)

    # Store current funding rate snapshot
    funding_time = datetime.fromtimestamp(funding_data['funding_time'] / 1000, tz=timezone.utc)
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO funding_rate_history
            (symbol, funding_rate, funding_time, swap_price, z_score, mean_rate, std_dev)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            current_rate,
            funding_time.isoformat(),
            swap_price,
            z_data['z_score'],
            z_data['mean'],
            z_data['std_dev'],
        ))
    except sqlite3.IntegrityError:
        pass

    # Store price snapshot
    cursor.execute('''
        INSERT INTO price_snapshots (symbol, swap_price)
        VALUES (?, ?)
    ''', (symbol, swap_price))

    conn.commit()
    conn.close()

    # Check for signals and handle trading
    check_and_execute_signals(z_data, funding_data, ticker, settings)


def check_and_execute_signals(z_data: Dict, funding_data: Dict,
                              ticker: Dict, settings: Dict):
    """Check for trading signals and execute if applicable."""
    symbol = settings['symbol']

    # Get current open trade
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM trades WHERE symbol = ? AND status = 'open'
        ORDER BY entry_time DESC LIMIT 1
    ''', (symbol,))
    open_trade = cursor.fetchone()
    conn.close()

    current_position = None
    if open_trade:
        current_position = open_trade['side']

    # Get signal
    signal = get_signal(z_data['z_score'], settings, current_position)

    if signal['action'] == 'hold':
        return

    swap_price = ticker.get('last_price', 0) if ticker else 0

    if signal['action'] == 'open':
        # Open new position
        execute_open_trade(
            symbol=symbol,
            side=signal['type'],
            price=swap_price,
            funding_rate=z_data['current_rate'],
            z_score=z_data['z_score'],
            size=settings['position_size'],
            paper_mode=settings['paper_mode']
        )

    elif signal['action'] == 'close' and open_trade:
        # Close existing position
        execute_close_trade(
            trade_id=open_trade['id'],
            price=swap_price,
            funding_rate=z_data['current_rate'],
            z_score=z_data['z_score'],
            reason=signal['reason'],
            paper_mode=settings['paper_mode']
        )


def execute_open_trade(symbol: str, side: str, price: float,
                       funding_rate: float, z_score: float,
                       size: float, paper_mode: bool):
    """Execute opening a new trade."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Execute actual order if not paper trading
    if not paper_mode:
        order_side = 'sell' if side == 'short' else 'buy'
        result = okx_client.place_order(symbol, order_side, size)
        if not result.get('success'):
            print(f"Failed to place order: {result.get('message')}")
            conn.close()
            return

    # Record trade in database
    cursor.execute('''
        INSERT INTO trades
        (symbol, side, entry_price, entry_funding_rate, entry_z_score,
         position_size, status, entry_time, paper_trade)
        VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
    ''', (
        symbol, side, price, funding_rate, z_score,
        size, datetime.now(timezone.utc).isoformat(),
        1 if paper_mode else 0
    ))

    conn.commit()
    conn.close()
    print(f"Opened {side.upper()} position at {price} (Z-score: {z_score:.2f})")


def execute_close_trade(trade_id: int, price: float, funding_rate: float,
                        z_score: float, reason: str, paper_mode: bool):
    """Execute closing an existing trade."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get trade details
    cursor.execute('SELECT * FROM trades WHERE id = ?', (trade_id,))
    trade = cursor.fetchone()

    if not trade:
        conn.close()
        return

    symbol = trade['symbol']
    side = trade['side']
    entry_price = trade['entry_price']
    size = trade['position_size']

    # Execute actual order if not paper trading
    if not paper_mode:
        result = okx_client.close_position(symbol)
        if not result.get('success'):
            print(f"Failed to close position: {result.get('message')}")
            conn.close()
            return

    # Calculate P&L
    if side == 'long':
        price_pnl = (price - entry_price) * size
    else:  # short
        price_pnl = (entry_price - price) * size

    # Get funding payments for this trade
    cursor.execute('''
        SELECT SUM(payment_amount) as total_funding
        FROM funding_payments WHERE trade_id = ?
    ''', (trade_id,))
    funding_result = cursor.fetchone()
    funding_pnl = funding_result['total_funding'] if funding_result['total_funding'] else 0

    total_pnl = price_pnl + funding_pnl

    # Count funding periods held
    cursor.execute('''
        SELECT COUNT(*) as periods FROM funding_payments WHERE trade_id = ?
    ''', (trade_id,))
    periods = cursor.fetchone()['periods']

    # Update trade record
    cursor.execute('''
        UPDATE trades SET
            exit_price = ?,
            exit_funding_rate = ?,
            exit_z_score = ?,
            price_pnl = ?,
            funding_pnl = ?,
            total_pnl = ?,
            funding_periods_held = ?,
            status = 'closed',
            exit_time = ?,
            exit_reason = ?
        WHERE id = ?
    ''', (
        price, funding_rate, z_score,
        price_pnl, funding_pnl, total_pnl, periods,
        datetime.now(timezone.utc).isoformat(), reason,
        trade_id
    ))

    conn.commit()
    conn.close()
    print(f"Closed {side.upper()} position at {price} (P&L: {total_pnl:.2f}, Reason: {reason})")


def record_funding_payment(trade_id: int, symbol: str, funding_rate: float,
                           position_size: float, side: str):
    """Record a funding payment for an open position."""
    # For shorts: positive funding = receive payment, negative = pay
    # For longs: positive funding = pay, negative = receive payment
    if side == 'short':
        payment = funding_rate * position_size
    else:  # long
        payment = -funding_rate * position_size

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO funding_payments (trade_id, symbol, funding_rate, payment_amount, funding_time)
        VALUES (?, ?, ?, ?, ?)
    ''', (trade_id, symbol, funding_rate, payment, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/settings')
def settings_page():
    """Settings page."""
    return render_template('settings.html')


@app.route('/trades')
def trades_page():
    """Trade history page."""
    return render_template('trades.html')


@app.route('/api/dashboard')
def api_dashboard():
    """Get dashboard data including current funding rate and Z-score."""
    settings = get_settings()
    symbol = settings['symbol']

    # Get current funding rate from OKX
    funding_data = okx_client.get_funding_rate(symbol)
    ticker = okx_client.get_ticker(symbol)

    # Get historical rates for Z-score
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT funding_rate FROM funding_rate_history
        WHERE symbol = ?
        ORDER BY funding_time DESC
        LIMIT ?
    ''', (symbol, settings['lookback_periods']))
    historical_rates = [row['funding_rate'] for row in cursor.fetchall()]

    # Get current open trade
    cursor.execute('''
        SELECT * FROM trades WHERE symbol = ? AND status = 'open'
        ORDER BY entry_time DESC LIMIT 1
    ''', (symbol,))
    open_trade = cursor.fetchone()
    conn.close()

    # Calculate Z-score
    current_rate = funding_data.get('funding_rate', 0) if funding_data else 0
    all_rates = [current_rate] + historical_rates
    z_data = calculate_z_score(all_rates)

    # Get signal
    current_position = open_trade['side'] if open_trade else None
    signal = get_signal(z_data['z_score'], settings, current_position)

    # Calculate time until next funding
    next_funding_ts = funding_data.get('next_funding_time', 0) if funding_data else 0
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    time_until_funding = max(0, next_funding_ts - now_ts)

    # Format countdown
    seconds_until = time_until_funding // 1000
    hours = seconds_until // 3600
    minutes = (seconds_until % 3600) // 60
    seconds = seconds_until % 60
    countdown = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Calculate entry thresholds in terms of funding rate
    entry_high_rate = z_data['mean'] + (settings['entry_std_dev'] * z_data['std_dev'])
    entry_low_rate = z_data['mean'] - (settings['entry_std_dev'] * z_data['std_dev'])

    response = {
        'symbol': symbol,
        'swap_price': ticker.get('last_price', 0) if ticker else 0,
        'funding_rate': current_rate,
        'funding_rate_pct': current_rate * 100,
        'next_funding_rate': funding_data.get('next_funding_rate') if funding_data else None,
        'next_funding_time': next_funding_ts,
        'countdown': countdown,
        'annualized_rate': calculate_annualized_rate(current_rate),
        'funding_direction': 'POSITIVE' if current_rate >= 0 else 'NEGATIVE',
        'z_score': z_data['z_score'],
        'mean_rate': z_data['mean'],
        'mean_rate_pct': z_data['mean'] * 100,
        'std_dev': z_data['std_dev'],
        'std_dev_pct': z_data['std_dev'] * 100,
        'signal': signal,
        'entry_high_threshold': entry_high_rate,
        'entry_high_threshold_pct': entry_high_rate * 100,
        'entry_low_threshold': entry_low_rate,
        'entry_low_threshold_pct': entry_low_rate * 100,
        'settings': settings,
        'open_trade': dict(open_trade) if open_trade else None,
        'data_points': len(historical_rates),
    }

    return jsonify(response)


@app.route('/api/chart-data')
def api_chart_data():
    """Get chart data for funding rate and Z-score visualization."""
    settings = get_settings()
    symbol = settings['symbol']
    limit = request.args.get('limit', 90, type=int)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT funding_rate, funding_time, swap_price, z_score, mean_rate, std_dev
        FROM funding_rate_history
        WHERE symbol = ?
        ORDER BY funding_time DESC
        LIMIT ?
    ''', (symbol, limit))

    rows = cursor.fetchall()
    conn.close()

    # Reverse to get chronological order
    rows = list(reversed(rows))

    data = {
        'labels': [row['funding_time'] for row in rows],
        'funding_rates': [row['funding_rate'] * 100 for row in rows],  # As percentage
        'z_scores': [row['z_score'] if row['z_score'] else 0 for row in rows],
        'swap_prices': [row['swap_price'] if row['swap_price'] else 0 for row in rows],
        'entry_threshold': settings['entry_std_dev'],
        'exit_threshold': settings['exit_std_dev'],
        'stop_loss_threshold': settings['stop_loss_std_dev'],
    }

    return jsonify(data)


@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """Get or update settings."""
    if request.method == 'GET':
        settings = get_settings()
        # Don't expose secrets in full
        if settings.get('api_secret'):
            settings['api_secret'] = '********'
        return jsonify(settings)

    elif request.method == 'POST':
        data = request.get_json()
        current = get_settings()

        # Update settings
        current['symbol'] = data.get('symbol', current['symbol'])
        current['lookback_periods'] = int(data.get('lookback_periods', current['lookback_periods']))
        current['entry_std_dev'] = float(data.get('entry_std_dev', current['entry_std_dev']))
        current['exit_std_dev'] = float(data.get('exit_std_dev', current['exit_std_dev']))
        current['stop_loss_std_dev'] = float(data.get('stop_loss_std_dev', current['stop_loss_std_dev']))
        current['position_size'] = float(data.get('position_size', current['position_size']))
        current['paper_mode'] = data.get('paper_mode', current['paper_mode'])

        # Only update API credentials if provided and not masked
        if data.get('api_key'):
            current['api_key'] = data['api_key']
        if data.get('api_secret') and data['api_secret'] != '********':
            current['api_secret'] = data['api_secret']
        if data.get('api_passphrase'):
            current['api_passphrase'] = data['api_passphrase']

        if save_settings(current):
            update_okx_client()
            return jsonify({'success': True, 'message': 'Settings saved'})
        return jsonify({'success': False, 'message': 'Failed to save settings'}), 500


@app.route('/api/trades')
def api_trades():
    """Get trade history."""
    symbol = request.args.get('symbol')
    status = request.args.get('status')
    limit = request.args.get('limit', 50, type=int)

    conn = get_db_connection()
    cursor = conn.cursor()

    query = 'SELECT * FROM trades WHERE 1=1'
    params = []

    if symbol:
        query += ' AND symbol = ?'
        params.append(symbol)
    if status:
        query += ' AND status = ?'
        params.append(status)

    query += ' ORDER BY entry_time DESC LIMIT ?'
    params.append(limit)

    cursor.execute(query, params)
    trades = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify(trades)


@app.route('/api/trade/<int:trade_id>/funding-payments')
def api_funding_payments(trade_id: int):
    """Get funding payments for a specific trade."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM funding_payments WHERE trade_id = ?
        ORDER BY funding_time DESC
    ''', (trade_id,))
    payments = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify(payments)


@app.route('/api/bootstrap', methods=['POST'])
def api_bootstrap():
    """Bootstrap funding rate history."""
    settings = get_settings()
    count = bootstrap_funding_history(settings['symbol'], settings['lookback_periods'])
    return jsonify({'success': True, 'records': count})


@app.route('/api/manual-trade', methods=['POST'])
def api_manual_trade():
    """Manually open or close a trade."""
    data = request.get_json()
    action = data.get('action')  # 'open' or 'close'
    settings = get_settings()

    # Get current market data
    ticker = okx_client.get_ticker(settings['symbol'])
    funding_data = okx_client.get_funding_rate(settings['symbol'])

    if not ticker or not funding_data:
        return jsonify({'success': False, 'message': 'Failed to fetch market data'}), 500

    price = ticker['last_price']
    funding_rate = funding_data['funding_rate']

    # Get Z-score
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT funding_rate FROM funding_rate_history
        WHERE symbol = ? ORDER BY funding_time DESC LIMIT ?
    ''', (settings['symbol'], settings['lookback_periods']))
    historical_rates = [row['funding_rate'] for row in cursor.fetchall()]
    conn.close()

    z_data = calculate_z_score([funding_rate] + historical_rates)

    if action == 'open':
        side = data.get('side')  # 'long' or 'short'
        if not side:
            return jsonify({'success': False, 'message': 'Side required'}), 400

        execute_open_trade(
            symbol=settings['symbol'],
            side=side,
            price=price,
            funding_rate=funding_rate,
            z_score=z_data['z_score'],
            size=settings['position_size'],
            paper_mode=settings['paper_mode']
        )
        return jsonify({'success': True, 'message': f'Opened {side} position'})

    elif action == 'close':
        trade_id = data.get('trade_id')
        if not trade_id:
            return jsonify({'success': False, 'message': 'Trade ID required'}), 400

        execute_close_trade(
            trade_id=trade_id,
            price=price,
            funding_rate=funding_rate,
            z_score=z_data['z_score'],
            reason='manual',
            paper_mode=settings['paper_mode']
        )
        return jsonify({'success': True, 'message': 'Position closed'})

    return jsonify({'success': False, 'message': 'Invalid action'}), 400


@app.route('/api/stats')
def api_stats():
    """Get trading statistics."""
    settings = get_settings()
    symbol = settings['symbol']

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get overall stats
    cursor.execute('''
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN total_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN total_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(total_pnl) as total_pnl,
            SUM(price_pnl) as total_price_pnl,
            SUM(funding_pnl) as total_funding_pnl,
            AVG(total_pnl) as avg_pnl,
            MAX(total_pnl) as best_trade,
            MIN(total_pnl) as worst_trade,
            SUM(funding_periods_held) as total_funding_periods
        FROM trades
        WHERE symbol = ? AND status = 'closed'
    ''', (symbol,))

    stats = dict(cursor.fetchone())

    # Calculate win rate
    if stats['total_trades'] and stats['total_trades'] > 0:
        stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
    else:
        stats['win_rate'] = 0

    conn.close()

    return jsonify(stats)


# =============================================================================
# Scheduler
# =============================================================================

scheduler = BackgroundScheduler()


def start_scheduler():
    """Start the background scheduler for data collection."""
    # Update funding data every minute
    scheduler.add_job(
        update_funding_data,
        'interval',
        minutes=1,
        id='update_funding_data',
        replace_existing=True
    )

    # Check for funding payments every 8 hours (at funding times)
    # OKX funding times: 00:00, 08:00, 16:00 UTC
    scheduler.add_job(
        check_funding_payments,
        'cron',
        hour='0,8,16',
        minute=5,  # 5 minutes after funding time
        id='check_funding_payments',
        replace_existing=True
    )

    scheduler.start()
    print("Scheduler started")


def check_funding_payments():
    """Check and record funding payments for open positions."""
    settings = get_settings()
    symbol = settings['symbol']

    # Get current funding rate
    funding_data = okx_client.get_funding_rate(symbol)
    if not funding_data:
        return

    # Get open trades
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM trades WHERE symbol = ? AND status = 'open'
    ''', (symbol,))
    open_trades = cursor.fetchall()
    conn.close()

    # Record funding payment for each open trade
    for trade in open_trades:
        record_funding_payment(
            trade_id=trade['id'],
            symbol=symbol,
            funding_rate=funding_data['funding_rate'],
            position_size=trade['position_size'],
            side=trade['side']
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Initialize database
    init_db()

    # Update OKX client with settings
    update_okx_client()

    # Bootstrap historical data
    settings = get_settings()
    print(f"Bootstrapping funding rate history for {settings['symbol']}...")
    bootstrap_funding_history(settings['symbol'], settings['lookback_periods'])

    # Start scheduler
    start_scheduler()

    # Run Flask app
    print("Starting Funding Rate Mean Reversion Trading System...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

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

Funding Rate Mechanism:
- Funding payments occur every 8 hours on OKX (00:00, 08:00, 16:00 UTC)
- Positive funding rate: LONGS pay SHORTS (go SHORT to receive payment)
- Negative funding rate: SHORTS pay LONGS (go LONG to receive payment)
- Payment = Position Size × Funding Rate
"""

import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request, g
from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables from .env file
load_dotenv()

# Import modular components
from app.api import OKXClient, create_okx_client

# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

DATABASE = os.environ.get('DATABASE_PATH', 'funding_rate.db')

# Default Settings (can be overridden by environment variables or database)
DEFAULT_SETTINGS = {
    'symbol': os.environ.get('TRADING_SYMBOL', 'BTC-USDT-SWAP'),
    'lookback_periods': int(os.environ.get('LOOKBACK_PERIODS', 90)),
    'entry_std_dev': float(os.environ.get('ENTRY_STD_DEV', 2.0)),
    'exit_std_dev': float(os.environ.get('EXIT_STD_DEV', 0.2)),
    'stop_loss_std_dev': float(os.environ.get('STOP_LOSS_STD_DEV', 6.0)),
    'position_size': float(os.environ.get('POSITION_SIZE', 1.0)),
    'paper_mode': os.environ.get('PAPER_MODE', 'true').lower() == 'true',
    'api_key': os.environ.get('OKX_API_KEY', ''),
    'api_secret': os.environ.get('OKX_API_SECRET', ''),
    'api_passphrase': os.environ.get('OKX_PASSPHRASE', ''),
    # Fee settings (as percentages, e.g., 0.02 = 0.02%)
    'maker_fee': float(os.environ.get('MAKER_FEE', 0.02)),
    'taker_fee': float(os.environ.get('TAKER_FEE', 0.05)),
    'estimated_slippage': float(os.environ.get('ESTIMATED_SLIPPAGE', 0.01)),
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
            maker_fee REAL DEFAULT 0.02,
            taker_fee REAL DEFAULT 0.05,
            estimated_slippage REAL DEFAULT 0.01,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Add fee columns if they don't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE settings ADD COLUMN maker_fee REAL DEFAULT 0.02')
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('ALTER TABLE settings ADD COLUMN taker_fee REAL DEFAULT 0.05')
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('ALTER TABLE settings ADD COLUMN estimated_slippage REAL DEFAULT 0.01')
    except sqlite3.OperationalError:
        pass

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
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            entry_funding_rate REAL,
            exit_funding_rate REAL,
            entry_z_score REAL,
            exit_z_score REAL,
            position_size REAL NOT NULL,
            price_pnl REAL DEFAULT 0,
            funding_pnl REAL DEFAULT 0,
            fee_cost REAL DEFAULT 0,
            slippage_cost REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            net_pnl REAL DEFAULT 0,
            funding_periods_held INTEGER DEFAULT 0,
            status TEXT DEFAULT 'open',
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP,
            exit_reason TEXT,
            paper_trade INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Add fee columns to trades if they don't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE trades ADD COLUMN fee_cost REAL DEFAULT 0')
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('ALTER TABLE trades ADD COLUMN slippage_cost REAL DEFAULT 0')
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('ALTER TABLE trades ADD COLUMN net_pnl REAL DEFAULT 0')
    except sqlite3.OperationalError:
        pass

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
                                  exit_std_dev, stop_loss_std_dev, position_size, paper_mode,
                                  api_key, api_secret, api_passphrase,
                                  maker_fee, taker_fee, estimated_slippage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            DEFAULT_SETTINGS['symbol'],
            DEFAULT_SETTINGS['lookback_periods'],
            DEFAULT_SETTINGS['entry_std_dev'],
            DEFAULT_SETTINGS['exit_std_dev'],
            DEFAULT_SETTINGS['stop_loss_std_dev'],
            DEFAULT_SETTINGS['position_size'],
            1 if DEFAULT_SETTINGS['paper_mode'] else 0,
            DEFAULT_SETTINGS['api_key'],
            DEFAULT_SETTINGS['api_secret'],
            DEFAULT_SETTINGS['api_passphrase'],
            DEFAULT_SETTINGS['maker_fee'],
            DEFAULT_SETTINGS['taker_fee'],
            DEFAULT_SETTINGS['estimated_slippage'],
        ))

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
        # Get fee settings with defaults for backward compatibility
        row_dict = dict(row)
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
            'maker_fee': row_dict.get('maker_fee', DEFAULT_SETTINGS['maker_fee']),
            'taker_fee': row_dict.get('taker_fee', DEFAULT_SETTINGS['taker_fee']),
            'estimated_slippage': row_dict.get('estimated_slippage', DEFAULT_SETTINGS['estimated_slippage']),
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
                maker_fee = ?,
                taker_fee = ?,
                estimated_slippage = ?,
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
            settings.get('maker_fee', DEFAULT_SETTINGS['maker_fee']),
            settings.get('taker_fee', DEFAULT_SETTINGS['taker_fee']),
            settings.get('estimated_slippage', DEFAULT_SETTINGS['estimated_slippage']),
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False
    finally:
        conn.close()


def calculate_trade_costs(entry_price: float, exit_price: float, size: float,
                          settings: Dict, use_maker: bool = False) -> Dict:
    """
    Calculate trading costs including fees and slippage.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        size: Position size
        settings: Settings dict with fee rates
        use_maker: If True, use maker fee; otherwise use taker fee

    Returns:
        Dict with fee_cost, slippage_cost, total_cost
    """
    fee_rate = settings.get('maker_fee', 0.02) if use_maker else settings.get('taker_fee', 0.05)
    slippage_rate = settings.get('estimated_slippage', 0.01)

    # Fee is charged on both entry and exit (percentage of notional value)
    entry_notional = entry_price * size
    exit_notional = exit_price * size

    # Fee calculation: fee_rate is in percentage (e.g., 0.05 = 0.05%)
    entry_fee = entry_notional * (fee_rate / 100)
    exit_fee = exit_notional * (fee_rate / 100)
    total_fee = entry_fee + exit_fee

    # Slippage estimation (one-way on each leg)
    entry_slippage = entry_notional * (slippage_rate / 100)
    exit_slippage = exit_notional * (slippage_rate / 100)
    total_slippage = entry_slippage + exit_slippage

    return {
        'fee_cost': total_fee,
        'slippage_cost': total_slippage,
        'total_cost': total_fee + total_slippage,
        'fee_rate': fee_rate,
        'slippage_rate': slippage_rate,
    }


# =============================================================================
# OKX Client Management
# =============================================================================

# Global OKX client instance
okx_client: Optional[OKXClient] = None


def get_okx_client() -> OKXClient:
    """Get or create OKX client with current settings."""
    global okx_client
    if okx_client is None:
        update_okx_client()
    return okx_client


def update_okx_client():
    """Update OKX client with current settings."""
    global okx_client
    settings = get_settings()
    okx_client = OKXClient(
        api_key=settings.get('api_key', ''),
        api_secret=settings.get('api_secret', ''),
        passphrase=settings.get('api_passphrase', '')
    )
    return okx_client


# =============================================================================
# Z-Score Calculation
# =============================================================================

def calculate_z_score(funding_rates: List[float]) -> Dict:
    """
    Calculate Z-score from a list of funding rates.

    The Z-score measures how many standard deviations the current
    funding rate is from the historical mean.

    Args:
        funding_rates: List of funding rates, with current rate at index 0

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

    # Calculate mean and standard deviation
    mean = sum(historical_rates) / len(historical_rates)
    variance = sum((r - mean) ** 2 for r in historical_rates) / len(historical_rates)
    std_dev = variance ** 0.5

    # Calculate Z-score
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

    Strategy:
    - Z >= +entry_std_dev: HIGH funding → Go SHORT (collect funding)
    - Z <= -entry_std_dev: LOW funding → Go LONG (collect funding)
    - |Z| <= exit_std_dev: Mean reversion complete → EXIT
    - |Z| >= stop_loss_std_dev: Extreme deviation → STOP LOSS

    Args:
        z_score: Current Z-score
        settings: Strategy settings
        current_position: 'long', 'short', or None

    Returns:
        Dict with action, type, reason, and z_score
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
        # Short position: entered because funding was HIGH (positive Z)
        # Exit if Z-score returns to mean OR if Z goes extremely negative
        if abs(z_score) <= exit_threshold:
            signal['action'] = 'close'
            signal['type'] = 'exit'
            signal['reason'] = 'mean_reversion'
        elif z_score <= -stop_loss_threshold:
            # Z-score went very negative - funding reversed extremely
            signal['action'] = 'close'
            signal['type'] = 'stop_loss'
            signal['reason'] = 'stop_loss'

    elif current_position == 'long':
        # Long position: entered because funding was LOW (negative Z)
        # Exit if Z-score returns to mean OR if Z goes extremely positive
        if abs(z_score) <= exit_threshold:
            signal['action'] = 'close'
            signal['type'] = 'exit'
            signal['reason'] = 'mean_reversion'
        elif z_score >= stop_loss_threshold:
            # Z-score went very positive - funding reversed extremely
            signal['action'] = 'close'
            signal['type'] = 'stop_loss'
            signal['reason'] = 'stop_loss'

    else:
        # No position - check for entry signals
        if z_score >= entry_threshold:
            # High funding rate (positive Z) → go SHORT to COLLECT funding
            # When funding is positive, shorts receive payment
            signal['action'] = 'open'
            signal['type'] = 'short'
            signal['reason'] = f'Z-score {z_score:.2f} >= {entry_threshold} (HIGH funding)'
        elif z_score <= -entry_threshold:
            # Low funding rate (negative Z) → go LONG to COLLECT funding
            # When funding is negative, longs receive payment
            signal['action'] = 'open'
            signal['type'] = 'long'
            signal['reason'] = f'Z-score {z_score:.2f} <= -{entry_threshold} (LOW funding)'

    return signal


def calculate_annualized_rate(funding_rate: float) -> float:
    """
    Calculate annualized funding rate.

    OKX funding is paid every 8 hours:
    - 3 times per day
    - 1095 times per year (3 × 365)

    Args:
        funding_rate: Single funding rate as decimal (e.g., 0.0001)

    Returns:
        Annualized rate as percentage (e.g., 10.95 for 10.95%)
    """
    return funding_rate * 3 * 365 * 100


# =============================================================================
# Data Collection
# =============================================================================

def bootstrap_funding_history(symbol: str, periods: int = 90) -> int:
    """
    Fetch historical funding rates on startup.

    OKX allows max 100 records per API call, so we may need multiple
    calls to fetch the full history.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USDT-SWAP')
        periods: Number of funding periods to fetch (default 90 = 30 days)

    Returns:
        Number of records fetched
    """
    client = get_okx_client()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Use the client's built-in pagination helper
    all_rates = client.get_funding_rate_history_all(symbol, periods)

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
                rate.get('realized_rate'),
            ))
        except sqlite3.IntegrityError:
            pass  # Ignore duplicates

    conn.commit()
    conn.close()

    print(f"Bootstrapped {len(all_rates)} historical funding rates for {symbol}")
    return len(all_rates)


# Old update_funding_data removed - replaced by update_funding_rate which runs every 8 hours


def check_and_execute_signals(z_data: Dict, funding_data: Dict,
                              ticker: Dict, settings: Dict):
    """
    Check for trading signals and execute if applicable.

    This function is called after each data update to evaluate
    whether to open or close positions based on Z-score signals.
    """
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


# =============================================================================
# Trade Execution
# =============================================================================

def execute_open_trade(symbol: str, side: str, price: float,
                       funding_rate: float, z_score: float,
                       size: float, paper_mode: bool):
    """
    Execute opening a new trade.

    Args:
        symbol: Trading symbol
        side: 'long' or 'short'
        price: Current swap price
        funding_rate: Current funding rate
        z_score: Current Z-score
        size: Position size in contracts
        paper_mode: If True, simulate trade without real order
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    client = get_okx_client()

    # Execute actual order if not paper trading
    if not paper_mode:
        # For SHORT: sell to open
        # For LONG: buy to open
        order_side = 'sell' if side == 'short' else 'buy'
        result = client.place_order(symbol, order_side, size)
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
    print(f"{'[PAPER] ' if paper_mode else ''}Opened {side.upper()} position at ${price:.2f} (Z-score: {z_score:.2f}σ)")


def execute_close_trade(trade_id: int, price: float, funding_rate: float,
                        z_score: float, reason: str, paper_mode: bool):
    """
    Execute closing an existing trade.

    Args:
        trade_id: Database ID of trade to close
        price: Current swap price
        funding_rate: Current funding rate
        z_score: Current Z-score
        reason: Exit reason ('mean_reversion', 'stop_loss', 'manual')
        paper_mode: If True, simulate trade without real order
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    client = get_okx_client()

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
        result = client.close_position(symbol)
        if not result.get('success'):
            print(f"Failed to close position: {result.get('message')}")
            conn.close()
            return

    # Calculate P&L
    # For LONG: profit when price goes UP
    # For SHORT: profit when price goes DOWN
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

    # Calculate trading costs (fees + slippage)
    settings = get_settings()
    costs = calculate_trade_costs(entry_price, price, size, settings, use_maker=False)
    fee_cost = costs['fee_cost']
    slippage_cost = costs['slippage_cost']

    # Total P&L before costs
    total_pnl = price_pnl + funding_pnl

    # Net P&L after costs
    net_pnl = total_pnl - fee_cost - slippage_cost

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
            fee_cost = ?,
            slippage_cost = ?,
            total_pnl = ?,
            net_pnl = ?,
            funding_periods_held = ?,
            status = 'closed',
            exit_time = ?,
            exit_reason = ?
        WHERE id = ?
    ''', (
        price, funding_rate, z_score,
        price_pnl, funding_pnl, fee_cost, slippage_cost, total_pnl, net_pnl, periods,
        datetime.now(timezone.utc).isoformat(), reason,
        trade_id
    ))

    conn.commit()
    conn.close()

    paper_prefix = '[PAPER] ' if trade['paper_trade'] else ''
    print(f"{paper_prefix}Closed {side.upper()} position at ${price:.2f}")
    print(f"  Price P&L: ${price_pnl:.2f}, Funding P&L: ${funding_pnl:.2f}")
    print(f"  Fees: ${fee_cost:.2f}, Slippage: ${slippage_cost:.2f}")
    print(f"  Gross P&L: ${total_pnl:.2f}, Net P&L: ${net_pnl:.2f}")
    print(f"  Reason: {reason}, Held for {periods} funding periods")


def record_funding_payment(trade_id: int, symbol: str, funding_rate: float,
                           position_size: float, side: str):
    """
    Record a funding payment for an open position.

    Funding payment logic:
    - Positive funding rate: LONGS pay SHORTS
      - SHORT position receives: +funding_rate × size
      - LONG position pays: -funding_rate × size
    - Negative funding rate: SHORTS pay LONGS
      - SHORT position pays: +funding_rate × size (negative, so pays)
      - LONG position receives: -funding_rate × size (negative, so receives)

    Args:
        trade_id: Database ID of open trade
        symbol: Trading symbol
        funding_rate: Current funding rate (positive or negative)
        position_size: Position size in contracts
        side: 'long' or 'short'
    """
    # For shorts: payment = +rate * size (receive when rate positive)
    # For longs: payment = -rate * size (receive when rate negative)
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

    direction = "received" if payment > 0 else "paid"
    print(f"Funding payment {direction}: ${abs(payment):.4f} (rate: {funding_rate*100:.4f}%)")


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


@app.route('/delta-neutral')
def delta_neutral_page():
    """Delta-neutral strategy calculator page."""
    return render_template('delta_neutral.html')


@app.route('/api/dashboard')
def api_dashboard():
    """Get dashboard data including current funding rate and Z-score."""
    settings = get_settings()
    symbol = settings['symbol']
    client = get_okx_client()

    # Get current funding rate from OKX
    funding_data = client.get_funding_rate(symbol)
    ticker = client.get_ticker(symbol)

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

    # Calculate time until next funding (00:00, 08:00, 16:00 UTC)
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour

    # Find next funding hour
    funding_hours = [0, 8, 16, 24]  # 24 = midnight next day
    next_funding_hour = next(h for h in funding_hours if h > current_hour)

    if next_funding_hour == 24:
        next_funding_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        next_funding_dt = now_utc.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)

    next_funding_ts = int(next_funding_dt.timestamp() * 1000)
    time_until_funding = max(0, int((next_funding_dt - now_utc).total_seconds() * 1000))

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
    client = get_okx_client()

    # Get current market data
    ticker = client.get_ticker(settings['symbol'])
    funding_data = client.get_funding_rate(settings['symbol'])

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

    # Get overall stats including fee and net P&L
    cursor.execute('''
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN net_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(total_pnl) as total_pnl,
            SUM(price_pnl) as total_price_pnl,
            SUM(funding_pnl) as total_funding_pnl,
            SUM(COALESCE(fee_cost, 0)) as total_fees,
            SUM(COALESCE(slippage_cost, 0)) as total_slippage,
            SUM(COALESCE(net_pnl, total_pnl)) as total_net_pnl,
            AVG(total_pnl) as avg_pnl,
            AVG(COALESCE(net_pnl, total_pnl)) as avg_net_pnl,
            MAX(COALESCE(net_pnl, total_pnl)) as best_trade,
            MIN(COALESCE(net_pnl, total_pnl)) as worst_trade,
            SUM(funding_periods_held) as total_funding_periods
        FROM trades
        WHERE symbol = ? AND status = 'closed'
    ''', (symbol,))

    stats = dict(cursor.fetchone())

    # Calculate win rate based on net P&L
    if stats['total_trades'] and stats['total_trades'] > 0:
        stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
    else:
        stats['win_rate'] = 0

    # Add fee settings for reference
    stats['maker_fee'] = settings.get('maker_fee', 0.02)
    stats['taker_fee'] = settings.get('taker_fee', 0.05)
    stats['estimated_slippage'] = settings.get('estimated_slippage', 0.01)

    conn.close()

    return jsonify(stats)


# =============================================================================
# Scheduler
# =============================================================================

scheduler = BackgroundScheduler()

# Cache for current price (updated frequently for UI)
_price_cache = {'price': 0, 'updated': None}


def update_price_only():
    """Update just the swap price for UI display. Runs every 30 seconds."""
    global _price_cache
    settings = get_settings()
    client = get_okx_client()

    ticker = client.get_ticker(settings['symbol'])
    if ticker:
        _price_cache['price'] = ticker.get('last_price', 0)
        _price_cache['updated'] = datetime.now(timezone.utc)


def update_funding_rate():
    """
    Update funding rate data - runs after each 8-hour funding period.
    This is the only time Z-score can change.
    """
    settings = get_settings()
    symbol = settings['symbol']
    client = get_okx_client()

    # Get current funding rate from OKX
    funding_data = client.get_funding_rate(symbol)
    if not funding_data:
        print(f"Failed to fetch funding rate for {symbol}")
        return

    # Get current ticker for swap price
    ticker = client.get_ticker(symbol)
    swap_price = ticker.get('last_price', 0) if ticker else 0

    # Update price cache
    global _price_cache
    _price_cache['price'] = swap_price
    _price_cache['updated'] = datetime.now(timezone.utc)

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

    # Store funding rate (only stores if funding_time is new)
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
        conn.commit()
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Funding rate updated: {current_rate*100:.4f}%, Z-score: {z_data['z_score']:.2f}")
    except Exception as e:
        print(f"Error storing funding rate: {e}")
    finally:
        conn.close()

    # Check for trading signals
    ticker_data = {'last_price': swap_price}
    check_and_execute_signals(z_data, funding_data, ticker_data, settings)


def start_scheduler():
    """Start the background scheduler for data collection."""

    # Price updates every 30 seconds (for UI only)
    scheduler.add_job(
        update_price_only,
        'interval',
        seconds=30,
        id='update_price',
        replace_existing=True
    )

    # Funding rate update after each 8-hour period (00:05, 08:05, 16:05 UTC)
    # This is when Z-score actually changes
    scheduler.add_job(
        update_funding_rate,
        'cron',
        hour='0,8,16',
        minute=5,
        id='update_funding_rate',
        replace_existing=True
    )

    # Check for funding payments (same time as rate update)
    scheduler.add_job(
        check_funding_payments,
        'cron',
        hour='0,8,16',
        minute=5,
        id='check_funding_payments',
        replace_existing=True
    )

    scheduler.start()
    print("Background scheduler started")
    print("  - Price update: every 30 seconds (UI only)")
    print("  - Funding rate: 00:05, 08:05, 16:05 UTC (when Z-score changes)")
    print("  - Funding payments: 00:05, 08:05, 16:05 UTC")

    # Initial data fetch (skip if API unavailable)
    print("Fetching initial data...")
    try:
        update_funding_rate()
    except Exception as e:
        print(f"Warning: Could not fetch initial data: {e}")


def check_funding_payments():
    """
    Check and record funding payments for open positions.

    This runs 5 minutes after each funding time (00:05, 08:05, 16:05 UTC)
    to ensure the funding has been settled before recording.
    """
    settings = get_settings()
    symbol = settings['symbol']
    client = get_okx_client()

    # Get current funding rate
    funding_data = client.get_funding_rate(symbol)
    if not funding_data:
        print("Failed to fetch funding rate for payment check")
        return

    # Get open trades
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM trades WHERE symbol = ? AND status = 'open'
    ''', (symbol,))
    open_trades = cursor.fetchall()
    conn.close()

    if not open_trades:
        return

    print(f"Recording funding payments for {len(open_trades)} open position(s)")

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

def main():
    """Main entry point for the trading system."""
    print("=" * 60)
    print("OKX Funding Rate Mean Reversion Trading System")
    print("=" * 60)

    # Initialize database
    print("\nInitializing database...")
    init_db()

    # Update OKX client with settings
    print("Configuring OKX client...")
    update_okx_client()

    # Bootstrap historical data (skip if API unavailable)
    settings = get_settings()
    print(f"\nBootstrapping funding rate history for {settings['symbol']}...")
    try:
        bootstrap_funding_history(settings['symbol'], settings['lookback_periods'])
    except Exception as e:
        print(f"Warning: Could not bootstrap historical data: {e}")
        print("The app will start without historical data. Data will be fetched when API is available.")

    # Start scheduler
    print("\nStarting background scheduler...")
    start_scheduler()

    # Display configuration
    print("\n" + "-" * 60)
    print("Configuration:")
    print(f"  Symbol: {settings['symbol']}")
    print(f"  Lookback: {settings['lookback_periods']} periods ({settings['lookback_periods'] * 8 / 24:.0f} days)")
    print(f"  Entry threshold: ±{settings['entry_std_dev']}σ")
    print(f"  Exit threshold: ±{settings['exit_std_dev']}σ")
    print(f"  Stop loss: ±{settings['stop_loss_std_dev']}σ")
    print(f"  Position size: {settings['position_size']} contracts")
    print(f"  Mode: {'PAPER TRADING' if settings['paper_mode'] else 'LIVE TRADING'}")
    print("-" * 60)

    # Run Flask app
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print(f"\nStarting web server on http://{host}:{port}")
    print("Press Ctrl+C to stop\n")

    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    main()

"""
Database functions - SQLite database management
"""

import os
import sqlite3
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Default settings
DEFAULT_SETTINGS = {
    'symbol': os.environ.get('TRADING_SYMBOL', 'BTC-USDT-SWAP'),
    'lookback_periods': int(os.environ.get('LOOKBACK_PERIODS', 90)),
    'entry_std_dev': float(os.environ.get('ENTRY_STD_DEV', 2.0)),
    'exit_std_dev': float(os.environ.get('EXIT_STD_DEV', 0.2)),
    'stop_loss_std_dev': float(os.environ.get('STOP_LOSS_STD_DEV', 6.0)),
    'position_size': float(os.environ.get('POSITION_SIZE', 1.0)),
    'leverage': int(os.environ.get('LEVERAGE', 2)),
    'paper_mode': os.environ.get('PAPER_MODE', 'true').lower() == 'true',
    'api_key': os.environ.get('OKX_API_KEY', ''),
    'api_secret': os.environ.get('OKX_API_SECRET', ''),
    'api_passphrase': os.environ.get('OKX_PASSPHRASE', ''),
    'maker_fee': float(os.environ.get('MAKER_FEE', 0.02)),
    'taker_fee': float(os.environ.get('TAKER_FEE', 0.05)),
    'estimated_slippage': float(os.environ.get('ESTIMATED_SLIPPAGE', 0.01)),
}

DATABASE_PATH = os.environ.get('DATABASE_PATH', 'funding_rate.db')


def get_db_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY,
            symbol TEXT DEFAULT 'BTC-USDT-SWAP',
            lookback_periods INTEGER DEFAULT 90,
            entry_std_dev REAL DEFAULT 2.0,
            exit_std_dev REAL DEFAULT 0.2,
            stop_loss_std_dev REAL DEFAULT 6.0,
            position_size REAL DEFAULT 1.0,
            leverage INTEGER DEFAULT 2,
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

    # Add columns if they don't exist (for existing databases)
    for col, default in [('maker_fee', 0.02), ('taker_fee', 0.05), ('estimated_slippage', 0.01)]:
        try:
            cursor.execute(f'ALTER TABLE settings ADD COLUMN {col} REAL DEFAULT {default}')
        except sqlite3.OperationalError:
            pass

    # Add leverage column for existing databases
    try:
        cursor.execute('ALTER TABLE settings ADD COLUMN leverage INTEGER DEFAULT 2')
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, funding_time)
        )
    ''')

    # Trades table
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

    # Add fee columns to trades if they don't exist
    for col in ['fee_cost', 'slippage_cost', 'net_pnl']:
        try:
            cursor.execute(f'ALTER TABLE trades ADD COLUMN {col} REAL DEFAULT 0')
        except sqlite3.OperationalError:
            pass

    # Funding payments table
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

    # Initialize default settings if not exists
    cursor.execute('SELECT COUNT(*) FROM settings')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO settings (symbol, lookback_periods, entry_std_dev,
                                  exit_std_dev, stop_loss_std_dev, position_size, leverage,
                                  paper_mode, api_key, api_secret, api_passphrase,
                                  maker_fee, taker_fee, estimated_slippage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            DEFAULT_SETTINGS['symbol'],
            DEFAULT_SETTINGS['lookback_periods'],
            DEFAULT_SETTINGS['entry_std_dev'],
            DEFAULT_SETTINGS['exit_std_dev'],
            DEFAULT_SETTINGS['stop_loss_std_dev'],
            DEFAULT_SETTINGS['position_size'],
            DEFAULT_SETTINGS['leverage'],
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
        row_dict = dict(row)
        return {
            'id': row['id'],
            'symbol': row['symbol'],
            'lookback_periods': row['lookback_periods'],
            'entry_std_dev': row['entry_std_dev'],
            'exit_std_dev': row['exit_std_dev'],
            'stop_loss_std_dev': row['stop_loss_std_dev'],
            'position_size': row['position_size'],
            'leverage': row_dict.get('leverage', DEFAULT_SETTINGS['leverage']),
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
                leverage = ?,
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
            settings.get('leverage', DEFAULT_SETTINGS['leverage']),
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

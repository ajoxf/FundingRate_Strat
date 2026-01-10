"""
Trading logic - Z-score calculation, trade execution, funding payments
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from .database import get_db_connection, get_settings, DEFAULT_SETTINGS
from .okx_client import get_okx_client


def calculate_trade_costs(entry_price: float, exit_price: float, size: float,
                          settings: Dict, use_maker: bool = False) -> Dict:
    """Calculate trading costs including fees and slippage."""
    fee_rate = settings.get('maker_fee', 0.02) if use_maker else settings.get('taker_fee', 0.05)
    slippage_rate = settings.get('estimated_slippage', 0.01)

    entry_notional = entry_price * size
    exit_notional = exit_price * size

    entry_fee = entry_notional * (fee_rate / 100)
    exit_fee = exit_notional * (fee_rate / 100)
    total_fee = entry_fee + exit_fee

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


def calculate_liquidation_price(entry_price: float, leverage: int, side: str,
                                 maintenance_margin: float = 0.5) -> float:
    """
    Calculate estimated liquidation price.

    Args:
        entry_price: Position entry price
        leverage: Leverage multiplier (e.g., 2, 3, 5, 10)
        side: 'long' or 'short'
        maintenance_margin: Maintenance margin rate (default 0.5% for OKX BTC)

    Returns:
        Estimated liquidation price
    """
    # Liquidation occurs when losses exceed margin minus maintenance margin
    # For cross margin: liq_price = entry * (1 ± (1/leverage) * (1 - maintenance_margin))
    margin_ratio = 1 / leverage
    buffer = 1 - (maintenance_margin / 100)  # ~99.5% of margin can be lost

    if side == 'long':
        # Long liquidates when price drops
        liq_price = entry_price * (1 - margin_ratio * buffer)
    else:
        # Short liquidates when price rises
        liq_price = entry_price * (1 + margin_ratio * buffer)

    return liq_price


def calculate_leveraged_metrics(entry_price: float, current_price: float, size: float,
                                 side: str, leverage: int, settings: Dict) -> Dict:
    """
    Calculate leveraged position metrics including ROI on margin.

    Returns:
        Dict with margin, notional, pnl, roi_on_margin, liquidation_price, etc.
    """
    notional = entry_price * size
    margin = notional / leverage

    # Calculate P&L
    if side == 'long':
        price_pnl = (current_price - entry_price) * size
    else:
        price_pnl = (entry_price - current_price) * size

    # Calculate costs
    costs = calculate_trade_costs(entry_price, current_price, size, settings)

    # Net P&L
    net_pnl = price_pnl - costs['total_cost']

    # ROI on margin (leveraged return)
    roi_on_margin = (net_pnl / margin) * 100 if margin > 0 else 0

    # ROI on notional (unleveraged return)
    roi_on_notional = (net_pnl / notional) * 100 if notional > 0 else 0

    # Liquidation price
    liq_price = calculate_liquidation_price(entry_price, leverage, side)

    # Distance to liquidation
    if side == 'long':
        liq_distance_pct = ((entry_price - liq_price) / entry_price) * 100
        price_to_liq_pct = ((current_price - liq_price) / current_price) * 100
    else:
        liq_distance_pct = ((liq_price - entry_price) / entry_price) * 100
        price_to_liq_pct = ((liq_price - current_price) / current_price) * 100

    # Warning level based on distance to liquidation
    if price_to_liq_pct < 5:
        liq_warning = 'CRITICAL'
    elif price_to_liq_pct < 10:
        liq_warning = 'HIGH'
    elif price_to_liq_pct < 20:
        liq_warning = 'MEDIUM'
    else:
        liq_warning = 'LOW'

    return {
        'notional': notional,
        'margin': margin,
        'leverage': leverage,
        'price_pnl': price_pnl,
        'net_pnl': net_pnl,
        'roi_on_margin': roi_on_margin,
        'roi_on_notional': roi_on_notional,
        'liquidation_price': liq_price,
        'liq_distance_pct': liq_distance_pct,
        'price_to_liq_pct': price_to_liq_pct,
        'liq_warning': liq_warning,
        'fee_cost': costs['fee_cost'],
        'slippage_cost': costs['slippage_cost'],
    }


def calculate_z_score(symbol: str, lookback_periods: int) -> Optional[Dict]:
    """Calculate current Z-score for funding rate."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT funding_rate FROM funding_rate_history
        WHERE symbol = ?
        ORDER BY funding_time DESC
        LIMIT ?
    ''', (symbol, lookback_periods))

    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 10:
        return None

    rates = [row['funding_rate'] for row in rows]
    current_rate = rates[0]
    mean_rate = sum(rates) / len(rates)
    variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
    std_dev = variance ** 0.5

    if std_dev == 0:
        return None

    z_score = (current_rate - mean_rate) / std_dev

    return {
        'z_score': z_score,
        'current_rate': current_rate,
        'mean_rate': mean_rate,
        'std_dev': std_dev,
        'data_points': len(rates),
    }


def get_open_trade(symbol: str) -> Optional[Dict]:
    """Get current open trade for symbol."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM trades
        WHERE symbol = ? AND status = 'open'
        ORDER BY entry_time DESC
        LIMIT 1
    ''', (symbol,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def execute_open_trade(symbol: str, side: str, price: float, funding_rate: float,
                       z_score: float, settings: Dict) -> Optional[int]:
    """Open a new trade."""
    conn = get_db_connection()
    cursor = conn.cursor()

    paper_mode = settings.get('paper_mode', True)
    position_size = settings.get('position_size', 1.0)

    # Place order if not paper trading
    if not paper_mode:
        client = get_okx_client()
        if client:
            order_side = 'buy' if side == 'long' else 'sell'
            result = client.place_order(symbol, order_side, position_size)
            if not result.get('success'):
                print(f"Order failed: {result.get('message')}")
                return None

    cursor.execute('''
        INSERT INTO trades (symbol, side, entry_price, entry_funding_rate,
                           entry_z_score, position_size, status, entry_time, paper_trade)
        VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
    ''', (
        symbol, side, price, funding_rate, z_score, position_size,
        datetime.now(timezone.utc).isoformat(),
        1 if paper_mode else 0
    ))

    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()

    prefix = '[PAPER] ' if paper_mode else ''
    print(f"{prefix}Opened {side.upper()} position at ${price:.2f}, Z-score: {z_score:.2f}")

    return trade_id


def execute_close_trade(trade_id: int, reason: str) -> bool:
    """Close an existing trade."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM trades WHERE id = ?', (trade_id,))
    trade = cursor.fetchone()

    if not trade:
        conn.close()
        return False

    trade = dict(trade)
    symbol = trade['symbol']
    side = trade['side']
    entry_price = trade['entry_price']
    size = trade['position_size']

    # Get current price and funding data
    client = get_okx_client()
    ticker = client.get_ticker(symbol) if client else None
    price = float(ticker['last']) if ticker else entry_price

    funding_data = client.get_funding_rate(symbol) if client else None
    funding_rate = float(funding_data['fundingRate']) if funding_data else 0

    z_data = calculate_z_score(symbol, 90)
    z_score = z_data['z_score'] if z_data else 0

    # Place closing order if not paper trading
    if not trade['paper_trade']:
        if client:
            order_side = 'sell' if side == 'long' else 'buy'
            result = client.place_order(symbol, order_side, size)
            if not result.get('success'):
                print(f"Close order failed: {result.get('message')}")

    # Calculate P&L
    if side == 'long':
        price_pnl = (price - entry_price) * size
    else:
        price_pnl = (entry_price - price) * size

    # Get funding payments
    cursor.execute('SELECT SUM(payment_amount) as total FROM funding_payments WHERE trade_id = ?', (trade_id,))
    result = cursor.fetchone()
    funding_pnl = result['total'] if result['total'] else 0

    # Calculate costs
    settings = get_settings()
    costs = calculate_trade_costs(entry_price, price, size, settings)
    fee_cost = costs['fee_cost']
    slippage_cost = costs['slippage_cost']

    total_pnl = price_pnl + funding_pnl
    net_pnl = total_pnl - fee_cost - slippage_cost

    # Count funding periods
    cursor.execute('SELECT COUNT(*) as periods FROM funding_payments WHERE trade_id = ?', (trade_id,))
    periods = cursor.fetchone()['periods']

    # Update trade
    cursor.execute('''
        UPDATE trades SET
            exit_price = ?, exit_funding_rate = ?, exit_z_score = ?,
            price_pnl = ?, funding_pnl = ?, fee_cost = ?, slippage_cost = ?,
            total_pnl = ?, net_pnl = ?, funding_periods_held = ?,
            status = 'closed', exit_time = ?, exit_reason = ?
        WHERE id = ?
    ''', (
        price, funding_rate, z_score,
        price_pnl, funding_pnl, fee_cost, slippage_cost, total_pnl, net_pnl, periods,
        datetime.now(timezone.utc).isoformat(), reason, trade_id
    ))

    conn.commit()
    conn.close()

    prefix = '[PAPER] ' if trade['paper_trade'] else ''
    print(f"{prefix}Closed {side.upper()} at ${price:.2f}")
    print(f"  Net P&L: ${net_pnl:.2f} (Fees: ${fee_cost:.2f})")

    return True


def record_funding_payment(trade_id: int, symbol: str, funding_rate: float,
                           position_size: float, side: str):
    """Record a funding payment for an open position."""
    # Longs pay when positive, receive when negative
    # Shorts receive when positive, pay when negative
    if side == 'long':
        payment = -funding_rate * position_size * 100  # Negative = paid
    else:
        payment = funding_rate * position_size * 100  # Positive = received

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO funding_payments (trade_id, symbol, funding_rate, payment_amount, funding_time)
        VALUES (?, ?, ?, ?, ?)
    ''', (trade_id, symbol, funding_rate, payment, datetime.now(timezone.utc).isoformat()))

    conn.commit()
    conn.close()


def bootstrap_funding_history(symbol: str, periods: int = 180) -> int:
    """Fetch and store historical funding rate data."""
    client = get_okx_client()
    if not client:
        return 0

    rates = client.get_funding_rate_history_all(symbol, periods)
    if not rates:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    count = 0
    for rate in rates:
        funding_time = datetime.fromtimestamp(rate['funding_time'] / 1000, tz=timezone.utc)
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO funding_rate_history
                (symbol, funding_rate, funding_time, realized_rate)
                VALUES (?, ?, ?, ?)
            ''', (symbol, rate['funding_rate'], funding_time.isoformat(), rate.get('realized_rate')))
            count += 1
        except Exception:
            pass

    conn.commit()
    conn.close()

    return count

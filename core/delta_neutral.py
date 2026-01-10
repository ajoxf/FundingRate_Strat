"""
Delta-Neutral Funding Rate Strategy

Strategy: Long spot + Short perpetual = Zero directional exposure
Profit: Collect positive funding rates (shorts receive when funding is positive)

Capital Structure:
- Spot: Full notional (e.g., $10,000 worth of BTC)
- Perp: Margin only (e.g., at 3x leverage, only $3,333 margin needed)
- Total capital: Spot + Perp margin

Returns:
- Pure funding rate collection
- No liquidation risk (spot covers perp losses)
- Expected: 10-20% annually in bull markets
"""

from datetime import datetime, timezone
from typing import Dict, Optional
from .database import get_db_connection, get_settings


def calculate_delta_neutral_metrics(
    spot_size: float,
    perp_size: float,
    spot_price: float,
    perp_price: float,
    perp_leverage: int,
    funding_rate: float,
    settings: Dict
) -> Dict:
    """
    Calculate metrics for a delta-neutral position.

    Args:
        spot_size: Amount of spot BTC held (positive = long)
        perp_size: Amount of perp BTC (positive = short position size)
        spot_price: Current spot price
        perp_price: Current perpetual price
        perp_leverage: Leverage used on perpetual
        funding_rate: Current funding rate
        settings: Trading settings

    Returns:
        Dict with position metrics
    """
    # Notional values
    spot_notional = spot_size * spot_price
    perp_notional = perp_size * perp_price

    # Delta calculation (should be ~0 for delta-neutral)
    # Long spot = positive delta, Short perp = negative delta
    net_delta = spot_size - perp_size
    delta_pct = (net_delta / spot_size * 100) if spot_size > 0 else 0

    # Capital required
    perp_margin = perp_notional / perp_leverage
    total_capital = spot_notional + perp_margin

    # Basis (spot vs perp price difference)
    basis = perp_price - spot_price
    basis_pct = (basis / spot_price) * 100

    # Funding payment (per 8-hour period)
    # Positive funding: shorts receive, longs pay
    funding_payment_8h = funding_rate * perp_notional
    funding_payment_daily = funding_payment_8h * 3
    funding_payment_annual = funding_payment_daily * 365

    # APY calculation
    funding_apy = (funding_payment_annual / total_capital) * 100

    # Annualized rate display
    annualized_rate = funding_rate * 3 * 365 * 100  # As percentage

    return {
        # Position info
        'spot_size': spot_size,
        'perp_size': perp_size,
        'spot_notional': spot_notional,
        'perp_notional': perp_notional,

        # Delta
        'net_delta': net_delta,
        'delta_pct': delta_pct,
        'is_delta_neutral': abs(delta_pct) < 1,  # Within 1% is considered neutral

        # Capital
        'perp_margin': perp_margin,
        'total_capital': total_capital,
        'perp_leverage': perp_leverage,

        # Basis
        'basis': basis,
        'basis_pct': basis_pct,

        # Funding
        'funding_rate': funding_rate,
        'funding_rate_pct': funding_rate * 100,
        'funding_payment_8h': funding_payment_8h,
        'funding_payment_daily': funding_payment_daily,
        'funding_apy': funding_apy,
        'annualized_rate': annualized_rate,
    }


def calculate_delta_neutral_pnl(
    entry_spot_price: float,
    entry_perp_price: float,
    current_spot_price: float,
    current_perp_price: float,
    spot_size: float,
    perp_size: float,
    total_funding_received: float,
    settings: Dict
) -> Dict:
    """
    Calculate P&L for a delta-neutral position.

    The beauty: Spot P&L and Perp P&L should largely cancel out,
    leaving only funding as profit.
    """
    # Spot P&L (long position)
    spot_pnl = (current_spot_price - entry_spot_price) * spot_size

    # Perp P&L (short position)
    perp_pnl = (entry_perp_price - current_perp_price) * perp_size

    # Price P&L (should be near zero if delta-neutral)
    price_pnl = spot_pnl + perp_pnl

    # Total P&L
    total_pnl = price_pnl + total_funding_received

    # Trading costs
    fee_rate = settings.get('taker_fee', 0.05) / 100
    spot_fees = (entry_spot_price * spot_size + current_spot_price * spot_size) * fee_rate
    perp_fees = (entry_perp_price * perp_size + current_perp_price * perp_size) * fee_rate
    total_fees = spot_fees + perp_fees

    net_pnl = total_pnl - total_fees

    return {
        'spot_pnl': spot_pnl,
        'perp_pnl': perp_pnl,
        'price_pnl': price_pnl,
        'funding_pnl': total_funding_received,
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'net_pnl': net_pnl,
    }


def estimate_delta_neutral_returns(
    capital: float,
    avg_funding_rate: float,
    perp_leverage: int = 3,
    days: int = 365
) -> Dict:
    """
    Estimate returns for delta-neutral strategy.

    Args:
        capital: Total capital to deploy
        avg_funding_rate: Average expected funding rate (e.g., 0.0003 = 0.03%)
        perp_leverage: Leverage on perpetual side
        days: Number of days to project

    Example with $10,000:
        - Spot allocation: $7,500 (buys 0.075 BTC at $100k)
        - Perp margin: $2,500 (at 3x, controls 0.075 BTC short)
        - Position size: 0.075 BTC both sides
    """
    # Capital split
    # spot_notional + perp_margin = capital
    # spot_notional = perp_notional (same size)
    # perp_margin = perp_notional / leverage
    # So: spot_notional + spot_notional/leverage = capital
    # spot_notional * (1 + 1/leverage) = capital
    # spot_notional = capital / (1 + 1/leverage)

    spot_allocation_ratio = 1 / (1 + 1/perp_leverage)
    spot_notional = capital * spot_allocation_ratio
    perp_margin = capital - spot_notional

    # Funding calculations
    funding_per_8h = avg_funding_rate * spot_notional
    funding_per_day = funding_per_8h * 3
    funding_total = funding_per_day * days

    # Returns
    gross_return = funding_total
    gross_return_pct = (gross_return / capital) * 100

    # Estimate costs (entry + exit)
    trading_costs = spot_notional * 0.001 * 2  # ~0.1% round trip

    net_return = gross_return - trading_costs
    net_return_pct = (net_return / capital) * 100

    # APY
    apy = (net_return_pct / days) * 365

    return {
        'capital': capital,
        'spot_notional': spot_notional,
        'perp_margin': perp_margin,
        'perp_leverage': perp_leverage,

        'avg_funding_rate': avg_funding_rate,
        'funding_per_8h': funding_per_8h,
        'funding_per_day': funding_per_day,
        'funding_total': funding_total,

        'gross_return': gross_return,
        'gross_return_pct': gross_return_pct,
        'trading_costs': trading_costs,
        'net_return': net_return,
        'net_return_pct': net_return_pct,
        'apy': apy,

        'days': days,
    }


def get_recommended_position_size(
    capital: float,
    spot_price: float,
    perp_leverage: int = 3
) -> Dict:
    """
    Calculate recommended position sizes for delta-neutral strategy.
    """
    # Split capital between spot and perp margin
    spot_allocation_ratio = 1 / (1 + 1/perp_leverage)
    spot_capital = capital * spot_allocation_ratio
    perp_capital = capital - spot_capital

    # Position size in BTC
    position_size = spot_capital / spot_price

    # Perp notional (same as spot)
    perp_notional = position_size * spot_price

    return {
        'total_capital': capital,
        'spot_capital': spot_capital,
        'perp_capital': perp_capital,
        'position_size_btc': position_size,
        'spot_notional': spot_capital,
        'perp_notional': perp_notional,
        'perp_leverage': perp_leverage,
        'capital_efficiency': (perp_notional / capital) * 100,  # How much exposure per dollar
    }

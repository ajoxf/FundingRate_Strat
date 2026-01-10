"""
Core module - Database, OKX client, and trading logic
Shared between Flask and Streamlit UIs
"""

from .database import (
    init_db,
    get_db_connection,
    get_settings,
    save_settings,
    DEFAULT_SETTINGS,
)

from .trading import (
    calculate_z_score,
    get_open_trade,
    execute_open_trade,
    execute_close_trade,
    record_funding_payment,
    bootstrap_funding_history,
    calculate_trade_costs,
    calculate_liquidation_price,
    calculate_leveraged_metrics,
)

from .okx_client import (
    OKXClient,
    OKXClientError,
    get_okx_client,
    create_okx_client,
)

from .delta_neutral import (
    calculate_delta_neutral_metrics,
    calculate_delta_neutral_pnl,
    estimate_delta_neutral_returns,
    get_recommended_position_size,
)

__all__ = [
    'init_db',
    'get_db_connection',
    'get_settings',
    'save_settings',
    'DEFAULT_SETTINGS',
    'calculate_z_score',
    'get_open_trade',
    'execute_open_trade',
    'execute_close_trade',
    'record_funding_payment',
    'bootstrap_funding_history',
    'calculate_trade_costs',
    'calculate_liquidation_price',
    'calculate_leveraged_metrics',
    'OKXClient',
    'OKXClientError',
    'get_okx_client',
    'create_okx_client',
    'calculate_delta_neutral_metrics',
    'calculate_delta_neutral_pnl',
    'estimate_delta_neutral_returns',
    'get_recommended_position_size',
]

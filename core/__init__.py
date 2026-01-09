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
)

from .okx_client import (
    OKXClient,
    OKXClientError,
    get_okx_client,
    create_okx_client,
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
    'OKXClient',
    'OKXClientError',
    'get_okx_client',
    'create_okx_client',
]

"""
OKX Funding Rate Mean Reversion Trading System - Streamlit UI

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import time

# Import backend functions from core module (Flask-independent)
from core import (
    init_db,
    get_settings,
    save_settings,
    get_db_connection,
    get_okx_client,
    calculate_z_score,
    get_open_trade,
    execute_open_trade,
    execute_close_trade,
    bootstrap_funding_history,
    calculate_trade_costs,
    DEFAULT_SETTINGS,
)

# Page configuration
st.set_page_config(
    page_title="Funding Rate Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        color: #888;
        font-size: 12px;
    }
    .positive { color: #00ff88; }
    .negative { color: #ff4444; }
    .neutral { color: #888; }
    .signal-long { background-color: rgba(0, 255, 136, 0.1); border-left: 4px solid #00ff88; }
    .signal-short { background-color: rgba(255, 68, 68, 0.1); border-left: 4px solid #ff4444; }
    .signal-none { background-color: rgba(136, 136, 136, 0.1); border-left: 4px solid #888; }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_db()


def get_funding_history(symbol: str, limit: int = 90) -> pd.DataFrame:
    """Get funding rate history from database."""
    conn = get_db_connection()
    query = """
        SELECT funding_rate, funding_time
        FROM funding_rate_history
        WHERE symbol = ?
        ORDER BY funding_time DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(symbol, limit))
    conn.close()
    if not df.empty:
        df['funding_time'] = pd.to_datetime(df['funding_time'])
        df = df.sort_values('funding_time')
    return df


def get_trades(symbol: str, status: str = None, limit: int = 50) -> pd.DataFrame:
    """Get trades from database."""
    conn = get_db_connection()
    query = """
        SELECT * FROM trades
        WHERE symbol = ?
    """
    params = [symbol]
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY entry_time DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_stats(symbol: str) -> dict:
    """Get trading statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(total_pnl) as total_pnl,
            SUM(price_pnl) as total_price_pnl,
            SUM(funding_pnl) as total_funding_pnl,
            SUM(COALESCE(fee_cost, 0)) as total_fees,
            SUM(COALESCE(slippage_cost, 0)) as total_slippage,
            SUM(COALESCE(net_pnl, total_pnl)) as total_net_pnl,
            AVG(total_pnl) as avg_pnl,
            MAX(COALESCE(net_pnl, total_pnl)) as best_trade,
            MIN(COALESCE(net_pnl, total_pnl)) as worst_trade
        FROM trades
        WHERE symbol = ? AND status = 'closed'
    ''', (symbol,))

    row = cursor.fetchone()
    conn.close()

    stats = dict(row) if row else {}
    if stats.get('total_trades') and stats['total_trades'] > 0:
        stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
    else:
        stats['win_rate'] = 0
    return stats


def get_current_data(settings: dict) -> dict:
    """Fetch current market data from OKX."""
    client = get_okx_client()
    if not client:
        return None

    symbol = settings['symbol']

    try:
        # Get current funding rate
        funding_data = client.get_funding_rate(symbol)
        if not funding_data:
            return None

        current_rate = float(funding_data.get('fundingRate', 0))
        next_funding_time = int(funding_data.get('nextFundingTime', 0))

        # Get current price
        ticker = client.get_ticker(symbol)
        current_price = float(ticker.get('last', 0)) if ticker else 0

        # Calculate Z-score
        z_data = calculate_z_score(symbol, settings['lookback_periods'])

        # Calculate time until next funding
        now = datetime.now(timezone.utc)
        next_funding = datetime.fromtimestamp(next_funding_time / 1000, tz=timezone.utc)
        time_diff = next_funding - now
        hours, remainder = divmod(int(time_diff.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        countdown = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            'symbol': symbol,
            'current_price': current_price,
            'funding_rate': current_rate,
            'funding_rate_pct': current_rate * 100,
            'annualized_rate': current_rate * 100 * 3 * 365,
            'z_score': z_data.get('z_score', 0) if z_data else 0,
            'mean_rate': z_data.get('mean_rate', 0) if z_data else 0,
            'std_dev': z_data.get('std_dev', 0) if z_data else 0,
            'data_points': z_data.get('data_points', 0) if z_data else 0,
            'countdown': countdown,
            'next_funding_time': next_funding,
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def render_dashboard():
    """Render the main dashboard page."""
    settings = get_settings()

    st.title(f"📈 {settings['symbol']}")

    # Paper mode badge
    if settings['paper_mode']:
        st.warning("📝 **Paper Trading Mode** - Trades are simulated")
    else:
        st.error("⚠️ **Live Trading Mode** - Real orders will be placed!")

    # Fetch current data
    with st.spinner("Fetching market data..."):
        data = get_current_data(settings)

    if not data:
        st.error("Unable to fetch market data. Check your API connection.")
        if st.button("🔄 Retry"):
            st.rerun()
        return

    # Top row: Price and Funding info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Swap Price", f"${data['current_price']:,.2f}")

    with col2:
        rate_color = "normal" if data['funding_rate'] >= 0 else "inverse"
        st.metric("Funding Rate", f"{data['funding_rate_pct']:.4f}%",
                  delta=f"Annualized: {data['annualized_rate']:.1f}%",
                  delta_color=rate_color)

    with col3:
        st.metric("Next Funding", data['countdown'])

    with col4:
        st.metric("Data Points", f"{data['data_points']} periods")

    st.divider()

    # Z-Score Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Z-Score Analysis")

        z_score = data['z_score']
        entry_threshold = settings['entry_std_dev']
        exit_threshold = settings['exit_std_dev']
        stop_loss = settings['stop_loss_std_dev']

        # Determine signal
        signal = "none"
        signal_text = "No Signal"
        if z_score >= entry_threshold:
            signal = "short"
            signal_text = "🔴 SHORT Signal (Funding High)"
        elif z_score <= -entry_threshold:
            signal = "long"
            signal_text = "🟢 LONG Signal (Funding Low)"
        elif abs(z_score) <= exit_threshold:
            signal_text = "⚪ Near Mean"

        # Display Z-score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=z_score,
            title={'text': "Current Z-Score"},
            gauge={
                'axis': {'range': [-6, 6]},
                'bar': {'color': "#00ff88" if z_score < 0 else "#ff4444"},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [-6, -entry_threshold], 'color': "rgba(0, 255, 136, 0.3)"},
                    {'range': [-entry_threshold, -exit_threshold], 'color': "rgba(136, 136, 136, 0.1)"},
                    {'range': [-exit_threshold, exit_threshold], 'color': "rgba(88, 166, 255, 0.2)"},
                    {'range': [exit_threshold, entry_threshold], 'color': "rgba(136, 136, 136, 0.1)"},
                    {'range': [entry_threshold, 6], 'color': "rgba(255, 68, 68, 0.3)"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': z_score
                }
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(signal_text)

    with col2:
        st.subheader("Signal Thresholds")
        st.write(f"🔴 **SHORT Entry**: ≥ +{entry_threshold}σ")
        st.write(f"🟢 **LONG Entry**: ≤ -{entry_threshold}σ")
        st.write(f"⚪ **Exit**: ±{exit_threshold}σ")
        st.write(f"🛑 **Stop Loss**: ±{stop_loss}σ")

        st.divider()

        st.write(f"**Mean Rate**: {data['mean_rate']*100:.4f}%")
        st.write(f"**Std Dev**: {data['std_dev']*100:.4f}%")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Z-Score History")
        history = get_funding_history(settings['symbol'], settings['lookback_periods'])

        if not history.empty:
            # Calculate Z-scores for history
            mean = history['funding_rate'].mean()
            std = history['funding_rate'].std()
            if std > 0:
                history['z_score'] = (history['funding_rate'] - mean) / std
            else:
                history['z_score'] = 0

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history['funding_time'],
                y=history['z_score'],
                mode='lines',
                name='Z-Score',
                line=dict(color='#58a6ff')
            ))
            # Add threshold lines
            fig.add_hline(y=entry_threshold, line_dash="dash", line_color="#ff4444",
                         annotation_text="Short Entry")
            fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="#00ff88",
                         annotation_text="Long Entry")
            fig.add_hline(y=0, line_dash="dot", line_color="#888")

            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#333'),
                yaxis=dict(gridcolor='#333'),
                font={'color': 'white'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data. Click 'Fetch Historical Data' in Settings.")

    with col2:
        st.subheader("Funding Rate History")
        if not history.empty:
            colors = ['#00ff88' if x >= 0 else '#ff4444' for x in history['funding_rate']]
            fig = go.Figure(go.Bar(
                x=history['funding_time'],
                y=history['funding_rate'] * 100,
                marker_color=colors,
                name='Funding Rate %'
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#333'),
                yaxis=dict(gridcolor='#333', title='Rate %'),
                font={'color': 'white'}
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Current Position
    st.subheader("Current Position")
    open_trade = get_open_trade(settings['symbol'])

    if open_trade:
        col1, col2, col3, col4 = st.columns(4)

        side_emoji = "🟢" if open_trade['side'] == 'long' else "🔴"

        with col1:
            st.metric("Side", f"{side_emoji} {open_trade['side'].upper()}")
        with col2:
            st.metric("Entry Price", f"${open_trade['entry_price']:,.2f}")
        with col3:
            st.metric("Size", open_trade['position_size'])
        with col4:
            st.metric("Entry Z-Score", f"{open_trade['entry_z_score']:.2f}σ")

        # Estimated costs
        costs = calculate_trade_costs(
            open_trade['entry_price'],
            data['current_price'],
            open_trade['position_size'],
            settings
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Est. Fees", f"-${costs['fee_cost']:.2f}")
        with col2:
            st.metric("Est. Slippage", f"-${costs['slippage_cost']:.2f}")
        with col3:
            st.metric("Total Est. Cost", f"-${costs['total_cost']:.2f}")

        st.caption(f"Opened: {open_trade['entry_time']}")

        if st.button("🔴 Close Position", type="primary"):
            with st.spinner("Closing position..."):
                execute_close_trade(open_trade['id'], "manual")
                st.success("Position closed!")
                time.sleep(1)
                st.rerun()
    else:
        st.info("No open position")

        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            if st.button("🟢 Open LONG", type="primary"):
                with st.spinner("Opening long position..."):
                    execute_open_trade(settings['symbol'], 'long', data['current_price'],
                                      data['funding_rate'], z_score, settings)
                    st.success("Long position opened!")
                    time.sleep(1)
                    st.rerun()
        with col2:
            if st.button("🔴 Open SHORT", type="secondary"):
                with st.spinner("Opening short position..."):
                    execute_open_trade(settings['symbol'], 'short', data['current_price'],
                                      data['funding_rate'], z_score, settings)
                    st.success("Short position opened!")
                    time.sleep(1)
                    st.rerun()

    st.divider()

    # Trading Stats
    st.subheader("Trading Statistics")
    stats = get_stats(settings['symbol'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", stats.get('total_trades', 0))
    with col2:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with col3:
        total_pnl = stats.get('total_pnl', 0) or 0
        st.metric("Gross P&L", f"${total_pnl:,.2f}",
                  delta_color="normal" if total_pnl >= 0 else "inverse")
    with col4:
        net_pnl = stats.get('total_net_pnl', 0) or 0
        st.metric("Net P&L", f"${net_pnl:,.2f}",
                  delta_color="normal" if net_pnl >= 0 else "inverse")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_fees = stats.get('total_fees', 0) or 0
        st.metric("Total Fees", f"-${total_fees:,.2f}")
    with col2:
        total_slip = stats.get('total_slippage', 0) or 0
        st.metric("Total Slippage", f"-${total_slip:,.2f}")
    with col3:
        best = stats.get('best_trade', 0) or 0
        st.metric("Best Trade", f"${best:,.2f}")
    with col4:
        worst = stats.get('worst_trade', 0) or 0
        st.metric("Worst Trade", f"${worst:,.2f}")


def render_trades():
    """Render the trades history page."""
    st.title("📊 Trade History")

    settings = get_settings()

    # Filters
    col1, col2 = st.columns([1, 3])
    with col1:
        status_filter = st.selectbox("Status", ["All", "Open", "Closed"])
    with col2:
        limit = st.slider("Show trades", 10, 200, 50)

    status = None if status_filter == "All" else status_filter.lower()

    # Get trades
    trades_df = get_trades(settings['symbol'], status, limit)

    if trades_df.empty:
        st.info("No trades found")
        return

    # Stats summary
    stats = get_stats(settings['symbol'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", stats.get('total_trades', 0))
    with col2:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with col3:
        best = stats.get('best_trade', 0) or 0
        st.metric("Best Trade", f"${best:,.2f}")
    with col4:
        worst = stats.get('worst_trade', 0) or 0
        st.metric("Worst Trade", f"${worst:,.2f}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_pnl = stats.get('total_pnl', 0) or 0
        st.metric("Gross P&L", f"${total_pnl:,.2f}")
    with col2:
        funding_pnl = stats.get('total_funding_pnl', 0) or 0
        st.metric("Funding P&L", f"${funding_pnl:,.2f}")
    with col3:
        total_fees = stats.get('total_fees', 0) or 0
        st.metric("Total Fees", f"-${total_fees:,.2f}")
    with col4:
        net_pnl = stats.get('total_net_pnl', 0) or 0
        st.metric("Net P&L", f"${net_pnl:,.2f}")

    st.divider()

    # Display trades table
    display_df = trades_df[[
        'id', 'side', 'entry_price', 'exit_price',
        'entry_z_score', 'exit_z_score',
        'price_pnl', 'funding_pnl', 'fee_cost', 'net_pnl',
        'status', 'exit_reason', 'entry_time'
    ]].copy()

    display_df.columns = [
        'ID', 'Side', 'Entry $', 'Exit $',
        'Entry Z', 'Exit Z',
        'Price P&L', 'Funding P&L', 'Fees', 'Net P&L',
        'Status', 'Exit Reason', 'Entry Time'
    ]

    # Format columns
    display_df['Entry $'] = display_df['Entry $'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "--")
    display_df['Exit $'] = display_df['Exit $'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "--")
    display_df['Entry Z'] = display_df['Entry Z'].apply(lambda x: f"{x:.2f}σ" if pd.notna(x) else "--")
    display_df['Exit Z'] = display_df['Exit Z'].apply(lambda x: f"{x:.2f}σ" if pd.notna(x) else "--")
    display_df['Price P&L'] = display_df['Price P&L'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    display_df['Funding P&L'] = display_df['Funding P&L'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    display_df['Fees'] = display_df['Fees'].apply(lambda x: f"-${x:,.2f}" if pd.notna(x) and x > 0 else "--")
    display_df['Net P&L'] = display_df['Net P&L'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    display_df['Side'] = display_df['Side'].apply(lambda x: f"{'🟢' if x == 'long' else '🔴'} {x.upper()}")
    display_df['Status'] = display_df['Status'].apply(lambda x: f"{'🟡' if x == 'open' else '🔵'} {x.upper()}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Actions for open trades
    open_trades = trades_df[trades_df['status'] == 'open']
    if not open_trades.empty:
        st.subheader("Open Trades Actions")
        for _, trade in open_trades.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Trade #{trade['id']}: {trade['side'].upper()} @ ${trade['entry_price']:,.2f}")
            with col2:
                if st.button(f"Close #{trade['id']}", key=f"close_{trade['id']}"):
                    execute_close_trade(trade['id'], "manual")
                    st.success(f"Trade #{trade['id']} closed!")
                    time.sleep(1)
                    st.rerun()


def render_settings():
    """Render the settings page."""
    st.title("⚙️ Settings")

    settings = get_settings()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strategy Settings")

        symbol = st.selectbox(
            "Symbol",
            ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"],
            index=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"].index(settings['symbol'])
        )

        lookback_periods = st.number_input(
            "Lookback Periods",
            min_value=10, max_value=500,
            value=settings['lookback_periods'],
            help="Number of 8-hour funding periods for mean calculation (90 = 30 days)"
        )

        entry_std_dev = st.number_input(
            "Entry Std Dev (σ)",
            min_value=0.5, max_value=5.0, step=0.1,
            value=float(settings['entry_std_dev']),
            help="Z-score threshold for entry signals"
        )

        exit_std_dev = st.number_input(
            "Exit Std Dev (σ)",
            min_value=0.0, max_value=1.0, step=0.1,
            value=float(settings['exit_std_dev']),
            help="Z-score threshold for exit (mean reversion complete)"
        )

        stop_loss_std_dev = st.number_input(
            "Stop Loss Std Dev (σ)",
            min_value=3.0, max_value=10.0, step=0.5,
            value=float(settings['stop_loss_std_dev']),
            help="Z-score threshold for stop loss"
        )

        position_size = st.number_input(
            "Position Size (Contracts)",
            min_value=0.001, step=0.001,
            value=float(settings['position_size']),
            help="Size of position to open"
        )

        paper_mode = st.toggle(
            "Paper Trading Mode",
            value=settings['paper_mode'],
            help="When enabled, trades are simulated"
        )

        if not paper_mode:
            st.warning("⚠️ Live trading mode! Real orders will be placed on OKX.")

        st.divider()
        st.subheader("Fee Settings")

        maker_fee = st.number_input(
            "Maker Fee (%)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(settings.get('maker_fee', 0.02)),
            help="Fee for limit orders (default: 0.02%)"
        )

        taker_fee = st.number_input(
            "Taker Fee (%)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(settings.get('taker_fee', 0.05)),
            help="Fee for market orders (default: 0.05%)"
        )

        estimated_slippage = st.number_input(
            "Estimated Slippage (%)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(settings.get('estimated_slippage', 0.01)),
            help="Expected slippage per trade"
        )

        st.info("Check your OKX VIP level at [okx.com/fees](https://www.okx.com/fees)")

        if st.button("💾 Save Strategy Settings", type="primary"):
            new_settings = {
                **settings,
                'symbol': symbol,
                'lookback_periods': lookback_periods,
                'entry_std_dev': entry_std_dev,
                'exit_std_dev': exit_std_dev,
                'stop_loss_std_dev': stop_loss_std_dev,
                'position_size': position_size,
                'paper_mode': paper_mode,
                'maker_fee': maker_fee,
                'taker_fee': taker_fee,
                'estimated_slippage': estimated_slippage,
            }
            if save_settings(new_settings):
                st.success("Settings saved!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to save settings")

    with col2:
        st.subheader("OKX API Credentials")

        api_key = st.text_input(
            "API Key",
            value=settings.get('api_key', ''),
            type="password"
        )

        api_secret = st.text_input(
            "API Secret",
            value=settings.get('api_secret', ''),
            type="password"
        )

        api_passphrase = st.text_input(
            "API Passphrase",
            value=settings.get('api_passphrase', ''),
            type="password"
        )

        st.warning("🔒 Only provide read and trade permissions. Never enable withdrawal!")

        if st.button("💾 Save API Credentials"):
            new_settings = {
                **settings,
                'api_key': api_key,
                'api_secret': api_secret,
                'api_passphrase': api_passphrase,
            }
            if save_settings(new_settings):
                st.success("API credentials saved!")
            else:
                st.error("Failed to save credentials")

        st.divider()
        st.subheader("Data Management")

        st.write("Bootstrap historical funding rate data from OKX.")

        if st.button("📥 Fetch Historical Data"):
            with st.spinner("Fetching historical data..."):
                result = bootstrap_funding_history(settings['symbol'])
                if result > 0:
                    st.success(f"Fetched {result} historical funding rates!")
                else:
                    st.error("Failed to fetch data. Check API connection.")

        st.divider()
        st.subheader("Strategy Information")

        st.write("**Funding Interval:** Every 8 hours")
        st.write("**Funding Times (UTC):** 00:00, 08:00, 16:00")
        st.write("**Data Update:** Every 1 minute")

        st.divider()
        st.subheader("Strategy Explanation")

        st.markdown("""
        **Mean Reversion Logic:**
        Funding rates on perpetual swaps tend to revert to their historical mean.
        When funding rate deviates significantly (measured by Z-score), we take a
        position expecting it to return to the mean.

        **Entry Signals:**
        - 🔴 **SHORT** when Z-score ≥ +2.0σ (funding unusually HIGH)
        - 🟢 **LONG** when Z-score ≤ -2.0σ (funding unusually LOW)

        **Funding Rate Payments:**
        - **Positive funding:** Longs pay shorts
        - **Negative funding:** Shorts pay longs

        **Exit Conditions:**
        - **Mean Reversion:** Z-score returns to ±0.2σ
        - **Stop Loss:** Z-score reaches ±6.0σ
        """)


def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["📈 Dashboard", "📊 Trades", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()

    # Quick stats in sidebar
    settings = get_settings()
    stats = get_stats(settings['symbol'])

    st.sidebar.subheader("Quick Stats")
    st.sidebar.metric("Symbol", settings['symbol'])
    st.sidebar.metric("Total Trades", stats.get('total_trades', 0))

    net_pnl = stats.get('total_net_pnl', 0) or 0
    st.sidebar.metric("Net P&L", f"${net_pnl:,.2f}")

    st.sidebar.divider()

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Auto-refresh toggle
    auto_refresh = st.sidebar.toggle("Auto-refresh (0.5s)", value=False)

    if auto_refresh:
        time.sleep(0.5)
        st.rerun()

    # Render selected page
    if page == "📈 Dashboard":
        render_dashboard()
    elif page == "📊 Trades":
        render_trades()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()

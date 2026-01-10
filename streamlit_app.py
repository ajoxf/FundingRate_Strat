"""
OKX Funding Rate Mean Reversion Trading System - Streamlit UI

Run with: streamlit run streamlit_app.py

NOTE: Streamlit Cloud apps sleep after inactivity. For 24/7 algo trading,
use the Flask version (trading_portal.py) on a local machine or VPS.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
    calculate_liquidation_price,
    calculate_leveraged_metrics,
    DEFAULT_SETTINGS,
)

# Page configuration
st.set_page_config(
    page_title="Funding Rate Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching Flask dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
    }
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #30363d;
    }
    .card-title {
        font-size: 16px;
        font-weight: 600;
        color: #e6edf3;
    }
    .stat-box {
        text-align: center;
        padding: 12px;
    }
    .stat-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: #e6edf3;
    }
    .stat-value.green { color: #3fb950; }
    .stat-value.red { color: #f85149; }
    .stat-value.blue { color: #58a6ff; }
    .stat-value.purple { color: #a371f7; }
    .signal-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid #30363d;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-green { background-color: rgba(63, 185, 80, 0.2); color: #3fb950; }
    .badge-red { background-color: rgba(248, 81, 73, 0.2); color: #f85149; }
    .badge-blue { background-color: rgba(88, 166, 255, 0.2); color: #58a6ff; }
    .badge-yellow { background-color: rgba(210, 153, 34, 0.2); color: #d29922; }
    .countdown {
        font-family: monospace;
        font-size: 20px;
        font-weight: 700;
        color: #e6edf3;
        background: #21262d;
        padding: 8px 16px;
        border-radius: 6px;
    }
    .divider {
        border-top: 1px solid #30363d;
        margin: 16px 0;
    }
    .threshold-table {
        width: 100%;
        border-collapse: collapse;
    }
    .threshold-table td {
        padding: 12px 8px;
        border-bottom: 1px solid #30363d;
        color: #e6edf3;
    }
    .text-muted { color: #8b949e; }
    .text-green { color: #3fb950; }
    .text-red { color: #f85149; }
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
    query = "SELECT * FROM trades WHERE symbol = ?"
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
            SUM(COALESCE(net_pnl, total_pnl)) as total_net_pnl
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
        funding_data = client.get_funding_rate(symbol)
        if not funding_data:
            return None

        current_rate = float(funding_data.get('fundingRate', 0))
        ticker = client.get_ticker(symbol)
        current_price = float(ticker.get('last', 0)) if ticker else 0

        # Get historical rates for Z-score
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT funding_rate FROM funding_rate_history
            WHERE symbol = ? ORDER BY funding_time DESC LIMIT ?
        ''', (symbol, settings['lookback_periods']))
        historical_rates = [row['funding_rate'] for row in cursor.fetchall()]
        conn.close()

        all_rates = [current_rate] + historical_rates
        data_points = len(all_rates)

        if len(all_rates) >= 10:
            mean_rate = sum(all_rates) / len(all_rates)
            variance = sum((r - mean_rate) ** 2 for r in all_rates) / len(all_rates)
            std_dev = variance ** 0.5
            z_score = (current_rate - mean_rate) / std_dev if std_dev > 0 else 0
        else:
            mean_rate = current_rate
            std_dev = 0
            z_score = 0

        # Calculate countdown to next funding (00:00, 08:00, 16:00 UTC)
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour
        funding_hours = [0, 8, 16, 24]
        next_funding_hour = next(h for h in funding_hours if h > current_hour)

        if next_funding_hour == 24:
            from datetime import timedelta
            next_funding_dt = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_funding_dt = now_utc.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)

        time_diff = next_funding_dt - now_utc
        hours, remainder = divmod(int(time_diff.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        countdown = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {
            'symbol': symbol,
            'current_price': current_price,
            'funding_rate': current_rate,
            'funding_rate_pct': current_rate * 100,
            'annualized_rate': current_rate * 100 * 3 * 365,
            'z_score': z_score,
            'mean_rate': mean_rate,
            'std_dev': std_dev,
            'data_points': data_points,
            'countdown': countdown,
            'funding_direction': 'POSITIVE' if current_rate >= 0 else 'NEGATIVE',
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def render_dashboard():
    """Render the main dashboard page - matching Flask layout."""
    settings = get_settings()
    data = get_current_data(settings)

    if not data:
        st.error("Unable to fetch market data. Check your API connection.")
        if st.button("🔄 Retry"):
            st.rerun()
        return

    # Two column layout like Flask
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # Symbol Header Card
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; color: #e6edf3; font-size: 24px;">{data['symbol']}</h2>
                    <span class="text-muted">Perpetual Swap</span>
                </div>
                <span class="badge badge-{'green' if data['funding_direction'] == 'POSITIVE' else 'red'}">{data['funding_direction']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Price & Funding Info Card
        ann_color = 'green' if data['annualized_rate'] >= 0 else 'red'
        st.markdown(f"""
        <div class="card">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; text-align: center;">
                <div class="stat-box">
                    <div class="stat-label">Swap Price</div>
                    <div class="stat-value">${data['current_price']:,.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Funding Rate</div>
                    <div class="stat-value">{data['funding_rate_pct']:.4f}%</div>
                    <div class="text-muted" style="font-size: 12px;">{data['funding_rate']:.6f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Annualized</div>
                    <div class="stat-value {ann_color}">{data['annualized_rate']:.2f}%</div>
                </div>
            </div>
            <div class="divider"></div>
            <div style="display: flex; justify-content: center; align-items: center; gap: 16px;">
                <span class="text-muted">Next Funding In:</span>
                <span class="countdown">{data['countdown']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Z-Score Analysis Card
        z_score = data['z_score']
        entry_threshold = settings['entry_std_dev']
        exit_threshold = settings['exit_std_dev']

        if z_score >= entry_threshold:
            signal_text = "SHORT SIGNAL"
            signal_class = "badge-red"
        elif z_score <= -entry_threshold:
            signal_text = "LONG SIGNAL"
            signal_class = "badge-green"
        elif abs(z_score) <= exit_threshold:
            signal_text = "NEAR MEAN"
            signal_class = "badge-blue"
        else:
            signal_text = "NO SIGNAL"
            signal_class = "badge-blue"

        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <span class="card-title">Z-Score Analysis</span>
                <span class="text-muted">{data['data_points']} periods</span>
            </div>
            <div class="signal-box">
                <div class="stat-label">Current Z-Score</div>
                <div class="stat-value" style="font-size: 48px;">{z_score:.2f}σ</div>
                <div style="margin-top: 12px;">
                    <span class="badge {signal_class}">{signal_text}</span>
                </div>
            </div>
            <div class="divider"></div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; text-align: center;">
                <div class="stat-box">
                    <div class="stat-label">Mean Rate</div>
                    <div class="stat-value blue">{data['mean_rate']*100:.4f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Std Dev</div>
                    <div class="stat-value purple">{data['std_dev']*100:.4f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Signal Thresholds Card
        entry_high_rate = data['mean_rate'] + (entry_threshold * data['std_dev'])
        entry_low_rate = data['mean_rate'] - (entry_threshold * data['std_dev'])

        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <span class="card-title">Signal Thresholds</span>
            </div>
            <table class="threshold-table">
                <tr>
                    <td><span class="badge badge-red">SHORT</span> Entry</td>
                    <td style="text-align: right;">≥ +{entry_threshold}σ</td>
                    <td style="text-align: right;" class="text-muted">{entry_high_rate*100:.4f}%</td>
                </tr>
                <tr>
                    <td><span class="badge badge-green">LONG</span> Entry</td>
                    <td style="text-align: right;">≤ -{entry_threshold}σ</td>
                    <td style="text-align: right;" class="text-muted">{entry_low_rate*100:.4f}%</td>
                </tr>
                <tr>
                    <td>Exit (Mean Reversion)</td>
                    <td style="text-align: right;">±{exit_threshold}σ</td>
                    <td style="text-align: right;" class="text-muted">Near mean</td>
                </tr>
                <tr>
                    <td>Stop Loss</td>
                    <td style="text-align: right;">±{settings['stop_loss_std_dev']}σ</td>
                    <td style="text-align: right;" class="text-muted">Extreme deviation</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        # Current Position Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="card-title">Current Position</span></div>', unsafe_allow_html=True)

        open_trade = get_open_trade(settings['symbol'])

        if open_trade:
            side_color = 'green' if open_trade['side'] == 'long' else 'red'
            leverage = settings.get('leverage', 2)

            # Calculate leveraged metrics
            metrics = calculate_leveraged_metrics(
                open_trade['entry_price'],
                data['current_price'],
                open_trade['position_size'],
                open_trade['side'],
                leverage,
                settings
            )

            # Liquidation warning colors
            liq_colors = {
                'CRITICAL': '#f85149',
                'HIGH': '#d29922',
                'MEDIUM': '#58a6ff',
                'LOW': '#3fb950'
            }
            liq_color = liq_colors.get(metrics['liq_warning'], '#8b949e')
            roi_color = 'green' if metrics['roi_on_margin'] >= 0 else 'red'

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; text-align: center;">
                <div class="stat-box">
                    <div class="stat-label">Side</div>
                    <div class="stat-value {side_color}">{open_trade['side'].upper()}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Leverage</div>
                    <div class="stat-value purple">{leverage}x</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Entry Price</div>
                    <div class="stat-value">${open_trade['entry_price']:,.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Current Price</div>
                    <div class="stat-value">${data['current_price']:,.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Margin</div>
                    <div class="stat-value">${metrics['margin']:,.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">ROI (Margin)</div>
                    <div class="stat-value {roi_color}">{metrics['roi_on_margin']:+.2f}%</div>
                </div>
            </div>
            <div class="divider"></div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; text-align: center;">
                <div class="stat-box">
                    <div class="stat-label">Liquidation Price</div>
                    <div class="stat-value" style="color: {liq_color};">${metrics['liquidation_price']:,.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Distance to Liq</div>
                    <div class="stat-value" style="color: {liq_color};">{metrics['price_to_liq_pct']:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show warning if close to liquidation
            if metrics['liq_warning'] == 'CRITICAL':
                st.markdown('<div style="background: rgba(248, 81, 73, 0.2); border: 1px solid #f85149; border-radius: 6px; padding: 12px; margin: 12px 0; text-align: center; color: #f85149; font-weight: 600;">⚠️ CRITICAL: Close to liquidation!</div>', unsafe_allow_html=True)
            elif metrics['liq_warning'] == 'HIGH':
                st.markdown('<div style="background: rgba(210, 153, 34, 0.2); border: 1px solid #d29922; border-radius: 6px; padding: 12px; margin: 12px 0; text-align: center; color: #d29922;">⚠️ Warning: Getting close to liquidation</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="divider"></div>
            <div style="text-align: center;" class="text-muted">
                Opened: {open_trade['entry_time'][:19]}
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔴 Close Position", key="close_pos"):
                execute_close_trade(open_trade['id'], "manual")
                st.success("Position closed!")
                time.sleep(1)
                st.rerun()
        else:
            st.markdown('<div style="text-align: center; padding: 20px;" class="text-muted">No open position</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🟢 Open Long", key="open_long", use_container_width=True):
                    execute_open_trade(settings['symbol'], 'long', data['current_price'],
                                      data['funding_rate'], z_score, settings)
                    st.success("Long position opened!")
                    time.sleep(1)
                    st.rerun()
            with col2:
                if st.button("🔴 Open Short", key="open_short", use_container_width=True):
                    execute_open_trade(settings['symbol'], 'short', data['current_price'],
                                      data['funding_rate'], z_score, settings)
                    st.success("Short position opened!")
                    time.sleep(1)
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Z-Score History Chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="card-title">Z-Score History</span></div>', unsafe_allow_html=True)

        history = get_funding_history(settings['symbol'], settings['lookback_periods'])

        if not history.empty:
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
                mode='lines+markers',
                name='Z-Score',
                line=dict(color='#58a6ff', width=2),
                marker=dict(size=4)
            ))
            fig.add_hline(y=entry_threshold, line_dash="dash", line_color="#f85149",
                         annotation_text="Entry High")
            fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="#3fb950",
                         annotation_text="Entry Low")
            fig.add_hline(y=0, line_dash="dot", line_color="#8b949e")

            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e')),
                yaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e')),
                showlegend=True,
                legend=dict(font=dict(color='#8b949e'))
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data. Go to Settings to fetch data.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Funding Rate History Chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="card-title">Funding Rate History</span></div>', unsafe_allow_html=True)

        if not history.empty:
            colors = ['#3fb950' if x >= 0 else '#f85149' for x in history['funding_rate']]
            fig = go.Figure(go.Bar(
                x=history['funding_time'],
                y=history['funding_rate'] * 100,
                marker_color=colors,
                name='Funding Rate %'
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e')),
                yaxis=dict(gridcolor='#30363d', tickfont=dict(color='#8b949e'), title=dict(text='Rate %', font=dict(color='#8b949e'))),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Trading Statistics Card - Full Width
    stats = get_stats(settings['symbol'])

    pnl_color = 'green' if (stats.get('total_pnl', 0) or 0) >= 0 else 'red'
    funding_color = 'green' if (stats.get('total_funding_pnl', 0) or 0) >= 0 else 'red'
    net_color = 'green' if (stats.get('total_net_pnl', 0) or 0) >= 0 else 'red'

    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <span class="card-title">Trading Statistics</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; text-align: center;">
            <div class="stat-box">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{stats.get('total_trades', 0)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value">{stats.get('win_rate', 0):.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total P&L (Gross)</div>
                <div class="stat-value {pnl_color}">${stats.get('total_pnl', 0) or 0:,.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Funding P&L</div>
                <div class="stat-value {funding_color}">${stats.get('total_funding_pnl', 0) or 0:,.2f}</div>
            </div>
        </div>
        <div class="divider"></div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; text-align: center;">
            <div class="stat-box">
                <div class="stat-label">Total Fees</div>
                <div class="stat-value red">-${stats.get('total_fees', 0) or 0:,.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Slippage</div>
                <div class="stat-value red">-${stats.get('total_slippage', 0) or 0:,.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Net P&L</div>
                <div class="stat-value {net_color}">${stats.get('total_net_pnl', 0) or 0:,.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Fee Rate</div>
                <div class="stat-value text-muted">{settings.get('taker_fee', 0.05):.3f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_trades():
    """Render the trades history page."""
    st.title("📊 Trade History")
    settings = get_settings()

    col1, col2 = st.columns([1, 3])
    with col1:
        status_filter = st.selectbox("Status", ["All", "Open", "Closed"])
    with col2:
        limit = st.slider("Show trades", 10, 200, 50)

    status = None if status_filter == "All" else status_filter.lower()
    trades_df = get_trades(settings['symbol'], status, limit)

    if trades_df.empty:
        st.info("No trades found")
        return

    stats = get_stats(settings['symbol'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", stats.get('total_trades', 0))
    with col2:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with col3:
        st.metric("Gross P&L", f"${stats.get('total_pnl', 0) or 0:,.2f}")
    with col4:
        st.metric("Net P&L", f"${stats.get('total_net_pnl', 0) or 0:,.2f}")

    st.divider()

    for _, trade in trades_df.iterrows():
        with st.expander(f"Trade #{trade['id']} - {trade['side'].upper()} @ ${trade['entry_price']:,.2f}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Entry:** ${trade['entry_price']:,.2f}")
                st.write(f"**Exit:** ${trade['exit_price']:,.2f}" if trade['exit_price'] else "**Exit:** --")
            with col2:
                st.write(f"**Price P&L:** ${trade['price_pnl'] or 0:,.2f}")
                st.write(f"**Funding P&L:** ${trade['funding_pnl'] or 0:,.2f}")
            with col3:
                st.write(f"**Fees:** -${trade['fee_cost'] or 0:,.2f}")
                st.write(f"**Net P&L:** ${trade['net_pnl'] or 0:,.2f}")

            if trade['status'] == 'open':
                if st.button(f"Close Trade #{trade['id']}", key=f"close_{trade['id']}"):
                    execute_close_trade(trade['id'], "manual")
                    st.success("Trade closed!")
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
            ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
            index=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"].index(settings['symbol']) if settings['symbol'] in ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"] else 0
        )

        lookback = st.number_input("Lookback Periods", 10, 500, settings['lookback_periods'])
        entry_std = st.number_input("Entry Std Dev", 0.5, 5.0, float(settings['entry_std_dev']), 0.1)
        exit_std = st.number_input("Exit Std Dev", 0.0, 1.0, float(settings['exit_std_dev']), 0.1)
        stop_loss_std = st.number_input("Stop Loss Std Dev", 3.0, 10.0, float(settings['stop_loss_std_dev']), 0.5)
        position_size = st.number_input("Position Size", 0.001, 100.0, float(settings['position_size']), 0.001)

        leverage_options = [1, 2, 3, 5, 10]
        leverage_labels = ["1x (Conservative)", "2x (Recommended)", "3x (Moderate)", "5x (Aggressive)", "10x (High Risk)"]
        current_leverage = settings.get('leverage', 2)
        leverage_index = leverage_options.index(current_leverage) if current_leverage in leverage_options else 1
        leverage = st.selectbox(
            "Leverage",
            options=leverage_options,
            format_func=lambda x: leverage_labels[leverage_options.index(x)],
            index=leverage_index,
            help="Higher leverage = higher returns but higher liquidation risk"
        )

        paper_mode = st.toggle("Paper Trading", settings['paper_mode'])

        st.divider()
        st.subheader("Fee Settings")

        maker_fee = st.number_input("Maker Fee (%)", 0.0, 1.0, float(settings.get('maker_fee', 0.02)), 0.001)
        taker_fee = st.number_input("Taker Fee (%)", 0.0, 1.0, float(settings.get('taker_fee', 0.05)), 0.001)
        slippage = st.number_input("Est. Slippage (%)", 0.0, 1.0, float(settings.get('estimated_slippage', 0.01)), 0.001)

        if st.button("💾 Save Settings", type="primary"):
            new_settings = {
                **settings,
                'symbol': symbol,
                'lookback_periods': lookback,
                'entry_std_dev': entry_std,
                'exit_std_dev': exit_std,
                'stop_loss_std_dev': stop_loss_std,
                'position_size': position_size,
                'leverage': leverage,
                'paper_mode': paper_mode,
                'maker_fee': maker_fee,
                'taker_fee': taker_fee,
                'estimated_slippage': slippage,
            }
            save_settings(new_settings)
            st.success("Settings saved!")
            st.rerun()

    with col2:
        st.subheader("API Credentials")

        api_key = st.text_input("API Key", settings.get('api_key', ''), type="password")
        api_secret = st.text_input("API Secret", settings.get('api_secret', ''), type="password")
        api_passphrase = st.text_input("Passphrase", settings.get('api_passphrase', ''), type="password")

        if st.button("💾 Save API Credentials"):
            new_settings = {**settings, 'api_key': api_key, 'api_secret': api_secret, 'api_passphrase': api_passphrase}
            save_settings(new_settings)
            st.success("Credentials saved!")

        st.divider()
        st.subheader("Data Management")

        if st.button("📥 Fetch Historical Data"):
            with st.spinner("Fetching..."):
                result = bootstrap_funding_history(settings['symbol'])
                if result > 0:
                    st.success(f"Fetched {result} records!")
                else:
                    st.error("Failed to fetch data")

        st.divider()
        st.warning("""
        **Note:** Streamlit Cloud apps sleep after inactivity.
        For 24/7 automated trading, use the Flask version locally or on a VPS.
        """)


def main():
    """Main app."""
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", ["📈 Dashboard", "📊 Trades", "⚙️ Settings"], label_visibility="collapsed")

    st.sidebar.divider()

    settings = get_settings()
    stats = get_stats(settings['symbol'])
    symbol_display = settings['symbol'].replace('-SWAP', '').replace('-', '/')

    leverage = settings.get('leverage', 2)
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; padding: 16px; border: 1px solid #30363d;">
        <div style="color: #8b949e; font-size: 11px; text-transform: uppercase;">Symbol</div>
        <div style="color: #fff; font-size: 20px; font-weight: 700; margin-bottom: 12px;">{symbol_display}</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #8b949e; font-size: 12px;">Leverage</span>
            <span style="color: #a371f7; font-weight: 600;">{leverage}x</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #8b949e; font-size: 12px;">Total Trades</span>
            <span style="color: #58a6ff; font-weight: 600;">{stats.get('total_trades', 0)}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #8b949e; font-size: 12px;">Net P&L</span>
            <span style="color: {'#3fb950' if (stats.get('total_net_pnl', 0) or 0) >= 0 else '#f85149'}; font-weight: 600;">${(stats.get('total_net_pnl', 0) or 0):,.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()

    if st.sidebar.button("🔄 Refresh"):
        st.rerun()

    auto_refresh = st.sidebar.toggle("Auto-refresh (5s)", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

    if page == "📈 Dashboard":
        render_dashboard()
    elif page == "📊 Trades":
        render_trades()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()

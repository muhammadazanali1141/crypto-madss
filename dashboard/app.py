"""
CryptoMADSS - Streamlit Dashboard
Multi-Agent Distributed System for Cryptocurrency Trading Signal Generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# ============================================
# PATH SETUP
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="CryptoMADSS - Trading Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD AGENTS
# ============================================
@st.cache_resource
def load_market_agent():
    """Load market agent with error handling"""
    try:
        from agents.market_data_agent import MarketDataAgent
        agent = MarketDataAgent()
        return agent
    except Exception as e:
        st.error(f"❌ Market Agent Error: {e}")
        return None

@st.cache_resource
def load_coordinator():
    """Load coordinator with error handling"""
    try:
        from agents.coordinator_agent import CoordinatorAgent
        coordinator = CoordinatorAgent(
            account_balance=10000,
            risk_per_trade_pct=1.0,
            stop_loss_pct=2.0
        )
        return coordinator
    except Exception as e:
        st.error(f"❌ Coordinator Error: {e}")
        return None

# ============================================
# MAIN APP
# ============================================

st.title("🚀 CryptoMADSS - Multi-Agent Trading Dashboard")
st.markdown("*Multi-Agent Distributed System for Cryptocurrency Trading Signal Generation*")

# Load agents
market_agent = load_market_agent()
coordinator = load_coordinator()

if market_agent is None:
    st.warning("⚠️ Market Agent not available. Using fallback data.")
    # Create dummy data for demonstration
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    dummy_data = pd.DataFrame({
        'open': np.random.randn(100) * 100 + 65000,
        'high': np.random.randn(100) * 100 + 65100,
        'low': np.random.randn(100) * 100 + 64900,
        'close': np.random.randn(100) * 100 + 65000,
        'volume': np.random.randn(100) * 1000 + 5000
    }, index=dates)
    dummy_data['close'] = dummy_data['close'].cumsum() + 65000
    data = dummy_data
    st.warning("⚠️ Using dummy data (Binance API unavailable)")
else:
    try:
        # Fetch real data
        with st.spinner("Fetching market data..."):
            data = market_agent.get_historical_data("BTCUSDT", "1h", "30 days ago UTC")
            
            # Check if data is empty
            if data is None or len(data) == 0:
                st.warning("⚠️ No data received from Binance. Using dummy data.")
                dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
                dummy_data = pd.DataFrame({
                    'open': np.random.randn(100) * 100 + 65000,
                    'high': np.random.randn(100) * 100 + 65100,
                    'low': np.random.randn(100) * 100 + 64900,
                    'close': np.random.randn(100) * 100 + 65000,
                    'volume': np.random.randn(100) * 1000 + 5000
                }, index=dates)
                dummy_data['close'] = dummy_data['close'].cumsum() + 65000
                data = dummy_data
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        st.warning("⚠️ Using dummy data for demonstration.")
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        dummy_data = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 65000,
            'high': np.random.randn(100) * 100 + 65100,
            'low': np.random.randn(100) * 100 + 64900,
            'close': np.random.randn(100) * 100 + 65000,
            'volume': np.random.randn(100) * 1000 + 5000
        }, index=dates)
        dummy_data['close'] = dummy_data['close'].cumsum() + 65000
        data = dummy_data

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("⚙️ Configuration")

# Symbol selection
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
selected_symbol = st.sidebar.selectbox("Trading Pair", symbols, index=0)

# Time interval
intervals = ["1h", "4h", "1d"]
selected_interval = st.sidebar.selectbox("Time Interval", intervals, index=0)

# Account settings
st.sidebar.header("💰 Account Settings")
account_balance = st.sidebar.number_input("Balance ($)", value=10000, step=1000, min_value=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)

# Signal button
generate_signal = st.sidebar.button("🔄 Generate Signal", type="primary", use_container_width=True)

# ============================================
# MAIN CONTENT
# ============================================

# Initialize session state
if 'signal' not in st.session_state:
    st.session_state['signal'] = None

try:
    # Check if data has values
    if len(data) > 0:
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
    else:
        current_price = 65000
        price_change = 0
        st.warning("⚠️ No price data available")

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=f"💰 {selected_symbol}",
            value=f"${current_price:,.2f}",
            delta=f"{price_change:.2f}%"
        )

    # Generate signal
    if generate_signal and coordinator:
        with st.spinner("🧠 Analyzing with AI agents..."):
            try:
                result = coordinator.generate_signal(selected_symbol, selected_interval, "7 days ago UTC")
                st.session_state['signal'] = result
                st.success("✅ Signal generated!")
            except Exception as e:
                st.error(f"Error generating signal: {e}")
                st.session_state['signal'] = None

    # Display signal
    if st.session_state['signal']:
        signal = st.session_state['signal']
        
        with col2:
            decision = signal.get('final_decision', 'HOLD')
            confidence = signal.get('confidence', 0)
            
            if decision == "BUY":
                emoji = "📈"
            elif decision == "SELL":
                emoji = "📉"
            else:
                emoji = "➖"
            
            st.metric(
                label=f"{emoji} Signal",
                value=decision,
                delta=f"Confidence: {confidence:.1f}%"
            )
        
        with col3:
            position_size = signal.get('position_size', 0)
            stop_loss = signal.get('stop_loss_price', 0)
            
            if decision in ["BUY", "SELL"] and position_size > 0:
                st.metric(
                    label="📊 Position",
                    value=f"{position_size:.4f} units",
                    delta=f"SL: ${stop_loss:,.2f}"
                )
            else:
                st.metric(
                    label="📊 Status",
                    value="No Position",
                    delta="Waiting"
                )

    # ============================================
    # CHART
    # ============================================
    if len(data) > 0:
        st.subheader(f"📈 {selected_symbol} - Price Chart")

        # Use last 300 candles or all if less
        chart_data = data if len(data) <= 300 else data.iloc[-300:]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("Price", "Volume", "RSI")
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data['open'],
                high=chart_data['high'],
                low=chart_data['low'],
                close=chart_data['close'],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # Moving averages (only if enough data)
        if len(chart_data) >= 20:
            sma_20 = chart_data['close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(x=chart_data.index, y=sma_20, 
                           name="SMA 20", line=dict(color='orange', width=1.5)),
                row=1, col=1
            )

        if len(chart_data) >= 50:
            sma_50 = chart_data['close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(x=chart_data.index, y=sma_50, 
                           name="SMA 50", line=dict(color='blue', width=1.5)),
                row=1, col=1
            )

        # Volume
        colors = ['green' if chart_data['close'].iloc[i] >= chart_data['open'].iloc[i] else 'red' 
                  for i in range(len(chart_data))]
        fig.add_trace(
            go.Bar(x=chart_data.index, y=chart_data['volume'], 
                   name="Volume", marker_color=colors),
            row=2, col=1
        )

        # RSI (only if enough data)
        if len(chart_data) >= 14:
            delta = chart_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            fig.add_trace(
                go.Scatter(x=chart_data.index, y=rsi, 
                           name="RSI", line=dict(color='purple', width=2)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(
            height=700,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # DECISION LOG
    # ============================================
    if coordinator:
        st.subheader("📝 Decision Log")
        
        log = coordinator.get_decision_log()
        if log:
            log_df = pd.DataFrame(log)
            st.dataframe(log_df, use_container_width=True)
            
            csv = log_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Log (CSV)",
                data=csv,
                file_name=f"decision_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No decisions logged. Click 'Generate Signal'.")

    # ============================================
    # AGENT STATUS
    # ============================================
    st.subheader("🤖 Agent Status")
    cols = st.columns(4)
    
    with cols[0]:
        if market_agent:
            st.success("✅ Market Data Agent")
        else:
            st.warning("⚠️ Market Agent (Fallback Mode)")
    
    with cols[1]:
        st.success("✅ Technical Agent")
    
    with cols[2]:
        if coordinator:
            st.success("✅ Coordinator Agent")
        else:
            st.error("❌ Coordinator Agent")
    
    with cols[3]:
        st.success("✅ Risk Agent")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check your internet connection and try again.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption(f"CryptoMADSS v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ Academic/Paper Trading Only | Not Financial Advice")
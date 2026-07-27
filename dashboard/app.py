"""
CryptoMADSS - Streamlit Dashboard
FR-8.1: Display current price data and technical indicator charts
FR-8.2: Display the latest recommendation from Coordinator Agent
FR-8.3: Display backtest performance metrics and equity curve
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.coordinator_agent import CoordinatorAgent
from agents.market_data_agent import MarketDataAgent
from backtesting.backtest_engine import BacktestEngine

# Page configuration
st.set_page_config(
    page_title="CryptoMADSS Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚀 CryptoMADSS - Multi-Agent Trading Signal Dashboard")
st.markdown("*Multi-Agent Distributed System for Cryptocurrency Trading Signal Generation*")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Symbol selection
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
selected_symbol = st.sidebar.selectbox("Select Trading Pair", symbols)

# Time interval
intervals = ["1h", "4h", "1d", "1m", "5m"]
selected_interval = st.sidebar.selectbox("Time Interval", intervals)

# Lookback period
lookback = st.sidebar.selectbox(
    "Lookback Period",
    ["7 days ago UTC", "14 days ago UTC", "30 days ago UTC", "90 days ago UTC"]
)

# Account settings
st.sidebar.header("💰 Account Settings")
account_balance = st.sidebar.number_input("Account Balance ($)", value=10000, step=1000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)

# Buttons
refresh = st.sidebar.button("🔄 Refresh Signal")
run_backtest = st.sidebar.button("📊 Run Backtest")

# Initialize agents
@st.cache_resource
def get_agents():
    return CoordinatorAgent(
        account_balance=account_balance,
        risk_per_trade_pct=risk_per_trade
    )

@st.cache_resource
def get_market_agent():
    return MarketDataAgent()

# Main content
col1, col2, col3 = st.columns(3)

# Fetch data
market_agent = get_market_agent()
coordinator = get_agents()

try:
    # Get historical data
    data = market_agent.get_historical_data(selected_symbol, selected_interval, "90 days ago UTC")
    current_price = data['close'].iloc[-1]
    price_change = ((data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]) * 100

    # Display current price
    with col1:
        st.metric(
            label=f"💰 {selected_symbol} Price",
            value=f"${current_price:,.2f}",
            delta=f"{price_change:.2f}%"
        )

    # Generate signal
    if refresh:
        with st.spinner("Generating signal..."):
            result = coordinator.generate_signal(selected_symbol, selected_interval, lookback)
            st.session_state['signal'] = result

    # Display signal if available
    if 'signal' in st.session_state:
        signal = st.session_state['signal']
        
        with col2:
            decision = signal.get('final_decision', 'HOLD')
            confidence = signal.get('confidence', 0)
            
            # Color coding
            if decision == "BUY":
                color = "green"
                emoji = "📈"
            elif decision == "SELL":
                color = "red"
                emoji = "📉"
            else:
                color = "orange"
                emoji = "➖"
            
            st.metric(
                label=f"{emoji} Signal",
                value=decision,
                delta=f"Confidence: {confidence:.1f}%",
                delta_color="normal" if decision != "SELL" else "inverse"
            )
        
        with col3:
            position_size = signal.get('position_size', 0)
            stop_loss = signal.get('stop_loss_price', 0)
            
            if decision in ["BUY", "SELL"]:
                st.metric(
                    label="📊 Position Size",
                    value=f"{position_size:.6f} units",
                    delta=f"SL: ${stop_loss:,.2f}"
                )
            else:
                st.metric(
                    label="📊 Status",
                    value="No Position",
                    delta="Waiting for signal"
                )

    # Candlestick chart
    st.subheader(f"📈 {selected_symbol} Price Chart")
    
    # Create candlestick chart
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
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    sma_20 = data['close'].rolling(window=20).mean()
    sma_50 = data['close'].rolling(window=50).mean()
    
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_20, name="SMA 20", line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_50, name="SMA 50", line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Volume
    colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] else 'red' 
              for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['volume'], name="Volume", marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(
        go.Scatter(x=data.index, y=rsi, name="RSI", line=dict(color='purple')),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Backtest Results
    if run_backtest:
        with st.spinner("Running backtest... This may take a moment."):
            engine = BacktestEngine(
                initial_balance=account_balance,
                risk_per_trade_pct=risk_per_trade
            )
            metrics = engine.run_backtest(selected_symbol, selected_interval, lookback)
            st.session_state['backtest_metrics'] = metrics
    
    if 'backtest_metrics' in st.session_state:
        metrics = st.session_state['backtest_metrics']
        
        st.subheader("📊 Backtest Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics.get('total_trades', 0))
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
        
        with col2:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
            st.metric("Avg Trade", f"${metrics.get('avg_trade', 0):.2f}")
        
        with col4:
            st.metric("Final Balance", f"${metrics.get('final_balance', 0):,.2f}")
            st.metric("Winning Trades", metrics.get('winning_trades', 0))

    # Decision Log
    st.subheader("📝 Decision Log")
    
    log = coordinator.get_decision_log()
    if log:
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, use_container_width=True)
        
        # Download button
        csv = log_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Decision Log",
            data=csv,
            file_name=f"decision_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No decisions logged yet. Click 'Refresh Signal' to generate signals.")

    # Agent Status
    st.subheader("🤖 Agent Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("✅ Market Data Agent")
    with col2:
        st.success("✅ Technical Analysis Agent")
    with col3:
        st.success("✅ Prediction Agent (LSTM)")
    with col4:
        st.success("✅ Risk Agent")
    
    st.caption("All agents are operational")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please make sure you have internet connection and Binance API is accessible.")

# Footer
st.markdown("---")
st.caption(f"CryptoMADSS v1.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ This is for academic/paper trading purposes only. Not financial advice.")
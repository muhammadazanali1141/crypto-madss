"""
Backtesting Engine (Optimized)
-------------------------------
Historical data par strategy ko test karta hai
Data sirf ek baar fetch karta hai, phir reuse karta hai
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.coordinator_agent import CoordinatorAgent
from agents.market_data_agent import MarketDataAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_agent import RiskAgent


class BacktestEngine:
    def __init__(self, initial_balance=10000, risk_per_trade_pct=1.0, stop_loss_pct=2.0):
        """
        Initialize backtest engine
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Track trades
        self.trades = []
        self.equity_curve = []
        
        # Current position
        self.position = None
        
        # Initialize agents once (reuse)
        print("🔄 Initializing agents for backtest...")
        self.market_agent = MarketDataAgent()
        self.prediction_agent = PredictionAgent()
        self.risk_agent = RiskAgent(
            account_balance=initial_balance,
            risk_per_trade_pct=risk_per_trade_pct,
            stop_loss_pct=stop_loss_pct
        )
        print("✅ Agents initialized!")
        
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Risk per Trade: {risk_per_trade_pct}%")
    
    def _get_technical_signal(self, data: pd.DataFrame, idx: int) -> dict:
        """
        Technical analysis for a specific point in time
        """
        try:
            # Get data up to current index
            current_data = data.iloc[:idx+1]
            close_prices = current_data['close']
            
            if len(close_prices) < 50:
                return {"direction": "neutral", "rsi": 50}
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # SMA crossover
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                if sma_20.iloc[-1] > sma_50.iloc[-1] and current_rsi < 70:
                    direction = "bullish"
                elif sma_20.iloc[-1] < sma_50.iloc[-1] and current_rsi > 30:
                    direction = "bearish"
                else:
                    direction = "neutral"
            else:
                direction = "neutral"
            
            return {
                "direction": direction,
                "rsi": round(current_rsi, 2)
            }
        except Exception as e:
            return {"direction": "neutral", "rsi": 50}
    
    def run_backtest(self, symbol="BTCUSDT", interval="1h", lookback="30 days ago UTC"):
        """
        Run backtest on historical data - OPTIMIZED (data fetched once)
        """
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST STARTED (Optimized)")
        print(f"   Symbol: {symbol}")
        print(f"   Period: {lookback}")
        print(f"{'='*60}")
        
        # ============================================
        # STEP 1: Fetch data ONCE
        # ============================================
        print("\n📥 STEP 1: Fetching historical data (ONE TIME)...")
        start_time = time.time()
        
        data = self.market_agent.get_historical_data(symbol, interval, lookback)
        close_prices = data["close"].tolist()
        
        print(f"   ✅ Fetched {len(data)} candles")
        print(f"   ⏱️ Fetch time: {time.time() - start_time:.2f} seconds")
        
        # ============================================
        # STEP 2: Process each candle
        # ============================================
        print("\n🔄 STEP 2: Processing signals sequentially...")
        
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self.position = None
        
        total_candles = len(data)
        progress_step = max(1, total_candles // 20)  # 5% progress steps
        
        # Pre-calculate predictions for all points (optional optimization)
        print("   Pre-calculating LSTM predictions...")
        predictions = {}
        for i in range(60, total_candles):
            prices = close_prices[:i+1]
            if len(prices) >= 60:
                try:
                    pred = self.prediction_agent.predict_next_price(prices)
                    if "error" not in pred:
                        predictions[i] = pred
                except:
                    pass
        
        print(f"   ✅ Pre-calculated {len(predictions)} predictions")
        
        # Process each candle
        for i in range(60, total_candles):  # Start from 60 for indicators
            current_data = data.iloc[:i+1]
            current_price = close_prices[i]
            current_time = data.index[i]
            
            # Progress
            if i % progress_step == 0:
                progress = ((i - 60) / (total_candles - 60)) * 100
                print(f"   Progress: {progress:.1f}%")
            
            try:
                # Get prediction for this point
                prediction = predictions.get(i, {"direction": "neutral", "change_pct": 0})
                
                # Technical analysis
                technical = self._get_technical_signal(data, i)
                
                # Risk assessment
                risk_params = self.risk_agent.evaluate(current_price, direction="BUY")
                
                # Combine signals
                tech_score = 1 if technical['direction'] == 'bullish' else -1 if technical['direction'] == 'bearish' else 0
                pred_score = 1 if prediction.get('direction', 'neutral') == 'bullish' else -1 if prediction.get('direction', 'neutral') == 'bearish' else 0
                
                risk_penalty = 0.3 if risk_params.get('flagged_high_risk', False) else 0
                weighted_score = (tech_score * 0.30) + (pred_score * 0.50) - risk_penalty
                
                # Final decision
                if weighted_score > 0.25:
                    decision = "BUY"
                elif weighted_score < -0.25:
                    decision = "SELL"
                else:
                    decision = "HOLD"
                
                # Process trade
                self._process_trade(decision, current_price, current_time, risk_params)
                
                # Update equity
                self._update_equity(current_price)
                
            except Exception as e:
                if i % 100 == 0:
                    print(f"   ⚠️ Error at index {i}: {e}")
                continue
        
        # ============================================
        # STEP 3: Calculate metrics
        # ============================================
        print("\n📊 STEP 3: Calculating metrics...")
        metrics = self._calculate_metrics()
        
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST COMPLETE")
        print(f"   Total time: {time.time() - start_time:.2f} seconds")
        print(f"{'='*60}")
        
        return metrics
    
    def _process_trade(self, decision, current_price, timestamp, risk_params):
        """
        Process trading decision
        """
        # Close existing position if opposite signal
        if self.position and decision in ['BUY', 'SELL']:
            if (self.position['type'] == 'BUY' and decision == 'SELL') or \
               (self.position['type'] == 'SELL' and decision == 'BUY'):
                # Close position
                trade = {
                    'entry_price': self.position['entry_price'],
                    'exit_price': current_price,
                    'size': self.position['size'],
                    'type': self.position['type'],
                    'entry_time': self.position['entry_time'],
                    'exit_time': timestamp,
                    'pnl': 0
                }
                
                if self.position['type'] == 'BUY':
                    trade['pnl'] = (current_price - self.position['entry_price']) * self.position['size']
                else:
                    trade['pnl'] = (self.position['entry_price'] - current_price) * self.position['size']
                
                self.trades.append(trade)
                self.balance += trade['pnl']
                self.position = None
        
        # Open new position
        if decision in ['BUY', 'SELL'] and not self.position:
            position_size = risk_params.get('position_size_units', 0)
            stop_loss = risk_params.get('stop_loss_price', 0)
            
            if position_size > 0:
                self.position = {
                    'type': decision,
                    'entry_price': current_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'entry_time': timestamp
                }
    
    def _update_equity(self, current_price):
        """Update equity curve"""
        equity = self.balance
        if self.position:
            if self.position['type'] == 'BUY':
                unrealized = (current_price - self.position['entry_price']) * self.position['size']
            else:
                unrealized = (self.position['entry_price'] - current_price) * self.position['size']
            equity += unrealized
        
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'balance': self.balance,
            'equity': equity
        })
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'final_balance': self.balance,
                'profit_factor': 0
            }
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (total_pnl / self.initial_balance) * 100
        
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': round(win_rate, 2),
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'final_balance': round(self.balance, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_trade': round(total_pnl / total_trades if total_trades > 0 else 0, 2)
        }


if __name__ == "__main__":
    print("\n🚀 Starting Optimized Backtest Engine")
    print("="*60)
    
    engine = BacktestEngine(
        initial_balance=10000,
        risk_per_trade_pct=1.0,
        stop_loss_pct=2.0
    )
    
    metrics = engine.run_backtest(
        symbol="BTCUSDT",
        interval="1h",
        lookback="30 days ago UTC"
    )
    
    print("\n📊 BACKTEST METRICS:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:,.2f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Backtest Complete!")
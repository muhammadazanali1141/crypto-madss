"""
Coordinator Agent
-----------------
Sab agents ke outputs ko combine karke final trading signal generate karta hai
SRS ke hisaab se FR-6.1 se FR-6.4 implement karta hai
"""

import os
import sys
from datetime import datetime
import pandas as pd  # ✅ IMPORT ADDED

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_agent import MarketDataAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_agent import RiskAgent


class CoordinatorAgent:
    """
    FR-6.1: Collects outputs from Technical, Sentiment, Prediction, and Risk agents
    FR-6.2: Applies combination/weighting logic to produce final decision
    FR-6.3: Attaches position size and stop-loss to Buy/Sell recommendations
    FR-6.4: Logs every decision along with contributing agent outputs
    """
    
    def __init__(self, account_balance=10000, risk_per_trade_pct=1.0, stop_loss_pct=2.0):
        """
        Initialize all agents with configuration
        """
        print("🔄 Initializing Coordinator Agent...")
        print("-" * 40)
        
        # Initialize all agents
        self.market_agent = MarketDataAgent()
        self.prediction_agent = PredictionAgent()
        self.risk_agent = RiskAgent(
            account_balance=account_balance,
            risk_per_trade_pct=risk_per_trade_pct,
            stop_loss_pct=stop_loss_pct
        )
        
        # Configuration
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Decision log (FR-6.4)
        self.decision_log = []
        
        print("✅ All agents initialized successfully!")
        print("=" * 50)
    
    def _get_technical_signal(self, data: pd.DataFrame) -> dict:
        """
        Technical Analysis Signal (FR-2.1 to FR-2.4)
        """
        try:
            close_prices = data['close']
            
            # Simple moving averages
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            
            current_price = close_prices.iloc[-1]
            sma_20_current = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
            sma_50_current = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
            
            # RSI calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # MACD calculation
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            current_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
            current_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
            
            # Determine direction
            if current_rsi > 70:
                rsi_signal = "bearish"
            elif current_rsi < 30:
                rsi_signal = "bullish"
            else:
                rsi_signal = "neutral"
            
            macd_signal = "bullish" if current_macd > current_signal else "bearish" if current_macd < current_signal else "neutral"
            
            # Combined signal
            if sma_20_current > sma_50_current and rsi_signal == "bullish":
                direction = "bullish"
            elif sma_20_current < sma_50_current and rsi_signal == "bearish":
                direction = "bearish"
            else:
                direction = "neutral"
            
            return {
                "direction": direction,
                "rsi": round(current_rsi, 2),
                "macd": round(current_macd, 4),
                "signal_line": round(current_signal, 4),
                "sma_20": round(sma_20_current, 2),
                "sma_50": round(sma_50_current, 2),
                "rsi_signal": rsi_signal,
                "macd_signal": macd_signal
            }
        except Exception as e:
            print(f"   ⚠️ Technical analysis error: {e}")
            return {
                "direction": "neutral",
                "rsi": 50,
                "macd": 0,
                "signal_line": 0,
                "sma_20": 0,
                "sma_50": 0,
                "rsi_signal": "neutral",
                "macd_signal": "neutral"
            }
    
    def generate_signal(self, symbol="BTCUSDT", interval="1h", lookback="7 days ago UTC") -> dict:
        """
        Generate final trading signal by combining all agents
        """
        print(f"\n{'='*60}")
        print(f"📊 CRYPTOMADSS - SIGNAL GENERATION")
        print(f"   Symbol: {symbol}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # ============================================
        # STEP 1: Fetch Market Data
        # ============================================
        print("\n📥 STEP 1: Fetching Market Data")
        print("-" * 40)
        
        try:
            data = self.market_agent.get_historical_data(symbol, interval, lookback)
            close_prices = data["close"].tolist()
            current_price = close_prices[-1]
            
            print(f"   ✅ Data fetched: {len(close_prices)} candles")
            print(f"   💰 Current Price: ${current_price:,.2f}")
            print(f"   📊 Period: {data.index[0]} to {data.index[-1]}")
            
        except Exception as e:
            error_msg = f"Market data fetch failed: {e}"
            print(f"   ❌ {error_msg}")
            return {"error": error_msg}
        
        # ============================================
        # STEP 2: Technical Analysis
        # ============================================
        print("\n📈 STEP 2: Technical Analysis")
        print("-" * 40)
        
        technical_signal = self._get_technical_signal(data)
        print(f"   📊 Direction: {technical_signal['direction'].upper()}")
        print(f"   📊 RSI: {technical_signal['rsi']}")
        print(f"   📊 MACD: {technical_signal['macd']:.4f}")
        print(f"   📊 Signal Line: {technical_signal['signal_line']:.4f}")
        print(f"   📊 SMA 20: {technical_signal['sma_20']:,.2f}")
        print(f"   📊 SMA 50: {technical_signal['sma_50']:,.2f}")
        
        # ============================================
        # STEP 3: LSTM Prediction
        # ============================================
        print("\n🤖 STEP 3: LSTM Price Prediction")
        print("-" * 40)
        
        try:
            prediction = self.prediction_agent.predict_next_price(close_prices)
            
            if "error" in prediction:
                print(f"   ⚠️ {prediction['error']}")
                prediction_direction = "neutral"
                prediction_change = 0
                predicted_price = current_price
            else:
                print(f"   ✅ Current: ${prediction['current_price']:,.2f}")
                print(f"   🔮 Predicted: ${prediction['predicted_price']:,.2f}")
                print(f"   📊 Change: {prediction['change_pct']:.2f}%")
                print(f"   🎯 Direction: {prediction['direction'].upper()}")
                
                prediction_direction = prediction['direction']
                prediction_change = prediction['change_pct']
                predicted_price = prediction['predicted_price']
                
        except Exception as e:
            print(f"   ⚠️ Prediction failed: {e}")
            prediction_direction = "neutral"
            prediction_change = 0
            predicted_price = current_price
            prediction = {"error": str(e)}
        
        # ============================================
        # STEP 4: Risk Assessment
        # ============================================
        print("\n🛡️ STEP 4: Risk Assessment")
        print("-" * 40)
        
        try:
            # Determine direction for risk calculation
            if technical_signal['direction'] == 'bullish' or prediction_direction == 'bullish':
                risk_direction = "BUY"
            elif technical_signal['direction'] == 'bearish' or prediction_direction == 'bearish':
                risk_direction = "SELL"
            else:
                risk_direction = "BUY"  # Default
            
            risk_params = self.risk_agent.evaluate(current_price, direction=risk_direction)
            
            if "error" in risk_params:
                print(f"   ⚠️ {risk_params['error']}")
            else:
                print(f"   💰 Risk Amount: ${risk_params['risk_amount_usd']:,.2f}")
                print(f"   📊 Position Size: {risk_params['position_size_units']:.6f} units")
                print(f"   💵 Position Value: ${risk_params['position_value_usd']:,.2f}")
                print(f"   📈 Position % of Account: {risk_params['position_pct_of_account']:.2f}%")
                print(f"   🛑 Stop-Loss Price: ${risk_params['stop_loss_price']:,.2f}")
                print(f"   ⚠️ High Risk Flag: {risk_params['flagged_high_risk']}")
                
        except Exception as e:
            print(f"   ⚠️ Risk assessment failed: {e}")
            risk_params = {"error": str(e)}
        
        # ============================================
        # STEP 5: Combine Signals
        # ============================================
        print("\n🎯 STEP 5: Combining Signals")
        print("-" * 40)
        
        # Technical score
        tech_score = 1 if technical_signal['direction'] == 'bullish' else -1 if technical_signal['direction'] == 'bearish' else 0
        
        # Prediction score
        pred_score = 1 if prediction_direction == 'bullish' else -1 if prediction_direction == 'bearish' else 0
        
        # Risk penalty
        risk_penalty = 0
        if isinstance(risk_params, dict) and risk_params.get('flagged_high_risk', False):
            risk_penalty = 0.3
            print(f"   ⚠️ High risk penalty applied: -{risk_penalty*100:.0f}%")
        
        # Weighted average
        weighted_score = (tech_score * 0.30) + (pred_score * 0.50)
        weighted_score = weighted_score - risk_penalty
        
        print(f"   📊 Technical Score: {tech_score} (weight: 30%)")
        print(f"   📊 Prediction Score: {pred_score} (weight: 50%)")
        print(f"   📊 Weighted Score: {weighted_score:.3f}")
        
        # Final decision
        if weighted_score > 0.25:
            final_direction = "BUY"
            confidence = min(weighted_score * 100, 95)
        elif weighted_score < -0.25:
            final_direction = "SELL"
            confidence = min(abs(weighted_score) * 100, 95)
        else:
            final_direction = "HOLD"
            confidence = 50 - (abs(weighted_score) * 100)
        
        confidence = max(5, min(95, confidence))
        
        print(f"\n   🎯 FINAL DECISION: {final_direction}")
        print(f"   📊 Confidence: {confidence:.1f}%")
        
        # ============================================
        # STEP 6: Prepare Result
        # ============================================
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "final_decision": final_direction,
            "confidence": round(confidence, 1),
            "weighted_score": round(weighted_score, 3),
            "technical_signal": technical_signal,
            "prediction": prediction if isinstance(prediction, dict) else {"error": str(prediction)},
            "risk_params": risk_params if isinstance(risk_params, dict) else {"error": str(risk_params)},
            "signals": {
                "technical_score": tech_score,
                "prediction_score": pred_score,
                "risk_penalty": round(risk_penalty, 2)
            }
        }
        
        # Add position size and stop-loss for Buy/Sell
        if final_direction in ["BUY", "SELL"] and isinstance(risk_params, dict) and "error" not in risk_params:
            result["position_size"] = risk_params.get("position_size_units", 0)
            result["stop_loss_price"] = risk_params.get("stop_loss_price", 0)
            result["position_value"] = risk_params.get("position_value_usd", 0)
        else:
            result["position_size"] = 0
            result["stop_loss_price"] = 0
            result["position_value"] = 0
        
        # Log decision
        log_entry = {
            "timestamp": result["timestamp"],
            "symbol": result["symbol"],
            "decision": result["final_decision"],
            "confidence": result["confidence"],
            "current_price": result["current_price"],
            "predicted_price": result["predicted_price"],
            "technical_direction": technical_signal['direction'],
            "prediction_direction": prediction_direction,
            "weighted_score": result["weighted_score"]
        }
        self.decision_log.append(log_entry)
        
        print(f"\n📝 Decision logged (Total: {len(self.decision_log)} decisions)")
        
        print("\n" + "="*60)
        print(f"✅ SIGNAL GENERATION COMPLETE")
        print(f"   🎯 Final Decision: {final_direction}")
        print(f"   📊 Confidence: {confidence:.1f}%")
        if final_direction in ["BUY", "SELL"]:
            print(f"   📊 Position Size: {result['position_size']:.6f} units")
            print(f"   🛑 Stop-Loss: ${result['stop_loss_price']:,.2f}")
            print(f"   💰 Position Value: ${result['position_value']:,.2f}")
        print("="*60)
        
        return result
    
    def generate_signal_from_data(self, data: pd.DataFrame, symbol="BTCUSDT") -> dict:
        """
        Generate signal from already fetched data (for backtesting)
        """
        print(f"\n{'='*60}")
        print(f"📊 CRYPTOMADSS - SIGNAL GENERATION (from data)")
        print(f"   Symbol: {symbol}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # ============================================
        # STEP 1: Use provided data
        # ============================================
        print("\n📥 STEP 1: Using provided market data")
        print("-" * 40)
        
        try:
            close_prices = data["close"].tolist()
            current_price = close_prices[-1]
            print(f"   ✅ Data available: {len(close_prices)} candles")
            print(f"   💰 Current Price: ${current_price:,.2f}")
            print(f"   📊 Period: {data.index[0]} to {data.index[-1]}")
        except Exception as e:
            return {"error": f"Data processing failed: {e}"}
        
        # ============================================
        # STEP 2: Technical Analysis
        # ============================================
        print("\n📈 STEP 2: Technical Analysis")
        print("-" * 40)
        
        technical_signal = self._get_technical_signal(data)
        print(f"   📊 Direction: {technical_signal['direction'].upper()}")
        print(f"   📊 RSI: {technical_signal['rsi']}")
        print(f"   📊 MACD: {technical_signal['macd']:.4f}")
        
        # ============================================
        # STEP 3: LSTM Prediction
        # ============================================
        print("\n🤖 STEP 3: LSTM Price Prediction")
        print("-" * 40)
        
        try:
            prediction = self.prediction_agent.predict_next_price(close_prices)
            if "error" in prediction:
                print(f"   ⚠️ {prediction['error']}")
                prediction_direction = "neutral"
                prediction_change = 0
                predicted_price = current_price
            else:
                print(f"   ✅ Current: ${prediction['current_price']:,.2f}")
                print(f"   🔮 Predicted: ${prediction['predicted_price']:,.2f}")
                print(f"   📊 Change: {prediction['change_pct']:.2f}%")
                print(f"   🎯 Direction: {prediction['direction'].upper()}")
                prediction_direction = prediction['direction']
                prediction_change = prediction['change_pct']
                predicted_price = prediction['predicted_price']
        except Exception as e:
            print(f"   ⚠️ Prediction failed: {e}")
            prediction_direction = "neutral"
            prediction_change = 0
            predicted_price = current_price
            prediction = {"error": str(e)}
        
        # ============================================
        # STEP 4: Risk Assessment
        # ============================================
        print("\n🛡️ STEP 4: Risk Assessment")
        print("-" * 40)
        
        try:
            risk_params = self.risk_agent.evaluate(current_price, direction="BUY")
            if "error" in risk_params:
                print(f"   ⚠️ {risk_params['error']}")
            else:
                print(f"   💰 Risk Amount: ${risk_params['risk_amount_usd']:,.2f}")
                print(f"   📊 Position Size: {risk_params['position_size_units']:.6f} units")
                print(f"   💵 Position Value: ${risk_params['position_value_usd']:,.2f}")
                print(f"   ⚠️ High Risk Flag: {risk_params['flagged_high_risk']}")
        except Exception as e:
            print(f"   ⚠️ Risk assessment failed: {e}")
            risk_params = {"error": str(e)}
        
        # ============================================
        # STEP 5: Combine Signals
        # ============================================
        print("\n🎯 STEP 5: Combining Signals")
        print("-" * 40)
        
        tech_score = 1 if technical_signal['direction'] == 'bullish' else -1 if technical_signal['direction'] == 'bearish' else 0
        pred_score = 1 if prediction_direction == 'bullish' else -1 if prediction_direction == 'bearish' else 0
        
        risk_penalty = 0
        if isinstance(risk_params, dict) and risk_params.get('flagged_high_risk', False):
            risk_penalty = 0.3
            print(f"   ⚠️ High risk penalty applied: -{risk_penalty*100:.0f}%")
        
        weighted_score = (tech_score * 0.30) + (pred_score * 0.50) - risk_penalty
        
        print(f"   📊 Technical Score: {tech_score} (weight: 30%)")
        print(f"   📊 Prediction Score: {pred_score} (weight: 50%)")
        print(f"   📊 Weighted Score: {weighted_score:.3f}")
        
        if weighted_score > 0.25:
            final_direction = "BUY"
            confidence = min(weighted_score * 100, 95)
        elif weighted_score < -0.25:
            final_direction = "SELL"
            confidence = min(abs(weighted_score) * 100, 95)
        else:
            final_direction = "HOLD"
            confidence = 50 - (abs(weighted_score) * 100)
        
        confidence = max(5, min(95, confidence))
        
        print(f"\n   🎯 FINAL DECISION: {final_direction}")
        print(f"   📊 Confidence: {confidence:.1f}%")
        
        # ============================================
        # STEP 6: Prepare Result
        # ============================================
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "final_decision": final_direction,
            "confidence": round(confidence, 1),
            "weighted_score": round(weighted_score, 3),
            "technical_signal": technical_signal,
            "prediction": prediction if isinstance(prediction, dict) else {"error": str(prediction)},
            "risk_params": risk_params if isinstance(risk_params, dict) else {"error": str(risk_params)},
            "signals": {
                "technical_score": tech_score,
                "prediction_score": pred_score,
                "risk_penalty": round(risk_penalty, 2)
            }
        }
        
        if final_direction in ["BUY", "SELL"] and isinstance(risk_params, dict) and "error" not in risk_params:
            result["position_size"] = risk_params.get("position_size_units", 0)
            result["stop_loss_price"] = risk_params.get("stop_loss_price", 0)
            result["position_value"] = risk_params.get("position_value_usd", 0)
        else:
            result["position_size"] = 0
            result["stop_loss_price"] = 0
            result["position_value"] = 0
        
        log_entry = {
            "timestamp": result["timestamp"],
            "symbol": result["symbol"],
            "decision": result["final_decision"],
            "confidence": result["confidence"],
            "current_price": result["current_price"],
            "predicted_price": result["predicted_price"]
        }
        self.decision_log.append(log_entry)
        
        print(f"\n📝 Decision logged (Total: {len(self.decision_log)} decisions)")
        
        print("\n" + "="*60)
        print(f"✅ SIGNAL GENERATION COMPLETE")
        print(f"   🎯 Final Decision: {final_direction}")
        print(f"   📊 Confidence: {confidence:.1f}%")
        if final_direction in ["BUY", "SELL"]:
            print(f"   📊 Position Size: {result['position_size']:.6f} units")
            print(f"   🛑 Stop-Loss: ${result['stop_loss_price']:,.2f}")
        print("="*60)
        
        return result
    
    def get_decision_log(self):
        """Return all logged decisions (FR-6.4)"""
        return self.decision_log
    
    def save_decision_log(self, filepath="data/decision_log.csv"):
        """Save decision log to CSV (FR-6.4)"""
        if not self.decision_log:
            print("No decisions to save")
            return
        
        df = pd.DataFrame(self.decision_log)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Decision log saved to {filepath}")
        return df


# ============================================
# Test / Main
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 CRYPTOMADSS - COORDINATOR AGENT TEST")
    print("="*60)
    
    # Initialize coordinator with custom settings
    coordinator = CoordinatorAgent(
        account_balance=10000,
        risk_per_trade_pct=1.0,
        stop_loss_pct=2.0
    )
    
    # Generate signal for BTC
    result = coordinator.generate_signal(
        symbol="BTCUSDT",
        interval="1h",
        lookback="7 days ago UTC"
    )
    
    # Print complete result
    print("\n\n📋 COMPLETE RESULT:")
    print("="*60)
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  '*indent}{key}:")
                print_dict(value, indent+1)
            elif isinstance(value, float):
                print(f"{'  '*indent}{key}: {value:,.2f}")
            else:
                print(f"{'  '*indent}{key}: {value}")
    
    print_dict(result)
    
    # Show decision log
    print("\n\n📝 DECISION LOG:")
    print("="*60)
    log = coordinator.get_decision_log()
    if log:
        for entry in log:
            print(f"  {entry['timestamp']} | {entry['symbol']} | {entry['decision']} | Confidence: {entry['confidence']:.1f}%")
    else:
        print("  No decisions logged yet")
    
    print("\n✅ Test Complete!")
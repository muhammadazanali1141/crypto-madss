"""
Technical Analysis Agent
-------------------------
Ye agent price data se technical indicators calculate karta hai:
RSI, MACD, Moving Averages, Bollinger Bands.
Phir in ki basis par Buy/Sell/Hold ka signal deta hai.
"""

import pandas as pd
import ta  # technical analysis library


class TechnicalAgent:
    def add_indicators(self, df):
        """
        df mein 'close', 'high', 'low', 'volume' columns hone chahiye
        """
        df = df.copy()

        # Moving Averages
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
        df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)

        # RSI (Relative Strength Index) - overbought/oversold batata hai
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        # MACD (trend direction aur momentum)
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # Bollinger Bands (volatility)
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()

        return df

    def generate_signal(self, df):
        """
        Latest row ke indicators dekh kar simple rule-based signal deta hai.
        Return: dict with signal aur reasoning
        """
        latest = df.iloc[-1]
        signals = []
        score = 0

        # RSI rule
        if latest["rsi"] < 30:
            signals.append("RSI oversold (<30) -> Bullish signal")
            score += 1
        elif latest["rsi"] > 70:
            signals.append("RSI overbought (>70) -> Bearish signal")
            score -= 1

        # MACD rule
        if latest["macd"] > latest["macd_signal"]:
            signals.append("MACD above signal line -> Bullish momentum")
            score += 1
        else:
            signals.append("MACD below signal line -> Bearish momentum")
            score -= 1

        # Moving Average crossover rule
        if latest["sma_20"] > latest["sma_50"]:
            signals.append("SMA20 above SMA50 -> Uptrend")
            score += 1
        else:
            signals.append("SMA20 below SMA50 -> Downtrend")
            score -= 1

        # Final decision score ke basis par
        if score >= 2:
            decision = "BUY"
        elif score <= -2:
            decision = "SELL"
        else:
            decision = "HOLD"

        return {
            "agent": "TechnicalAgent",
            "decision": decision,
            "score": score,
            "reasoning": signals
        }


if __name__ == "__main__":
    # Quick test dummy data ke sath
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    dummy = pd.DataFrame({
        "close": np.random.normal(100, 5, 100).cumsum() + 1000,
        "high": np.random.normal(105, 5, 100).cumsum() + 1000,
        "low": np.random.normal(95, 5, 100).cumsum() + 1000,
        "volume": np.random.randint(100, 1000, 100)
    }, index=dates)

    agent = TechnicalAgent()
    df_with_indicators = agent.add_indicators(dummy)
    result = agent.generate_signal(df_with_indicators)
    print(result)
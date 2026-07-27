"""
Market Data Agent
------------------
Ye agent Binance API se crypto ka historical/live data fetch karta hai.
API key ki zaroorat nahi hai kyunki hum sirf PUBLIC market data
(price, volume, candles) le rahe hain — ye Binance ka har koi access kar sakta hai.
"""

from binance.client import Client
import pandas as pd


class MarketDataAgent:
    def __init__(self):
        # Public data ke liye API key/secret ki zaroorat nahi hoti
        self.client = Client()

    def get_historical_data(self, symbol="BTCUSDT", interval="1h", lookback="90 day ago UTC"):
        """
        symbol: konsi crypto pair (jaise BTCUSDT, ETHUSDT)
        interval: 1m, 5m, 1h, 4h, 1d waghera
        lookback: kitna purana data chahiye (e.g. '90 day ago UTC', '1 year ago UTC')
        """
        klines = self.client.get_historical_klines(symbol, interval, lookback)

        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(klines, columns=columns)

        # Numeric columns ko float mein convert karna
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Timestamp ko readable date mein convert karna
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Sirf zaroori columns rakhna
        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        df.set_index("open_time", inplace=True)

        return df

    def save_to_csv(self, df, path="data/historical_data.csv"):
        df.to_csv(path)
        print(f"Data saved to {path}")


# Test karne ke liye (agar ye file direct run ki jaye)
if __name__ == "__main__":
    agent = MarketDataAgent()
    data = agent.get_historical_data("BTCUSDT", "1h", "90 day ago UTC")
    print(data.head())
    print(f"\nTotal rows: {len(data)}")
    agent.save_to_csv(data, "../data/historical_data.csv")
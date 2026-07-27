"""
Market Data Agent
------------------
Ye agent Binance API se crypto ka historical/live data fetch karta hai.
API key ki zaroorat nahi hai kyunki hum sirf PUBLIC market data
(price, volume, candles) le rahe hain — ye Binance ka har koi access kar sakta hai.
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import time


class MarketDataAgent:
    def __init__(self):
        # Public data ke liye API key/secret ki zaroorat nahi hoti
        try:
            self.client = Client()
            # Test connection
            self.client.ping()
            print("✅ Binance API connected successfully!")
        except BinanceAPIException as e:
            print(f"⚠️ Binance API Error: {e}")
            print("🔄 Retrying with alternative method...")
            # Alternative: Use requests library directly
            self.client = None
        except Exception as e:
            print(f"⚠️ Connection error: {e}")
            self.client = None

    def get_historical_data(self, symbol="BTCUSDT", interval="1h", lookback="90 day ago UTC"):
        """
        symbol: konsi crypto pair (jaise BTCUSDT, ETHUSDT)
        interval: 1m, 5m, 1h, 4h, 1d waghera
        lookback: kitna purana data chahiye (e.g. '90 day ago UTC', '1 year ago UTC')
        """
        # Agar client None hai toh alternative method use karein
        if self.client is None:
            return self._get_historical_data_alternative(symbol, interval, lookback)
        
        try:
            klines = self.client.get_historical_klines(symbol, interval, lookback)
            return self._process_klines(klines)
        except BinanceAPIException as e:
            print(f"⚠️ Binance API Error: {e}")
            print("🔄 Trying alternative method...")
            return self._get_historical_data_alternative(symbol, interval, lookback)
        except Exception as e:
            print(f"⚠️ Error fetching data: {e}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def _get_historical_data_alternative(self, symbol="BTCUSDT", interval="1h", lookback="90 day ago UTC"):
        """
        Alternative method using direct HTTP requests if binance client fails
        """
        import requests
        import json
        
        try:
            # Binance public API endpoint
            base_url = "https://api.binance.com/api/v3/klines"
            
            # Convert interval and lookback
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
            }
            
            # Calculate limit from lookback
            limit = 500  # Default
            
            params = {
                "symbol": symbol,
                "interval": interval_map.get(interval, "1h"),
                "limit": limit
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                klines = response.json()
                return self._process_klines(klines)
            else:
                print(f"⚠️ Alternative API error: {response.status_code}")
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                
        except Exception as e:
            print(f"⚠️ Alternative method failed: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def _process_klines(self, klines):
        """Process raw klines data into DataFrame"""
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
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        print(f"Data saved to {path}")


# Test karne ke liye (agar ye file direct run ki jaye)
if __name__ == "__main__":
    agent = MarketDataAgent()
    data = agent.get_historical_data("BTCUSDT", "1h", "90 day ago UTC")
    print(data.head())
    print(f"\nTotal rows: {len(data)}")
    agent.save_to_csv(data, "../data/historical_data.csv")
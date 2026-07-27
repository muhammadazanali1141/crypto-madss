import numpy as np
import pickle
import os
import sys
from tensorflow.keras.models import load_model


class PredictionAgent:
    def __init__(self, model_path=None, scaler_path=None, sequence_length=60):
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths
        if model_path is None:
            model_path = os.path.join(current_dir, "models", "lstm_model.h5")
        if scaler_path is None:
            scaler_path = os.path.join(current_dir, "models", "scaler.pkl")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
        
        # Load model and scaler
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.sequence_length = sequence_length
        print("✅ PredictionAgent initialized successfully!")

    def predict_next_price(self, recent_prices: list) -> dict:
        """
        recent_prices: list of the most recent close prices (at least sequence_length long)
        """
        if len(recent_prices) < self.sequence_length:
            return {
                "agent": "PredictionAgent",
                "error": f"Need at least {self.sequence_length} recent prices, got {len(recent_prices)}"
            }

        prices_array = np.array(recent_prices[-self.sequence_length:]).reshape(-1, 1)
        scaled = self.scaler.transform(prices_array)
        X_input = scaled.reshape(1, self.sequence_length, 1)

        scaled_prediction = self.model.predict(X_input, verbose=0)
        predicted_price = self.scaler.inverse_transform(scaled_prediction)[0][0]

        current_price = recent_prices[-1]
        change_pct = ((predicted_price - current_price) / current_price) * 100

        if change_pct > 0.5:
            direction = "bullish"
        elif change_pct < -0.5:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "agent": "PredictionAgent",
            "current_price": round(current_price, 2),
            "predicted_price": round(float(predicted_price), 2),
            "change_pct": round(change_pct, 2),
            "direction": direction,
        }


if __name__ == "__main__":
    # Add parent directory to path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from agents.market_data_agent import MarketDataAgent
        
        print("Fetching market data...")
        market_agent = MarketDataAgent()
        data = market_agent.get_historical_data("BTCUSDT", "1h", "10 days ago UTC")
        recent_prices = data["close"].tolist()
        
        print(f"Fetched {len(recent_prices)} price points")
        
        print("Initializing Prediction Agent...")
        predictor = PredictionAgent()
        result = predictor.predict_next_price(recent_prices)
        
        print("\n📊 Prediction Result:")
        print("=" * 40)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 40)
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        print("Trying to import market_data_agent directly...")
        
        # Try direct import
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from market_data_agent import MarketDataAgent
        
        print("Fetching market data...")
        market_agent = MarketDataAgent()
        data = market_agent.get_historical_data("BTCUSDT", "1h", "10 days ago UTC")
        recent_prices = data["close"].tolist()
        
        print("Initializing Prediction Agent...")
        predictor = PredictionAgent()
        result = predictor.predict_next_price(recent_prices)
        
        print("\n📊 Prediction Result:")
        print("=" * 40)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 40)
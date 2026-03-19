import os
import pandas as pd
from dotenv import load_dotenv

from data.zerodha_fetcher import ZerodhaDataFetcher
from features.technical_indicators import process_features
from env.advanced_trading_env import AdvancedTradingEnvironment
from models.rl_agent import RLAgent
from live_trading.zerodha_executor import ZerodhaExecutor

# Load secrets from .env securely
load_dotenv()

def main():
    print("=== Advanced LSTM-PPO Indian Market Trading Engine ===")
    
    # 1. Check Auth & Data Integration
    if not os.getenv("KITE_API_KEY"):
        print("CRITICAL: .env file not set up. Please copy .env.example to .env and add your Kite API Key.")
        print("Running in DEMO offline mode using simulated historic minute-data...\n")
        raw_data = generate_dummy_data()
    else:
        print("Authenticating with Zerodha Kite Connect...")
        fetcher = ZerodhaDataFetcher()
        # Fetch true intraday 5-minute data logic block
        token = fetcher.get_instrument_token("INFY")
        if token:
            raw_data = fetcher.fetch_historical_data(token, "2023-10-01", "2023-10-30", "5minute")
        else:
            raw_data = generate_dummy_data()
            
    # 2. Process High Frequency Strategy Features
    feature_data = process_features(raw_data)
    print(f"Features mapped with seq lengths: {len(feature_data)}")
    
    # 3. Setup Intraday Environment with Real Indian Taxes & Slippage
    env = AdvancedTradingEnvironment(feature_data)
    agent = RLAgent(env=env)
    
    print("\n[SUCCESS] Advanced Agent Initialized with Recurrent LSTM Memory Cells.")
    print("To begin PPO sequence training on Indian Market data, run `agent.train()`.\n")
    print("A Streamlit dashboard viewer is accessible via: `streamlit run dashboard/app.py`")
    
def generate_dummy_data():
    """Generates synthetic intraday data if Kite auth fails for offline testing."""
    import numpy as np
    dates = pd.date_range("2023-10-01 09:15", "2023-10-05 15:30", freq="5min")
    df = pd.DataFrame(index=dates, columns=["Open", "High", "Low", "Close", "Volume"])
    df["Close"] = np.cumsum(np.random.randn(len(dates))) + 1500
    df["Open"] = df["Close"] + np.random.randn(len(dates)) * 2
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.random.rand(len(dates)) * 5
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.random.rand(len(dates)) * 5
    df["Volume"] = np.random.randint(1000, 50000, size=len(dates))
    return df

if __name__ == "__main__":
    main()

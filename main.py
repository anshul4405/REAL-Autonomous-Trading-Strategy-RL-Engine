import os
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from data.fyers_fetcher import FyersDataFetcher
from features.technical_indicators import process_features
from env.advanced_trading_env import AdvancedTradingEnvironment
from models.rl_agent import RLAgent
from backtesting.engine import BacktestEngine

load_dotenv()

def main():
    print("=== Advanced LSTM-PPO Indian Market Trading Engine ===")
    
    # 1. Fetch High-Resolution Intraday Data
    if os.getenv("FYERS_APP_ID"):
        print("Authenticating with Fyers API for Live Indian Data...")
        fetcher = FyersDataFetcher()
        raw_data = fetcher.fetch_historical_data()
    else:
        print("No Fyers API keys found. Fetching FREE High-Frequency Intraday NSE Data via YFinance...")
        print("Downloading 60 days of 5-Minute Data for Reliance Industries (RELIANCE.NS)...")
        # Yahoo Finance provides 60 days of free 5-minute data which is perfect for training!
        raw_data = yf.download("RELIANCE.NS", period="60d", interval="5m", progress=False)
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.droplevel(1)
        raw_data.dropna(inplace=True)
            
    # 2. Process High Frequency Strategy Features
    print("Engineering Technical Indicators (MACD, RSI, Bollinger Bands)...")
    feature_data = process_features(raw_data)
    print(f"Usable 5-minute Intraday Candles generated: {len(feature_data)}")
    
    # Splitting Data
    train_size = int(len(feature_data) * 0.8)
    train_data = feature_data.iloc[:train_size]
    test_data = feature_data.iloc[train_size:]
    
    # 3. Setup Intraday Environment with Normalization for *Better Predictions*
    print("\n--- Initializing Recurrent LSTM-PPO Network ---")
    train_env = AdvancedTradingEnvironment(train_data)
    agent = RLAgent(env=train_env)
    
    print("\n[TRAINING] Commencing Neural Network Training for 15,000 steps...")
    print("The agent will now learn to actively trade the Reliance 5-minute chart!")
    agent.train(total_timesteps=15000)
    agent.save_model("models/lstm_reliance_agent")
    
    # 4. Backtesting on completely unseen Testing Data
    print("\n--- Evaluating LSTM Strategy on Unseen Test Data ---")
    test_env = AdvancedTradingEnvironment(test_data)
    backtest_engine = BacktestEngine(env=test_env, model=agent)
    
    # The LSTM model evaluates trades step-by-step
    results = backtest_engine.run_backtest()
    baseline = backtest_engine.baseline_buy_and_hold()
    
    print("\n[RESULTS] RL Agent Performance (Factoring in Indian Taxes & Slippage):")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")
        
    print("\n[RESULTS] Traditional Buy & Hold Performance:")
    for k, v in baseline.items():
        print(f"  {k}: {v:.2f}")

    print("\nA Streamlit dashboard viewer is accessible via: `streamlit run dashboard/app.py`")

if __name__ == "__main__":
    main()

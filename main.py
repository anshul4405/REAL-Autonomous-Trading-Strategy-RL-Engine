import os
import pandas as pd

from data.data_fetcher import HistoricalDataFetcher
from features.technical_indicators import process_features
from env.trading_env import TradingEnvironment
from models.rl_agent import RLAgent
from backtesting.engine import BacktestEngine

def main():
    print("=== Autonomous Trading System using RL ===")
    
    # 1. Data Collection
    print("\n--- 1. Data Collection ---")
    fetcher = HistoricalDataFetcher(ticker="AAPL", start_date="2020-01-01", end_date="2023-01-01")
    raw_data = fetcher.fetch_data(save_csv=True)
    if raw_data.empty:
        print("Failed to fetch data. Ensure yfinance is installed and online.")
        return
        
    # 2. Feature Engineering
    print("\n--- 2. Feature Engineering ---")
    feature_data = process_features(raw_data)
    print(f"Features prepared. Usable Datapoints: {len(feature_data)}")
    
    # Split Data (Train: 80%, Test: 20%)
    train_size = int(len(feature_data) * 0.8)
    train_data = feature_data.iloc[:train_size]
    test_data = feature_data.iloc[train_size:]
    
    # 3. RL Environment & Agent
    print("\n--- 3. RL Environment Setup ---")
    train_env = TradingEnvironment(train_data)
    agent = RLAgent(env=train_env)
    
    print("Agent initialized successfully. (Training is disabled in the main demo script)")
    
    # Optional: Un-comment to actually train
    # print("Training Agent...")
    # agent.train(total_timesteps=10000)
    # agent.save_model("models/ppo_aapl_agent")
    
    # 4. Backtesting
    print("\n--- 4. Backtesting Setup ---")
    test_env = TradingEnvironment(test_data)
    backtest_engine = BacktestEngine(env=test_env, model=agent.model)
    
    baseline_metrics = backtest_engine.baseline_buy_and_hold()
    print("Baseline (Buy & Hold) Performance over Test Period:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.2f}")
        
    print("\nSetup complete! To visualize the trading interface, run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()

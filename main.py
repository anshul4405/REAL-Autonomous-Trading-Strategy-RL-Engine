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
    print("=== Universal LSTM-PPO Indian Market Trading Engine ===")
    
    # Core Strategy: Generalize the AI across the Top 5 NIFTY50 heavyweight stocks!
    # By training sequentially, the LSTM learns universal price action rules instead of memorizing just one stock.
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    
    print(f"\n--- Universal Free Pipeline: Fetching {len(tickers)} Heavyweight Assets via Yahoo Finance ---")
    
    # 1. Initialize RL Framework strictly on the first asset to dictate Neural Architecture shape
    first_ticker = tickers[0]
    print(f"\n[AI SEQUENCE 1] Processing foundational base asset: {first_ticker}")
    
    raw_data = yf.download(first_ticker, period="60d", interval="5m", progress=False)
    if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.droplevel(1)
            
    first_features = process_features(raw_data)
    train_size = int(len(first_features) * 0.8)
    first_train_data = first_features.iloc[:train_size]
    
    # Booting Environment and AI
    base_env = AdvancedTradingEnvironment(first_train_data)
    agent = RLAgent(env=base_env)
    
    print(f"Commencing Initial Neural Sequence Training (10,000 steps)...")
    agent.train(total_timesteps=10000)
    
    # 2. Iterate flawlessly through the rest of the market universe!
    sequence_num = 2
    for ticker in tickers[1:]:
        print(f"\n[AI SEQUENCE {sequence_num}] Migrating Universal Memory to: {ticker}")
        
        curr_raw = yf.download(ticker, period="60d", interval="5m", progress=False)
        if isinstance(curr_raw.columns, pd.MultiIndex):
            curr_raw.columns = curr_raw.columns.droplevel(1)
            
        curr_features = process_features(curr_raw)
        curr_train = curr_features.iloc[:int(len(curr_features) * 0.8)]
        
        # HOT-SWAP the Environment! Keep the original standard deviation weights locked in.
        new_env = AdvancedTradingEnvironment(curr_train)
        agent.set_env(new_env)
        
        print(f"Resuming Universal Engine Training (10,000 steps)...")
        agent.train(total_timesteps=10000)
        sequence_num += 1
        
    print("\nSaving the Fully Generalized Master Universal Network!")
    agent.save_model("models/lstm_universal_nifty_master")
    
    # 3. Evaluate the highly rigorous model against entirely unseen data context
    test_ticker = tickers[-1]
    print(f"\n--- Evaluating Universal LSTM Intelligence strictly on {test_ticker} Test Dataset ---")
    
    # Slice the unseen evaluation chunk out of the final processed token loop 
    test_data = curr_features.iloc[int(len(curr_features) * 0.8):]
    test_env = AdvancedTradingEnvironment(test_data)
    
    agent.set_env(test_env)
    agent.env.training = False  # Disable statistical mutation for accurate live backtesting!
    
    backtest_engine = BacktestEngine(env=test_env, model=agent)
    results = backtest_engine.run_backtest()
    baseline = backtest_engine.baseline_buy_and_hold()
    
    print("\n[RESULTS] Universal AI Generalization Performance (Post Indian Slippage/Taxes):")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")
        
    print("\n[RESULTS] Baseline Traditional Holding metric over test phase:")
    for k, v in baseline.items():
        print(f"  {k}: {v:.2f}")

    print("\nA Streamlit realtime interactive interface can be launched via: `streamlit run dashboard/app.py`")

if __name__ == "__main__":
    main()

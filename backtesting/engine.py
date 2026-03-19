import numpy as np
import pandas as pd

class BacktestEngine:
    def __init__(self, env, model):
        self.env = env
        self.model = model
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252 * 375) # Approximately 375 minutes in an Indian trading day

    def calculate_max_drawdown(self, portfolio_values):
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd

    def run_backtest(self):
        """Evaluates the LSTM model over the provided environment."""
        obs, _ = self.env.reset()
        portfolio_values = [self.env.net_worth]
        
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        
        while not done:
            # We explicitly pass lstm context to the agent
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_starts = np.zeros((1,), dtype=bool)
            
            portfolio_values.append(self.env.net_worth)
            done = terminated or truncated
            
        returns = pd.Series(portfolio_values).pct_change().dropna()
        total_return = (portfolio_values[-1] - self.env.initial_balance) / self.env.initial_balance
        sharpe = self.calculate_sharpe_ratio(returns)
        max_dd = self.calculate_max_drawdown(portfolio_values)
        
        win_steps = len(returns[returns > 0])
        win_rate = win_steps / len(returns) if len(returns) > 0 else 0

        return {
            "Total Return (%)": total_return * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100,
            "Win Rate (%)": win_rate * 100
        }

    def baseline_buy_and_hold(self):
        """Calculates buy and hold performance as a baseline."""
        initial_price = self.env.df.iloc[0]['Close']
        final_price = self.env.df.iloc[-1]['Close']
        total_return = (final_price - initial_price) / initial_price
        
        prices = self.env.df['Close'].values
        returns = pd.Series(prices).pct_change().dropna()
        sharpe = self.calculate_sharpe_ratio(returns)
        max_dd = self.calculate_max_drawdown(prices)
        
        return {
            "Total Return (%)": total_return * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100
        }

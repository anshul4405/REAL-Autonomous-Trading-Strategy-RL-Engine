import time
import numpy as np
import pandas as pd
from risk_management.controls import RiskManager

class PaperTrader:
    """Simulates live continuous execution with LSTM internal state tracking."""
    
    def __init__(self, model, data_fetcher, initial_balance=100000.0):
        self.model = model
        self.data_fetcher = data_fetcher
        self.balance = initial_balance
        self.shares_held = 0
        self.risk_manager = RiskManager()
        self.portfolio_history = []
        
        self.avg_entry_price = 0.0
        
        # LSTM specific context states
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        
    def simulate_tick(self, latest_observation, current_price):
        """Simulates one step of live trading maintaining state across ticks."""
        
        if self.shares_held > 0 and self.avg_entry_price > 0:
            exit_decision = self.risk_manager.check_exit_conditions(
                entry_price=self.avg_entry_price, 
                current_price=current_price
            )
            if exit_decision == "SELL":
                revenue = self.shares_held * current_price
                print(f"Risk Management Triggered SELL at {current_price:.2f}. Revenue: ₹{revenue:.2f}")
                self.balance += revenue
                self.shares_held = 0
                self.avg_entry_price = 0.0
                return
                
        # Ask Model for Prediction using previous states
        action, self.lstm_states = self.model.predict(
            latest_observation, 
            lstm_states=self.lstm_states, 
            episode_start=self.episode_starts
        )
        # Toggle False after the first step since sequence is continuous
        self.episode_starts = np.zeros((1,), dtype=bool)
        
        if action == 1 and self.risk_manager.can_trade(): # Buy
            shares_to_buy = self.risk_manager.get_position_size(self.balance, current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                total_cost = (self.shares_held * self.avg_entry_price) + cost
                self.shares_held += shares_to_buy
                self.avg_entry_price = total_cost / self.shares_held
                self.balance -= cost
                self.risk_manager.trades_today += 1
                print(f"BOUGHT {shares_to_buy} shares at ₹{current_price:.2f}")
                
        elif action == 2 and self.shares_held > 0: # Sell
            revenue = self.shares_held * current_price
            self.balance += revenue
            self.shares_held = 0
            self.avg_entry_price = 0.0
            self.risk_manager.trades_today += 1
            print(f"SOLD shares at ₹{current_price:.2f}")
            
        net_worth = self.balance + (self.shares_held * current_price)
        self.portfolio_history.append(net_worth)
        print(f"Current Net Worth: ₹{net_worth:.2f}")

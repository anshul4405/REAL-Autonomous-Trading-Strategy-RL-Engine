class RiskManager:
    """Handles risk limits and position sizing rules to prevent catastrophic losses."""
    
    def __init__(self, max_capital_allocation=0.02, stop_loss_pct=0.05, take_profit_pct=0.10, max_trades_per_day=5):
        # Only risk 2% of total capital per trade
        self.max_capital_allocation = max_capital_allocation
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trades_per_day = max_trades_per_day
        self.trades_today = 0
        
    def reset_daily_limits(self):
        """Reset limits at the start of a trading day."""
        self.trades_today = 0
        
    def can_trade(self):
        """Check if trading frequency limit is reached."""
        return self.trades_today < self.max_trades_per_day
        
    def get_position_size(self, current_balance, current_price):
        """Calculate max shares to buy based on max capital allocation."""
        allocatable = current_balance * self.max_capital_allocation
        return int(allocatable // current_price)
        
    def check_exit_conditions(self, entry_price, current_price):
        """Returns action to take: 'SELL', 'HOLD' based on SL/TP settings."""
        if current_price <= entry_price * (1 - self.stop_loss_pct):
            return "SELL" # Stop Loss Hit
        elif current_price >= entry_price * (1 + self.take_profit_pct):
            return "SELL" # Take Profit Hit
        return "HOLD"

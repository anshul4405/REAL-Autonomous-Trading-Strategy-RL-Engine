import time
import logging

class ZerodhaExecutor:
    """Translates Agent actions into real NSE Market/Limit Orders via Kite Connect."""
    def __init__(self, kite, trading_symbol="INFY", exchange="NSE"):
        self.kite = kite
        self.trading_symbol = trading_symbol
        self.exchange = exchange
        self.active_position = 0 # Track net quantity
        logging.basicConfig(level=logging.INFO)
        
    def _place_order(self, transaction_type, quantity, order_type="MARKET", price=None):
        try:
            # Product MIS specifies Margin Intraday Squareoff required for day trading
            order_id = self.kite.place_order(
                tradingsymbol=self.trading_symbol,
                exchange=self.exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                variety=self.kite.VARIETY_REGULAR,
                order_type=order_type,
                product=self.kite.PRODUCT_MIS,
                validity=self.kite.VALIDITY_DAY,
                price=price
            )
            logging.info(f"Order Live Success: {order_id} | {transaction_type} {quantity} @ {order_type}")
            return order_id
        except Exception as e:
            logging.error(f"Live Execution Error (Check API Auth/Limits): {e}")
            return None

    def execute_agent_action(self, action, recommended_qty):
        """
        Receives action [0=HOLD, 1=BUY, 2=SELL] from LSTM Agent
        and automatically handles Intraday precision sizing.
        """
        if action == 1 and self.active_position == 0:
            # Entering Long Sequence
            order = self._place_order(self.kite.TRANSACTION_TYPE_BUY, recommended_qty)
            if order:
                self.active_position += recommended_qty
                
        elif action == 2 and self.active_position > 0:
            # Exiting Long Sequence
            order = self._place_order(self.kite.TRANSACTION_TYPE_SELL, self.active_position)
            if order:
                self.active_position = 0
                
        # Action 0 = HOLD, do nothing
